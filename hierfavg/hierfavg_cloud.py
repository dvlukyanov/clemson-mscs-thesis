import os
import sys
sys.path.insert(0, '../')
from enum import Enum
import yaml
from message import Message
import json
from util.json_encoder import MessageEncoder
import pickle as pkl
from model.simple_cnn import SimpleCNN
import torch
from torch.utils.data import DataLoader
from util.dataload import ImageDataset, transform
import asyncio


# init config

with open('hierfavg_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# init base model
    
model = SimpleCNN().to(device)


# save base model

def save_model(model, round):
    model_name = config['storage']['model']['name']['cloud'].format(round)
    torch.save(model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
    print(f"Model {model_name} saved")
    return model_name

save_model(model, 0)


# classes

ModelState = Enum('ModelState', ['WAITING', 'UPDATED'])

class Edge:
    def __init__(self, id, writer):
        self.id = id
        self.writer = writer
        self.status = ModelState.WAITING
        self.model_name = None
        self.model = None


class Cloud():
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.model = None
        self.round = 1
        self.edges = []

    async def init(self):
        server = await asyncio.start_server(self.handle_edges, self.address, self.port)
        print(f"Cloud server started")
        return server
    
    def getEdge(self, edge_id):
        return next((edge for edge in self.edges if edge.id == edge_id), None)
    
    def addEdge(self, id, writer):
        edge = Edge(id, writer)
        self.edges.append(edge)
        print(f"New edge is registered: {edge}")
        return edge
    
    def updateEdges(self, status):
        for edge in self.edges:
            edge.status = status

    def notifyEdges(self, model_name):
        for edge in self.edges:
            edge.writer.write(model_name.encode('utf-8'))
    
    async def handle_edges(self, reader, writer):
        address = writer.get_extra_info('peername')
        print(f"Connection from {address}")
        try:
            while True:
                data = await reader.read(100)
                if not data:
                    break
                body = json.loads(data.decode('utf-8'))
                message = Message(**body)
                print(f"Received from {address}: {message}")

                edge = self.getEdge(message.sender_id)
                if edge is None:
                    edge = self.addEdge(message.sender_id, writer)
                edge.model_name = message.model_name
                edge.status = ModelState.UPDATED

                writer.write("Confirmed".encode('utf-8'))
                await writer.drain()

                if len(self.edges) == config['edge']['qnt'] and all(edge.status == ModelState.UPDATED for edge in self.edges):
                    print("All edges aggregated models")
                    self.loadModels(config['storage']['model']['path'])
                    self.aggregate()
                    self.test()
                    model_name = self.saveModel()
                    self.incrementRound()
                
                    self.updateEdges(ModelState.WAITING)
                    self.notifyEdges(model_name)
        
        except asyncio.CancelledError:
            pass

        finally:
            print(f"Connection with {address} closed.")
            writer.close()

    def loadModel(self, path):
        model = SimpleCNN()
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model {path} loaded")
        return model

    def loadModels(self, model_folder):
        for edge in self.edges:
            path = os.path.join(model_folder, edge.model_name)
            edge.model = self.loadModel(path)

    def aggregate(self):
        states = [edge.model.state_dict() for edge in self.edges]
        self.model = SimpleCNN()
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        self.model.load_state_dict(state)
        self.model.eval()

    def test(self):
        data_file = open(os.path.join(config['storage']['data']['path'], 'test.pkl'), 'rb')
        data = pkl.load(data_file)
        test_dataset = ImageDataset(data, config['seed'], transform)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {total} test images at {self.round} round: {100 * correct / total} %')

    def saveModel(self):
        model_name = config['storage']['model']['name']['cloud'].format(self.round)
        torch.save(self.model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
        print(f"Model {model_name} saved")
        return model_name
    
    def incrementRound(self):
        self.round += 1


async def main():
    cloud = Cloud(config['cloud']['address'], config['cloud']['port'])
    server = await cloud.init()
    async with server:
            await server.serve_forever()
            print("Cloud stopped")


if __name__ == "__main__":
    asyncio.run(main())