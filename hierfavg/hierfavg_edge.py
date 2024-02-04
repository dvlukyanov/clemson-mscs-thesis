import os
import sys
sys.path.insert(0, '../')
import yaml
import json
from util.message import Message
from util.json_encoder import MessageEncoder
import pickle as pkl
from enum import Enum
from model.simple_cnn import SimpleCNN
import torch
from torch.utils.data import DataLoader
from util.dataload import ImageDataset, transform
import asyncio

# init config

with open('hierfavg_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# classes

ModelState = Enum('ModelState', ['WAITING', 'UPDATED'])


class Cloud():
    def __init__(self, address, port):
        self.address = address
        self.port = port
        self.model = None

    async def init(self):
        self.reader, self.writer = await asyncio.open_connection(self.address, self.port)
        print(f"Edge connected to cloud {self.address}:{self.port}")

    async def notify(self, edge_id, model_name, iteration):
        message = Message(edge_id, model_name, iteration)
        body = json.dumps(message, cls=MessageEncoder)
        self.writer.write(body.encode('utf-8'))
        await self.writer.drain()
        print(f"Cloud notified by edge")

    async def listen(self):
        message = None
        while message is None or message.startswith("model_cloud") is False:
            message = (await self.reader.read(1024)).decode('utf-8')
        print(f"Received from cloud: {message}")
        return message


class Client:
    def __init__(self, id, writer):
        self.id = id
        self.writer = writer
        self.status = ModelState.WAITING
        self.model_name = None
        self.model = None


class Edge():
    def __init__(self, id, address, port, cloud):
        self.id = id
        self.iteration = 1
        self.cloud = cloud
        self.address = address
        self.port = port
        self.clients = []

    def getClient(self, client_id):
        return next((client for client in self.clients if client.id == client_id), None)
    
    def addClient(self, id, writer):
        client = Client(id, writer)
        self.clients.append(client)
        print(f"New client is registered: {client}")
        return client
    
    def updateClients(self, status):
        for client in self.clients:
            client.status = status
            print(f"Client {client.id} status updated to {status}")

    def notifyClients(self, model_name):
        for client in self.clients:
            client.writer.write(model_name.encode('utf-8'))
            print(f"Edge {self.id} notified client {client.id}")

    async def handle_client(self, reader, writer):
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

                client = self.getClient(message.sender_id)
                if client is None:
                    client = self.addClient(message.sender_id, writer)
                client.model_name = message.model_name
                client.status = ModelState.UPDATED

                writer.write("Confirmed".encode('utf-8'))
                await writer.drain()

                if len(self.clients) == config['client']['qnt'] and all(client.status == ModelState.UPDATED for client in self.clients):
                    print("All clients trained models")
                    self.loadModels(config['storage']['model']['path'])
                    self.aggregate()
                    self.test()
                    model_name = self.saveModel()
                    self.incrementIteration()
                    
                    self.updateClients(ModelState.WAITING)
                    if (self.iteration - 1) % config['edge']['training']['iterations_per_round'] == 0:
                        await self.cloud.notify(self.id, model_name, self.iteration)
                        model_name = await self.cloud.listen()

                    self.notifyClients(model_name)

        except asyncio.CancelledError:
            pass

        finally:
            print(f"Connection with {address} closed.")
            writer.close()

    async def init(self):
        server = await asyncio.start_server(self.handle_client, self.address, self.port + self.id)
        print(f"Edge {self.id} started")
        return server
    
    def loadModel(self, path):
        model = SimpleCNN()
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model {path} loaded")
        return model

    def loadModels(self, model_folder):
        for client in self.clients:
            path = os.path.join(model_folder, client.model_name)
            client.model = self.loadModel(path)

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
        print(f'Accuracy of the model on the {total} test images at {self.iteration} iteration: {100 * correct / total} %')

    def aggregate(self):
        states = [client.model.state_dict() for client in self.clients]
        self.model = SimpleCNN()
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model aggregated")

    def saveModel(self):
        model_name = config['storage']['model']['name']['edge'].format(self.iteration, self.id)
        torch.save(self.model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
        print(f"Model {model_name} saved")
        return model_name
    
    def incrementIteration(self):
        self.iteration += 1
        print(f"Iteration incremented to {self.iteration}")
    

async def main():
    cloud = Cloud(config['cloud']['address'], config['cloud']['port'])
    await cloud.init()
    edge = Edge(int(sys.argv[1]), config['edge']['address'], config['edge']['port'], cloud)
    server = await edge.init()
    async with server:
            await server.serve_forever()
            print("Edge stopped")

if __name__ == '__main__':
    asyncio.run(main())