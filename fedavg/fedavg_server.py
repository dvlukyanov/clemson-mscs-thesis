import sys
sys.path.insert(0, '../')
import os
import random
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import asyncio
import yaml
from enum import Enum
from model.simple_cnn import SimpleCNN
from util.dataload import ImageDataset, transform


# init config

with open('fedavg_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# init base model
    
model = SimpleCNN().to(device)


# save base model

def save_model(model, iteration):
    model_name = config['storage']['model']['name']['server'].format(iteration)
    torch.save(model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
    print(f"Model {model_name} saved")
    return model_name

save_model(model, 0)


# classes

ModelState = Enum('ModelState', ['WAITING', 'UPDATED'])


class Client:
    def __init__(self, id, writer):
        self.id = id
        self.writer = writer
        self.status = ModelState.WAITING
        self.model = None


class Control:
    def __init__(self):
        self.clients = []
        self.iteration = 0

    def getById(self, id):
        return next((client for client in self.clients if client.id == id), None)
    
    def add(self, id, writer):
        client = Client(id, writer)
        self.clients.append(client)

    def list(self):
        return self.clients

    def updateStatus(self, id, status):
        client = next((client for client in self.clients if client.id == id), None)
        if client is None:
            raise Exception(f"Client {id} not found")
        client.status = status

    def updateStatuses(self, status):
        for client in self.clients:
            client.status = status

    def loadModels(self, model_folder):
        for client in self.clients:
            path = os.path.join(model_folder, config['storage']['model']['name']['client'].format(self.iteration, client.id))
            client.model = SimpleCNN()
            client.model.load_state_dict(torch.load(path))
            client.model.eval()

    def aggregateModels(self):
        states = [client.model.state_dict() for client in self.clients]
        model_new = SimpleCNN()
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        model_new.load_state_dict(state)
        model_new.eval()
        return model_new

    def notifyClients(self, model_name):
        for client in self.clients:
            client.writer.write(model_name.encode('utf-8'))

    def incrementIteration(self):
        self.iteration += 1
    
    def test(self, model):
        data_file = open(os.path.join(config['storage']['data']['path'], 'test.pkl'), 'rb')
        data = pkl.load(data_file)
        test_dataset = ImageDataset(data, config['seed'], transform)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {total} test images at {self.iteration} iteration: {100 * correct / total} %')


control = Control()


# listen to clients

async def handle_client(reader, writer):
    address = writer.get_extra_info('peername')
    print(f"Connection from {address}")

    try:
        while True:
            data = await reader.read(100)
            if not data:
                break

            client_id = data.decode('utf-8')
            print(f"Received from {address}: {client_id}")

            client = control.getById(client_id)
            if client is None:
                control.add(client_id, writer)
                client = control.getById(client_id)
                print(f"New client is registered: {client}")
            control.updateStatus(client_id, ModelState.UPDATED)

            writer.write("Confirmed".encode('utf-8'))
            await writer.drain()

            if len(control.list()) == config['client']['qnt'] and all(client.status == ModelState.UPDATED for client in control.list()):
                print("All clients trained models")
                control.loadModels(config['storage']['model']['path'])
                model = control.aggregateModels()
                control.test(model)
                control.incrementIteration()
                model_name = save_model(model, control.iteration)
                
                control.updateStatuses(ModelState.WAITING)
                control.notifyClients(model_name)

    except asyncio.CancelledError:
        pass

    finally:
        print(f"Connection with {address} closed.")
        writer.close()


async def main():
    server = await asyncio.start_server(handle_client, config['server']['address'], config['server']['port'])
    print("Server started")

    async with server:
        await server.serve_forever()
        print("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())