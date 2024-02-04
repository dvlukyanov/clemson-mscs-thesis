import sys
sys.path.insert(0, '../')
import os
import random
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import asyncio
import yaml
from util.message import Message
from response import Response
import json
from util.json_encoder import MessageEncoder
from enum import Enum
from model.simple_cnn import SimpleCNN
from util.dataload import ImageDataset, transform
from copy import deepcopy


# init config

with open('fedbuff_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# classes

class Server():
    def __init__(self, address, port):
        self.address = address
        self.port = port

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.address, self.port)

    async def notify(self, client_id, model_name, iteration):
        message = Message(client_id, model_name, iteration)
        body = json.dumps(message, cls=MessageEncoder)
        self.writer.write(body.encode('utf-8'))
        await self.writer.drain()
        print(f"Client {client_id} notified server: {body}")

    async def listen(self):
        body = None
        while body is None:
            body = (await self.reader.read(100)).decode('utf-8')
        body = json.loads(body)
        response = Response(**body)
        print(f"Received from server: {response}")
        return response


class Client():
    def __init__(self, id, blueprint, device):
        self.id = id
        self.device = device
        self.blueprint = blueprint
        self.iteration = 1
        print(f"Client {self.id} initialized with {self.blueprint.__class__.__name__} model")

    async def init(self, address, port):
        self.server = Server(address, port)
        await self.server.connect()
        print(f"Client {self.id} connected to server {self.server.address}:{self.server.port}")

    async def run(self):
        model_name = None
        while True:
            model_name = model_name if model_name else self.find_latest_model()
            self.load_model(model_name)
            self.train_model()
            model_name = self.save_model()
            await self.server.notify(self.id, model_name, self.iteration)
            response = await self.server.listen()
            if response.reload:
                model_name = response.model_name
            self.increment_iteration()

    def find_latest_model(self):
        model_names = os.listdir(config['storage']['model']['path'])
        model_names = [name for name in model_names if name.startswith(config['storage']['model']['name']['server'].split('_')[0])]
        model_names = sorted(model_names, key=lambda name: int((name.split('_')[-1]).split('.')[0]))
        print(f"Latest model: {model_names[-1]}")
        return model_names[-1]

    def load_model(self, model_name):
        model_path = os.path.join(config['storage']['model']['path'], model_name)
        model = deepcopy(self.blueprint).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model = model
        print(f"Model {model_name} is loaded to client {self.id}")

    def train_model(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        data_file = open(os.path.join(config['storage']['data']['path'], f'partition_{self.id}.pkl'), 'rb')
        data = pkl.load(data_file)
        
        train_dataset = ImageDataset(data['train'], config['seed'], transform)
        test_dataset = ImageDataset(data['test'], config['seed'], transform)
        
        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        for epoch in range(config['client']['training']['epochs_per_iteration']):
            running_loss = 0.0

            correct_predictions_train = 0
            total_samples_train = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
        
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted_train = torch.max(outputs, 1)
                total_samples_train += labels.size(0)
                correct_predictions_train += (predicted_train == labels).sum().item()

                running_loss += loss.item()

            accuracy_train = 100 * correct_predictions_train / total_samples_train

            correct_predictions_test = 0
            total_samples_test = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = self.model(inputs)
                    _, predicted_test = torch.max(outputs, 1)
                    total_samples_test += labels.size(0)
                    correct_predictions_test += (predicted_test == labels).sum().item()

            accuracy_test = 100 * correct_predictions_test / total_samples_test

            print(f'Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

            scheduler.step()
            
        print(f'Iteration {self.iteration} is completed')

    def save_model(self):
        model_name = config['storage']['model']['name']['client'].format(self.iteration, self.id)
        model_path = os.path.join(config['storage']['model']['path'], model_name)
        torch.save(self.model.state_dict(), model_path)
        print(f'Model {model_path} saved')
        return model_name
    
    def increment_iteration(self):
        self.iteration += 1
        print(f"Iteration incremented to {self.iteration}")


# launch

async def main():
    try:
        client = Client(int(sys.argv[1]), SimpleCNN(), device)
        await client.init(config['server']['address'], config['server']['port'])
        await client.run()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    asyncio.run(main())