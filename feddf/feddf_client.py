import sys
sys.path.insert(0, '../')
import os
import random
import yaml
from util.message import Message
import json
from util.json_encoder import MessageEncoder
import pickle as pkl
from model.model_factory import ModelFactory
from util.dataload import ImageDataset, transform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import asyncio

# init config

with open('feddf_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(device)

# classes

class Server():
    def __init__(self, address, port):
        self.address = address
        self.port = port
        pass

    async def init(self):
        self.reader, self.writer = await asyncio.open_connection(self.address, self.port)
        print(f"Client connected to server {self.address}:{self.port}")

    async def notify(self, client_id, model_type, model_name, iteration):
        message = Message(sender_id=client_id, model_type=model_type, model_name=model_name, iteration=iteration)
        body = json.dumps(message, cls=MessageEncoder)
        self.writer.write(body.encode('utf-8'))
        await self.writer.drain()
        print(f"Server notified by client {client_id}: {body}")


class Client():
    def __init__(self, id, server, device):
        self.id = id
        self.device = device
        self.model = None
        self.model_type = None
        self.iteration = 1
        self.server = server
        print(f"Client {self.id} initialized")

    def load_model(self, model_type=None, model_name=None):
        model = ModelFactory.create(model_type, self.device)
        if model_name:
            model_path = os.path.join(config['storage']['model']['path'], model_name)
            model.load_state_dict(torch.load(model_path))
            model.eval()
        self.model = model
        self.model_type = model_type
        print(f"Model {model_name if model_name else model_type} is loaded to client {self.id}")

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
            
        print(f'Iteration is completed')

    def save_model(self):
        model_name = config['storage']['model']['name']['client'].format(self.iteration, self.id)
        model_path = os.path.join(config['storage']['model']['path'], model_name)
        torch.save(self.model.state_dict(), model_path)
        print(f'Model {model_path} saved')
        return model_name
    
    async def listen(self):
        message = None
        while message is None or (message.startswith("model_client") is False and message.startswith("model_client") is False):
            message = (await self.server.reader.read(100)).decode('utf-8')
        print(f"Received from server: {message}")
        return message

    def increment_iteration(self):
        self.iteration += 1


async def main():
    try:
        server = Server(config['server']['address'], config['server']['port'])
        await server.init()
        client = Client(sys.argv[1], server, device)
        server_model_name = None
        while True:
            client.load_model(sys.argv[2], server_model_name)
            client.train_model()
            model_name = client.save_model()
            await client.server.notify(client.id, client.model_type, model_name, client.iteration)
            server_model_name = await client.listen()
            client.increment_iteration()
    except Exception as e:
        print(f"Error connecting to the server: {e}")


if __name__ == '__main__':
    asyncio.run(main())