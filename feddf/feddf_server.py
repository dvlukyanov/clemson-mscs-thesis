import os
import sys
sys.path.insert(0, '../')
import random
from enum import Enum
import yaml
from util.message import Message
import json
from util.json_encoder import MessageEncoder
import pickle as pkl
from model.model_factory import ModelFactory
from model.model_enum import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util.dataload import ImageDataset, transform
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


# classes

ModelState = Enum('ModelState', ['WAITING', 'UPDATED'])

class Client:
    def __init__(self, id, writer):
        self.id = id
        self.writer = writer
        self.status = ModelState.WAITING
        self.model_type = None
        self.model_name = None
        self.model = None


class Server():
    def __init__(self, address, port, device):
        self.address = address
        self.port = port
        self.device = device
        self.model = None
        self.round = 1
        self.clients = []

    async def init(self):
        server = await asyncio.start_server(self.handle_clients, self.address, self.port)
        print(f"Server started")
        return server
    
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
    
    async def handle_clients(self, reader, writer):
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
                client.model_type = message.model_type
                client.model_name = message.model_name
                client.status = ModelState.UPDATED

                writer.write("Confirmed".encode('utf-8'))
                await writer.drain()

                if len(self.clients) == config['client']['qnt'] and all(client.status == ModelState.UPDATED for client in self.clients):
                    print("All clients trained models")
                    self.load_models(config['storage']['model']['path'])

                    selected_clients = self.select_clients()
                    self.train_central_model(selected_clients)
                    self.save_central_model()
                    self.increment_round()
                    self.update_client_models()
                    # self.train_client_models()
                    self.save_client_models()
                
                    self.updateClients(ModelState.WAITING)
                    self.notify_clients()
        
        except asyncio.CancelledError:
            pass

        finally:
            print(f"Connection with {address} closed.")
            writer.close()

    def load_model(self, model_type, path):
        model = ModelFactory.create(model_type, device)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model {path} loaded")
        return model

    def load_models(self, model_folder):
        for client in self.clients:
            path = os.path.join(model_folder, client.model_name)
            client.model = self.load_model(client.model_type, path)

    def select_clients(self):
        groups = {}
        for client in self.clients:
            if client.model_type not in groups:
                groups[client.model_type] = []
            groups[client.model_type].append(client)

        [random.shuffle(group) for group in groups.values()]

        selected = []
        for _, group in groups.items():
            selected.append(random.choice(group))
            if len(selected) == config['server']['knowledge_transfer']['client_qnt']:
                break
        return selected

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

    def train_central_model(self, selected_clients):
        self.init_central_model(selected_clients[0].model_type)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        data_file = open(os.path.join(config['storage']['data']['path'], f'partition_0.pkl'), 'rb')
        data = pkl.load(data_file)

        train_dataset = ImageDataset(data['train'], config['seed'], transform)
        test_dataset = ImageDataset(data['test'], config['seed'], transform)

        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        softmax_temperature = config['server']['knowledge_transfer']['temperature']
        soft_target_loss_weight = config['server']['knowledge_transfer']['soft_target_loss_weight']
        criterion_loss_weight = config['server']['knowledge_transfer']['criterion_loss_weight']

        criterion_kldl = torch.nn.KLDivLoss(reduction="batchmean")

        for epoch in range(config['server']['knowledge_transfer']['epochs']):
            running_loss = 0.0

            correct_predictions_train = 0
            total_samples_train = 0

            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
###
                # optimizer.zero_grad()
                # outputs = self.model(inputs)
                # loss = (softmax_temperature ** 2) * criterion_kldl(
                #     torch.nn.functional.log_softmax(
                #         outputs / softmax_temperature, dim=1
                #     ),
                #     torch.nn.functional.softmax(
                #         selected_clients[0].model(inputs).detach()
                #         / softmax_temperature,
                #         dim=1,
                #     ),
                # )
                # loss.backward()
                # optimizer.step()
###
                optimizer.zero_grad()

                teacher_outputs = self.get_averaged_logits(list(map(lambda c: c.model, selected_clients)), inputs)
                outputs = self.model(inputs)

                # soft_targets = nn.functional.softmax(teacher_outputs / softmax_temperature, dim=-1)
                # soft_prob = nn.functional.log_softmax(outputs / softmax_temperature, dim=-1)
                # soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (softmax_temperature**2)

                # loss_label = criterion(outputs, labels)
                # loss = soft_target_loss_weight * soft_targets_loss + criterion_loss_weight * loss_label

                loss = (softmax_temperature ** 2) * criterion_kldl(
                    torch.nn.functional.log_softmax(
                        outputs / softmax_temperature, dim=1
                    ),
                    torch.nn.functional.softmax(
                        teacher_outputs / softmax_temperature,
                        dim=1,
                    ),
                )

                loss.backward()
                optimizer.step()
###
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

        for client in self.clients:
                self.evaluate(client, testloader, epoch)
            
        print(f'Training is completed')
        pass

    def evaluate(self, client, testloader, epoch):
        correct_predictions_test = 0
        total_samples_test = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = client.model(inputs)
                _, predicted_test = torch.max(outputs, 1)
                total_samples_test += labels.size(0)
                correct_predictions_test += (predicted_test == labels).sum().item()

        accuracy_test = 100 * correct_predictions_test / total_samples_test

        print(f'Client {client.id} model. Epoch: {epoch + 1}. Test accuracy: {accuracy_test:.2f}%')

    def init_central_model(self, model_type):
        model = ModelFactory.create(model_type, self.device)
        model.eval()
        self.model = model
        self.model_type = model_type
        print(f"Model {model_type} is loaded to server")

    def get_averaged_logits(self, models, inputs):
        outputs = []
        for model in models:
            with torch.no_grad():
                _output = model(inputs)
                outputs.append(_output)
        avg = torch.mean(torch.stack(outputs, dim=0), dim=0, keepdim=False)
        return avg

    def save_central_model(self):
        model_name = config['storage']['model']['name']['server'].format(self.round)
        torch.save(self.model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
        print(f"Model {model_name} saved")
        return model_name
    
    def update_client_models(self):
        for client in self.clients:
            client.model = self.model
            client.model_name = config['storage']['model']['name']['client'].format(self.round, client.id)
    
    # def train_client_models(self):
    #     models = {}
    #     for client in self.clients:
    #         if client.model_type not in models:
    #             model = ModelFactory.create(client.model_type, self.device)
    #             model.eval()
    #             client.model = model
    #             client.model_name = config['storage']['model']['name']['client'].format(self.round, client.id)



    def save_client_models(self):
        for client in self.clients:
            torch.save(self.model.state_dict(), os.path.join(config['storage']['model']['path'], client.model_name))
            print(f"Model {client.model_name} saved")

    def notify_clients(self):
        for client in self.clients:
            client.writer.write(client.model_name.encode('utf-8'))
            print(f'Client {client.id} is notified')
    
    def increment_round(self):
        self.round += 1


async def main():
    server = Server(config['server']['address'], config['server']['port'], device)
    server = await server.init()
    async with server:
            await server.serve_forever()
            print("Server stopped")


if __name__ == "__main__":
    asyncio.run(main())