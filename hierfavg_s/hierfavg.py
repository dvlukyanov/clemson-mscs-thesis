import sys
sys.path.insert(0, '../')
import os
import random
import math
from statistics import mean
import yaml
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model_factory import ModelFactory
from util.dataload import ImageDataset


class Cloud():
    def __init__(self, model_type, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_type = model_type
        self.config = config
        self.rng = random.Random()
        self.rng.seed(self.config['seed'])
        self.round = 0
        self.model = ModelFactory.create(self.model_type, self.device, self.config['seed'])
        self.model_name = self.save_model()
        self.edges = [Edge(idx, self, self.model_type, self.device) for idx in range(1, self.config['edge']['qnt'] + 1)]
        self.accuracy = {}

    def save_model(self, round=0):
        model_name = self.config['storage']['model']['name']['cloud'].format(round)
        model_path = os.path.join(self.config['storage']['model']['path'], model_name)
        Utils.save_model(self.model, model_path)
        return model_name

    def train(self):
        for round in range(math.ceil(self.config['cloud']['epochs'] / (self.config['edge']['training']['iterations'] * self.config['client']['training']['epochs']))):
            self.round = round
            for edge in self.edges:
                edge.train(self.config['storage']['model']['name']['cloud'].format(self.round))
            self.model = Utils.aggregate([edge.model for edge in self.edges], self.model_type, self.device)
            self.test_aggregated()
            self.model_name = self.save_model(self.round)

    def test_aggregated(self):
        [correct, total] = Utils.test(self.device, self.model, self.config['client']['training']['batch_size'], self.config['seed'])
        epochs = (self.round + 1) * self.config["edge"]["training"]["iterations"] * self.config["client"]["training"]["epochs"]
        accuracy = 100 * correct / total
        print(f'Accuracy of the cloud model on the {total} test images at the ({epochs} epochs): {accuracy} %')
        return [epochs, accuracy]


class Edge():
    def __init__(self, id, cloud, model_type, device):
        self.id = id
        self.cloud = cloud
        self.device = device
        self.rng = random.Random()
        self.rng.seed(self.cloud.config['seed'])
        self.model_type = model_type
        self.model = None
        self.model_name = None
        self.clients = [Client((self.id - 1) * self.cloud.config['client']['qnt'] + idx, self, self.model_type, self.device) 
                        for idx in range(1, self.cloud.config['client']['qnt'] + 1)]
        self.iteration = 0

    def train(self, model_name):
        self.model_name = model_name
        for iteration in range(1, self.cloud.config['edge']['training']['iterations'] + 1):
            self.iteration = iteration
            for client in self.clients:
                client.train(self.model_name)
            self.test_clients_average()
            self.model = Utils.aggregate([client.model for client in self.clients], self.model_type, self.device)
            self.test_aggregated()
            self.model_name = self.save_model()
    
    def test_clients_average(self):
        performance = []
        for client in self.clients:
            [correct, total] = client.test()
            performance.append(100 * correct / total)
        print(f'Average client accuracy on ({self.cloud.round * self.iteration * self.cloud.config["client"]["training"]["epochs"] + self.iteration * self.cloud.config["client"]["training"]["epochs"]} epochs): {mean(performance)} %')
    
    def test_aggregated(self):
        [correct, total] = Utils.test(self.device, self.model, self.cloud.config['client']['training']['batch_size'], self.cloud.config['seed'])
        epochs = self.cloud.round * self.iteration * self.cloud.config["client"]["training"]["epochs"] + self.iteration * self.cloud.config["client"]["training"]["epochs"]
        accuracy = 100 * correct / total
        print(f'Accuracy of the edge model on the {total} test images at the ({epochs} epochs): {accuracy} %')
        return [epochs, accuracy]
    
    def save_model(self):
        model_name = self.cloud.config['storage']['model']['name']['edge'].format(self.cloud.round + 1, self.iteration, self.id)
        model_path = os.path.join(self.cloud.config['storage']['model']['path'], model_name)
        Utils.save_model(self.model, model_path)
        return model_name


class Client():
    def __init__(self, id, edge, model_type, device):
        self.id = id
        self.edge = edge
        self.device = device
        self.model_type = model_type
        self.model = None

    def train(self, model_name):
        random.seed(self.edge.cloud.config['seed'])
        torch.manual_seed(self.edge.cloud.config['seed'])
        torch.cuda.manual_seed(self.edge.cloud.config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = Utils.load_model(self.device, self.model_type, os.path.join(self.edge.cloud.config['storage']['model']['path'], model_name), 
                                 self.edge.cloud.config['seed'])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        data_file = open(os.path.join(self.edge.cloud.config['storage']['data']['path'], f'partition_{self.id}.pkl'), 'rb')
        data = pkl.load(data_file)

        train_dataset = ImageDataset(data['train'], self.edge.cloud.config['seed'])
        test_dataset = ImageDataset(data['test'], self.edge.cloud.config['seed'])

        trainloader = DataLoader(train_dataset, batch_size=self.edge.cloud.config['client']['training']['batch_size'], shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=self.edge.cloud.config['client']['training']['batch_size'], shuffle=False)

        for epoch in range(self.edge.cloud.config['client']['training']['epochs']):
            running_loss = 0.0
            correct_predictions_train = 0
            total_samples_train = 0

            for _, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
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
                    outputs = model(inputs)
                    _, predicted_test = torch.max(outputs, 1)
                    total_samples_test += labels.size(0)
                    correct_predictions_test += (predicted_test == labels).sum().item()

            accuracy_test = 100 * correct_predictions_test / total_samples_test

            print(f'Round: {self.edge.cloud.round + 1}. Edge: {self.edge.id}. Iteration: {self.edge.iteration}. Client: {self.id}. Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

            scheduler.step()

        self.model = model
    
    def test(self):
        return Utils.test(self.device, self.model, self.edge.cloud.config['client']['training']['batch_size'], self.edge.cloud.config['seed'])
    

class Utils:

    @staticmethod
    def load_model(device, model_type, model_path, seed):
        model = ModelFactory.create(model_type, device, seed)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    @staticmethod
    def save_model(model, model_path):
        torch.save(model.state_dict(), model_path)
        print(f'Model {model_path} saved')

    @staticmethod
    def aggregate(models, model_type, device):
        states = [model.state_dict() for model in models]
        model = ModelFactory.create(model_type, device)
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        model.load_state_dict(state)
        model.eval()
        return model

    @staticmethod
    def test(device, model, batch_size, seed):
        data_file = open(os.path.join(config['storage']['data']['path'], 'test.pkl'), 'rb')
        data = pkl.load(data_file)
        test_dataset = ImageDataset(data, seed)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return [correct, total]
    

if __name__ == "__main__":
    with open('hierfavg_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    cloud = Cloud(sys.argv[1], config)
    cloud.train()