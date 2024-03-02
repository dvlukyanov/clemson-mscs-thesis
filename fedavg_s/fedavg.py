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

class FedAvg():
    def __init__(self, model_type, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_type = model_type
        self.config = config
        self.rng = random.Random()
        self.rng.seed(self.config['seed'])
        self.round = 0
        self.model = ModelFactory.create(self.model_type, self.device, self.config['seed'])
        self.save_model(self.model, self.config['storage']['model']['name']['server'].format(self.round))
        self.accuracy = {}

    def train(self):
        for _ in range(math.ceil(self.config['server']['epochs'] / self.config['client']['training']['epochs'])):
            for client_id in range(1, self.config['client']['qnt'] + 1):
                model = self.train_client(client_id)
                self.save_model(model, config['storage']['model']['name']['client'].format(self.round + 1, client_id))
            self.increment_round()
            self.test_clients_average()
            self.model = self.aggregate()
            [epochs, accuracy] = self.test_aggregated(self.model)
            self.accuracy[epochs] = accuracy
            self.save_model(self.model, self.config['storage']['model']['name']['server'].format(self.round))
        print(self.accuracy)

    def train_client(self, client_id):
        # random.seed(config['seed'])
        # torch.manual_seed(config['seed'])
        # torch.cuda.manual_seed(config['seed'])
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        model = self.load_model(self.config['storage']['model']['name']['server'].format(self.round))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

        data_file = open(os.path.join(config['storage']['data']['path'], f'partition_{client_id}.pkl'), 'rb')
        data = pkl.load(data_file)

        train_dataset = ImageDataset(data['train'], config['seed'])
        test_dataset = ImageDataset(data['test'], config['seed'])

        trainloader = DataLoader(train_dataset, batch_size=self.config['client']['training']['batch_size'], shuffle=False)
        testloader = DataLoader(test_dataset, batch_size=self.config['client']['training']['batch_size'], shuffle=False)

        for epoch in range(config['client']['training']['epochs']):
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

            print(f'Round: {self.round + 1}. Client: {client_id}. Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

            scheduler.step()

        return model

    def aggregate(self):
        print("Random seed:", self.rng.getstate())
        chosen = self.rng.sample(range(1, self.config['client']['qnt'] + 1), self.config['client']['chosen'])
        print(f'Chosen: {chosen}')
        states = [self.load_model(config['storage']['model']['name']['client'].format(self.round, client_id)).state_dict() for client_id in chosen]
        model = ModelFactory.create(self.model_type, self.device)
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        model.load_state_dict(state)
        model.eval()
        return model

    def test_clients_average(self):
        performance = []
        for client_id in range(1, self.config['client']['qnt'] + 1):
            model_name = config['storage']['model']['name']['client'].format(self.round, client_id)
            model = self.load_model(model_name)
            [total, correct] = self.test(model)
            performance.append(100 * correct / total)
        print(f'Average client accuracy on {self.round} round ({self.round * self.config["client"]["training"]["epochs"]} epochs): {mean(performance)}')

    def test_aggregated(self, model):
        [total, correct] = self.test(model)
        epochs = self.round * self.config["client"]["training"]["epochs"]
        accuracy = 100 * correct / total
        print(f'Accuracy of the aggregated model on the {total} test images at the {self.round} round ({epochs} epochs): {accuracy} %')
        return [epochs, accuracy]

    def test(self, model):
        data_file = open(os.path.join(config['storage']['data']['path'], 'test.pkl'), 'rb')
        data = pkl.load(data_file)
        test_dataset = ImageDataset(data, config['seed'])
        testloader = DataLoader(test_dataset, batch_size=self.config['client']['training']['batch_size'], shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return [total, correct]

    def save_model(self, model, model_name):
        model_path = os.path.join(config['storage']['model']['path'], model_name)
        torch.save(model.state_dict(), model_path)
        print(f'Model {model_path} saved')

    def load_model(self, model_name):
        model_path = os.path.join(self.config['storage']['model']['path'], model_name)
        model = ModelFactory.create(self.model_type, self.device, self.config['seed'])
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def increment_round(self):
        self.round += 1


if __name__ == "__main__":
    with open('fedavg_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    fed_avg = FedAvg(sys.argv[1], config)
    fed_avg.train()