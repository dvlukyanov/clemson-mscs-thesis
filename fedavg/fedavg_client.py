import sys
import os
import random
import pickle as pkl
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fedavg_model import SimpleCNN
from fedavg_data import ImageDataset, transform
import asyncio


# set client id

client_id = sys.argv[1]
print(f"Client with ID {client_id} started")


# init config

with open('fedavg_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# function

def getModel(model_name):
    model_path = os.path.join(config['storage']['model']['path'], model_name)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def trainModel(model, client_id):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    data_file = open(os.path.join(config['storage']['data']['path'], f'partition_{client_id}.pkl'), 'rb')
    data = pkl.load(data_file)

    train_dataset = ImageDataset(data['train'], transform)
    test_dataset = ImageDataset(data['test'], transform)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(config['client']['training']['epochs_per_iteration']):
        running_loss = 0.0

        correct_predictions_train = 0
        total_samples_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
        
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
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted_test = torch.max(outputs, 1)
                total_samples_test += labels.size(0)
                correct_predictions_test += (predicted_test == labels).sum().item()

        accuracy_test = 100 * correct_predictions_test / total_samples_test

        print(f'Epoch: {epoch + 1}. Loss: {running_loss / 100:.3f}. Train accuracy: {accuracy_train:.2f}%. Test accuracy: {accuracy_test:.2f}%')

        scheduler.step()
            
    print(f'Iteration is completed')
    return model


def saveModel(model, iteration, client_id):
    model_name = config['storage']['model']['name']['client'].format(iteration, client_id)
    model_path = os.path.join(config['storage']['model']['path'], model_name)
    torch.save(model.state_dict(), model_path)
    print(f'Model {model_path} saved')


async def send_data(writer, data):
    writer.write(data.encode('utf-8'))
    await writer.drain()


# connect to server

async def main():
    try:
        server_model_name = None
        iteration = 0
        reader, writer = await asyncio.open_connection(config['server']['address'], config['server']['port'])
        while True:
            model = getModel(server_model_name if server_model_name is not None else config['storage']['model']['name']['server'].format(iteration))
            model = trainModel(model, client_id)
            saveModel(model, iteration, client_id)
            await send_data(writer, client_id)
            message = None
            while message is None or message.startswith("model_server") is False:
                message = (await reader.read(100)).decode('utf-8')
            server_model_name = message
            print(f"Received from server: {server_model_name}")
            iteration += 1

    except Exception as e:
        print(f"Error connecting to the server: {e}")

if __name__ == "__main__":
    asyncio.run(main())