import sys
sys.path.insert(0, '../')
import os
import random
import json
from util.json_encoder import MessageEncoder
import pickle as pkl
import torch
from torch.utils.data import DataLoader
import asyncio
import yaml
from model.simple_cnn import SimpleCNN
from util.dataload import ImageDataset, transform
from util.message import Message
from response import Response
from threading import Lock
from copy import deepcopy
from time import sleep


# init config

with open('fedbuff_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# clean storage
    
for file in os.listdir(config['storage']['model']['path']):
    os.remove(os.path.join(config['storage']['model']['path'], file))


# classes
    
class Server():
    def __init__(self, address, port, blueprint, device):
        self.blueprint = blueprint
        self.device = device
        self.address = address
        self.port = port
        self.models = []
        self.iteration = 0
        self.mutex = Lock()

    async def init(self):
        self.init_model()
        server = await asyncio.start_server(self.handle_clients, self.address, self.port)
        print(f"Server started")
        return server
    
    def init_model(self):
        model = deepcopy(self.blueprint)
        model.eval()
        model_name = config['storage']['model']['name']['server'].format(self.iteration)
        torch.save(model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
        print(f"Model {model_name} saved")
        return model
    
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

                while self.mutex.locked():
                    sleep(1)

                response = Response(reload=False, model_name=None)
                if message.model_name is not None:
                    self.models.append(message.model_name)
                    print(f"Model {message.model_name} added to buffer: {len(self.models)}")
                    if len(self.models) == config['server']['buffer_size']:
                        print("Buffer is full")
                        self.mutex.acquire()
                        self.aggregate()
                        self.flush()
                        self.test()
                        model_name = self.save_model()
                        self.increment_iteration()
                        response = Response(reload=True, model_name=model_name)
                        self.mutex.release()

                body = json.dumps(response, cls=MessageEncoder)
                writer.write(body.encode('utf-8'))
                await writer.drain()
        
        except asyncio.CancelledError:
            pass

    
    def aggregate(self):
        models = [self.load_model(model_name) for model_name in self.models]
        states = [model.state_dict() for model in models]
        self.model = deepcopy(self.blueprint).to(self.device)
        state = {}
        for key in states[0].keys():
            state[key] = torch.mean(torch.stack([state[key].to(torch.float32) for state in states]), dim=0, keepdim=False)
        self.model.load_state_dict(state)
        self.model.eval()
        print("Models aggregated")

    def load_model(self, model_name):
        model = deepcopy(self.blueprint).to(self.device)
        model.load_state_dict(torch.load(os.path.join(config['storage']['model']['path'], model_name)))
        model.eval()
        print(f"Model {model_name} loaded")
        return model

    def flush(self):
        self.models = []
        print("Buffer flushed")

    def test(self):
        print(f"Testing model at {self.iteration + 1} iteration")
        data_file = open(os.path.join(config['storage']['data']['path'], 'test.pkl'), 'rb')
        data = pkl.load(data_file)
        test_dataset = ImageDataset(data, config['seed'], transform)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the {total} test images at {self.iteration + 1} iteration: {100 * correct / total} %')

    def save_model(self):
        model_name = config['storage']['model']['name']['server'].format(self.iteration + 1)
        torch.save(self.model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
        print(f"Model {model_name} saved")
        return model_name
    
    def increment_iteration(self):
        self.iteration += 1


async def main():
    server = await Server(config['server']['address'], config['server']['port'], SimpleCNN(), device).init()
    async with server:
        await server.serve_forever()
        print("Cloud stopped")


if __name__ == '__main__':
    asyncio.run(main())