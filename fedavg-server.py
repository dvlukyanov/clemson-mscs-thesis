import os
import random
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import yaml

# init config

with open('fedavg-config.yaml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(config['seed'])
torch.manual_seed(config['seed'])
torch.cuda.manual_seed(config['seed'])
device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

clients = {i: None for i in range(config['client']['qnt'])}
progress = {i: None for i in range(config['client']['qnt'])}

# init model

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.convolutional_layer = nn.Sequential(            
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                        
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=8192, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear_layer(x)
        return x
    
model = SimpleCNN().to(device)

for name, params in model.named_parameters():
    print(params.shape)

# save model

model_name = config['storage']['model']['name'].format(0)
torch.save(model.state_dict(), os.path.join(config['storage']['model']['path'], model_name))
print("Base model is saved")

# listen to clients

async def handle_client(reader, writer):
    address = writer.get_extra_info('peername')
    print(f"Connection from {address}")

    # Store the client's writer in the clients dictionary
    clients[address] = writer

    try:
        while True:
            # Receive data from the client
            data = await reader.read(100)
            if not data:
                break

            print(f"Received from {address}: {data.decode('utf-8')}")

            # Send a response back to the client
            writer.write("Confirmed".encode('utf-8'))
            await writer.drain()

    except asyncio.CancelledError:
        pass

    finally:
        print(f"Connection with {address} closed.")
        writer.close()

        # Remove the client from the registry upon disconnection
        if address in clients:
            clients.pop(address)

async def main():
    server = await asyncio.start_server(handle_client, config['server']['ip'], config['server']['port'])
    print("Server started")

    # Start the task for sending periodic notifications
    # asyncio.create_task(send_notifications())

    async with server:
        await server.serve_forever()
        print("Server stopped")

if __name__ == "__main__":
    asyncio.run(main())