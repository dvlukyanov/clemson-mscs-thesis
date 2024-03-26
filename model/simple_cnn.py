import torch
import torch.nn as nn

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
    
    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0007, weight_decay=0.001)
    
    def scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.11)