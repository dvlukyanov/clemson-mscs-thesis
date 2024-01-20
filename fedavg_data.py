import random
import yaml
import torch
from torchvision import transforms


with open('fedavg_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


random.seed(config['seed'])


transform = transforms.Compose([
    transforms.ToTensor()
])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        tuples = [(image, label) for label in data.keys() for image in data[label]]
        random.shuffle(tuples)
        self.data, self.labels = zip(*tuples)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx]) if self.transform else self.data[idx], self.labels[idx]