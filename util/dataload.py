import random
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, seed, transform=None):
        tuples = [(image, label) for label in data.keys() for image in data[label]]
        random.seed(seed)
        random.shuffle(tuples)
        self.data, self.labels = zip(*tuples)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx]) if self.transform else self.data[idx], self.labels[idx]