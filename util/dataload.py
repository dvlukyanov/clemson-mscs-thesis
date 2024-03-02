import random
import torch
from torchvision import transforms

normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, seed):
        tuples = [(image, label) for label in data.keys() for image in data[label]]
        random.seed(seed)
        random.shuffle(tuples)
        self.data, self.labels = zip(*tuples)
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(degrees=(-15, 15)),
            # transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.3),
            transforms.ToTensor(),
            normalize
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.transform(self.data[idx]) if self.transform else self.data[idx], self.labels[idx]