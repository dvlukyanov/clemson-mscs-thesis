import random
import torch
from .simple_cnn import SimpleCNN
from .densenet import DenseNet
from .mobilenetv2 import MobileNetV2
from .resnet import ResNet_cifar
from .model_enum import Model

class ModelFactory:

    @staticmethod
    def create(model_type, device='cpu', seed=12345):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if model_type == Model.SimpleCNN.value:
            return SimpleCNN().to(device)
        elif model_type == Model.DenseNet:
            return DenseNet().to(device)
        elif model_type == Model.MobileNetV2:
            return MobileNetV2().to(device)
        elif model_type == Model.ResNet:
            return ResNet_cifar().to(device)
        else:
            raise ValueError("Wrong model type: " + model_type)