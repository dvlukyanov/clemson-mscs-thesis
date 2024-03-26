from enum import Enum

class Model(str, Enum):
    SimpleCNN = 'simple_cnn'
    DenseNet = 'densenet'
    MobileNetV2 = 'mobilenetv2'
    ResNet = 'resnet'

