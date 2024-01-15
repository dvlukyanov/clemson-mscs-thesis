import sys
import os
import random
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# set client id

client_id = sys.argv[1]
print("Client ID: ", client_id)

# init config

data_path = "data"

seed=12345
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device=torch.device("cuda" if torch.cuda.is_available() else "mps")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# load the model

# train the model

# save the model

