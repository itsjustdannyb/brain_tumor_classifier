import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms

# VGG19 model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(3, 64, kernel_size=(3,3), padding="same")
        self.c2 = nn.Conv2d(64, 128, (3,3), padding="same")
        self.c3 = nn.Conv2d(128, 256, (3,3), padding="same")
        self.c4 = nn.Conv2d(256, 512, (3,3), padding="same")
        self.s = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.f1 = nn.Linear(512*7*7, 4096)
        self.f2 = nn.Linear(4096, 1000)
        self.f3 = nn.Linear(1000, 1)
    
    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.s(x)
        
        x = F.relu(self.c2(x))
        x = self.s(x)

        x = F.relu(self.c3(x))
        x = self.s(x)

        x = F.relu(self.c4(x))
        x = self.s(x)

        x = torch.flatten(x,1)

        x = self.f1(x)
        x = self.f2(x)
        return torch.sigmoid(self.f3(x))
