import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms

from data_prep import to_torch_dataset, prep
from sklearn.model_selection import train_test_split

print("imports ok!")

# data 
path = r"C:\Users\Daniel\Documents\from_bams_prime\Datasets\brain_tumor_dataset"
X, y = prep(path)
X = X.astype(np.float32)

print(X[0].shape)
print(y.shape)

# split data into train, validation and test set
x_tmp, x_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_tmp, y_tmp, test_size=(0.2/0.8), random_state=42, shuffle=True)

# create torch dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = to_torch_dataset(x_train, y_train, transform)
val_dataset = to_torch_dataset(x_val, y_val, transform)
test_dataset = to_torch_dataset(x_test, y_test, transform)

# create dataloader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

# setup device for cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



net = Net().to(device)
# print(net)


criterion = nn.BCELoss()
optimizer = optim.Adam(params=net.parameters(), lr=1e-4)
EPOCHS = 32

for epoch in tqdm(range(EPOCHS)):
    for batch, (images, labels) in enumerate(train_loader):

        print("Input shape: ", images.shape)
        print("Labels shape: ", labels.shape)
        images = images.to(device)
        labels = labels.float().to(device)

        output = net(images)
        loss = criterion(output, labels)

        optimizer.zero_grad() # to stop gradients from accumulating
        loss.backward()

        optimizer.step()

        print(f"epoch: {epoch} --- loss: {loss}")





