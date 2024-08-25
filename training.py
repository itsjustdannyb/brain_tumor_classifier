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

# the model
from model import Net

print("imports ok!")

# data 
path = r"C:\Users\Daniel\Documents\from_bams_prime\Datasets\brain_tumor_dataset"
X, y = prep(path)

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


net = Net().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(params=net.parameters(), lr=1e-4)
EPOCHS = 20

for epoch in range(EPOCHS):
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

        print(f"epoch: {epoch} --- batch: {batch} --- loss: {loss:.4f}")



def check_accuracy(loader, model):

    correct = 0
    samples = 0
    model.eval() # evaluation mode

    # don't calculate gradients
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            output = model(x)
            prediction = (output >= 0.5).float()

            correct += (prediction == y).sum()
            samples += prediction.size(0)

        acc = (float(correct)/float(samples))*100
        # print(f"Accuracy: {acc:.2f}")

    model.train()
    return acc

print(f"checking accuracy on training data\naccuracy on train_set: {check_accuracy(train_loader, net):.2f}",)
print(f"checking accuracy on validation data\naccuracy on validation_set: {check_accuracy(val_loader, net):.2f}")
print(f"checking accuracy on test data\naccuracy on test_set: {check_accuracy(test_loader, net):.2f}")

torch.save(net.state_dict(), "brain_tumor_model")






