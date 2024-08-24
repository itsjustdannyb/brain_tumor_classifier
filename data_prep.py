import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class to_torch_dataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        sample = Image.fromarray((sample * 255).astype(np.uint8))
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, torch.tensor([label], dtype=torch.float) 


def prep(file_path, size=(125,125)):

    os.chdir(file_path)

    images = []
    labels = []
    for label in os.listdir(file_path):
        label_path = os.path.join(file_path, label)
        for img in os.listdir(label_path):
            img_path = os.path.join(label_path, img)
            # print(img_path)

            # image processing
            img = Image.open(img_path).convert("RGB")
            img = img.resize((125,125), Image.Resampling.LANCZOS)

            # convert image to array and scale it
            img = np.array(img) / 255
            # img = np.transpose(img, (2, 0, 1))
            images.append(img)
            labels.append(int(label))

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
