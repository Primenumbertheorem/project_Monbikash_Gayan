# dataset.py

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from config import resize_x, resize_y, batchsize

# Dataset class (optional, reusing ImageFolder)
def UnicornImgDataset(data_path, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((resize_x, resize_y)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        
    dataset = ImageFolder(root=data_path, transform=transform)
    return dataset

# Dataloader
def unicornLoader(dataset, shuffle=True):
    loader = DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=2, pin_memory=True)
    return loader
