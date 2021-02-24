import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


class Cifar10(Dataset):
    def __init__(self, path, transform = None, mode = "train"):
        if transform is None:
                transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        self.dataset = torchvision.datasets.CIFAR10(root = path, train = (mode == "train"), transform= transform, download = True)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)