import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

class ListDataset(Dataset):
    def __init__(self, list_imgs, transform = None):
        '''
        target: initialize dataset have list images input
        inputs:
            - list_imgs: list of image
            - transform: transformer apply to each images
        '''
        self.list_imgs = list_imgs
        self.transform = transform
    
    def __len__(self):
        '''
        target: return the size of dataset
        '''
        return len(self.list_imgs)
    
    def __getitem__(self, idx):
        '''
        target: return the image from given idx
        input:
            - idx: interger - index of element
        output:
            - image after transform
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.list_imgs[idx]
        image = image[:, :, :3]

        return self.transform(image)

class SegmentationDataset(Dataset):
    def __init__(self, dataset_args, transform = None, mode = "train"):
        folder_path = dataset_args["folder_path"]
        list_image_name = dataset_args["list_image_name"]
        images_folder_name = dataset_args["images_folder_name"]
        labels_folder_name = dataset_args["labels_folder_name"]
        suffix_label_name = dataset_args["suffix_label_name"]
        img_size = dataset_args["img_size"]

        """
        folder structure:
        |--images
        |   |--train
        |   |--val
        |   |--test
        |--color_labels
        |   |--train
        |   |--val
        |   |--test
        """
        image_folder = os.path.join(folder_path, images_folder_name)
        label_folder = os.path.join(folder_path, labels_folder_name)

        self.image_path = os.path.join(image_folder, mode)
        self.label_path = os.path.join(label_folder, mode)

        self.list_image_name = list_image_name
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
            ])
        self.suffix_label_name = suffix_label_name
    def __len__(self):
        return len(self.list_image_name)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.list_image_name[idx]
        label_name = img_name.replace(".jpg", self.suffix_label_name+".png")
        img_path = os.path.join(self.image_path, img_name)
        label_path = os.path.join(self.label_path, label_name)

        image = io.imread(img_path)
        label = io.imread(label_path)
        #some label have 4 channels
        label = label[:,:,:3]

        return self.transform(image), self.transform(label)
def cifar10(dataset_args, transform = None, mode = "train"):
    '''
    config example:
    dataset = {
    "name":"cifar10",
    "class":cifar10,
    "argument":{
        "path":"cifar10"
    }
    }
    '''
    if transform is None:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
    path = dataset_args["path"]
    return torchvision.datasets.CIFAR10(root = path, train = (mode == "train"), transform= transform, download = True)
