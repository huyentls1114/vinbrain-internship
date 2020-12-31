import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

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


def test_segmentationdataset():
    folder_path = "/home/huyen/data/bdd100k_seg/bdd100k/seg"
    list_img = os.listdir("/home/huyen/data/bdd100k_seg/bdd100k/seg/images/train")
    segDat = SegmentationDataset(folder_path, list_img, mode = "train")

    for i in range(len(segDat)):
        img, label = segDat[i]
        img = np.transpose(img, (1, 2, 0))
        label = np.transpose(label, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        plt.imshow(label)
        plt.show()
        if i == 3:
            break

def test_classifyDataset():
    folder_path = "/home/huyen/data/cifar10"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    dataset = torchvision.datasets.CIFAR10(root = folder_path, 
                                            train=True, 
                                            transform= transform
                                            )
    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = (img/2+0.5)
        img = np.transpose(img, (2, 1, 0))
        plt.imshow(img)
        plt.show()
        print(classes[label])
        if i == 3:
            break
if __name__ == "__main__":
    # test_segmentationdataset()
    test_classifyDataset()