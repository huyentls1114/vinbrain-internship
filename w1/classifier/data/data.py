import sys
sys.path.append("..")
from utils.utils import read_json, show_img
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

class CIFARDataLoader:
    def __init__(self, configs):
        x = 1
        #declare Transformer
        transform = extract_transform(configs['transform'])

        #declare Dataset
        path = configs["dataset"]["folder_path"]
        self.train_dataset = torchvision.datasets.CIFAR10(root = path, train = True, transform= transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root = path, train = False, transform= transform)

        #declare Dataloader
        self.batch_size = configs["batch_size"]
        split_train_val = configs["split_train_val"]
        self.num_sample = len(train_dataset)

        #split train val
        train_sampler, valid_sampler = self.split_sampler(split_train_val)

        #declari data loader
        self.train_loader = DataLoader(train_dataset, 
                                        batch_size = self.batch_size,
                                        num_workers = 2,
                                        sampler = train_sampler)
        self.val_loader = DataLoader(train_dataset, 
                                        batch_size = self.batch_size,
                                        num_workers = 2,
                                        sampler = valid_sampler)
        self.test_loader = DataLoader(test_dataset, 
                                        batch_size = self.batch_size,
                                        shuffle = False,
                                        num_workers = 2)
        
        #define list class
        self.classes = configs["classes"]

    def show_batch(self, mode = "train"):
        data_loader_dict = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test":self.test_loader
        }
        data_iter = iter(data_loader_dict[mode])
        images, labels = data_iter.next()
        print("class", " ".join(self.classes[labels[i]] for i in range(self.batch_size)))
        show_img(torchvision.utils.make_grid(images))
        

    def split_sampler(self, split):
        if split == 0:
            return None, None
        idx_full = np.arange(self.num_sample)
        np.random.seed(0)
        np.random.shuffle(idx_full)

        assert split > 0
        assert split < 1, "split must be from 0 to 1"
        len_train = int(split*self.num_sample)

        valid_idx = idx_full[len_train:]
        train_idx = idx_full[:len_train]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.num_sample = len(train_sampler)
        return train_sampler, valid_sampler

def extract_transform(transforms_dict):
    transform_list = []
    for transform in transforms_dict:
        if transform == "ToTensor":
            transform_list.append(transforms.ToTensor())
        if transform == "Normalize":
            mean = transform["mean"]
            std = transform["std"]
            transform_list.append(transforms.Normalize(mean, std))
        if transform == "Rescale":
            transform_list.append(Rescale(transform.size))
    
    return transforms.Compose(transform_list)


if __name__ == "__main__":
    configs = read_json("../config/config.json")
    cifaLoader = CIFARDataLoader(configs)
    cifaLoader.show_batch("val")