import sys
sys.path.append("..")
from utils.utils import show_img
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms


class CIFARData:
    def __init__(self, configs):
        '''
        target: initialize Cifa dataset, data loader
        input: configs imported from configfile in config folder,
            contains parameters: 
            - dataset: contain class dataset and arguments
            - transform_train, transform_test
            - split_train_val: float - use to split train_val
            - classes: list label of class name            
        '''
        x = 1
        #declare Dataset
        DatasetClass = configs.dataset["class"]
        self.train_dataset = DatasetClass(**configs.dataset["dataset_args"],transform = configs.transform_train, mode = "train")
        self.test_dataset = DatasetClass(**configs.dataset["dataset_args"],transform = configs.transform_test, mode = "test")

        #declare Dataloader
        self.batch_size = configs.batch_size
        split_train_val = configs.split_train_val
        self.num_sample = len(self.train_dataset)

        #split train val
        self.train_sampler, self.valid_sampler = self.split_sampler(split_train_val)
        self.test_sampler = SubsetRandomSampler(range(len(self.test_dataset)))

        #declare data loader
        self.train_loader = DataLoader(self.train_dataset, 
                                        batch_size = self.batch_size,
                                        num_workers = 2,
                                        sampler = self.train_sampler)
        self.val_loader = DataLoader(self.train_dataset, 
                                        batch_size = self.batch_size,
                                        num_workers = 2,
                                        sampler = self.valid_sampler)
        self.test_loader = DataLoader(self.test_dataset, 
                                        batch_size = self.batch_size,
                                        shuffle = False,
                                        num_workers = 2)
        
        #define list class
        self.classes = configs.classes

    def show_batch(self, mode = "train"):
        '''
        target: show image and labels
        input: 
            - mode: String - value ["train", "val", "test"]
        output:
            - batch images with labels
        '''
        dataset_dict = {
            "train": self.train_dataset,
            "val": self.train_dataset,
            "test":self.test_dataset
        }
        sampler_dict = {
            "train": self.train_sampler,
            "val": self.valid_sampler,
            "test": self.test_sampler
        }
        # data_iter = iter(data_loader_dict[mode])
        # images, labels = data_iter.next()
        list_imgs = []
        list_labels = []

        #random list idx
        list_idx = list(sampler_dict[mode])
        np.random.shuffle(list_idx)
        list_idx = list_idx[0:self.batch_size]
        
        #get image and label from dataset
        dataset = dataset_dict[mode]
        for i in range(self.batch_size):
            image, label = dataset[list_idx[i]]
            list_imgs.append(image)
            list_labels.append(label)

        print("class", " ".join(self.classes[list_labels[i]] for i in range(self.batch_size)))
        show_img(torchvision.utils.make_grid(list_imgs))
        

    def split_sampler(self, split):
        '''
        target: create SubsetRandomSamplers of train and val
        input:
            - split: float 0-1
        output:
            - train_sampler: SubsetRandomSamplers of train
            - valid_sampler: SubsetRandomSamplers of valid
        '''
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