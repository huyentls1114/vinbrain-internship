import sys
sys.path.append("..")
from utils.utils import show_img
import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import random
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

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
        args = configs.dataset["dataset_args"]
        self.train_dataset = DatasetClass(**configs.dataset["dataset_args"],transform = configs.transform_train, mode = "train")
        self.test_dataset = DatasetClass(**configs.dataset["dataset_args"],transform = configs.transform_test, mode = "test")

        #declare Dataloader
        self.batch_size = configs.batch_size
        split_train_val = configs.split_train_val
        self.num_sample = len(self.train_dataset)

        #define fold
        if hasattr(configs, "num_fold"):
            self.num_fold = configs.num_fold
        else:
            self.num_fold = None
        
        #split train val
        if self.num_fold is None:
            self.train_sampler, self.valid_sampler = self.split_sampler(self.train_dataset, split_train_val)
        else:
            self.list_fold = self.get_list_fold(self.num_fold)
            self.train_sampler, self.valid_sampler = self.get_fold_sampler(fold =0)
        self.init_loaders()
        #define list class
        self.classes = configs.dataset["dataset_args"]["classes"]

    def init_loaders(self):
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
        self.loader_dict = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test": self.test_loader
        }


    def show_batch(self, mode = "train", num_images = None, _class = None):
        '''
        target: show image and labels
        input: 
            - mode: String - value ["train", "val", "test"]
        output:
            - batch images with labels
        '''
        if num_images is None:
            num_images = self.batch_size
        # data_iter = iter(data_loader_dict[mode])
        # images, labels = data_iter.next()
        list_imgs = []
        list_labels = []

        #random list idx
        loader = self.loader_dict[mode]
        list_idx = list(loader.sampler)
        np.random.shuffle(list_idx)
        list_idx = list_idx[0:num_images]
        
        #get image and label from dataset
        dataset = loader.datatset
        
        for i in range(num_images):
            if _class is not None:
                image, label = self.choose_img_class(dataset, _class)
                list_imgs.append(image)
                list_labels.append(label)
            else:
                image, label = dataset[list_idx[i]]
                list_imgs.append(image)
                list_labels.append(label)

        print("class", " ".join(self.classes[list_labels[i]] for i in range(num_images)))
        show_img(torchvision.utils.make_grid(list_imgs))

    def choose_img_class(self,dataset, _class):
        len_dataset = len(dataset) 
        index = random.randint(0, len_dataset-1)
        image, label = dataset[index]
        while(label != _class):
            index = random.randint(0, len_dataset-1)
            image, label = dataset[index]
        return image, label

    def split_sampler(self, dataset, split):
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
        idx_full = np.arange(len(dataset))
        if hasattr(dataset, 'list_label'):
            y = dataset.list_label
        else:
            y = np.array(list(x[1] for x in dataset))

        assert split > 0
        assert split < 1, "split must be from 0 to 1"
        train_idx, valid_idx, y_train, y_val = train_test_split(idx_full, y, test_size=1-split, stratify=y)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.num_sample = len(train_sampler)
        return train_sampler, valid_sampler

    def get_list_fold(self, num_fold):
        idx_full = np.arange(len(dataset))
        if hasattr(dataset, 'list_label'):
            y = dataset.list_label
        else:
            y = np.array(list(x[1] for x in dataset))

        kfold = StratifiedKFold(n_splits = num_fold, random_state = 1996,shuffle = True)
        list_fold = list(kfold.split(X, y))
        return list_fold

    def get_fold_sampler(self, fold = 0): 
        '''
        target: create SubsetRandomSamplers of train and val
        input:
            - split: float 0-1
        output:
            - train_sampler: SubsetRandomSamplers of train
            - valid_sampler: SubsetRandomSamplers of valid
        '''
        assert self.list_fold is not None
        
        train_idx, valid_idx, y_train, y_val = self.list_fold[fold]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.num_sample = len(train_sampler)
        return train_sampler, valid_sampler

    def update_fold(self, fold = 0):
        self.train_sampler, self.valid_sampler = self.get_fold_sampler(fold)
        self.init_loaders()


    def caculate_num_per_labels(self, mode = "train"):
        list_idx = list(self.sampler_dict[mode])
        dataset = self.dataset_dict[mode]

        labels = torch.zeros(len(self.classes), dtype=torch.long)
        for index in list_idx:
            img, target = dataset[index]
            labels += torch.nn.functional.one_hot(torch.tensor(target), num_classes=len(self.classes))
        return labels