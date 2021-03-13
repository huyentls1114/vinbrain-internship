from torch.utils.data import DataLoader
import torch

class SegmentationData:
    def __init__(self, configs):
        self.batch_size = configs.batch_size

        self.transform_train = configs.transform_train
        self.transform_test = configs.transform_test
        self.transform_label = configs.transform_label

        dataset = configs.dataset
        self.train_dataset = dataset["class"](dataset["dataset_args"], self.transform_train, self.transform_label, "train")
        self.test_dataset = dataset["class"](dataset["dataset_args"], self.transform_test, self.transform_label, "test")
        self.val_dataset = dataset["class"](dataset["dataset_args"], self.transform_test, self.transform_label, "val")

        self.train_loader = DataLoader(self.train_dataset, 
                                        shuffle = True, 
                                        batch_size= self.batch_size,
                                        drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, 
                                        shuffle = False, 
                                        batch_size= self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, 
                                        shuffle = False, 
                                        batch_size= self.batch_size)
    def update_train_ds(self, **update_ds_args):
        self.train_dataset.update_train_ds(**update_ds_args)
        self.train_loader = DataLoader(self.train_dataset, 
                                shuffle = True, 
                                batch_size= self.batch_size,
                                drop_last=True)

    def show_batch(self, mode = "train"):
        dataset_dict = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test":self.test_dataset
        }
        dataset_dict[mode].show_sample(self.batch_size)
    def load_batch(self, mode = "train"):
        dataset_dict = {
            "train": self.train_dataset,
            "val": self.val_dataset,
            "test":self.test_dataset
        }
        return dataset_dict[mode].load_sample(self.batch_size)

    def caculate_num_per_labels(self, mode = "train"):
        loader_dict = {
            "train": self.train_loader,
            "val": self.val_loader,
            "test":self.test_loader
        }
        list_idx = list(loader_dict[mode].sampler)
        dataset = loader_dict[mode].dataset

        label_img = [0, 0]
        label_pixel = [0, 0]
        for index in list_idx:
            img, target = dataset[index]
            w, h = img.shape[-2, -1]
            # import pdb; pdb.set_trace()
            postive_pixel =  torch.sum(target.long())
            negative_pixel = w * h - postive_pixel
            label_pixel[0]+= negative_pixel
            label_pixel[1]+= postive_pixel
            if postive_pixel == 0:
                label_img[0] +=1
            else:
                label_img[1] +=1
            
        return label_img, label_pixel

        