from torch.utils.data import DataLoader

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
                                        batch_size= self.batch_size)
        self.val_loader = DataLoader(self.val_dataset, 
                                        shuffle = False, 
                                        batch_size= self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, 
                                        shuffle = False, 
                                        batch_size= self.batch_size)
    def show_batch(self, mode = "train"):
        dataset_dict = {
            "train": self.train_dataset,
            "val": self.train_dataset,
            "test":self.test_dataset
        }
        dataset_dict[mode].show_sample(self.batch_size)
    def load_batch(self, mode = "train"):
        dataset_dict = {
            "train": self.train_dataset,
            "val": self.train_dataset,
            "test":self.test_dataset
        }
        return dataset_dict[mode].load_sample(self.batch_size)



        