import torch
from torch.utils.data import Dataset
from skimage import io
import os
import numpy as np

'''
config example
dataset = {
"class":MenWomanDataset,
"argument":{
    "path":"E:\data\data"
}
}

'''
class MenWomanDataset(Dataset):
    def __init__(self, path, classes = None, transform = None, mode = "train"):
        '''
        target: initialize MenWoman Dataset
        dataset structure:
        ----------------------
        MenWoman
        |--men
        |--woman
        |--train.txt
        |--test.txt
        ----------------------
        inputs:
            - dataset_args: dictionary contain some information of dataset
                - paths: String - path of data directory
            - transform: torchvision.transform.Compose() - transformer apply for each images
            - mode: String - value in ["train", "test"]
        '''
        self.mode = mode
        self.transform = transform
        self.data_dir = dataset_args["path"]
        self.data_path = os.path.join(self.data_dir, mode +".txt")
        self.list_image_name, self.list_label = self.load_train_img(self.data_path)

    def __len__(self):
        '''
        target: length of dataset
        '''
        return len(self.list_image_name)

    def __getitem__(self, idx):
        '''
        target: get element of dataset given idx
        input:
            - idx: index of element
        output
            - image
            - label
        '''
        assert len(self.list_image_name) == len(self.list_label)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.list_image_name[idx]
        label = self.list_label[idx]

        img_path = os.path.join(self.data_dir, img_name)
        image = io.imread(img_path)
        image = image[:,:,:3]
        return self.transform(image), label

    def load_train_img(self, file_path):
        '''
        target: load images and labels from txt file
        input:
            - file_path: String - txt file contain links of images
        output:
            - list_images_name
            - list_labels
        '''

        file_ = open(file_path, "r")
        list_ = file_.readlines()
        list_img = []
        list_label = []
        for line in list_:
            class_, img_name, label = line.replace("\n","").split(",")
            list_img.append(os.path.join(class_, img_name))
            list_label.append(int(label))
        return list_img, list_label



def split_train_test_folder(input_folder, split_range):
    '''
    split train - test data of dataset have no test folder
    folder structure
    -----
    class1
    class2
    '''
    list_class = os.listdir(input_folder)
    list_img_name = []
    list_label = []
    class_number = 0
    for i, _class in enumerate(list_class):
        if os.path.isfile(os.path.join(input_folder, _class)):
            continue
        for img_name in os.listdir(os.path.join(input_folder, _class)):
            list_img_name.append(_class + "," + img_name)
            list_label.append(class_number)
        class_number+=1
    length_ = len(list_img_name)
    list_index = np.arange(length_)
    np.random.shuffle(list_index)
    
    list_img_name = np.array(list_img_name)
    list_label = np.array(list_label)

    list_train = list_img_name[list_index[:int(length_*split_range)]]
    list_test = list_img_name[list_index[int(length_*split_range):]]
    list_label_train = list_label[list_index[:int(length_*split_range)]]
    list_label_test = list_label[list_index[int(length_*split_range):]]

    save_train_img(list_train, list_label_train, os.path.join(input_folder, 'train.txt'))
    save_train_img(list_test, list_label_test, os.path.join(input_folder, 'test.txt'))

def save_train_img(list_img, list_label, file_path):
    '''
    save list image name to file
    inputs:
        - list_img: List of string - list image name
        - list_label: list of string - list label name
        - file_path: String - file output
    output:
        - file path contain image and label folowing pattern: img_name,label
    '''
    assert(len(list_img) == len(list_label))
    if os.path.isfile(file_path):
        os.remove(file_path)
    file_ = open(file_path, "w")
    for (img, label) in zip(list_img, list_label):
        file_.writelines(img+","+str(label)+"\n")
    file_.close()
