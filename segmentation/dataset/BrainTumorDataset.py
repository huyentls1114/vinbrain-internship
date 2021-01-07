from torch.utils.data import Dataset
from utils.utils import conver_numpy_image, contour
import os
import h5py
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import torch

class BrainTumorDataset(Dataset):
    def __init__(self, dataset_args, transform_image, transform_label, mode = "train"):
        self.input_folder = dataset_args["input_folder"]
        self.image_folder = os.path.join(self.input_folder, "images")
        self.mask_folder = os.path.join(self.input_folder, "masks")
        self.list_img_name = self.read_txt(os.path.join(self.input_folder, "%s.txt"%(mode)))
        self.transform_image = transform_image
        self.transform_label = transform_label
    def __len__(self):
        return len(self.list_img_name)
    
    def __getitem__(self, idx):
        img_name = self.list_img_name[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = plt.imread(img_path)
        image = image[:, :, 0]

        mask_path = os.path.join(self.mask_folder, img_name)
        mask = plt.imread(mask_path)
        mask = mask[:, :, 0]
        return self.transform_image(image), self.transform_label(mask)
    
    def load_sample(self, batch_size = 4):
        list_imgs = []
        list_masks = []

        #random list idx
        list_idx = np.arange(self.__len__())
        np.random.shuffle(list_idx)
        list_idx = list_idx[0:batch_size]

        for i in range(batch_size):
            image, mask = self.__getitem__(list_idx[i])
            list_imgs.append(image[None, :, :])
            list_masks.append(mask[None, :, :])
        return torch.cat(list_imgs), torch.cat(list_masks)


    def show_sample(self, batch_size = 4):
        list_imgs = []
        list_masks = []

        #random list idx
        list_idx = np.arange(self.__len__())
        np.random.shuffle(list_idx)
        list_idx = list_idx[0:batch_size]

        for i in range(batch_size):
            image, mask = self.__getitem__(list_idx[i])
            list_imgs.append(image)
            list_masks.append(mask)
        self.show_img(list_imgs, list_masks, batch_size)
    

    def show_img(self, list_imgs, list_masks, batch_size):
        list_combine = []
        fig = plt.figure(figsize=(batch_size, 3), dpi = 512)
        for i in range(batch_size):
            img = conver_numpy_image(list_imgs[i])
            mask = conver_numpy_image(list_masks[i])
            ct = contour(img, mask)
            combine = np.hstack([img, mask, ct])
            list_combine.append(combine)
        plt.imshow(np.vstack(list_combine)[:,:,0])
        plt.axis('off')
        plt.show()

    def read_txt(self, txt_file):        
        file_ = open(txt_file, "r")
        list_ = file_.readlines()
        list_img_name = []
        for line in list_:
            img_name = line.replace("\n","")
            list_img_name.append(img_name)
        file_.close()
        return list_img_name


class PrepairBrainTumorDataset:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.train_file = os.path.join(output_folder, 'train.txt')
        self.test_file = os.path.join(output_folder, 'test.txt')
        self.val_file = os.path.join(output_folder, 'val.txt')

        self.images_output_folder = os.path.join(self.output_folder, "images")
        self.create_new_dir(self.images_output_folder)
        self.masks_output_folder = os.path.join(self.output_folder, "masks")
        self.create_new_dir(self.masks_output_folder)
            
    def split_train_test_val(self):
        list_img, list_label = self.save_img()
        img_train, img_test, label_train, label_test = train_test_split(list_img, list_label, test_size =  0.2)
        img_train, img_val, label_train, label_val = train_test_split(img_train, label_train, test_size = 0.125)
        self.save_txt(img_train, self.train_file)
        self.save_txt(img_test, self.test_file)
        self.save_txt(img_val, self.val_file)

    def save_txt(self, list_img, file_path):
        '''
        save list image name to file
        inputs:
            - list_img: List of string - list image name
            - file_path: String - file output
        output:
            - file path contain image and label folowing pattern: img_name
        '''
        if os.path.isfile(file_path):
            os.remove(file_path)
        file_ = open(file_path, "w")
        for img in list_img:
            file_.writelines(img+"\n")
        file_.close()


    def save_img(self):
        '''
        convert mat data to image data with folder structure
        pattern image name: imgId_label.jpg
        BrainTumor
        |--images
        |--masks
        '''
        list_img = []
        list_mask = []
        list_label = []
        for folder in os.listdir(self.input_folder):    
            folder = os.path.join(self.input_folder, folder)
            if os.path.isfile(folder):
                continue
            for img_file in os.listdir(folder):
                cjdata = h5py.File(os.path.join(folder, img_file), 'r')['cjdata']
                image_order = img_file.replace(".mat","")
                label = np.array(cjdata["label"])[0][0]
                img_name = "%s_%d.jpg"%(image_order, label)
                image = np.array(cjdata["image"])
                # image = image[:,:,np.newaxis]
                mask = np.array(cjdata["tumorMask"])

                plt.imsave(os.path.join(self.images_output_folder, img_name), image, cmap = "gray")
                plt.imsave(os.path.join(self.masks_output_folder, img_name), mask, cmap = "gray")

                list_img.append(img_name)
                list_label.append(label)
        return list_img, list_label
    def create_new_dir(self, dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)