import pydicom
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import cv2
from utils.utils import conver_numpy_image, contour
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class PneumothoraxDataset(Dataset):
    def __init__(self, dataset_args, transform_image, transform_label, mode = "train"):
        self.input_folder = dataset_args["input_folder"]
        if "augmentation" in dataset_args.keys():
            self.augmentation = dataset_args["augmentation"]
        else:
            self.augmentation = None
        self.mode = mode
        self.image_folder = os.path.join(self.input_folder, "images")
        self.mask_folder = os.path.join(self.input_folder, "masks")
        
        #data process
        self.df_img_all = self.read_txt(os.path.join(self.input_folder, "%s.txt"%(mode)))
        if mode == "train":
            self.df_img = self.downsample_data(self.df_img_all, dataset_args["update_ds"]["weight_positive"])
        else:
            self.df_img = self.df_img_all
        self.list_img_name =self.df_img["img_name"].values

        self.transform_image = transform_image
        self.transform_label = transform_label

    def update_train_ds(self, weight_positive = 0.8):
        if self.mode == "train":
            self.df_img = self.downsample_data(self.df_img_all, weight_positive)
        else:
            self.df_img = self.df_img_all
        self.list_img_name =self.df_img["img_name"].values
    
    def downsample_data(self, df, weight_positive):
        df_label0 = df[df["label"] == 0]
        df_label1 = df[df["label"] == 1]

        positive_length = len(df_label1)
        negative_length = int(positive_length*(1-weight_positive)/weight_positive)
        df_label0 = df_label0.sample(negative_length)
        return pd.concat([df_label0, df_label1])

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

        if (self.mode == "train") and (self.augmentation is not None):
            # print(self.mode)
            augmented = self.augmentation(image = image, mask = mask)
            image, mask = augmented['image'], augmented['mask']
        return self.transform_image(np.array(image)), self.transform_label(np.array(mask))
        
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
        list_label = []
        for line in list_:
            img_name, label = line.replace("\n","").split(",")
            list_img_name.append(img_name)
            list_label.append(int(label))
        file_.close()
        return pd.DataFrame({"img_name": list_img_name, "label":list_label})

from dataset.PneumothoraxDataset import *
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
import pydicom
import random

from dataset.PneumothoraxDataset import *
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
import pydicom
import random

class PneumothoraxPreprocess:
    def __init__(self, input_folder, output_folder, val_size=0.3, reset_folder=True):
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.masks = pd.read_csv(os.path.join(self.input_folder, "stage_2_train.csv"))
        self.masks["label"] = (self.masks["EncodedPixels"] != "-1").astype(np.float)

        self.train_df = self.init_df("train")
        self.test_df = self.init_df("test")
        self.train_df, self.val_df, y_train, y_val = train_test_split(self.train_df, self.train_df["label"], test_size=val_size, random_state = 100)

        self.images_output_folder = os.path.join(self.output_folder, "images")
        self.masks_output_folder = os.path.join(self.output_folder, "masks")
        if reset_folder:
            self.create_new_dir(self.images_output_folder)
            self.create_new_dir(self.masks_output_folder)

        self.combine_df = pd.concat([self.train_df, self.test_df, self.val_df])
        self.list_df = {
            "train": self.train_df,
            "test": self.test_df,
            "val": self.val_df
        }

    def init_df(self, mode="train"):
        list_path = sorted(glob(self.input_folder+os.sep+"dicom-images-%s/*/*/*.dcm" % (mode)))
        
        list_uid = list(map(find_uid, list_path))
        df = pd.DataFrame({"ImageId": list_uid, "path": list_path})
        df = df.join(self.masks.set_index("ImageId"), on="ImageId")
        df = df[df["label"].notna()]
        df = df.drop_duplicates("ImageId")
        
        return df

    def create_new_dir(self, dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def save_txt(self, mode="train"):
        file_path = os.path.join(self.output_folder, "%s.txt" % mode)
        if os.path.isfile(file_path):
            os.remove(file_path)
        file_ = open(file_path, "w")
        for index, row in tqdm(self.list_df[mode].iterrows()):
            uid = row["ImageId"]
            label = row["label"]
            file_.writelines("%s.jpg,%d\n"%(uid, int(label)))
        file_.close()

    def save_img(self, path, encoded_pixels):
        if not os.path.isfile(path):
            return
        uid, img, mask = load_sample(path, encoded_pixels)
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))
        plt.imsave(os.path.join(self.images_output_folder, uid+".jpg"), img, cmap="gray")
        plt.imsave(os.path.join(self.masks_output_folder, uid+".jpg"), mask, cmap="gray")

    def save_imgs(self):
        for index, row in tqdm(self.combine_df.iterrows()):
            path = row["path"]
            encoded_pixels = self.masks[self.masks["ImageId"]==row["ImageId"]]["EncodedPixels"].values
            # print("encoded_pixels", encoded_pixels)
            self.save_img(path, encoded_pixels)
    def plot_pneumothorax(self):
        index = random.randint(0, len(self.combine_df))
        row = self.combine_df.iloc[index]
        encoded_pixels = self.masks[self.masks["ImageId"]==row["ImageId"]]["EncodedPixels"].values
        while (len(encoded_pixels) <=1):
            index = random.randint(0, len(self.combine_df))
            row = self.combine_df.iloc[index]
            encoded_pixels = self.masks[self.masks["ImageId"]==row["ImageId"]]["EncodedPixels"].values
        uid, img, mask = load_sample(row["path"], encoded_pixels)
        plot_sample(img, mask)
    def plot_non_pneumothorax(self):
        index = random.randint(0, len(self.combine_df))
        row = self.combine_df.iloc[index]
        encoded_pixels = self.masks[self.masks["ImageId"]==row["ImageId"]]["EncodedPixels"].values
        while (len(encoded_pixels) >1):
            index = random.randint(0, len(self.combine_df))
            row = self.combine_df.iloc[index]
            encoded_pixels = self.masks[self.masks["ImageId"]==row["ImageId"]]["EncodedPixels"].values
        uid, img, mask = load_sample(row["path"], encoded_pixels)
        plot_sample(img, mask)


def plot_sample(img, mask):
    plt.imshow(img, cmap = "bone")
    plt.imshow(mask, cmap = "binary", alpha = 0.3)
    plt.show()
def find_uid(path):
  return path.split(os.sep)[-1].replace(".dcm","")

def load_sample(img_path, encoded_pixels, width = 1024, height = 1024):
    data = pydicom.dcmread(img_path)
    img = pydicom.read_file(img_path).pixel_array
    
    uid = data.SOPInstanceUID
    mask = np.zeros((width, height)).astype(np.float)
    for encoded_pixel in encoded_pixels:
        if encoded_pixel != "-1":
            mask += rle2mask(encoded_pixel, width, height)
    mask = np.clip(mask, 0.0, 255.0)
    return uid, img, mask.T

def positive_negative_count(df):
    positive = len(df[df["label"] == 1])
    negative = len(df[df["label"] == 0])
    return positive, negative, positive/(negative+1e-9)

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
