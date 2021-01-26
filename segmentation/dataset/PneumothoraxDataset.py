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
        for line in list_:
            img_name = line.replace("\n","")
            list_img_name.append(img_name)
        file_.close()
        return list_img_name

class PneumothoraxPreprocess:
    def __init__(self, input_folder, output_folder, val_size = 0.3, reset_folder = True):
        self.input_folder = input_folder
        self.output_folder = output_folder
        train = sorted(glob(input_folder+os.sep+"dicom-images-train/*/*/*.dcm"))
        self.train, self.val = train_test_split(train, test_size = val_size)
        self.test = sorted(glob(input_folder+os.sep+"dicom-images-test/*/*/*.dcm"))
        self.masks = pd.read_csv(os.path.join(self.input_folder, "stage_2_train.csv"))
        self.all_img_path = np.concatenate([self.train, self.val, self.test], axis =0)

        self.images_output_folder = os.path.join(self.output_folder, "images")
        self.masks_output_folder = os.path.join(self.output_folder, "masks")
        if reset_folder:
            self.create_new_dir(self.images_output_folder)
            self.create_new_dir(self.masks_output_folder)

        # self.save_txt(self.train, os.path.join(self.output_folder, "train.txt"))
        # self.save_txt(self.test, os.path.join(self.output_folder, "test.txt"))
        # self.save_txt(self.val, os.path.join(self.output_folder, "val.txt"))
        self.list_img_dict = {
            "train": self.train,
            "test": self.test,
            "val": self.val
        }

    def create_new_dir(self, dir_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def save_txt(self, mode = "train"):
        file_path_dict = {
            "train":os.path.join(self.output_folder, "train.txt"),
            "test":os.path.join(self.output_folder, "test.txt"),
            "val":os.path.join(self.output_folder, "val.txt")
        }
        file_path = file_path_dict[mode]
        list_img = self.list_img_dict[mode]

        if os.path.isfile(file_path):
            os.remove(file_path)
        file_ = open(file_path, "w")
        for img in list_img:
            uid = img.split(os.sep)[-1].replace(".dcm",".jpg")
            # print(os.path.join(self.output_folder+os.sep+"images", uid))
            if not os.path.isfile(os.path.join(self.output_folder+os.sep+"images", uid)):
                continue
            file_.writelines(uid+"\n")
        file_.close()

    def save_img(self):
        for img_path in self.all_img_path:
            try:
                uid, img, mask = load_sample(img_path, self.masks)
                img = cv2.resize(img, (512, 512))
                mask = cv2.resize(mask, (512, 512))
            except Exception as e:
                print(e)
                continue
            plt.imsave(os.path.join(self.images_output_folder, uid+".jpg"), img, cmap = "gray")
            plt.imsave(os.path.join(self.masks_output_folder, uid+".jpg"), mask, cmap = "gray")            

def load_sample(img_path, masks, width = 1024, height = 1024):
    data = pydicom.dcmread(img_path)
    img = pydicom.read_file(img_path).pixel_array
    
    uid = data.SOPInstanceUID
    encoded_pixels = masks[masks["ImageId"] == uid].values[0][2]
    if encoded_pixels == "-1":
        mask = np.zeros((width, height)).astype(np.float)
    else:
        mask = rle2mask(encoded_pixels, width, height)
    return uid, img, mask.T


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