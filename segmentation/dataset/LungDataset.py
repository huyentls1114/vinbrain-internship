from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import conver_numpy_image, contour

class LungDataset(Dataset):
    def __init__(self, dataset_args, transform_image, transform_label, mode = "train"):
        super(LungDataset, self).__init__()
        if "augmentation" in dataset_args.keys():
            self.augmentation = dataset_args["augmentation"]
        else:
            self.augmentation = None


        self.covid_chesxray_folder = dataset_args["covid_chesxray_folder"]
        self.covid_chesxray_image_folder = os.path.join(self.covid_chesxray_folder, "images")
        self.covid_chesxray_mask_folder = os.path.join(self.covid_chesxray_folder, "masks")
        self.covid_chesxray_names = self.read_txt(self.covid_chesxray_folder, mode)

        self.transform_image = transform_image
        self.transform_label = transform_label

        self.mode = mode

    def __len__(self):
        return len(self.covid_chesxray_names)
    
    def __getitem__(self, idx):
        img_name = self.covid_chesxray_names[idx]
        img_path = os.path.join(self.covid_chesxray_image_folder, img_name)
        image = plt.imread(img_path)
        mask_path = os.path.join(self.covid_chesxray_mask_folder, img_name)
        mask = plt.imread(mask_path)

        
        if mask.shape ==3:
            mask = mask[:, :, 0]

        if (self.mode == "train") and (self.augmentation is not None):
            # print(self.mode)
            augmented = self.augmentation(image = image, mask = mask)
            image, mask = augmented['image'], augmented['mask']
        
        image = process_img(image, channel = 3)
        # if image.shape == 3:
            # import pdb; pdb.set_trace()
        print(image.shape)
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

    def read_txt(self, input_folder, mode):
        if mode == "test":
            mode = "val"
        train_txt_file = os.path.join(input_folder, "%s.txt"%(mode))
        file_ = open(train_txt_file)
        list_ = file_.readlines()
        list_img_name = []
        for line in list_:
            img_name = line.replace("\n","")
            list_img_name.append(img_name)
        file_.close()
        return list_img_name

    def show_sample(self, batch_size = 4):
        list_imgs = []
        list_masks = []

        #random list idx
        list_idx = np.arange(self.__len__())
        np.random.shuffle(list_idx)
        list_idx = list_idx[0:batch_size]

        for i in range(batch_size):
            image, mask = self.__getitem__(list_idx[i])
            if image.shape == 3:
                image = image[:,:,0]
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
            if img.shape[2]!=mask.shape[2]:
               mask = np.concatenate([mask]*3, axis = 2)
            combine = np.hstack([img, mask, ct])
            list_combine.append(combine)
        plt.imshow(np.vstack(list_combine)[:,:,0]/255.0, cmap = "gray")
        plt.axis('off')
        plt.show()
def process_img(image, channel = 3):
    "return image with shape [w, h, channel]"
    print(image.shape)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        return np.dstack((image, )*channel)
    if (channel >1) and (image.shape[2] == 1):
        return np.dstack((image, )*channel)
    if (channel == 1) and (image.shape[2] == 3):
        return image[:,:, 0:1]
    if image.shape[2] == channel:
        return image
    if image.shape[2] > channel:
        return image[:,:,:channel]
