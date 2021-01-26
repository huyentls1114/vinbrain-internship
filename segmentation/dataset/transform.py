import random
from utils.utils import conver_numpy_image
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import collections
import albumentations as A
from albumentations.core.transforms_interface import DualTransform
from .functional import *

def visualize(transform, image, mask):
    "image, mask: numpy array images"
    image = np.array(image[:,:,0])
    mask = np.array(mask[:,:,0])
    augmented= transform(image = image, mask = mask)
    image_tf, mask_tf = augmented["image"], augmented["mask"]
    before = np.hstack([image, mask])
    plt.imshow(before)
    plt.show()
    after = np.hstack([image_tf, mask_tf])
    plt.imshow(after)
    plt.show()

class RandomCrop(A.DualTransform):
    def __init__(self, height, width, always_apply = False, p =1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
    def apply(self, img, h_start = 0, w_start = 0, **params):
        return crop(img, self.height, self.width, h_start, w_start)
    def get_params(self):
        return {"h_start": random.random(), "w_start": random.random()}
    def get_transform_init_args_names(self):
        return ("height","width")


class RandomVerticalFlip(A.DualTransform):
    def apply(self, img, **params):
        return vertical_flip(img)
    def get_transform_init_args_names(self):
        return ()

class RandomHorizontalFlip(A.DualTransform):
    def apply(self, img, **params):
        return horizontal_flip(img)
    def get_transform_init_args_names(self):
        return ()

class RandomRotate(A.DualTransform):
    def __init__(self, limit, always_apply = False, p = 1.0):
        super(RandomRotate, self).__init__(always_apply, p)
        self.limit = tuple(limit)
    def apply(self, img, angle, **params):
        return rotate(img, angle)
    def get_params(self):
        return {"angle":random.uniform(self.limit[0], self.limit[1])}
    def get_transform_init_args_names(self):
        return ("limit")

class RandomBlur(A.ImageOnlyTransform):
    def __init__(self, blur_limit, always_apply = False, p = 1.0):
        "value of blur_limit from 1 to 3"
        assert blur_limit>3
        super(RandomBlur, self).__init__(always_apply, p)
        self.blur_limit = tuple([3, blur_limit])
    def apply(self, img, ksize, **params):
        return cv2.blur(img, (ksize, ksize))
    def get_params(self):
        min_ = self.blur_limit[0]
        max_ = self.blur_limit[1]
        return {"ksize": int(random.choice(np.arange(min_, max_, 2)))}
    def get_transform_init_args_names(self):
        return ("blur_limit")
class RandomTranspose(A.DualTransform):
    def apply(self, img, **params):
        return transpose(img)
    def get_transform_init_args_names(self):
        return ()
class RandomBrightnessContrast(A.ImageOnlyTransform):
    def __init__(self, brightness_limit = 0.2, contrast_limit= 0.2, always_apply = False, p = 1):
        super(RandomBrightnessContrast, self).__init__(always_apply, p)
        if isinstance(brightness_limit, float):
            self.brightness_limit = tuple([-brightness_limit, brightness_limit])
        else:
            self.brightness_limit = brightness_limit
        if isinstance(contrast_limit, float):
            self.contrast_limit = tuple([-contrast_limit, contrast_limit])
        else:
            self.contrast_limit = contrast_limit
    def apply(self, img, alpha = 1, beta = 0, **params):
        return brightness_and_constrast(img, alpha, beta)
    def get_params(self):
        return {
            "alpha": 1 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
            "beta": random.uniform(self.brightness_limit[0], self.brightness_limit[1])
        }
    def get_transform_init_args_names(self):
        return ("brightness_limit", "contrast_limit")

class CLAHE(A.ImageOnlyTransform):
    def __init__(self, tileGridSize = (8,8), always_apply = False, p = 1):
        super(CLAHE, self).__init__(always_apply, p)
        self.tileGridSize = tileGridSize

    def apply(self, img, clipLimit, **params):
        return clahe(img, clipLimit, self.tileGridSize)
    
    def get_params(self):
        return {
            "clipLimit": random.uniform(3, 4)
        }
    def get_transform_init_args_names(self):
        return ("tileGridSize")

