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