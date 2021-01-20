import random
from utils.utils import conver_numpy_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import numpy as np

class ComposeTransform:
    def __init__(self, transform, transform_image = True, transform_mask = True):
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.transform = transform
    def __call__(self, image, mask):
        "image, mask: tensor"
        seed = random.randint(0, 100000) 
        torch.manual_seed(seed)
        if self.transform_image:
            image = self.transform(image)
        torch.manual_seed(seed)
        if self.transform_mask:
            mask = self.transform(mask)
        return image, mask
    def visualize(self, image, mask):
        "image, mask: numpy array images"
        image = np.array(image[:,:,0])
        mask = np.array(mask[:,:,0])
        image_tensor = transforms.ToTensor()(image)
        mask_tensor = transforms.ToTensor()(mask)
        image_tf, mask_tf = self.__call__(image_tensor, mask_tensor)
        image_tf, mask_tf = conver_numpy_image(image_tf), conver_numpy_image(mask_tf)
        before = np.hstack([image, mask])
        after = np.hstack([image_tf, mask_tf])
        plt.imshow(np.vstack([before, after[:,:,0]]))
        plt.show()

        

# class ElasticDeformation:
#     def __init__(self, alpha, sigma):
#         self.alpha = alpha
#         self.sigma = sigma
#     def __call__(self, image, mask):
#         seed = random.randint(10000)
#         # ker