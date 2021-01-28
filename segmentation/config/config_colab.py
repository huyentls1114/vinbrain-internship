import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.BrainTumorDataset import BrainTumorDataset
from model.metric import Dice_Score
from model.unet import Unet, UnetDynamic
from model.backbone_densenet import BackboneDense121
from model.backbone import BackboneResnet18VGG, BackboneDensenet121VGG,BackboneEfficientB0VGG
from utils.utils import len_train_datatset
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
#data config
image_size = 256
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_Pneumothorax/BackboneEfficientB0VGG_focalloss_augment_1e-4"
loss_file = "loss_file.txt"
config_file_path = "/content/vinbrain-internship/segmentation/config/config_colab.py"


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size)
])
transform_label = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size)
])

import albumentations as A
from dataset.transform import *
from dataset.PneumothoraxDataset import *
dataset = {
    "class": PneumothoraxDataset,
    "dataset_args":{
        "input_folder":"/content/data/Pneumothorax",
        "augmentation": A.Compose([
            A.Resize(512, 512),
            RandomCrop(450, 450, p = 0.5),
            A.OneOf([
                RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
                RandomTranspose(p = 0.5),
            ]),
            RandomRotate((0, 270), p = 0.5),
            RandomBlur(blur_limit = 10, p = 0.1),
            CLAHE(p = 0.1),
            RandomBrightnessContrast(p = 0.1)
        ]),
        "update_ds": "downsample"
    }
}

#train config
import os
from model.unet import UnetCRF
from model.backbone import BackboneEfficientB0VGG
num_classes = 1
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneEfficientB0VGG,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "bilinear": False,
            "pixel_shuffle":True
        }
    }
}

device = "gpu"
gpu_id = 0

batch_size = 32
num_epochs = 200

metric = {
    "class":Dice_Score,
    "metric_args":{
        "threshold":0.5,
        "epsilon":1e-4
    }
}
num_classes = 1
from loss.loss import FocalLoss
loss_function = {
    "class": FocalLoss,
    "loss_args":{
        "alpha": 0.25,
        "gamma": 2
    }
}

#optimizer
lr = 1e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

from torch.optim.lr_scheduler import OneCycleLR
steps_per_epoch = int(len_train_datatset(dataset, transform_train, transform_label, 1)/batch_size)
num_epochs = 40
lr_scheduler = {
    "class":OneCycleLR,
    "metric": None,
    "step_type":"batch",
    "schedule_args":{
        "max_lr": 1e-4,
        "epochs":num_epochs,
        "steps_per_epoch":steps_per_epoch+1,
        "final_div_factor":10,
    }    
}

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
lr_scheduler_crf = {
    "class":CosineAnnealingWarmRestarts,
    "metric": None,
    "step_type":"iteration",
    "schedule_args":{
        "T_0": 1,
        "T_mult":2
    }    
}

update_ds = {
    "method":"downsample"
}