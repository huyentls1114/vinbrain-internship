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
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_Pneumothorax/BackboneResnet34VGG_mixloss_metter_augment2_Plateau"
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
                # RandomHorizontalFlip(p=0.5),
                # RandomTranspose(p = 0.5),
            ]),
            RandomRotate((-30, 30), p = 0.5),
            RandomBlur(blur_limit = 3.1, p = 0.1),
            CLAHE(p = 0.1),
            RandomBrightnessContrast(p = 0.1)
        ]),
        "update_ds": {
            "weight_positive": 1
        }
    }
}

#train config
import os
from model.unet import UnetCRF
from model.backbone import BackboneResnet34VGG
num_classes = 1
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneResnet34VGG,
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

from pattern_model import Meter
metric = {
    "class":Meter,
    "metric_args":{
    }
}
from pattern_model import MixedLoss
loss_function = {
    "class":MixedLoss,
    "loss_args":{
        "alpha":0.5,
        "gamma":2
    }
}

#optimizer
lr = 1e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

lr_scheduler = {
    "class": ReduceLROnPlateau,
    "metric":"val_loss_avg",
    "step_type":"epoch",
    "schedule_args":{
        "mode":"min",
        "factor":0.1,
        "patience":4,
        "threshold":1e-4,
        "min_lr":1e-6
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