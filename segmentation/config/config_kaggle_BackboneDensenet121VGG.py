import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from dataset.BrainTumorDataset import BrainTumorDataset
from model.metric import Dice_Score
from model.unet import Unet, UnetDynamic
from model.backbone_densenet import BackboneDense121
from model.backbone import BackboneResnet18VGG, BackboneDensenet121VGG,BackboneEfficientB0VGG
from utils.utils import len_train_datatset
from loss.loss import DiceLoss

#data config
image_size = 512
output_folder = "/kaggle/working/model/BackboneDensenet121VGG_comboloss_ROLR_1e-4"
loss_file = "loss_file.txt"
config_file_path = "/kaggle/working/vinbrain-internship/segmentation/config/config_kaggle_BackboneDensenet121VGG.py"

from model.unet import Unet
from model.backbone import BackboneDensenet121VGG
num_classes = 1
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneDensenet121VGG,
        "encoder_args":{
            "pretrained":True,           
        },
        "decoder_args":{
            "bilinear": False,
            "pixel_shuffle":True
        }
    }
}

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
    transforms.Resize(image_size)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
    transforms.Resize(image_size)
])
transform_label = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
    transforms.Resize(image_size)
])

import albumentations as A
from dataset.transform import *
from dataset.PneumothoraxDataset import *
dataset = {
    "class": PneumothoraxDataset,
    "dataset_args":{
        "input_folder":"/kaggle/input/pneumothorax-512/Pneumothorax",
        "augmentation": A.Compose([
            A.Resize(576, 576),
            RandomRotate((-30, 30), p = 0.5),
            A.OneOf([
                # RandomVerticalFlip(p=0.5),
                RandomHorizontalFlip(p=0.5),
                # RandomTranspose(p = 0.5),
            ]),
            RandomBlur(blur_limit = 3.1, p = 0.1),
            # CLAHE(p = 0.1),
            RandomBrightnessContrast(p = 0.1),
            RandomCrop(512, 512, p = 0.5)
        ]),
        "update_ds": {
            "weight_positive": 0.8
        }
    }
}


device = "gpu"
gpu_id = 0

batch_size = 4
num_epochs = 20

# from pattern_model import 
from won.loss import DiceMetric
metric = {
    "class":DiceMetric,
    "metric_args":{
        "num_classes": num_classes,
        "threshold": 0.5
    }
}
# from pattern_model import MixedLoss
# from loss.loss import MixedLoss
from won.loss import ComboLoss
loss_function = {
    "class":ComboLoss,
    "loss_args":{
        "weights": {
            "bce":3,
            "dice":1,
            "focal":4
        }
    }
}


#optimizer
lr = 5e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = {
    "class": ReduceLROnPlateau,
    "metric":"val_loss_avg",
    "step_type":"epoch",
    "schedule_args":{
        "mode":"min",
        "factor":0.5,
        "patience":4,
        "threshold":1e-2,
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