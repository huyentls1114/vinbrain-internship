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
import segmentation_models_pytorch as smp

#data config
image_size = 256
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_Pneumothorax/HRnet_comboloss_augment_RLOP1e-4"
loss_file = "loss_file.txt"
config_file_path = "/content/drive/MyDrive/vinbrain_internship/configs/config_hr.py"
num_classes = 1
from model.seg_hrnet import get_seg_model
net = {
    "class": get_seg_model,
    "net_args":{
        "cfg" :{
            "DATASET"{
                "NUM_CLASSES":1,
            },
            "MODEL":{
                "PRETRAINED":'/content/vinbrain-internship/segmentation/pretrained_models/hrnet_w18_small_model_v1.pth',
                "EXTRA":{
                    "FINAL_CONV_KERNEL":1,
                    "STAGE1":{
                        "NUM_MODULES":1,
                        "NUM_BRANCHES":1,
                        "BLOCK": "BOTTLENECK",
                        "NUM_BLOCKS":[1],
                        "NUM_CHANNELS":[32],
                        "FUSE_METHOD":"SUM"
                    },
                    "STAGE2":{
                        "NUM_MODULES":1,
                        "NUM_BRANCHES":2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS":[2, 2],
                        "NUM_CHANNELS":[16, 32],
                        "FUSE_METHOD":"SUM"
                    },
                    "STAGE3":{
                        "NUM_MODULES":1,
                        "NUM_BRANCHES":3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS":[2, 2, 2],
                        "NUM_CHANNELS":[16, 32, 64],
                        "FUSE_METHOD":"SUM"
                    },
                    "STAGE4":{
                        "NUM_MODULES":1,
                        "NUM_BRANCHES":4,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS":[2, 2, 2, 2],
                        "NUM_CHANNELS":[16, 32, 64, 128],
                        "FUSE_METHOD":"SUM"
                    }
                }
            }
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
        "input_folder":"/content/data/Pneumothorax",
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

batch_size = 16
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
lr = 1e-4
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
        "factor":0.3,
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