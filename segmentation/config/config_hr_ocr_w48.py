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
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_Pneumothorax/HRnet_OCR_w48_comboloss_augment_RLOP1e-22"
loss_file = "loss_file.txt"
config_file_path = "/content/vinbrain-internship/segmentation/config/config_hr_ocr_w48.py"
num_classes = 1
from model.seg_hrnet_ocr import get_seg_model
net = {
    "class": get_seg_model,
    "net_args":{
        "cfg" :{
            "DATASET":{
                "NUM_CLASSES":1,
            },
            "MODEL":{
                "NUM_OUTPUTS":2,
                "OCR":{
                    "MID_CHANNELS":512,
                    "KEY_CHANNELS":256,
                    "DROPOUT":0.05,
                    "SCALE":1,
                },
                "ALIGN_CORNERS":True,
                "PRETRAINED":'/content/drive/MyDrive/vinbrain_internship/pretrained_model/HRNet/hrnetv2_w48_imagenet_pretrained.pth',
                "EXTRA":{
                    "FINAL_CONV_KERNEL":1,
                    "STAGE1":{
                        "NUM_MODULES":1,
                        "NUM_BRANCHES":1,
                        "BLOCK": "BOTTLENECK",
                        "NUM_BLOCKS":[4],
                        "NUM_CHANNELS":[64],
                        "FUSE_METHOD":"SUM"
                    },
                    "STAGE2":{
                        "NUM_MODULES":1,
                        "NUM_BRANCHES":2,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS":[4, 4],
                        "NUM_CHANNELS":[48, 96],
                        "FUSE_METHOD":"SUM"
                    },
                    "STAGE3":{
                        "NUM_MODULES":4,
                        "NUM_BRANCHES":3,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS":[4, 4, 4],
                        "NUM_CHANNELS":[48, 96, 192],
                        "FUSE_METHOD":"SUM"
                    },
                    "STAGE4":{
                        "NUM_MODULES":3,
                        "NUM_BRANCHES":4,
                        "BLOCK": "BASIC",
                        "NUM_BLOCKS":[4, 4, 4, 4],
                        "NUM_CHANNELS":[48, 96, 192, 384],
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
from metric.metric_hr import DiceMetric
metric = {
    "class":DiceMetric,
    "metric_args":{
        "num_classes": num_classes,
        "threshold": 0.5
    }
}

from loss.loss_hr import CrossEntropyOCR
loss_function = {
    "class":CrossEntropyOCR,
    "loss_args":{
        # "ignore_label": 255
        "align_corners" : net["net_args"]["cfg"]["MODEL"]["ALIGN_CORNERS"], 
        "num_outputs" : net["net_args"]["cfg"]["MODEL"]["NUM_OUTPUTS"], 
        "balance_weights" : [0.4, 1]
        }
}


#optimizer
lr = 1e-2
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
        "patience":8,
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