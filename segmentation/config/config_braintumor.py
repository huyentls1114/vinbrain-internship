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
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_BrainTumor/BackboneDensenet121VGG_BCE_augment_RLOP1e-3"
loss_file = "loss_file.txt"
config_file_path = "/content/drive/MyDrive/vinbrain_internship/configs/config_braintumor.py"

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
from dataset.BrainTumorDataset import *
dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"/content/data/BrainTumor",
        "augmentation": A.Compose([
            A.Resize(512, 512),
            RandomCrop(450, 450),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5)
        ])
    }
}


device = "gpu"
gpu_id = 0

batch_size = 16
num_epochs = 40

# from pattern_model import 
from won.loss import DiceMetric
metric = {
    "class":DiceMetric,
    "metric_args":{
        "num_classes": num_classes,
        "threshold": 0.5
    }
}

#loss function
loss_function = {
    "class": nn.BCEWithLogitsLoss,
    "loss_args":{
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