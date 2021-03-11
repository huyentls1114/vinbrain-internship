import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.unet import Unet, UnetDynamic
from model.backbone_densenet import BackboneDense121
from model.backbone import BackboneResnet18VGG, BackboneDensenet121VGG,BackboneEfficientB0VGG
from utils.utils import len_train_datatset
from torch.optim.lr_scheduler import OneCycleLR
import albumentations as A
import segmentation_models_pytorch as smp

#data config
image_size = 257
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_BrainTumor/PSP_BCE_rate0.8_augment_RLOP1e-3"
loss_file = "loss_file.txt"
config_file_path = "/content/vinbrain-internship/segmentation/config/config_psp.py"

num_classes = 1
from model.pspnet import PSPNet
net = {
    "class":PSPNet,
    "net_args":{
        "layers":50,
        "classes": num_classes,
        "zoom_factor": 8
    }
}

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
    transforms.Resize((image_size, image_size))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
    transforms.Resize((image_size, image_size))
])
transform_label = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.540,0.540,0.540), std = (0.264,0.264,0.264)),
    transforms.Resize((image_size, image_size))
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

from loss.loss_psp import BCEWithLogits_Compose
loss_function = {
    "class": BCEWithLogits_Compose,
    "loss_args":{
        "aux_weight": 0.4,
    }
}


#optimizer
from torch.optim import SGD
lr = 1e-2
optimizer = {
    "class":SGD,
    "optimizer_args":{
        "momentum": 0.9,
        "weight_decay": 0.0001
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