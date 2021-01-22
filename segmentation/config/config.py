import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.BrainTumorDataset import BrainTumorDataset
from model.metric import Dice_Score
from model.unet import Unet
from model.backbone import BackboneOriginal, BackBoneResnet18, BackBoneResnet101
#data config
image_size = 192
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
dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"E:\data\BrainTumor",
        "augmentation": A.Compose([
            A.Resize(512, 512),
            RandomCrop(450, 450),
            RandomVerticalFlip(p=0.5),
            RandomHorizontalFlip(p=0.5),
            RandomRotate((0, 270), p = 0.5),
            RandomBlur(blur_limit = 10, p = 0.5)
        ])
    }
}

#train config
from model.backbone import BackboneOriginal
num_classes = 1
net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackboneOriginal,
        "encoder_args":{},
        "decoder_args":{
            "bilinear": True
        }
    }
}

device = "cpu"
gpu_id = 0

batch_size = 4
num_epochs = 10

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
    }
}

output_folder = "E:\model\segmentation\BrainTumor"
loss_file = "loss_file.txt"
config_file_path = "E:\\vinbrain-internship\segmentation\config\config.py"

#optimizer
lr = 1e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}
lr_scheduler = {
    "class": ReduceLROnPlateau,
    "metric":"val_loss",
    "step_type":"epoch",
    "schedule_args":{
        "mode":"min",
        "factor":0.5,
        "patience":4,
        "threshold":1e-2,
        "min_lr":1e-5
    }
}
steps_save_loss = 2
steps_save_image = 2