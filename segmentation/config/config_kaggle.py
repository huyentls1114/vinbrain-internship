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
from loss.loss import DiceLoss

#data config
image_size = 256
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()

])
transform_label = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

dataset = {
    "class": BrainTumorDataset,
    "dataset_args":{
        "input_folder":"/kaggle/input/braintumor/BrainTumor"
    }
}

#train config
num_classes = 1
from model.backbone import BackboneEfficientB0VGG
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

batch_size = 16
# num_epochs = 200

metric = {
    "class":Dice_Score,
    "metric_args":{
        "threshold":0.5,
        "epsilon":1e-4
    }
}
from loss.loss import FocalLoss
loss_function = {
    "class": FocalLoss,
    "loss_args":{
        "alpha": 0,
        "gamma": 0
    }
}

output_folder = "/kaggle/working/BrainTumor_BackboneEfficientB0VGG_focaloss_onecyle0_0_2e-3"
loss_file = "loss_file.txt"
config_file_path = "/kaggle/working/vinbrain-internship/segmentation/config/config_colab.py"

#optimizer
lr = 1e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

from torch.optim.lr_scheduler import OneCycleLR
steps_per_epoch = int(len_train_datatset(dataset, transform_train, transform_label, 1)/batch_size)
num_epochs = 60
lr_scheduler = {
    "class":OneCycleLR,
    "metric": None,
    "step_type":"batch",
    "schedule_args":{
        "max_lr":0.006,
        "epochs":num_epochs,
        "steps_per_epoch":steps_per_epoch+1
    }    
}