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
        "input_folder":"/content/data/BrainTumor"
    }
}

#train config
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

batch_size = 16
num_epochs = 200

metric = {
    "class":Dice_Score,
    "metric_args":{
        "threshold":0.5,
        "epsilon":1e-4
    }
}
num_classes = 1
loss_function = {
    "class": nn.BCEWithLogitsLoss,
    "loss_args":{
    }
}

output_folder = "/content/drive/MyDrive/vinbrain_internship/model/BrainTumor_EfficientB0VGG_removelastCN_OCLR_1e-2"
loss_file = "loss_file.txt"
config_file_path = "/content/vinbrain-internship/segmentation/config/config_colab.py"

#optimizer
lr = 1e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

from torch.optim.lr_scheduler import OneCycleLR
num_epochs = 30
lr_schedule = {
    "class":OneCycleLR,
    "metric": None,
    "step_type":"batch",
    "schedule_args":{
        "max_lr":0.01,
        "epochs":configs.num_epochs,
        "steps_per_epoch":configs.steps_per_epoch+1
    }    
}
