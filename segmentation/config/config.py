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
        "input_folder":"E:\data\BrainTumor"
    }
}

#train config
num_classes = 1

net = {
    "class":Unet,
    "net_args":{
        "backbone_class": BackBoneResnet101,
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
loss_function = {
    "class": nn.BCEWithLogitsLoss,
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
        "min_lr":1-5
    }
}
steps_save_loss = 2
steps_save_image = 2