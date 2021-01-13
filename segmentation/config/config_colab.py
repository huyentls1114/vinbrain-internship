import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.BrainTumorDataset import BrainTumorDataset
from model.metric import Dice_Score
from model.unet import Unet, UnetDynamic
from model.backbone import BackboneOriginal, BackBoneResnet18, BackBoneResnet101, BackBoneResnet101Dynamic
from model.backbone import BackBoneResnet18Dynamic
from utils.utils import len_train_datatset

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
    "class":UnetDynamic,
    "net_args":{
        "backbone_class": BackBoneResnet18Dynamic,
        "encoder_args":{
            "pretrained":True           
        },
        "decoder_args":{
            "img_size": image_size
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

output_folder = "/content/drive/MyDrive/vinbrain_internship/model/BrainTumor_Resnet18_Dynamic_CosineAnnealingWarmRestarts_T0_1_1e-3"
loss_file = "loss_file.txt"
config_file_path = "/content/vinbrain-internship/segmentation/config/config_colab.py"

#optimizer
lr = 1e-3
optimizer = {
    "class":Adam,
    "optimizer_args":{
    }
}

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
lr_scheduler = {
    "class": CosineAnnealingWarmRestarts,
    "metric":"epoch",
    "step_type":"iteration",
    "schedule_args":{
        "T_0":1,
        "T_mult":3,
        "eta_min":1e-5,
    }
}

steps_save_loss = 100
steps_save_image = 100