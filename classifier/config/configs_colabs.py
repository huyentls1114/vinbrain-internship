import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset.transform import Rescale
from dataset.MenWomanDataset import MenWomanDataset
from model.CNN import CNN, TransferNet
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, OneCycleLR
from utils.utils import len_train_datatset
from model.optimizer import RAdam
from torchvision.models import resnet18, vgg16, densenet121
from utils.metric import Accuracy

output_folder = "/content/drive/MyDrive/vinbrain_internship/model/menWoman_densenet121_RMSProp_1e-3"
config_files = "/content/drive/MyDrive/vinbrain_internship/vinbrain-internship/classifier/config/configs_colabs.py"
#data config
batch_size = 64
split_train_val = 0.7
device = "gpu"
gpu_id = 0
img_size = 224

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(img_size*115/100)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(img_size*115/100)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
dataset = {
    "class":MenWomanDataset,
    "dataset_args":{
        "path":"/content/data",
        "classes" : ["men", "woman"],
    }
}

#train config
net = {
    "class":TransferNet,
    "net_args":{
        "model_base":densenet121,
        "fc_channels":[1024*7*7, 1024],
        "pretrain":True,
        "num_classes":2
    }
}

loss = {
    "class": nn.CrossEntropyLoss,
    "loss_args":{

    }
lr = 1e-3
steps_per_epoch = int(len_train_datatset(dataset, transform_train, split_train_val)/batch_size)
# lr_schedule = {
#     "class": StepLR,
#     "metric":None,
#     "step_type":"epoch",
#     "schedule_args":{
#         "step_size":1,
#         "gamma":0.1,
#     }
# }
# optimizer={
#     "class":Adam,
#     "optimizer_args":{
#     }}
optimizer = {
    "class":RMSprop,
    "optimizer_args":{
    }
}

num_epochs = 20

loss_file = "loss_file.txt"
metric = {
    "class":Accuracy,
    "metric_args":{
        "threshold": 0.5,
        "from_logits":True
    }
}

lr_scheduler = {
    "class": OneCycleLR,
    "metric":None,
    "step_type":"iteration",
    "schedule_args":{
        "max_lr":.0001,
        "epochs": num_epochs,
        "steps_per_epoch":steps_per_epoch+1
    }
}
