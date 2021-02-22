import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, densenet121
from dataset.transform import Rescale
from dataset.dataset import cifar10
from dataset.MenWomanDataset import MenWomanDataset
from model.CNN import CNN, TransferNet
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, OneCycleLR
from utils.utils import len_train_datatset
from model.optimizer import RAdam
from torchvision.models import resnet18, vgg16
from utils.metric import Accuracy

config_files = "/content/vinbrain-internship/classifier/config/config_cifar.py"
#data config
batch_size = 8
split_train_val = 0.7
device = "gpu"
gpu_id = 0
classes = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                        ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                        ])

dataset = {
    "name":"cifar10",
    "class":cifar10,
    "argument":{
        "path":"/content/drive/MyDrive/vinbrain_internship/data/cifar10"
    }
}

#train config
# net = {
#     "class":CNN,
#     "net_args":{}
# }
configs.net = {
    "class": TransferNet,
    "net_args":{
        "model_base": resnet18,
        "fc_channels":[512],
        "pretrain": False,
        "num_classes":2
    }
}
loss_function = nn.CrossEntropyLoss
lr = 0.01

optimizer ={
    "class": SGD,
    "optimizer_args":{
        "momentum":0.9
    }
}
num_epochs = 10
output_folder = "/content/drive/MyDrive/vinbrain_internship/model_classify/cifar10_resnet18_SGD_StepLR"

loss_file = "loss_file.txt"
metric = {
    "class":Accuracy,
    "metric_args":{
        "threshold": 0.5,
        "from_logits":True
    }
}
steps_save_loss = 500
lr_schedule = {
    "class": StepLR,
    "metric":None,
    "step_type":"epoch",
    "schedule_args":{
        "step_size":1,
        "gamma":0.1,
    }
}

# lr_schedule = {
#     "class": OneCycleLR,
#     "metric":None,
#     "step_type":"iteration",
#     "schedule_args":{
#         "max_lr":.0001,
#         "epochs": num_epochs,
#         "steps_per_epoch":steps_per_epoch+1
#     }
# }
