import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset.transform import Rescale
from dataset.dataset import cifar10
from net.CNN import CNN
from net.optimizer import *

config_files = "config/configs.py"
#data config
batch_size = 16
split_train_val = 0.7
device = "cpu"
gpu_id = 0
classes = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
dataset = {
    "name":"cifar10",
    "class":cifar10,
    "argument":{
        "path":"E:\data\cifar10"
    }
}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                        ])

#train config
net = CNN
loss_function = nn.CrossEntropyLoss
lr = 0.001
optimizer ={
    "class": SGDoptimizer,
    "optimizer_args":{
        "momentum":0.9
    }
}
num_epochs = 2
output_folder = "E:\model\classify_cifar"