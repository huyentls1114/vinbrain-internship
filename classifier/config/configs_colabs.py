import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset.transform import Rescale
from dataset.dataset import cifar10
from model.CNN import CNN
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from utils.utils import len_train_datatset
from model.optimizer import RAdam

config_files = "/content/drive/MyDrive/vinbrain_internship/vinbrain-internship/classifier/config/configs_colabs.py"
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
        "path":"cifar10"
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
steps_per_epoch = int(len_train_datatset(dataset, transform, split_train_val)/batch_size)
# lr_schedule = {
#     "class": StepLR,
#     "metric":None,
#     "step_type":"epoch",
#     "schedule_args":{
#         "step_size":1,
#         "gamma":0.1,
#     }
# }
lr_schedule = None
optimizer ={
    "class": SGD,
    "optimizer_args":{
        "momentum":0.9
    }
}
num_epochs = 10
output_folder = "/content/drive/MyDrive/vinbrain_internship/model/cifar10_optimizer"

loss_file = "loss_file.txt"