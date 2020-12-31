import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset.transform import Rescale
from dataset.dataset import cifar10
from dataset.MenWomanDataset import MenWomanDataset
from model.CNN import CNN, TransferNet
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from utils.utils import len_train_datatset
from model.optimizer import RAdam
from torchvision.models import resnet18
from utils.metric import Accuracy

config_files = "/content/drive/MyDrive/vinbrain_internship/vinbrain-internship/classifier/config/configs_colabs.py"
#data config
batch_size = 64
split_train_val = 0.7
device = "gpu"
gpu_id = 0
classes = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])
dataset = {
    "class":MenWomanDataset,
    "argument":{
        "path":"/content/data"
    }
}

#train config
net = {
    "class":TransferNet,
    "net_args":{
        "model_base":resnet18,
        "pretrain":True
    }
}
loss_function = nn.CrossEntropyLoss
lr = 0.001
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
lr_schedule = None
optimizer ={
    "class": SGD,
    "optimizer_args":{
        "momentum":0.9
    }
}
num_epochs = 10
output_folder = "/content/drive/MyDrive/vinbrain_internship/model/menWoman"

loss_file = "loss_file.txt"
metric = {
    "class":Accuracy,
    "metric_args":{
        "threshold": 0.5,
        "from_logits":True
    }
}
steps_save_loss = 100