import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, img_size = 32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class TransferNet(nn.Module):
    def __init__(self, model_base, num_classes, pretrain = False, requires_grad = True):
        super(TransferNet, self).__init__()
        self.model_base = model_base(pretrain)
        for param in self.model_base.parameters():
            param.requires_grad = requires_grad

        l = [module for module in self.model_base.modules() if type(module) != nn.Sequential]
        self.model_base = nn.Sequential(
                        *l,
        )
        self.model_base[-1].out_features = num_classes
    
    def forward(self, x):
        return self.model_base(x)

