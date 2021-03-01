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
  def __init__(self, model_base,
                      fc_channels, 
                      num_classes, 
                      pretrain = False):
    super(TransferNet, self).__init__()
    
    self.model_extractor = 
      self.create_extractor(model_base, pretrained)
    self.classify_layers = 
      self.create_fully_layers()
    
    self.net = nn.Sequential(
      self.model_extractor,
      nn.Flatten(),
      self.classify_layers,
      nn.Linear(fc_channels[-1], num_classes)
    )        
  def forward(self, x):
      return self.net(x)


    def create_model_extractor(model_base):
        model_base = model_base(pretrain)
        model_base_list = list(model_base.children())[:-1]
        
        for model in model_base_list:
            for param in model.parameters():
                param.requires_grad = requires_grad
        return nn.Sequential(*model_base_list)

    def create_fully_layers(fc_channels):
        classify_layers = []
        for i in range(len(fc_channels)-1):
            classify_layers.append(nn.Linear(fc_channels[i], fc_channels[i+1])),
            classify_layers.append(nn.ReLU()),
            classify_layers.append(nn.Dropout(0.5))
        return nn.Sequential(*classify_layers)
        
