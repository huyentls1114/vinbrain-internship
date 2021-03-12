import torch
import torch.nn as nn

class Deeplabv3(nn.Module):
    def __init__(self, repo_or_dir, model, pretrained):
        super(Deeplabv3, self).__init__()
        self.model = torch.hub.load(repo_or_dir, model, pretrained)

    def forward(self, x):
        return self.model(x)["out"]
