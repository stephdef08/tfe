import torchvision.models as models
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.densenet161(pretrained=True).cuda()
        self.model.eval()

    def forward(self, input):
        return self.model(input)
