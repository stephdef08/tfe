import torchvision.models as models
import torch
import torch.nn as nn
import utils

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.densenet161(pretrained=True).cuda()
        self.model.eval()

    def forward(self, input):
        tensor = self.model(input).to(device='cpu')
        bin = utils.binarize(tensor)
        return bin
