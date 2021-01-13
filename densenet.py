import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.densenet121(pretrained=True).cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1024, 128).cuda()
        self.model.load_state_dict(torch.load("model"))
        self.model.eval()

    def forward(self, input):
        tensor = self.model(input)
        tensor = F.relu(tensor, inplace=True).to(device='cpu')
        bin = utils.binarize(tensor)
        return bin
