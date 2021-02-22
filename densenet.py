import torchvision.models as models
import torch
import torch.nn as nn
import utils
import torch.optim as optim
from torchvision import transforms
import dataset
import numpy as np
import cv2
from utils import Extract
from signal import signal, SIGINT
import time


def handler(signal, frame):
    global SIGINTTRIG
    SIGINTTRIG = True

class Model(nn.Module):
    def __init__(self, eval=True, batch_size=32):
        super(Model, self).__init__()
        self.model = models.densenet121(pretrained=True).cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1024, 32).cuda()
        self.relu = nn.LeakyReLU().cuda()
        self.model.load_state_dict(torch.load('model32'))

        if eval == True:
            self.model.eval()
            self.eval = True
        else:
            self.model.train()
            self.eval = False
            self.tmp_loss = torch.zeros(batch_size, device='cuda:0')
            self.batch_size = batch_size

    def forward(self, input):
        if self.eval is True:
            with torch.no_grad():
                tensor = self.model(input)
                tensor = self.relu(tensor).to(device='cpu')
                bin = utils.binarize(tensor)
                return bin
        else:
            tensor = self.model(input)
            tensor = self.relu(tensor).to(device='cpu')
            return tensor

    def train(self, dir, epochs):
        global SIGINTTRIG
        SIGINTTRIG = False

        lr = 0.001

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.5)

        data = dataset.DRDataset(root=dir)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=4,
                                             pin_memory=True)

        loss_function = torch.nn.TripletMarginLoss()

        signal(SIGINT, handler)

        image0_gpu = torch.zeros((self.batch_size, 224, 224, 3), device='cuda:0')
        image1_gpu = torch.zeros((self.batch_size, 224, 224, 3), device='cuda:0')
        image2_gpu = torch.zeros((self.batch_size, 224, 224, 3), device='cuda:0')

        loss_list = []

        for epoch in range(epochs):
            if SIGINTTRIG == True:
                break

            start_time = time.time()

            for i, (image0, image1, image2) in enumerate(loader):
                image0_gpu = image0.to(device='cuda:0')
                image1_gpu = image1.to(device='cuda:0')
                image2_gpu = image2.to(device='cuda:0')

                out0 = self.model(image0_gpu)
                out1 = self.model(image1_gpu)
                out2 = self.model(image2_gpu)

                loss = loss_function(out0, out1, out2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())

                if i % 100 == 0:
                    print("epoch {}, batch {}, loss = {}".format(epoch, i,
                                                                 np.mean(loss_list)))
                    loss_list.clear()

            print("time for epoch {}".format(time.time()- start_time))

            torch.save(self.model.state_dict(), 'model16')

            if (epoch + 1) % 4:
                for param in optimizer.param_groups:
                    lr /= 2
                    param['lr'] = lr

if __name__ == "__main__":
    m = Model(False)
    m.train("tmp/", 5)
