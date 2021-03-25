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
import time


class Model(nn.Module):
    def __init__(self, eval=True, batch_size=32, num_features=128, threshold=.5):
        super(Model, self).__init__()
        self.conv_net = models.densenet121(pretrained=True).cuda()

        for param in self.conv_net.parameters():
            param.requires_grad = False

        self.conv_net.classifier = nn.Linear(1024, 4096).cuda()
        self.relu = nn.LeakyReLU().cuda()

        self.first_conv1 = nn.Conv2d(3, 96, kernel_size=8, padding=1, stride=16).cuda()
        self.first_conv2 = nn.MaxPool2d(3, 4, 1).cuda()

        self.second_conv1 = nn.Conv2d(3, 96, kernel_size=7, padding=4, stride=32).cuda()
        self.second_conv2 = nn.MaxPool2d(7, 2, 3).cuda()

        self.linear = nn.Linear(7168, num_features).cuda()

        self.load_state_dict(torch.load("model" + str(num_features)))

        self.threshold = threshold

        if eval == True:
            self.eval()
            self.eval = True
        else:
            self.train()
            self.eval = False
            self.batch_size = batch_size

    def forward(self, input):
        norm = nn.functional.normalize
        if self.eval is True:
            with torch.no_grad():
                tensor1 = self.conv_net(input)
                tensor1 = norm(self.relu(tensor1))

                tensor2 = self.first_conv1(input)
                tensor2 = self.first_conv2(tensor2)
                tensor2 = torch.flatten(tensor2, start_dim=1)

                tensor3 = self.second_conv1(input)
                tensor3 = self.second_conv2(tensor3)
                tensor3 = torch.flatten(tensor3, start_dim=1)

                tensor4 = norm(torch.cat((tensor2, tensor3), 1))

                return self.linear(torch.cat((tensor1, tensor4), 1)).to(device='cpu')
        else:
            tensor1 = self.conv_net(input)
            tensor1 = norm(self.relu(tensor1))

            tensor2 = self.first_conv1(input)
            tensor2 = self.first_conv2(tensor2)
            tensor2 = norm(torch.flatten(tensor2, start_dim=1))

            tensor3 = self.second_conv1(input)
            tensor3 = self.second_conv2(tensor3)
            tensor3 = norm(torch.flatten(tensor3, start_dim=1))

            tensor4 = norm(torch.cat((tensor2, tensor3), 1))

            return self.linear(torch.cat((tensor1, tensor4), 1))

    def train_epochs(self, dir, epochs):
        lr = 0.01

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        data = dataset.DRDataset(root=dir)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
                                             shuffle=True, num_workers=4,
                                             pin_memory=True)

        loss_function = torch.nn.TripletMarginLoss()

        image0_gpu = torch.zeros((self.batch_size, 3, 224, 224), device='cuda:0')
        image1_gpu = torch.zeros((self.batch_size, 3, 224, 224), device='cuda:0')
        image2_gpu = torch.zeros((self.batch_size, 3, 224, 224), device='cuda:0')

        loss_list = []
        try:
            for epoch in range(epochs):
                start_time = time.time()

                for i, (image0, image1, image2) in enumerate(loader):
                    image0_gpu = image0.to(device='cuda:0')
                    image1_gpu = image1.to(device='cuda:0')
                    image2_gpu = image2.to(device='cuda:0')

                    out0 = self.forward(image0_gpu)
                    out1 = self.forward(image1_gpu)
                    out2 = self.forward(image2_gpu)

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

                if (epoch + 1) % 4:
                    lr /= 2
                    for param in optimizer.param_groups:
                        param['lr'] = lr

                torch.save(self.state_dict(), 'model32')

        except KeyboardInterrupt:
            print("Interrupted")

if __name__ == "__main__":
    m = Model(False)
    m.train_epochs("tmp/", 5)
