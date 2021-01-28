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


def handler(signal, frame):
    global SIGINTTRIG
    SIGINTTRIG = True

class Model(nn.Module):
    def __init__(self, eval=True):
        super(Model, self).__init__()
        self.model = models.densenet121(pretrained=True).cuda()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Linear(1024, 128).cuda()
        self.relu = nn.ReLU(inplace=True).cuda()
        self.model.load_state_dict(torch.load("model"))

        if eval == True:
            self.model.eval()
            self.eval = True
        else:
            self.model.train()
            self.eval = False

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
            bin = utils.binarize(tensor)
            return bin

    def _process(self, mosaic, tensor_cpu, tensor_gpu, func):
        counter = 0
        for m in mosaic:
            m = transforms.Resize((224, 224))(m)
            tensor_cpu[counter] = transforms.ToTensor()(m)
            counter += 1

            if counter == max_tensor_size:
                tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
                out = self.model(tensor_gpu)
                for o in out:
                    func(o)

                counter = 0

        if counter != 0:
            tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
            out = self.model(tensor_gpu)
            for j in range(counter):
                func(out[j])

            counter = 0

    def train(self, dir, epochs):
        global SIGINTTRIG
        SIGINTTRIG = False
        opt = torch.optim.SGD(self.model.parameters(), lr=.001, momentum=.9)
        triplet_loss = nn.TripletMarginLoss()

        data = dataset.DRDataset(root=dir)
        loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True,
                                             num_workers=4, pin_memory=True)

        extractor = Extract()

        max_tensor_size = 32

        tensor_cpu = torch.zeros(max_tensor_size, 3, 224, 224)
        tensor_gpu = torch.zeros(max_tensor_size, 3, 224, 224, device='cuda:0')

        transform = torch.nn.Sequential(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

        features_image = []

        similar = []
        not_similar = []

        loss_similar = torch.zeros(1, device='cuda:0')
        loss_not_similar = torch.zeros(1, device='cuda:0')

        signal(SIGINT, handler)

        for epoch in range(epochs):
            if SIGINTTRIG == True:
                break

            for i, (image0, image1, image2) in enumerate(loader):
                print(image0.shape)
                print(image1.shape)
                print(image2.shape)
                return
                print(i)

                images0 = cv2.cvtColor(np.array(images0), cv2.COLOR_RGB2BGR)
                mosaic = extractor.extract_patches(images0)

                self._process(mosaic, tensor_cpu, tensor_gpu,
                              lambda a : features_image.append(a))

                images1 = cv2.cvtColor(np.array(images1), cv2.COLOR_RGB2BGR)
                mosaic = extractor.extract_patches(images1)

                self._process(mosaic, tensor_cpu, tensor_gpu,
                              lambda a : similar.append(a))

                images2 = cv2.cvtColor(np.array(images2), cv2.COLOR_RGB2BGR)
                mosaic = extractor.extract_patches(images2)

                self._process(mosaic, tensor_cpu, tensor_gpu,
                              lambda a : not_similar_image.append(a))

                loss_similar[0] = sum(similar)
                loss_not_similar[0] = sum(not_similar)




                optimizer.zero_grad(set_to_none=True)
