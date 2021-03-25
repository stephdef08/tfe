import os
import torchvision
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import pickle
from collections import defaultdict

# code inspired from
# https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py

class DRDataset(Dataset):

    def __init__(self, root='image_folder', transform=None, train=True):
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0, saturation=1, hue=.5),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        self.train = 'train' if train else 'test'

        self.root = root
        self.transform = transform
        self.rev_dict = {}
        self.image_dict = {}
        self.big_dict = {}
        L = []

        self.num_classes = 0

        self.num_elements = 0

        for i, j in enumerate(os.listdir(os.path.join(root))):
            self.rev_dict[i] = j
            self.image_dict[j] = np.array(os.listdir(os.path.join(root, j)))
            for k in os.listdir(os.path.join(root, j)):
                self.big_dict[self.num_elements] = (k, i)
                self.num_elements += 1

            self.num_classes += 1

    def _sample(self, idx):
        im, im_class = self.big_dict[idx]
        im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])
        numbers = list(range(im_class)) + list(range(im_class+1, self.num_classes))
        class3 = np.random.choice(numbers)
        im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
        p1 = os.path.join(self.root, self.rev_dict[im_class], im)
        p2 = os.path.join(self.root, self.rev_dict[im_class], im2)
        p3 = os.path.join(self.root, self.rev_dict[class3], im3)
        return [p1, p2, p3]

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        paths = self._sample(idx)
        images = []
        for i in paths:
            tmp = Image.open(i).convert('RGB')
            tmp = self.transform(tmp)
            images.append(tmp)

        return (images[0], images[1], images[2])

class AddDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.list_img = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        for subdir, dirs, files in os.walk(root):
            for f in files:
                self.list_img.append(os.path.join(subdir, f))

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        return self.transform(
            Image.open(self.list_img[idx]).convert('RGB')
            ), self.list_img[idx]
