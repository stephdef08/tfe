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
                    transforms.Resize((32, 32)),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
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

    """
    def __init__(self, root='image_folder', transform=None):
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomVerticalFlip(.5),
                    transforms.RandomHorizontalFlip(.5),
                    transforms.ColorJitter(brightness=0, contrast=0,
                                           saturation=1, hue=.5),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        self.root = root
        self.transform = transform

        self.test = pickle.load(open(os.path.join(root, 'test'), 'rb'),
                                encoding='bytes')
        self.train = pickle.load(open(os.path.join(root, 'train'), 'rb'),
                                 encoding='bytes')

        self.dict_label = defaultdict(list)
        self.size = 0

        self.len_test = len(self.test[b'fine_labels'])

        for i, label in enumerate(self.test[b'fine_labels']):
            self.dict_label[label].append(i)
            self.size += 1
        for i, label in enumerate(self.train[b'fine_labels']):
            self.dict_label[label].append(i + self.len_test)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        label = -1
        images = []

        if idx < len(self.test[b'fine_labels']):
            label = self.test[b'fine_labels'][idx]
            images.append(self.test[b'data'][idx])
        else:
            idx -= self.len_test
            label = self.train[b'fine_labels'][idx]
            images.append(self.train[b'data'][idx])

        idx = np.random.choice(self.dict_label[label])

        if idx < len(self.test[b'fine_labels']):
            images.append(self.test[b'data'][idx])
        else:
            images.append(self.train[b'data'][idx-self.len_test])

        choices_negative = list(range(label)) + list(range(label+1, 100))
        negative = np.random.choice(choices_negative)
        idx = np.random.choice(self.dict_label[negative])

        if idx < len(self.test[b'fine_labels']):
            images.append(self.test[b'data'][idx])
        else:
            images.append(self.train[b'data'][idx-self.len_test])

        for i in range(3):
            images[i] = cv2.merge([np.array(images[i][:1024]).reshape((32, 32)),
                                   np.array(images[i][1024:2048]).reshape((32, 32)),
                                   np.array(images[i][2048:]).reshape((32, 32))])
            images[i] = Image.fromarray(images[i])
            images[i] = self.transform(images[i])

        return (images[0], images[1], images[2])
        """
