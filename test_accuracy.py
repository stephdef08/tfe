import torch
from torch.utils.data import Dataset
import os
from collections import defaultdict
import numpy as np
from torchvision import transforms
import redis
from collections import Counter
import utils
from densenet import Model
from retrieve_images import retrieve_image
from joblib import Parallel, delayed
import multiprocessing
import cv2
import json
from argparse import ArgumentParser, ArgumentTypeError

class TestDataset(Dataset):
    def __init__(self, root='image_folder/val'):
        self.root = root

        list_classes = list(range(67))

        self.dic_img = defaultdict(list)
        self.img_list = []

        for i in list_classes:
            for img in os.listdir(os.path.join(root, str(i))):
                self.dic_img[i].append(os.path.join(root, str(i), img))

        nbr_empty = 0
        to_delete = []

        while True:
            for key in self.dic_img:
                if (not self.dic_img[key]) is False:
                    img = np.random.choice(self.dic_img[key])
                    self.dic_img[key].remove(img)
                    self.img_list.append(img)
                else:
                    to_delete.append(key)

            for key in to_delete:
                self.dic_img.pop(key, None)

            to_delete.clear()

            if len(self.img_list) > 1000 or len(self.dic_img) == 0:
                break

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        return self.img_list[idx]

@torch.no_grad()
def extract_patches(file, extractor):
    img = cv2.imread(file)
    mosaic = extractor.extract_patches(img)
    return (mosaic, file)

@torch.no_grad()
def test(num_features, threshold, extraction):
    r = redis.Redis(host='localhost', port='6379', db=0)

    model = Model(num_features=num_features, threshold=threshold)

    data = TestDataset()

    max_tensor_size = 32
    loader = torch.utils.data.DataLoader(data, batch_size=max_tensor_size,
                                         shuffle=True, num_workers=4, pin_memory=True)

    top_1_acc = 0
    top_5_acc = 0

    tensor_cpu = torch.zeros(max_tensor_size, 3, 224, 224)
    tensor_gpu = torch.zeros(max_tensor_size, 3, 224, 224, device='cuda:0')
    i = 0

    transform = torch.nn.Sequential(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    extractor = [utils.Extract(extraction=extraction)] * max_tensor_size

    counter = Counter()

    num_cores = multiprocessing.cpu_count()
    with Parallel(n_jobs=num_cores, prefer="threads") as parallel:
        nbr_images = 0
        hit_first = 0
        hit_five = 0
        for i, images in enumerate(loader):
            results = parallel(delayed(extract_patches)(f, e)
                              for f, e in zip(images, extractor))
            for res in results:
                nbr_images += 1
                mosaic, name = res
                idx_tensor = 0
                hit_rate = Counter()
                for j in range(len(mosaic)):
                    #add rotations of input image
                    img = transforms.Resize((224, 224))(mosaic[j])
                    tensor_cpu[idx_tensor] = transforms.ToTensor()(img)
                    idx_tensor += 1

                    if idx_tensor == max_tensor_size:
                        tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
                        out = model(tensor_gpu)
                        for k in range(max_tensor_size):
                            names = r.lrange(np.array2string(out[k]), 0, -1)
                            hit = {}
                            for l in range(len(names)):
                                js = json.loads(names[l].decode("utf-8"))
                                counter[js["name"]] += js["value"]
                                if js["name"] not in hit:
                                    hit[js["name"]] = True
                                    hit_rate[js["name"]] += 1
                        idx_tensor = 0

                if idx_tensor != 0:
                    tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
                    out = model(tensor_gpu)
                    for j in range(idx_tensor):
                        names = r.lrange(np.array2string(out[j]), 0, -1)
                        hit = {}
                        for k in range(len(names)):
                            js = json.loads(names[k].decode("utf-8"))
                            counter[js["name"]] += js["value"]
                            if js["name"] not in hit:
                                hit[js["name"]] = True
                                hit_rate[js["name"]] += 1

                similar = [tmp[0] for tmp in counter.most_common(5)]
                hit = [hit_rate[v] for v in similar]
                counter.clear()

                already_found_5 = False

                for j in range(len(similar)):
                    begin_retr = similar[j].find("/", similar[j].find("/")+1) + 1
                    end_retr = similar[j].find("/", begin_retr)

                    begin_test = name.find("/", name.find("/")+1) + 1
                    end_test = name.find("/", begin_test)

                    if similar[j][begin_retr:end_retr] == name[begin_test: end_test] \
                        and already_found_5 is False:
                        top_5_acc += 1
                        already_found_5 = True
                        if j == 0:
                            top_1_acc += 1

                if len(hit) > 0:
                    hit_first += hit[0] / len(mosaic)
                    hit_five += np.mean(hit) / len(mosaic)

                # if len(hit) > 0:
                #     print("top 1 accuracy {}, hit rate : {}, round {}".format((top_1_acc / nbr_images), hit[0] / len(mosaic), nbr_images))
                # else:
                #     print("top 1 accuracy {}, hit rate : 0, round {}".format((top_1_acc / nbr_images), nbr_images))
                # print("top 5 accuracy {}, round {} ".format((top_5_acc / nbr_images), nbr_images))


    print("top 1 accuracy : ", top_1_acc / data.__len__())
    print("top 5 accuracy : ", top_5_acc / data.__len__())
    print("Hit rate first : ", hit_first / data.__len__())
    print("Mean hit rate first five : ", hit_five / data.__len__())

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--num_features',
        help='number of features to extract',
        default=32,
        type=int
    )

    parser.add_argument(
        '--threshold',
        help='threshold to use for binarization of features',
        default=.5,
        type=float
    )

    parser.add_argument(
        '--extraction',
        help='method used to compute the mosaic',
        default='kmeans'
    )

    args = parser.parse_args()

    test(args.num_features, args.threshold, args.extraction)
