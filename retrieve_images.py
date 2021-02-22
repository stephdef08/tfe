import redis
from argparse import ArgumentParser, ArgumentTypeError
from PIL import Image
import torch
from torchvision import transforms
import densenet
import numpy as np
import json
import utils
import os
from collections import Counter
import cv2

@torch.no_grad()
def retrieve_image(r, path, model, extractor, tensor_cpu, tensor_gpu,
                   max_tensor_size):
    transform = torch.nn.Sequential(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    img = cv2.imread(path)
    mosaic = extractor.extract_patches(img)
    i = 0

    counter = Counter()

    for j in range(len(mosaic)):
        img = transforms.Resize((224, 224))(mosaic[j])
        tensor_cpu[i] = transforms.ToTensor()(img)
        i += 1

        if i == max_tensor_size:
            tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
            out = model(tensor_gpu)
            for k in range(max_tensor_size):
                names = r.lrange(np.array2string(out[k]), 0, -1)

                for l in range(len(names)):
                    js = json.loads(names[l].decode("utf-8"))
                    print(js["name"], js["value"])
                    counter[js["name"]] += js["value"]
            i = 0

    if i != 0:
        tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
        out = model(tensor_gpu)
        for j in range(i):
            names = r.lrange(np.array2string(out[j]), 0, -1)

            for k in range(len(names)):
                js = json.loads(names[k].decode("utf-8"))
                print(js["name"], js["value"])
                counter[js["name"]] += js["value"]

    return counter

if __name__ == "__main__":
    usage = "python3 retrieve_images.py --path <patch> [--extractor <algorithm>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path to the query patch',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor to use',
        default='densenet'
    )

    args = parser.parse_args()

    if args.path is None:
        print(usage)
        exit()

    model = None

    if args.extractor == 'densenet':
        model = densenet.Model()

    if model is None:
        print("Unkown feature extractor")
        exit()

    r = redis.Redis(host='localhost', port='6379', db=0)

    max_tensor_size = 32

    tensor_cpu = torch.zeros(max_tensor_size, 3, 224, 224)
    tensor_gpu = torch.zeros(max_tensor_size, 3, 224, 224, device='cuda:0')

    extractor = utils.Extract()

    counter = retrieve_image(r, args.path, model, extractor, tensor_cpu,
                             tensor_gpu, max_tensor_size)

    print(counter.most_common(100))
