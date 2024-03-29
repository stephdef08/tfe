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
                   max_tensor_size, threshold):
    img = cv2.imread(path)
    mosaic = extractor.extract_patches(img)
    i = 0

    counter = Counter()

    for j in range(len(mosaic)):
        img = transforms.Resize((224, 224))(mosaic[j])
        img = transforms.ToTensor()(img)
        tensor_cpu[i] = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)
        i += 1

        if i == max_tensor_size:
            tensor_gpu = tensor_cpu.to(device='cuda:0')
            with torch.no_grad():
                out = model(tensor_gpu).cpu()
            for k in range(max_tensor_size):
                names = r.lrange(np.array2string(utils.binarize(out[k], threshold)), 0, -1)

                for l in range(len(names)):
                    js = json.loads(names[l].decode("utf-8"))
                    counter[js["name"]] += js["value"]
            i = 0

    if i != 0:
        tensor_gpu = tensor_cpu.to(device='cuda:0')
        with torch.no_grad():
            out = model(tensor_gpu).cpu()
        for j in range(i):
            names = r.lrange(np.array2string(utils.binarize(out[j], threshold)), 0, -1)

            for k in range(len(names)):
                js = json.loads(names[k].decode("utf-8"))
                counter[js["name"]] += js["value"]

    return counter

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        '--path',
        help='path to the query image',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor to use',
        default='densenet'
    )

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

    parser.add_argument(
        '--num_patches',
        help='number of patches extracted for random extraction',
        default=0,
        type=int
    )

    parser.add_argument(
        '--weights',
        help='file storing the weights of the model'
    )

    args = parser.parse_args()

    if args.path is None:
        print(usage)
        exit()

    model = None

    if args.extractor == 'densenet':
        model = densenet.Model(num_features=args.num_features,
                               threshold=args.threshold, weights=args.weights)

    if model is None:
        print("Unkown feature extractor")
        exit()

    r = redis.Redis(host='localhost', port='6379', db=0)

    max_tensor_size = 32

    tensor_cpu = torch.zeros(max_tensor_size, 3, 224, 224)
    tensor_gpu = torch.zeros(max_tensor_size, 3, 224, 224, device='cuda:0')

    extractor = utils.Extract(extraction=args.extraction, num_features=args.num_features,
                               num_patches=args.num_patches)

    counter = retrieve_image(r, args.path, model, extractor, tensor_cpu,
                             tensor_gpu, max_tensor_size, args.threshold)

    print(counter.most_common(5))
