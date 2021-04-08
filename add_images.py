import redis
import os
from argparse import ArgumentParser, ArgumentTypeError
from PIL import Image
import torch
from torchvision import transforms
import densenet
import numpy as np
import json
import utils
import cv2
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter

@torch.no_grad()
def extract_patches(subdir, file, extractor):
    img = cv2.imread(os.path.join(subdir, file))
    mosaic = extractor.extract_patches(img)
    return (mosaic, os.path.join(subdir, file))

@torch.no_grad()
def add_redis(tensor_cpu, tensor_gpu, model, i, r, results, name_list, max_tensor_size, transform):
    for res in results:
        mosaic, name = res
        counter = Counter()
        for j in range(len(mosaic)):
            name_list.append((name, 1 / len(mosaic)))
            img = transforms.Resize((224, 224))(mosaic[j])
            tensor_cpu[i] = transforms.ToTensor()(img)
            i += 1

            if i == max_tensor_size:
                tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
                out = model(tensor_gpu)
                for k in range(max_tensor_size):
                    r.lpush(np.array2string(out[k]),
                            json.dumps({"name": name_list[k][0],
                                        "value": name_list[k][1]}))
                name_list.clear()
                i = 0
    return i

if __name__ == "__main__":
    usage = "python3 add_images.py --path <folder> [--extractor <algorithm>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path to the folder that contains the images to add',
    )

    parser.add_argument(
        '--extractor',
        help='feature extractor that is used',
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

    args = parser.parse_args()

    if args.path is None:
        print(usage)
        exit()

    if args.path[-1] != '/':
        print("The path mentionned is not a folder")
        exit()

    model = None

    if args.extractor == 'densenet':
        model = densenet.Model(num_features=args.num_features,
                               threshold=args.threshold)

    if model is None:
        print("Unkown feature extractor")
        exit()

    r = redis.Redis(host='localhost', port='6379', db=0)

    transform = torch.nn.Sequential(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    max_tensor_size = 32

    tensor_cpu = torch.zeros(max_tensor_size, 3, 224, 224)
    tensor_gpu = torch.zeros(max_tensor_size, 3, 224, 224, device='cuda:0')
    name_list = []
    counter = 0

    extractor = [utils.Extract(extraction=args.extraction,
                               num_patches=args.num_patches)] * max_tensor_size

    num_cores = multiprocessing.cpu_count()
    with Parallel(n_jobs=num_cores, prefer="threads") as parallel:
        for subdir, dirs, files in os.walk(args.path):
            nbr_iter = len(files) // max_tensor_size
            sub = [subdir] * max_tensor_size

            for i in range(nbr_iter):
                result = parallel(delayed(extract_patches)(s, f, e)
                                  for s, f, e in zip(sub, files[max_tensor_size * i: max_tensor_size * (i+1)], extractor))
                counter = add_redis(tensor_cpu, tensor_gpu, model, counter, r,
                                    result, name_list, max_tensor_size, transform)

            rest = len(files) % max_tensor_size

            if rest != 0:
                sub = [subdir] * rest
                result = parallel(delayed(extract_patches)(s, f, e)
                                  for s, f, e in zip(sub, files[max_tensor_size * nbr_iter:], extractor[:rest+1]))
                counter = add_redis(tensor_cpu, tensor_gpu, model, counter, r,
                                    result, name_list, max_tensor_size, transform)

        if counter != 0:
            tensor_gpu = transform(tensor_cpu).to(device='cuda:0')
            out = model(tensor_gpu)
            for j in range(counter):
                r.lpush(np.array2string(out[j]),
                        json.dumps({"name": name_list[j][0],
                                    "value": name_list[j][1]}))
