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

    args = parser.parse_args()

    if args.path is None:
        print(usage)
        exit()

    if args.path[-1] != '/':
        print("The path mentionned is not a folder")
        exit()

    model = None

    if args.extractor == 'densenet':
        model = densenet.Model()

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
    i = 0

    for subdir, dirs, files in os.walk(args.path):
        for file in files:
            img = cv2.imread(os.path.join(subdir, file))
            mosaic = utils.extract_patches(img)
            for j in range(len(mosaic)):
                name_list.append(os.path.join(subdir, file))
                img = transforms.Resize((224, 224))(mosaic[j])
                tensor_cpu[i] = transforms.ToTensor()(img)
                i += 1

                if i == max_tensor_size:
                    tensor_gpu = transform(tensor_cpu.to(device='cuda:0'))
                    out = model(tensor_gpu)
                    for k in range(max_tensor_size):
                        r.lpush(np.array2string(out[k]),
                                json.dumps({"name": name_list[k]}))
                    name_list.clear()
                    i = 0

    if i != 0:
        tensor_gpu = transform(tensor_cpu).to(device='cuda:0')
        out = model(tensor_gpu)
        for j in range(i):
            r.lpush(np.array2string(out[j]),
                    json.dumps({"name": name_list[j]}))
