import redis
import os
from argparse import ArgumentParser, ArgumentTypeError
from PIL import Image
import torch
from torchvision import transforms
import densenet
import numpy as np
import json

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

    max_tensor_size = 16

    tensor_cpu = torch.zeros(max_tensor_size, 3, 224, 224)
    tensor_gpu = torch.zeros(max_tensor_size, 3, 224, 224, device='cuda:0')
    name_list = []
    i = 0

    for subdir, dirs, files in os.walk(args.path):
        for file in files:
            #print(i)
            name_list.append(os.path.join(subdir, file))
            img = Image.open(name_list[i]).convert('RGB')
            img = transforms.Resize((224, 224))(img)
            tensor_cpu[i] = transforms.ToTensor()(img)

            if i < max_tensor_size - 1:
                i += 1
            else:
                tensor_gpu = transform(tensor_cpu).to(device='cuda:0')
                out = model(tensor_gpu)
                for j in range(max_tensor_size):
                    r.lpush(np.array2string(out[j]),
                            json.dumps({"name": name_list[j]}))
                name_list.clear()
                i = 0
            #print(json.loads(r.lrange(np.array2string(out), 0, 0)[0].decode("utf-8"))["name"])

    if i != 0:
        tensor_gpu = transform(tensor_cpu).to(device='cuda:0')
        out = model(tensor_gpu)
        for j in range(i):
            r.lpush(np.array2string(out[j]),
                    json.dumps({"name": name_list[j]}))
