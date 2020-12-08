import redis
import os
from argparse import ArgumentParser, ArgumentTypeError
from PIL import Image
from torchvision import transforms
import torch
import densenet

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

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for subdir, dirs, files in os.walk(args.path):
        for file in files:
            img = Image.open(os.path.join(subdir, file)).convert('RGB')
            img_t = transform(img).cuda()
            batch = torch.unsqueeze(img_t, 0)
            out = model(batch)
