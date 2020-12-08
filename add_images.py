import redis
import os
from argparse import ArgumentParser, ArgumentTypeError
import densenet
import cv2

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

    model = None

    if args.extractor == 'densenet':
        model = densenet.model()

    if model is None:
        print("Unkown feature extractor")
        exit()

    r = redis.Redis(host='localhost', port='6379', db=0)

    for subdir, dirs, files in os.walk(args.path):
        for file in files:
            img = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, (224, 224))
            img_resized_planar = np.zeros((3, 224, 224))
            for i in range(3):
                img_resized_planar[i, :, :] = img_resized[:, :, i]

            img_tensor = torch.from_numpy(img_resized_planar)
