from openslide import OpenSlide
import redis
from argparse import ArgumentParser, ArgumentTypeError
import densenet
import utils
import cv2
import numpy as np
from torchvision import transforms
import torch
import KMeansRex
from joblib import Parallel, delayed
import multiprocessing
import json


@torch.no_grad()
def detect_flesh(j, slide, nbr_patch_x):
    threshold = 224 * 224 * 225 * .7
    patch_list = []
    for i in range(1, nbr_patch_x):
        patch = slide.read_region((i * 224, j * 224), 0,
                                  (224, 224)).convert('L')
        #wrong, have to change that
        #totally white patches will be accepted
        #have to threshold the thumb
        _, patch_threshold = cv2.threshold(np.array(patch), 0, 255,
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if patch_threshold.sum() < threshold:
            patch_list.append((i, j))

    return patch_list

@torch.no_grad()
def read_rgb_regions(index, slide):
    x, y = index
    return slide.read_region((x * 224, y * 224), 0,
                                  (224, 224)).convert('RGB')

@torch.no_grad()
def extract_mosaic(index, slide, extractor):
    i, j = index
    patch = slide.read_region((i * 224, j * 224), 0,
                              (224, 224)).convert('RGB')
    patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
    mosaic = extractor.extract_patches(patch)
    return (mosaic, index)

class AddSlide:
    @torch.no_grad()
    def __init__(self, model, r, path):
        self.max_tensor_size = 32
        self.tensor_cpu = torch.zeros(self.max_tensor_size, 3, 224, 224)
        self.tensor_gpu = torch.zeros(self.max_tensor_size, 3, 224, 224, device='cuda:0')
        self.model = model
        self.r = r
        self.transform = torch.nn.Sequential(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
        self.path = path

    @torch.no_grad()
    def add_redis(self, counter, results, coordinates):
        for res in results:
            mosaic, x = res
            for j in range(len(mosaic)):
                coordinates.append(x)
                img = transforms.Resize((224, 224))(mosaic[j])
                self.tensor_cpu[counter] = transforms.ToTensor()(img)
                counter += 1

                if counter == self.max_tensor_size:
                    self.tensor_gpu = self.transform(self.tensor_cpu.to(device='cuda:0'))
                    out = self.model(self.tensor_gpu)
                    for l in range(self.max_tensor_size):
                        abscissa, ordinate = coordinates[l]
                        r.lpush(np.array2string(out[l]),
                                json.dumps({"name": self.path, "x": abscissa,
                                            "y": ordinate}))
                    coordinates.clear()
                    counter = 0
        return counter

    @torch.no_grad()
    def add_slide(self):
        num_cores = multiprocessing.cpu_count()
        with Parallel(n_jobs=num_cores, prefer="threads") as parallel:
            slide = OpenSlide(self.path)

            width, height = slide.dimensions

            (nbr_patch_x, nbr_patch_y) = (width // 224, height //224)

            patch_list = []

            rng = range(1, nbr_patch_y-1)
            s = [slide] * (nbr_patch_y-2)
            n = [nbr_patch_x-1] * (nbr_patch_y-2)

            result = parallel(delayed(detect_flesh)(i, j, k) \
                              for i, j, k in zip(rng, s, n))

            patch_list = []

            for res in result:
                patch_list += res

            tensor_out = torch.zeros(len(patch_list), 128)

            nbr_iter = len(patch_list) // self.max_tensor_size
            s = [slide] * self.max_tensor_size

            for i in range(nbr_iter):
                result = parallel(delayed(read_rgb_regions)(index, j)
                                  for index, j in zip(patch_list[self.max_tensor_size * i : \
                                          self.max_tensor_size * (i+1)], s))
                for j in range(self.max_tensor_size):
                    self.tensor_cpu[j] = transforms.ToTensor()(cv2.cvtColor(np.array(result[j]),
                                                                            cv2.COLOR_RGB2BGR))
                self.tensor_gpu = self.transform(self.tensor_cpu.to(device='cuda:0'))
                out = self.model(self.tensor_gpu)
                tensor_out[self.max_tensor_size * i : self.max_tensor_size * (i+1)] = \
                    transforms.ToTensor()(out)

            rest = len(patch_list) % self.max_tensor_size

            if rest != 0:
                s = [slide] * rest
                result = parallel(delayed(read_rgb_regions)(index, j) \
                                  for index, j in zip(patch_list[self.max_tensor_size * nbr_iter:], s))
                for j in range(rest):
                    self.tensor_cpu[j] = transforms.ToTensor()(cv2.cvtColor(np.array(result[j]),
                                                                   cv2.COLOR_RGB2BGR))
                self.tensor_gpu = self.transform(self.tensor_cpu.to(device='cuda:0'))
                out = self.model(self.tensor_gpu)
                tensor_out[self.max_tensor_size * nbr_iter:] = transforms.ToTensor()(out[:rest])

            tensor_out_np = np.array(tensor_out.numpy(), dtype=np.float64)

            centroids, assignements = KMeansRex.RunKMeans(tensor_out_np, 500,
                                                          initname=b"random")
            assignements = np.array(assignements, dtype=np.int32).reshape(len(patch_list))

            centroid_list = [np.inf] * len(centroids)
            final_patch_list = [0] * len(centroids)

            for i in range(len(patch_list)):
                distance = np.absolute(centroids[assignements[i]], tensor_out_np[i]).sum()
                if centroid_list[assignements[i]] > distance:
                    centroid_list[assignements[i]] = distance
                    final_patch_list[assignements[i]] = patch_list[i]

            coordinates_list = []
            final_patch_list = [x for x in final_patch_list if isinstance(x, tuple)]

            nbr_iter = len(final_patch_list) // self.max_tensor_size
            s = [slide] * self.max_tensor_size
            extractor = [utils.Extract()] * self.max_tensor_size

            counter = 0

            for i in range(nbr_iter):
                result = parallel(delayed(extract_mosaic)(index, j, e)
                                  for index, j, e in zip(final_patch_list[self.max_tensor_size * i : \
                                          self.max_tensor_size * (i+1)], s, extractor))
                counter = self.add_redis(counter, result, coordinates_list)


            rest = len(final_patch_list) % self.max_tensor_size

            if rest != 0:
                s = [slide] * rest
                result = parallel(delayed(extract_mosaic)(index, j, e)
                                  for index, j, e in zip(final_patch_list[self.max_tensor_size * nbr_iter:],
                                                         s, extractor))
                counter = self.add_redis(counter, result, coordinates_list)

            if counter != 0:
                self.tensor_gpu = self.transform(self.tensor_cpu.to(device='cuda:0'))
                out = self.model(self.tensor_gpu)
                for i in range(counter):
                    abscissa, ordinate = coordinates_list[i]
                    r.lpush(np.array2string(out[i]),
                            json.dumps({"name": self.path, "x": abscissa,
                                        "y": ordinate}))

if __name__ == "__main__":
    usage = "python3 add_slide.py --path <patch> [--extractor <algorithm>]"

    parser = ArgumentParser(usage)

    parser.add_argument(
        '--path',
        help='path the slide to add in the database'
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

    a = AddSlide(model, r, args.path)
    a.add_slide()
