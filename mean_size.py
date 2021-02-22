import cv2
from argparse import ArgumentParser
import os
from joblib import Parallel, delayed
import multiprocessing
import numpy

class_list = list(range(67))

folder = "image_folder/train"

def loop(str):
    nbr = 0
    size = 0
    for subdir, dirs, files in os.walk(str):
        for file in files:
            img = cv2.imread(os.path.join(subdir, file))
            size += img.size / 3
            nbr += 1

    return (size, nbr)

res = Parallel(n_jobs=multiprocessing.cpu_count(), prefer="threads")(delayed(loop)(os.path.join(folder, str(cls)))
                                                               for cls in class_list)

img_size = 0
nbr = 0
for s, n in res:
    img_size += s
    nbr += n

print("mean size : ", img_size / nbr)
