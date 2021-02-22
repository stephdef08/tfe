import numpy as np
import cv2
from PIL import Image
import KMeansRex
import time
import sys
import torch
from joblib import Parallel, delayed

#Replace with something more efficient
def binarize(tensor, train=False):
    if train == False:
        binary_rep = np.zeros((tensor.shape[0], tensor.shape[1]))
        binary_rep[tensor > .5] = 1
        binary_rep[tensor <= .5] = 0
        return binary_rep
    else:
        binary_rep = torch.zeros((tensor.shape[0], tensor.shape[1]), device='cuda:0')
        binary_rep[tensor > .5] = 1
        binary_rep[tensor <= .5] = 0
        return binary_rep

class Extract:
    def __init__(self):
        self.hist_list = np.zeros((64, 256), dtype=np.int32)
        self.hist_list_gpu = cv2.cuda_GpuMat(64, 256, 4)
        self.img_gpu = cv2.cuda_GpuMat(1024, 1024, 16)

    def extract_patches(self, img):
        img = cv2.resize(img, (1024, 1024)).astype(np.uint8)

        _, img_grey = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        threshold = 128 * 128 * 255 * 0.9


        patch_list = []

        for j in range(8):
            for i in range(8):
                patch_thresh = img_grey[j*128:(j+1)*128, i*128:(i+1)*128]
                if patch_thresh.sum() < threshold:
                    patch_list.append(j * 10 + i)

        mosaic = []

        if len(patch_list) >= 8:
            self.img_gpu.upload(img)
            cv2.cuda.computeHistograms(self.img_gpu, self.hist_list_gpu,
                                       patch_list, 8, 8)

            self.hist_list = self.hist_list_gpu.download()

            centroids, assignements = KMeansRex.RunKMeans(self.hist_list[:len(patch_list)].astype(np.float64),
                                                          8)
            assignements = np.array(assignements, dtype=np.int32).reshape(len(patch_list))

            label_list = [[] for i in range(8)]
            label_list_indices = [[] for i in range(8)]

            for i, idx in enumerate(patch_list):
                    label_list[assignements[i]].append(self.hist_list[i])
                    label_list_indices[assignements[i]].append(idx)

            for i in range(8):
                nbr_clusters = int(.25 * len(label_list[i]) + .5) if len(label_list[i]) > 0 else 0

                if nbr_clusters > 1:
                    centroids, assignements = KMeansRex.RunKMeans(np.array(label_list[i],
                                                                           dtype=np.float64),
                                                                  nbr_clusters)
                    indices = [x for x in range(nbr_clusters)]
                    for j in range(len(assignements)):
                        if assignements[j] in indices:
                            indices.remove(assignements[j])
                            mosaic.append(label_list_indices[i][j])
                else:
                    for j in range(len(label_list_indices[i])):
                        mosaic.append(label_list_indices[i][j])
            #mosaic = list(np.random.choice(patch_list, 8 + round(np.log2(len(patch_list)))))
        else:
            for idx in patch_list:
                mosaic.append(idx)

        b, g, r = cv2.split(img)
        for i in range(len(mosaic)):
            y = mosaic[i] // 10
            x = mosaic[i] % 10

            b_patch = b[y * 128 : (y+1) * 128 - 1, x * 128 : (x+1) * 128 -1]
            g_patch = g[y * 128 : (y+1) * 128 - 1, x * 128 : (x+1) * 128 -1]
            r_patch = r[y * 128 : (y+1) * 128 - 1, x * 128 : (x+1) * 128 -1]

            mosaic[i] = cv2.merge([r_patch, g_patch, b_patch])
            mosaic[i] = Image.fromarray(mosaic[i])

        return mosaic


def test(extractor, img):
    mosaic = extractor.extract_patches(img)


if __name__ == "__main__":
    extractor = [Extract()] * 32
    imgs = [cv2.imread("/home/stephan/Documents/tfe1/image_folder/test/39/0527.png")] * 32

    with Parallel(n_jobs=24, prefer="threads") as parallel:
        result = parallel(delayed(test)(e, i) for e, i in zip(extractor, imgs))
