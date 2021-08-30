import numpy as np
import cv2
from PIL import Image
import time
import sys
import torch
from joblib import Parallel, delayed
from sklearn.cluster import KMeans

def binarize(tensor, threshold):
    binary_rep = np.zeros(tensor.shape)
    binary_rep[tensor > threshold] = 1
    binary_rep[tensor <= threshold] = 0
    return binary_rep

class Extract:
    def __init__(self, extraction="kmeans", num_features=32, num_patches=0):
        self.extraction = extraction
        self.num_patches = num_patches
        self.num_features = num_features
        self.num_patches = num_patches

    def extract_patches(self, img):
        if self.extraction == "compl_random":
            img = cv2.resize(img, (224, 224))
            sizes = np.random.randint(16, 223, self.num_patches)

            positions = []

            for s in sizes:
                positions.append(np.random.randint(0, 223 - s))
                positions.append(np.random.randint(0, 223 - s))

            positions = np.array(positions).reshape((self.num_patches, 2))

            mosaic = []
            for i in range(self.num_patches):
                x, y = positions[i][0], positions[i][1]
                size = sizes[i]

                patch = img[y : y + size, x : x + size, :]
                mosaic.append(Image.fromarray(patch))

            return mosaic

        img = cv2.resize(img, (1952, 1952)).astype(np.uint8)

        _, img_grey = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        threshold = 224 * 224 * 255 * 0.9

        patch_list = []

        for j in range(8):
            for i in range(8):
                patch_thresh = img_grey[j*224:(j+1)*224, i*224:(i+1)*224]
                if patch_thresh.sum() < threshold:
                    patch_list.append((j, i))

        histograms = np.zeros((64, 256))


        mosaic = []

        if len(patch_list) >= 8:
            if self.extraction == 'kmeans':
                for k, (j, i) in enumerate(patch_list):
                    planes = cv2.split(img[j*224:(j+1)*224, i*224:(i+1)*224])
                    for c in range(3):
                        hist = cv2.calcHist(planes[c], [0], None, [256], [0,256])
                        histograms[k, :] += hist[:, 0]

                kmeans = KMeans(n_clusters=8).fit(histograms[:len(patch_list)])

                label_list = [[] for i in range(8)]
                label_list_indices = [[] for i in range(8)]

                for i, idx in enumerate(patch_list):
                    label_list[kmeans.labels_[i]].append(histograms[i])
                    label_list_indices[kmeans.labels_[i]].append(idx)

                for i in range(8):
                    nbr_clusters = int(.25 * len(label_list[i]) + .5) if len(label_list[i]) > 0 else 0

                    if nbr_clusters > 1:
                        kmeans = KMeans(n_clusters=nbr_clusters).fit(label_list[i])
                        indices = set(range(nbr_clusters))
                        for j in range(len(kmeans.labels_)):
                            if kmeans.labels_[j] in indices:
                                indices.remove(kmeans.labels_[j])
                                mosaic.append(label_list_indices[i][j])
                    else:
                        for j in range(len(label_list_indices[i])):
                            mosaic.append(label_list_indices[i][j])
            else:
                mosaic = list(np.random.choice(patch_list, 8 + round(np.log2(len(patch_list)))))
        else:
            for idx in patch_list:
                mosaic.append(idx)

        for i in range(len(mosaic)):
            y, x = mosaic[i]

            mosaic[i] = img[y * 224 : (y+1) * 224, x * 224 : (x+1) * 224, :]
            mosaic[i] = Image.fromarray(mosaic[i])

        return mosaic


def test(extractor, img):
    mosaic = extractor.extract_patches(img)


if __name__ == "__main__":
    # extractor = [Extract(extraction="compl_random")] * 32
    # imgs = [cv2.imread("/home/stephan/Documents/tfe1/image_folder/test/39/0527.png")] * 32
    #
    # with Parallel(n_jobs=24, prefer="threads") as parallel:
    #     result = parallel(delayed(test)(e, i) for e, i in zip(extractor, imgs))
    extractor = Extract(extraction="random")
    patches = extractor.extract_patches(cv2.imread("/home/stephan/Documents/tfe1/image_folder/test/39/0527.png"))

    for p in patches:
        p.show()
