import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import KMeansRex
from time import time

#Replace with something more efficient
def binarize(tensor):
    binary_rep = np.zeros((tensor.shape[0], tensor.shape[1]))
    row = binary_rep.shape[0]
    col = binary_rep.shape[1]
    for y in range(row):
        for i in range(col):
            binary_rep[y][i] = 0 if tensor[y][i] < .5 else 1

    return binary_rep

def extract_patches(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (1000, 1000))

    hist_list = np.zeros((16, 256), dtype=np.float64)
    b, g, r = cv2.split(img)

    for j in range(4):
        for i in range(4):
            hist_list[j*4+i, :] += np.reshape(cv2.calcHist([b[j*250:(j+1)*250-1, i*250:(i+1)*250-1]],
                                                         [0], None, [256], [0,256]), 256)
            hist_list[j*4+i, :] += np.reshape(cv2.calcHist([g[j*250:(j+1)*250-1, i*250:(i+1)*250-1]],
                                                         [0], None, [256], [0,256]), 256)
            hist_list[j*4+i, :] += np.reshape(cv2.calcHist([r[j*250:(j+1)*250-1, i*250:(i+1)*250-1]],
                                                         [0], None, [256], [0,256]), 256)

    centroids, assignements = KMeansRex.RunKMeans(hist_list, 4, initname=b"random")
    assignements = np.array(assignements, dtype=np.int32).reshape(16)

    label_list = [[], [], [], []]
    label_list_indices = [[], [], [], []]

    for j in range(4):
        for i in range(4):
            label_list[assignements[j*4+i]].append(hist_list[j*4+i])
            label_list_indices[assignements[j*4+i]].append((j, i))

    mosaic = []
    for i in range(4):
        nbr_clusters = int(.5 * len(label_list[i]))

        if nbr_clusters > 1:
            centroids, assignements = KMeansRex.RunKMeans(np.array(label_list[i],
                                                                   dtype=np.float64),
                                                          nbr_clusters,
                                                          initname=b"random")
            for j in range(len(label_list[i])):
                if assignements[j] == 0:
                    mosaic.append(label_list_indices[i][j])
        else:
            for j in range(len(label_list[i])):
                mosaic.append(label_list_indices[i][j])

    for i in range(len(mosaic)):
        y = mosaic[i][0]
        x = mosaic[i][1]

        b_patch = b[y * 250 : (y+1) * 250 - 1, x * 250 : (x+1) * 250 -1]
        g_patch = g[y * 250 : (y+1) * 250 - 1, x * 250 : (x+1) * 250 -1]
        r_patch = r[y * 250 : (y+1) * 250 - 1, x * 250 : (x+1) * 250 -1]

        mosaic[i] = cv2.merge([r_patch, g_patch, b_patch])
        mosaic[i] = Image.fromarray(mosaic[i])

    return mosaic

if __name__ == "__main__":
    for i in range(10):
        extract_patches("/home/stephan/Pictures/slice.png")
