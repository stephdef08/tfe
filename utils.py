import torch
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

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

    hist_list = np.zeros((16, 256))
    b, g, r = cv2.split(img)

    for j in range(4):
        for i in range(4):
            hist_list[j*4+i, :] += np.reshape(cv2.calcHist([b[j*250:(j+1)*250-1, i*250:(i+1)*250-1]],
                                                         [0], None, [256], [0,256]), 256)
            hist_list[j*4+i, :] += np.reshape(cv2.calcHist([g[j*250:(j+1)*250-1, i*250:(i+1)*250-1]],
                                                         [0], None, [256], [0,256]), 256)
            hist_list[j*4+i, :] += np.reshape(cv2.calcHist([r[j*250:(j+1)*250-1, i*250:(i+1)*250-1]],
                                                         [0], None, [256], [0,256]), 256)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(hist_list)

    label_list = [[], [], [], []]
    label_list_indices = [[], [], [], []]

    for j in range(4):
        for i in range(4):
            label_list[kmeans.labels_[j*4+i]].append(hist_list[j*4+i])
            label_list_indices[kmeans.labels_[j*4+i]].append((j, i))

    mosaic = []
    for i in range(4):
        nbr_clusters = int(.25 * len(label_list[i]))
        nbr_clusters = nbr_clusters if nbr_clusters > 0 else 1
        kmeans = KMeans(nbr_clusters, random_state=0).fit(label_list[i])
        for j in range(len(label_list[i])):
            if kmeans.labels_[j] == 0:
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
