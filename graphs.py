import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MultipleLocator

file = open("output.txt")
lines = file.readlines()

dic = defaultdict(lambda: defaultdict(list))
items = []

for line in lines:
    res = line.find("features")
    if res != -1:
        items = line.split()
    else:
        colon = line.find(":")
        if colon != -1:
            dic[items[1]][line[:colon]].append(float(line[colon+2:]))

for keys in dic.keys():
    for keys2 in dic[keys].keys():
        plt.plot(np.arange(.1, 1, .1), dic[keys][keys2], label=keys2)

    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("treshold for binarisation")

    plt.title("kmeans extraction for " + keys + " features")
    plt.show()
