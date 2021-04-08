import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MultipleLocator

file = open("output_sizes.txt")
lines = file.readlines()

numbers = []
top1 = []
top5 = []
hitrate = []
hitratefive = []
items = []
i = 0

for line in lines:
    res = line.find("number")
    if res != -1:
        items = line.split()
        numbers.append(items[3])
    else:
        colon = line.find(":")
        if colon != -1:
            if line[:colon] == "top 1 accuracy ":
                top1.append(float(line[colon+2:]))
            elif line[:colon] == "top 5 accuracy ":
                top5.append(float(line[colon+2:]))
            elif line[:colon] == "Hit rate first ":
                hitrate.append(float(line[colon+2:]))
            elif line[:colon] == "Mean hit rate first five ":
                hitratefive.append(float(line[colon+2:]))

plt.plot(numbers, top1, label="top 1 accuracy")
plt.plot(numbers, top5, label="top 5 accuracy")
plt.plot(numbers, hitrate, label="Mean hit rate most similar")
plt.plot(numbers, hitratefive, label="Mean hit rate first five most similar")

plt.legend()
plt.ylabel("accuracy")
plt.xlabel("number of extracted patches")

plt.show()

"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MultipleLocator

file = open("output.txt")
lines = file.readlines()

dic = defaultdict(lambda: defaultdict(list))
items = []
i = 0

for line in lines:
    res = line.find("features")
    res1 = line.find("thresh")
    res2 = line.find("kmeans")
    if res != -1 and res1 != -1 and res2 != -1:
        items = line.split()
    elif items != []:
        colon = line.find(":")
        if colon != -1 and i > 2:
            dic[items[1]][line[:colon]].append(float(line[colon+2:]))

        i += 1
        if i == 7:
            i = 0
            items = []

for keys in dic.keys():
    for keys2 in dic[keys].keys():
        if keys2 == "Hit rate first ":
            plt.plot(np.arange(.1, 1, .1), dic[keys][keys2], label="Mean hit rate most similar")
        elif keys2 == "Mean hit rate first five ":
            plt.plot(np.arange(.1, 1, .1), dic[keys][keys2], label="Mean hit rate first five most similar")
        else:
            plt.plot(np.arange(.1, 1, .1), dic[keys][keys2], label=keys2)

    plt.legend()
    plt.ylabel("accuracy")
    plt.xlabel("treshold for binarisation")

    plt.title("kmeans extraction for " + keys + " features")
    plt.show()

"""
