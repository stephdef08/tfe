#modified from https://www.geeksforgeeks.org/how-to-download-google-images-using-python/
#git clone https://github.com/Joeclinton1/google-images-download.git
#cd google-images-download && sudo python setup.py install
# importing google_images_download module
from google_images_download import google_images_download
import json
from joblib import Parallel, delayed
from PIL import Image
import os

# creating object
response = google_images_download.googleimagesdownload()

search_queries = []

with open('imagenet_class_names.txt') as myfile:
    lines = myfile.readlines()
    for line in lines:
        search_queries.append(line.split(" ", 2)[2].replace("\n", "", 1).replace("_", " ", 1))

def downloadimages(query):
    # keywords is the search query
    # format is the image file format
    # limit is the number of images to be downloaded
    # print urs is to print the image file url
    # size is the image size which can
    # be specified manually ("large, medium, icon")
    # aspect ratio denotes the height width ratio
    # of images to download. ("tall, square, wide, panoramic")
    arguments = {"keywords": query,
                 "format": "jpg",
                 "limit":150,
                 "print_urls":True,
                 "size": "medium",
                 "aspect_ratio":"square",
                 "chromedriver":"/home/stephan/Documents/google-images-download/google_images_download/chromedriver_linux64/chromedriver"}
    try:
        response.download(arguments)

    # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit":150,
                     "print_urls":True,
                     "size": "medium",
                     "chromedriver":"/home/stephan/Documents/google-images-download/google_images_download/chromedriver_linux64/chromedriver"}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass
    except:
        pass

def download(query):
    downloadimages(query)
    print()

# Driver Code

Parallel(n_jobs=24, prefer='threads')(delayed(download)(query) for query in search_queries)
