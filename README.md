# tfe

Parts of code are taken from https://github.com/SathwikTejaswi/deep-ranking/blob/master/Code/data_utils.py

The required libraries are in the file requirements.txt

The files were modified since the results were computed with the mosaic, and it was not possible to obtain once again the same results reported in the master thesis

## Train a new network
The following script is executed
```bash
python densenet.py [arguments]
```
with the following arguments
- --num_features (default: 128): the size of the last linear layer (i.e. the number of features)
- --weights (required): the file that will contain the weights, a different file is saved after every epoch, with the number of the epoch appended to the name of the file
- --path (required): path to the training images

The folder that contains the training images should be organised as follows:
```
folder:
|------ class1:
          |------ image1
          |------ image2
          |------ ...
|------ class2:
          |------ image1
          |------ image2
          |------ ..
|------ ...
```

## Index images
```bash
redis-server
python add_images.py [arguments]
```
with the following arguments
- --path (required): path to the images to index
- --extractor (default: densenet): densenet
- --num_features (default: 32): the size of the last linear layer (i.e. the number of features)
- --threshold (default: 0.5): threshold value that is used to binarise the features
- --extraction (default: kmeans): kmeans or compl_random
- --num_patches (default: 0): number of patches extracted when using compl_random extraction
- --weights (required): file storing the weights of the model

The folder that contains the images to index must have the same structure that the one used for training

## Retrieve images
The redis server that was used to index the images must be running

```bash
python retrieve_images.py [arguments]
```
with the following arguments
- --path (required): path to the query image
- --extractor (default: densenet): densenet
- --num_features (default: 32): the size of the last linear layer (i.e. the number of features)
- --threshold (default: 0.5): threshold value that is used to binarise the features
- --extraction (default: kmeans): kmeans or compl_random
- --num_patches (default: 0): number of patches extracted when using compl_random extraction
- --weights (required): file storing the weights of the model

## Testing the accuracy
The redis server that was used to index the images must be running

```bash
python test_accuracy.py [arguments]
```
with the following arguments
- --path (required): path to the query images
- --num_features (default: 32): the size of the last linear layer (i.e. the number of features)
- --threshold (default: 0.5): threshold value that is used to binarise the features
- --extraction (default: kmeans): kmeans or compl_random
- --num_patches (default: 0): number of patches extracted when using compl_random extraction
- --weights (required): file storing the weights of the model

The folder that contains the images must have the same structure that the one used for training
