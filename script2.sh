#!/bin/bash

for features in 16 32 64 128
do echo adding for $features features
    redis-server --daemonize yes

    python add_images.py --path image_folder/test/ --num_features $features

    python test_accuracy.py --num_features $features

    redis-cli shutdown
done
