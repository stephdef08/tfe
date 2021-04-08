#!/bin/bash

for features in 16 32 64 128
do
    echo training for $features features
    python densenet.py --num_features $features
    for extraction in random compl_random kmeans
    do
        for thresh in $(seq 0.1 0.1 0.9)
        do
            redis-server --daemonize yes

            echo features $features thresh $thresh extraction $extraction

            for class in {32..48}
            do

                class1="image_folder/test/"
                class1="$class1$class/"
                python add_images.py --path $class1 --num_features $features --threshold $thresh --extraction $extraction
            done

            redis-cli INFO | grep "db0:keys"

            python test_accuracy.py --num_features $features --threshold $thresh --extraction $extraction

            redis-cli shutdown
        done
    done
done
