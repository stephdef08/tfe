#!/bin/bash

for extraction in random compl_random
do
    for thresh in $(seq 0.1 0.1 0.9)
    do
        for features in 16 32 64 128
        do
            redis-server --daemonize yes

            echo features $features thresh $thresh extraction $extraction
            echo
            echo

            for class in {32..48}
            do

                class1="image_folder/test/"
                class1="$class1$class/"
                python add_images.py --path $class1 --num_features $features --threshold $thresh
            done

            python test_accuracy.py --num_features $features --threshold $thresh

            redis-cli shutdown
        done
    done
done
