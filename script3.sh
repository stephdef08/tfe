#!/bin/bash

for num in $(seq 4 4 64)
do
    echo number of patches $num
    redis-server --daemonize yes

    for class in {32..48}
    do

        class1="image_folder/test/"
        class1="$class1$class/"
        python add_images.py --path $class1 --extraction compl_random --num_patches $num
    done

    python test_accuracy.py --extraction compl_random --num_patches $num

    redis-cli shutdown
done
