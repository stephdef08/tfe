#!/bin/bash

for var in {32..48}
do
    var1="image_folder/test/"
    var1="$var1$var/"
    python add_images.py --path $var1
done
