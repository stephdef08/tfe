#!/bin/bash

for dir in image_folder/val/*/
do
    var=$(ls -l $dir | wc -l)
    let "var = $var - 1"
    echo $dir '&' $var '\'
done
