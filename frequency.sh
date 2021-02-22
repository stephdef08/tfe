#!/bin/bash
 let "count = 0"
for var in {32..48}
do
        var1="image_folder/test/"
        var1="$var1$var/"
        tmp=$(ls $var1 | wc -l)
        ((count += $tmp))
        printf "$var & $tmp \\\\\\\\ \n"
done

echo $count
