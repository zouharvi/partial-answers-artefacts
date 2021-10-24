#!/bin/bash

files=(data/final/1v1_*)
for (( index=2; index<${#files[@]}; index+=3 )); do
    f=${files[index]}
    echo $f
    python src/train_lm_classifier.py -to craft -ti craft -ep 2 --max-length 512 -bs 8 --input $f
done 
