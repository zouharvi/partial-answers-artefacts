#!/usr/bin/bash

for i in `seq 0 20`; do
    echo "Running seed $i" &
    ./main.py -i train_NE.txt -ts test_NE_input.txt -o "output_model_5_bigger/seed_$i" -vp 0.0 --experiment pred --seed $i
done
