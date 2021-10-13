#!/usr/bin/env python3

import argparse
from utils import *
import numpy as np

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data", default="data/final/COP.all.json",
        help="Location of joined data JSON",
    )
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data)
    binarizer, data = streamline_data(data, y_filter="subject")
    
    print(len(data), "samples loaded")
    print(list(binarizer.classes_))
    print(len(binarizer.classes_), "classes in total")
    
    example = data[10]
    print("exampleX:", example[0])
    print("exampleY:", example[1])
    for i, val in enumerate(example[1]):
        if val:
            print(binarizer.classes_[i])