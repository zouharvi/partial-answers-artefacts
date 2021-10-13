#!/usr/bin/env python3

import argparse
from utils import *

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
    # data[2].pop("body")
    # data[2].pop("raw_text")
    # print(data[2])
    binarizer, data = streamline_data(data, y_filter="newspaper")
    print(list(binarizer.classes_))
    print(len(binarizer.classes_), "classes in total")
    print("X[0]:", data[0][0]) # headline
    print("Y[0]:", data[0][1]) # class
