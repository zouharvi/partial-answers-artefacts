#!/usr/bin/env python3

import argparse
from utils import *

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data-in", default="data/final/COP.all.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--data-out", default="data/final/COP.clean.json",
        help="Location of cleaned data JSON",
    )
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = load_data_raw(args.data_in)
    print(len(data), "samples loaded")
    data = filter_data(data, cutoff=True)
    print(len(data), "samples after cleaning")
    print("Fields X:", data[0][0].keys())
    print("Fields Y:", data[0][1].keys())
    save_data(args.data_out, data)