#!/usr/bin/env python3

"""
Script for preparing and cleaning the data
"""

import sys
sys.path.append("src")
from utils import *
import argparse
from collections import Counter

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data-in", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--data-out", default="data/final/{LABEL}.json",
        help="Location of creafted data JSON, the {LABEL} token (including curly brakets) is going to be replaced by the data label",
    )
    args.add_argument(
        "--seed", type=int, default=0,
    )
    return args.parse_args()

def rel_freq_key(data, key):
    data_counter = Counter([y[key] for y in data])
    total = len(data)
    return {k:f"{x/total:.2%}" for k,x in data_counter.items()}


if __name__ == "__main__":
    args = parse_args()
    # take only labels
    data = [x[1] for x in load_data(args.data_in)]

    print(len(data))

    print(rel_freq_key(data, "month"))
    print(rel_freq_key(data, "year"))