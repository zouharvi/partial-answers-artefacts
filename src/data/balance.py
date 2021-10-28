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
    data_counter = Counter([y[key] for x, y in data])
    total = len(data)
    output = sorted(list(data_counter.items()),
                    key=lambda x: x[1], reverse=True)
    output = [(k, f"{x/total:.2%}") for k, x in output]
    output = [f"{k}: {x}" for k, x in output]
    return output


def rel_freq_key_plus(data, key):
    data_counter = Counter([item for x, y in data for item in y[key]])
    total = len(data)
    output = sorted(list(data_counter.items()),
                    key=lambda x: x[1], reverse=True)
    output = [(
        k.replace("&", "\\&"),
        f"{x/total:.1%}".replace("%", "\\%")
    )
        for k, x in output
        if x / total >= 0.1
    ]
    output = [f"{k} & {x} \\\\" for k, x in output]
    return "\n".join(output)


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data_in)

    print(
        "Average number of words:",
        format(np.average([len(x["body"].split()) for x, y in data]), ".2f")
    )

    print("\n", rel_freq_key(data, "month"))
    print("\n", rel_freq_key(data, "year"))
    print("\n", rel_freq_key(data, "newspaper"))
    print("\n", rel_freq_key_plus(data, "subject"))
    print("\n", rel_freq_key_plus(data, "geographic"))
