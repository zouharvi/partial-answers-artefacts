#!/usr/bin/env python3

"""
Script for generating overviews for variables (generates LaTeX table contents).
"""

import sys
sys.path.append("src")
from utils import *
import argparse
from collections import Counter


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--data-in", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    return args.parse_args()


def rel_freq_key(data, key):
    """
    Generate a LaTeX table for single-label variables.
    """
    data_counter = Counter([y[key] for x, y in data])
    total = len(data)
    output = sorted(
        list(data_counter.items()),
        key=lambda x: x[1], reverse=True
    )
    output = [(k, f"{x/total:.2%}") for k, x in output]
    output = [f"{k} & {x} \\\\" for k, x in output]
    return "\n".join(output)


def rel_freq_key_plus(data, key):
    """
    Generate a LaTeX table for multi-label variables.
    """
    data_counter = Counter([item for x, y in data for item in y[key]])
    total = len(data)
    output = sorted(
        list(data_counter.items()),
        key=lambda x: x[1], reverse=True
    )
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
        "Average number of words per article:",
        format(np.average([len(x["body"].split()) for x, y in data]), ".2f")
    )

    print("\n", rel_freq_key(data, "month"), sep="")
    print("\n", rel_freq_key(data, "year"), sep="")
    print("\n", rel_freq_key(data, "newspaper"), sep="")
    print("\n", rel_freq_key_plus(data, "subject"), sep="")
    print("\n", rel_freq_key_plus(data, "geographic"), sep="")
