#!/usr/bin/env python3

"""
Vizualization of data size effect on model performance
"""

# Visualization libs
import matplotlib.pyplot as plt

# Misc
import argparse
from argparse import Namespace
import pickle
import numpy as np
from collections import defaultdict


def parse_args() -> Namespace:
    """Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """

    parser = argparse.ArgumentParser()  # Argument parsing object

    # Arguments
    parser.add_argument("--data-in", default="ass3/errors_tfidf.pkl")

    # Parse the args
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load pickled data
    with open(args.data_in, "rb") as f:
        data_tfidf = pickle.load(f)

    # create figure
    plt.figure(figsize=(4, 3))

    BUCKET_X = [25, 50, 75, 100, 125, 150, 200, 250, 300, 350]

    def bucket_index(x):
        for i, b in enumerate(BUCKET_X):
            if x <= b:
                return i
        return len(BUCKET_X)

    data_buckets = defaultdict(lambda: [])
    for article, y_true, y_pred in data_tfidf:
        data_buckets[bucket_index(len(article))].append((y_true, y_pred))

    plt.bar(
        [bucket_i for bucket_i, bucket in data_buckets.items()],
        [
            len([None for y_true, y_pred in bucket if y_true != y_pred]) / len(bucket)
            for bucket_i, bucket in data_buckets.items()
        ],
        label="FP+FN",
    )
    plt.bar(
        [bucket_i-0.15 for bucket_i, bucket in data_buckets.items()],
        [
            len([None for y_true, y_pred in bucket if not y_pred and y_true]) / len(bucket)
            for bucket_i, bucket in data_buckets.items()
        ],
        color="tab:orange",
        width=0.3,
        label="FN",
    )
    plt.bar(
        [bucket_i+0.15 for bucket_i, bucket in data_buckets.items()],
        [
            len([None for y_true, y_pred in bucket if y_pred and not y_true]) / len(bucket)
            for bucket_i, bucket in data_buckets.items()
        ],
        color="tab:green",
        width=0.3,
        label="FP",
    )
    plt.xticks(
        list(range(len(BUCKET_X) + 1)),
        [
            f"<{bucket_x} ({len(data_buckets[bucket_i])})" if bucket_x > 0 else f">{-1*bucket_x} ({len(data_buckets[bucket_i])})"
            for bucket_i, bucket_x in enumerate(BUCKET_X + [-BUCKET_X[-1]])
        ],
        rotation=60,
        size=8,
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
