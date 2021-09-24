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
        data = pickle.load(f)

    # create figure
    plt.figure(figsize=(4, 3))
    plt.plot(
        [x[1] for x in data[("bow", None)]["train"]],
        marker=".",
        markeredgecolor="black",
        label="BoW",
        color="tab:blue",
    )
    plt.plot(
        [x[1] for x in data[("bow", "english")]["train"]],
        marker=".",
        markeredgecolor="black",
        label="BoW + stopwords",
        linestyle=":",
        color="tab:blue",
    )
    plt.plot(
        [x[1] for x in data[("tfidf", None)]["train"]],
        marker=".",
        markeredgecolor="black",
        label="TF-IDF",
        color="tab:red",
    )
    plt.plot(
        [x[1] for x in data[("tfidf", "english")]["train"]],
        marker=".",
        markeredgecolor="black",
        label="TF-IDF + stopwords",
        linestyle=":",
        color="tab:red",
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Train size")
    plt.xticks(
        list(range(len(data[("tfidf", None)]["train"]))),
        [x[0] if x[0] < 1000 else str(x[0]/1000) + "k" for x in data[("tfidf", None)]["train"]],
        rotation=45,
    )
    plt.legend()
    plt.tight_layout(rect=(-0.05, -0.05, 1, 1))
    plt.show()
