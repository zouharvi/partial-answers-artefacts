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


def parse_args() -> Namespace:
    """
    Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--data-in", default="ass3/errors_tfidf.pkl")

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
        [x[1] for x in data[("bow", None)]["features"]],
        marker=".",
        markeredgecolor="black",
        label="BoW",
        color="tab:blue",
    )
    plt.plot(
        [x[1] for x in data[("bow", "english")]["features"]],
        marker=".",
        markeredgecolor="black",
        label="BoW + stopwords",
        linestyle=":",
        color="tab:blue",
    )
    plt.plot(
        [x[1] for x in data[("tfidf", None)]["features"]],
        marker=".",
        markeredgecolor="black",
        label="TF-IDF",
        color="tab:red",
    )
    plt.plot(
        [x[1] for x in data[("tfidf", "english")]["features"]],
        marker=".",
        markeredgecolor="black",
        label="TF-IDF + stopwords",
        linestyle=":",
        color="tab:red",
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Number of features")
    plt.xticks(
        list(range(len(data[("tfidf", None)]["features"]))),
        [x[0] if x[0] < 1000 else str(x[0]//1000) + "k" for x in data[("tfidf", None)]["features"]],
        rotation=45,
    )
    plt.legend()
    plt.tight_layout(rect=(-0.05, -0.05, 1, 1))
    plt.show()
