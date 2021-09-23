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
    """Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """

    parser = argparse.ArgumentParser()  # Argument parsing object

    # Arguments
    parser.add_argument("--data-bow", default="ass3/features_bow.pkl")
    parser.add_argument("--data-tfidf", default="ass3/features_tfidf.pkl")
    parser.add_argument("-s", "--sentiment", action="store_true")
    parser.add_argument("-l", "--legend", action="store_true")

    # Parse the args
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load pickled data
    with open(args.data_bow, "rb") as f:
        data_coefs_bow = pickle.load(f)
    with open(args.data_tfidf, "rb") as f:
        data_coefs_tfidf = pickle.load(f)

    # create figure
    plt.figure(figsize=(4, 4.2))

    coefs_bow_map = {k: v for k, v in data_coefs_bow}

    coefs_pos_tfidf = data_coefs_tfidf[-8:]
    pivot = len(data_coefs_tfidf) // 2
    coefs_ntr_tfidf = data_coefs_tfidf[pivot:pivot + 2]
    coefs_neg_tfidf = data_coefs_tfidf[:8]

    widths_tfidf = \
        [x[1] for x in coefs_pos_tfidf][::-1] +\
        [x[1] for x in coefs_ntr_tfidf][::-1] +\
        [x[1] for x in coefs_neg_tfidf][::-1]
    labels_tfidf = \
        [x[0] for x in coefs_pos_tfidf] +\
        [x[0] for x in coefs_ntr_tfidf] +\
        [x[0] for x in coefs_neg_tfidf]
    ys_tfidf = [
        -x
        for x in range(len(coefs_pos_tfidf) + len(coefs_ntr_tfidf) + len(coefs_neg_tfidf))
    ]
    ys_bow = [
        -x
        for x in range(len(coefs_pos_tfidf) + len(coefs_ntr_tfidf) + len(coefs_neg_tfidf))
    ]

    # plot each model configuration
    plt.barh(
        y=ys_tfidf,
        width=widths_tfidf,
        color="tab:blue",
        label="TF-IDF",
    )
    plt.barh(
        y=ys_bow,
        width=[coefs_bow_map[k] for k in labels_tfidf],
        color="tab:green",
        height=0.4,
        label="BoW",
    )
    plt.yticks(
        ticks=ys_tfidf,
        labels=labels_tfidf,
        family="monospace",
        size=10,
    )
    plt.xlabel("SVM Coefficient")
    plt.ylabel("Token")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # TODO: compare with count vectorizer
