#!/usr/bin/env python3

"""
Tile-intensity graph for LR performance with individual artefacts
Dependency between variables modelled as p(-,xi|theta).
"""

import argparse
import sys
sys.path.append("src")
from utils import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--logfile", default="computed/feature_dependency.out",
        help="From where to store logged values"
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.logfile, "rb") as f:
        logdata = pickle.load(f)

    # redefine Y_KEYS and fix order
    Y_KEYS = [
        "newspaper", "ncountry", "ncompas",
        "month", "year", "subject", "geographic"
    ]
    Y_KEYS_INDEX = {k: i for i, k in enumerate(Y_KEYS)}

    plotdata = np.zeros((len(Y_KEYS) + 1, len(Y_KEYS)))

    for item in logdata:
        y_key_index_1 = Y_KEYS_INDEX[item["y_filter_1"]]
        y_key_index_2 = Y_KEYS_INDEX[item["y_filter_2"]]
        if "acc" in item:
            val = item["acc"]
        else:
            val = item["rprec"]

        plotdata[y_key_index_1 + 1, y_key_index_2] = val
        plotdata[0, y_key_index_2] = item["dummy"]

    fig = plt.figure(figsize=(4.4, 3.9))
    ax = plt.gca()

    im = ax.imshow(plotdata, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(Y_KEYS)))
    ax.set_yticks(np.arange(len(Y_KEYS) + 1))
    ax.set_xticklabels([Y_KEYS_PRETTY[y] for y in Y_KEYS])
    ax.set_yticklabels(["Random"] + [Y_KEYS_PRETTY[y] for y in Y_KEYS])

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=30, ha="right",
        rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    for i in range(len(Y_KEYS) + 1):
        for j in range(len(Y_KEYS)):
            text = ax.text(
                j, i, f"{plotdata[i, j]:.0%}",
                ha="center", va="center",
                color="black" if plotdata[i,
                                          j] <= 0.7 and plotdata[i, j] > 0.3 else "white",
            )

    # turn spines off
    ax.spines[:].set_visible(False)

    # add separator between dummy
    ax.add_patch(Rectangle((-0.5, 0.4), len(Y_KEYS) + 1, 0.1, color="white"))

    # remove all whitespace
    plt.tight_layout(rect=(-0.025, -0.025, 1.025, 1.03))
    plt.show()
