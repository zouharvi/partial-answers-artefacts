#!/usr/bin/env python3

import argparse
from ast import parse
import sys
from types import CodeType
sys.path.append("src")
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools as it


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--logfile-1v1", default="data/eval/1v1_results.py",
    )
    args.add_argument(
        "--logfile-0v1", default="data/eval/0v1_results.py",
    )
    return args.parse_args()

KEY_ORDER = [
        "newspaper", "ncountry", "ncompas",
        "month", "year", "subject", "geographic"
    ]

if __name__ == "__main__":
    args = parse_args()
    with open(args.logfile_1v1, "r") as f:
        scores = eval(f.read())
    with open(args.logfile_0v1, "r") as f:
        scores_baseline = eval(f.read())

    # redefine Y_KEYS and fix order
    Y_KEYS = [
        "newspaper", "ncountry", "ncompas",
        "month", "year", "subject", "geographic"
    ]
    Y_KEYS_LOCAL = [
        "newspaper", "ncountry", "ncompas",
        "month", "year"
    ]
    Y_KEYS_INDEX = {k: i for i, k in enumerate(Y_KEYS)}
    Y_KEYS_INDEX_LOCAL = {k: i for i, k in enumerate(Y_KEYS)}
    plotdata = np.zeros((len(Y_KEYS_LOCAL)+1, len(Y_KEYS)))

    for xk,yk in it.product(Y_KEYS, Y_KEYS_LOCAL):
        yk_idx = Y_KEYS_INDEX_LOCAL[yk]
        xk_idx = Y_KEYS_INDEX[xk]

        plotdata[yk_idx+1, xk_idx] = scores[(yk,xk)]

    for xk in Y_KEYS:
        xk_idx = Y_KEYS_INDEX[xk]
        plotdata[0, xk_idx] = scores_baseline[xk]

    fig = plt.figure(figsize=(4.4,3.9))
    ax = plt.gca()

    im = ax.imshow(plotdata, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(Y_KEYS)))
    ax.set_yticks(np.arange(len(Y_KEYS_LOCAL)+1))
    ax.set_xticklabels([Y_KEYS_PRETTY[x] for x in Y_KEYS])
    ax.set_yticklabels(["Bert-Single"] + [Y_KEYS_PRETTY[y] for y in Y_KEYS_LOCAL])

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=30, ha="right",
        rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    for i in range(len(Y_KEYS_LOCAL)+1):
        for j in range(len(Y_KEYS)):
            text = ax.text(
                j, i, f"{plotdata[i, j]:.0%}",
                ha="center", va="center",
                color="black" if plotdata[i, j] <= 0.8 and plotdata[i, j] > 0.3 else "white",
            )

    # turn spines off
    ax.spines[:].set_visible(False)

    # add separator between dummy
    ax.add_patch(Rectangle((-0.5, 0.4), len(Y_KEYS), 0.1, color="white"))


    # remove all whitespace
    plt.tight_layout(rect=(-0.025, -0.025, 1.025, 1.03))
    plt.show()
    # plt.savefig("test.png")
