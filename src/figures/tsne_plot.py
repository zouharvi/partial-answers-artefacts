#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
from utils.data import *

import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default='data/final/clean.json',
                        help="Path to the news dataset.")
    parser.add_argument("-e", "--embeddings", required=True,
                        help="Path to the embeddings file to plot.")
    parser.add_argument("-l", "--label-key", default="newspaper")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Read data
    data = utils.load_data(args.input)
    _, data = zip(*data)

    with open(args.embeddings, "rb") as f:
        embeddings = pickle.load(f)

    df = pd.DataFrame(embeddings)

    df["newspaper"] = [x["newspaper"] for x in data]
    df["ncompas"] = [NEWSPAPER_TO_COMPAS[x["newspaper"]] for x in data]
    df["ncountry"] = [COUNTRY_TO_PRETTY[NEWSPAPER_TO_COUNTRY[x["newspaper"]]]
                      for x in data]

    plt.figure(figsize=(4.5, 4))

    for label in df[args.label_key].unique():
        t = df[df[args.label_key] == label]

        sc = plt.scatter(
            t[0], t[1],
            alpha=0.6,
            s=2,
            label=label
        )

    lg = plt.legend(
        bbox_to_anchor=(0.1, 1, 0.8, 0),
        frameon=False,
        handlelength=1,
        loc="lower center",
        markerscale=4,
        ncol=4,
    )
    for h in lg.legendHandles:
        h.set_alpha(1)

    # hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout(rect=(-0.06, 0, 1.06, 1))
    plt.show()
