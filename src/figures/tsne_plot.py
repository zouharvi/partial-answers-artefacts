#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
import utils_data

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
    parser.add_argument("-o", "--output", default='data/figures/scatter_{e}_label_{l}.png',
                        help="Path where to store the figure.")
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

    print(len(embeddings))
    print(embeddings)
    df = pd.DataFrame(embeddings)

    df["newspaper"] = [x["newspaper"] for x in data]
    df["ncompas"] = [utils_data.NEWSPAPER_TO_COMPAS[x["newspaper"]] for x in data]
    df["ncountry"] = [utils_data.NEWSPAPER_TO_COUNTRY[x["newspaper"]] for x in data]

    plt.figure(figsize=(5, 4))

    for label in df[args.label_key].unique():
        t = df[df[args.label_key] == label]

        sc = plt.scatter(
            t[0], t[1],
            alpha=0.6,
            s=2,
            label=label
        )

    lg = plt.legend(markerscale=4)
    for h in lg.legendHandles:
        h.set_alpha(1)

    plt.show()