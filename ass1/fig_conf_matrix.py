#!/usr/bin/env python3

"""
Vizualization of model misclassification
"""

# Scientific / Numeric libs
import math
import numpy as np

# Visualization libs
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Misc
import pickle
import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    """Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """
    # Argument parsing object
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--data", default="tmp.pkl",
                        help="Where to find experiment data")
    parser.add_argument("-s", "--sentiment", action="store_true")

    # Parse the args
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Load pickled data
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    # Find the min amount of exponential steps (base 2) to cover the magnitude
    # of all data
    max_val = np.max(data["c_mat"])
    steps = math.ceil(math.log2(max_val))

    # Compute reference checkpoints for the colorbar
    cbar_ticks = [2**i for i in range(0, steps + 1)]

    # Create formater to manipulate the strings formated in the plot
    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)

    # Create a figure
    plt.figure(figsize=(4.7, 3.7))

    # pass all the data to seaborn heatmap with lgonorm color scaling
    sns.heatmap(
        data["c_mat"],
        cmap="Greens",
        annot=True,
        norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=2**steps),
        xticklabels=data["labels"],
        yticklabels=data["labels"],
        cbar_kws=dict(
            ticks=cbar_ticks,
            format=formatter),
        fmt="g")

    # Set title
    if args.sentiment:
        plt.title("Sentiment task")
    else:
        plt.title("Topic task")

    # Set additional informational string
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()
