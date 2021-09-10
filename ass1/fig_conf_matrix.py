#!/usr/bin/env python3

import math
import argparse
import numpy as np
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mpl_color
import matplotlib.ticker as tkr
import seaborn as sns

PRETTY_NAME_MODEL = {
    "nb": "Naive Bayes",
    "lr": "Logistic Regression",
    "mccc": "Baseline",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="tmp.pkl",
                        help="Where to find experiment data")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    with open(args.data, "rb") as f:
        data = pickle.load(f)
        
    max_val = np.max(data["c_mat"])
    steps = math.ceil(math.log2(max_val))
    
    cbar_ticks = [2**i for i in range(0,steps+1)]
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)

    plt.figure(figsize=(4.7, 3.7))
    sns.heatmap(
        data["c_mat"],
        cmap="Greens",
        annot=True,
        norm=mpl_color.LogNorm(vmin=0.5,vmax=2**steps),
        xticklabels=data["labels"],
        yticklabels=data["labels"],
        cbar_kws=dict(
            ticks=cbar_ticks,
            format=formatter),
        fmt="g")
    # plt.title("Averaged confusion matrix\nModel: {}, Task: {}".format(PRETTY_NAME_MODEL[data["model"]],data["task"]))
    # plt.title("Topic task")
    plt.title("Sentiment task")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()
