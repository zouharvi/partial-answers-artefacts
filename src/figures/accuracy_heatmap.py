#!/usr/bin/env python3

import argparse
import sys
sys.path.append("src")
from utils import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import itertools as it
scores = {
    ("ncountry", "ncountry"): 1.0,
    ("ncountry", "geographic"): 0.803,
    ("ncountry", "newspaper"): 0.875,
    ("ncountry", "ncompas"): 0.926,
    ("ncountry", "year"): 0.418,
    ("ncountry", "month"): 0.605,
    ("ncountry", "subject"): 0.671,
    ("month", "ncountry"): 0.978,
    ("month", "month"): 1.0,
    ("month", "newspaper"): 0.853,
    ("month", "subject"): 0.752,
    ("month", "year"): 0.485,
    ("month", "geographic"): 0.708,
    ("month", "ncompas"): 0.926,
    ("newspaper", "newspaper"): 1.0,
    ("newspaper", "geographic"): 0.846,
    ("newspaper", "month"): 0.603,
    ("newspaper", "ncompas"): 1.0,
    ("newspaper", "ncountry"): 1.0,
    ("newspaper", "year"): 0.402,
    ("newspaper", "subject"): 0.653,
    ("ncompas", "ncompas"): 1.0,
    ("ncompas", "ncountry"): 0.975,
    ("ncompas", "newspaper"): 0.892,
    ("ncompas", "geographic"): 0.832,
    ("ncompas", "year"): 0.408,
    ("ncompas", "month"): 0.435,
    ("ncompas", "subject"): 0.706,
    ("year", "year"): 1.0,
    ("year", "ncountry"): 0.969,
    ("year", "month"): 0.8,
    ("year", "newspaper"): 0.878,
    ("year", "subject"): 0.763,
    ("year", "geographic"): 0.768,
    ("year", "ncompas"): 0.934
}

KEY_ORDER = [
        "newspaper", "ncountry", "ncompas",
        "month", "year", "subject", "geographic"
    ]

if __name__ == "__main__":

    # redefine Y_KEYS and fix order
    Y_KEYS = sorted(set([k[0] for k in scores.keys()]), key=KEY_ORDER.index)
    X_KEYS = sorted(set([k[1] for k in scores.keys()]), key=KEY_ORDER.index)
    
    Y_KEYS_INDEX = {k: i for i, k in enumerate(Y_KEYS)}
    X_KEYS_INDEX = {k: i for i, k in enumerate(X_KEYS)}

    plotdata = np.zeros((len(Y_KEYS), len(X_KEYS)))

    for yk,xk in it.product(Y_KEYS,X_KEYS):
        yk_idx = Y_KEYS_INDEX[yk]
        xk_idx = X_KEYS_INDEX[xk]

        val = scores[(yk,xk)]
        
        plotdata[yk_idx, xk_idx] = val

    fig = plt.figure(figsize=(4.4,3.9))
    ax = plt.gca()

    im = ax.imshow(plotdata, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(len(X_KEYS)))
    ax.set_yticks(np.arange(len(Y_KEYS)))
    ax.set_xticklabels([Y_KEYS_PRETTY[x] for x in X_KEYS])
    ax.set_yticklabels([Y_KEYS_PRETTY[y] for y in Y_KEYS])

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(), rotation=30, ha="right",
        rotation_mode="anchor"
    )

    # Loop over data dimensions and create text annotations.
    for i,yk in enumerate(Y_KEYS):
        for j,xk in enumerate(X_KEYS):
            text = ax.text(
                j, i, f"{scores[(yk, xk)]:.0%}",
                ha="center", va="center",
                color="black" if scores[(yk, xk)] < 0.80 and scores[(yk, xk)] > 0.3 else "white",
            )

    # turn spines off
    ax.spines[:].set_visible(False)

    # add separator between dummy
    #ax.add_patch(Rectangle((-0.5, 0.4), len(Y_KEYS), 0.1, color="white"))


    # remove all whitespace
    plt.tight_layout(rect=(-0.025, -0.025, 1.025, 1.03))
    plt.savefig("test.png")
