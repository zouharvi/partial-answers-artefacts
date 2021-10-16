#!/usr/bin/env python3

"""
Script for preparing and cleaning the data
"""

import argparse
from collections import Counter
from utils import *


def craft_rv1(data, x_filter="headline", y_filter="newspaper"):
    """
    Rest vs. 1: Prepends all answers to the input apart from the one specified by "y_filter_key"

    TODO returns?
    """
    assert y_filter in Y_KEYS
    assert x_filter in X_KEYS

    # TODO: for now drop these lists because they are long and there is no clean way to fuse them into the model
    Y_KEYS_LOCAL = Y_KEYS - {"subject", "geographic"}

    def x_manipulator(x, y):
        y = {**y, y_filter: "None"}
        return {
            **x,
            "craft": ' | '.join([y[k] for k in Y_KEYS_LOCAL]) + " | " + x[x_filter]
        }

    return [
        (
            x_manipulator(x, y),
            {**y, "craft": y[y_filter]}
        )
        for x, y in data
    ]


def parse_args():
    args=argparse.ArgumentParser()
    args.add_argument(
        "--data-in", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--data-out", default="data/final/{LABEL}.json",
        help="Location of creafted data JSON, the {LABEL} token (including curly brakets) is going to be replaced by the data label",
    )
    return args.parse_args()


if __name__ == "__main__":
    args=parse_args()
    data=load_data(args.data_in)
    for y_filter in Y_KEYS:
        print("Crafting Rv1", y_filter)
        data_new = craft_rv1(data, x_filter="headline", y_filter=y_filter)
        save_data(args.data_out.replace("{LABEL}", y_filter+"_Rv1"), data_new)