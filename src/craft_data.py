#!/usr/bin/env python3

"""
Script for preparing and cleaning the data
"""

import argparse
import random
from utils import *


def craft_rv1(data, x_filter="headline", y_filter="newspaper", dropout=0):
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
        # apply dropout to artefacts
        artefacts = [
            y[k] if random.random() >= dropout else "None"
            for k in Y_KEYS_LOCAL
        ]
        return {
            **x,
            "craft": ' | '.join(artefacts) + " | " + x[x_filter]
        }

    return [
        (
            x_manipulator(x, y),
            {**y, "craft": y[y_filter]}
        )
        for x, y in data
    ]


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data-in", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--data-out", default="data/final/{LABEL}.json",
        help="Location of creafted data JSON, the {LABEL} token (including curly brakets) is going to be replaced by the data label",
    )
    args.add_argument(
        "--seed", type=int, default=0,
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data_in)

    random.seed(args.seed)

    # Rv1_00
    for y_filter in Y_KEYS:
        print("Crafting Rv1_00", y_filter)
        data_new = craft_rv1(
            data, x_filter="headline",
            y_filter=y_filter, dropout=0.0
        )
        save_data(args.data_out.replace(
            "{LABEL}", "Rv1_00_" + y_filter), data_new)

    # Rv1_50
    for y_filter in Y_KEYS:
        print("Crafting Rv1_50", y_filter)
        data_new = craft_rv1(
            data, x_filter="headline",
            y_filter=y_filter, dropout=0.5
        )
        save_data(args.data_out.replace(
            "{LABEL}", "Rv1_50_" + y_filter), data_new)

    # Rv1_75
    for y_filter in Y_KEYS:
        print("Crafting Rv1_75", y_filter)
        data_new = craft_rv1(
            data, x_filter="headline",
            y_filter=y_filter, dropout=0.75
        )
        save_data(args.data_out.replace(
            "{LABEL}", "Rv1_75_" + y_filter), data_new)
