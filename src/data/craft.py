#!/usr/bin/env python3

"""
Script for preparing and cleaning the data
"""

import argparse
import random
import sys
sys.path.append("src")
from utils import *


def plus_var_processor(plus_var):
    return ", ".join(
        random.sample(plus_var, k=min(5, len(plus_var)))
    ).lower()


def craft_rv1(data, x_filter, y_filter, dropout=0):
    """
    Rest vs. 1: Prepends all answers to the input apart from the one specified by "y_filter_key"

    Returns a list of tuples with dictionaries.
    Use streamline_data(data, x_filter="craft", y_filter="craft") to get the intended dataset format.
    """
    assert y_filter in Y_KEYS
    assert x_filter in X_KEYS

    def x_manipulator(x, y):
        # apply dropout to artefacts
        artefacts = [
            y[k] if random.random() >= dropout else "None"
            for k in {"newspaper", "month", "year"} - set([y_filter])
        ]
        return {
            **x,
            "craft": ' \n'.join(artefacts) + " \n" + x[x_filter]
        }

    return [
        (
            x_manipulator(x, y),
            {**y, "craft": y[y_filter]}
        )
        for x, y in data
    ]


def craft_rv1_plus(data, x_filter, y_filter, dropout=0):
    """
    Rest vs. 1: Prepends all answers to the input apart from the one specified by "y_filter_key"

    Returns a list of tuples with dictionaries.
    Use streamline_data(data, x_filter="craft", y_filter="craft") to get the intended dataset format.
    """
    assert y_filter in Y_KEYS
    assert x_filter in X_KEYS

    def x_manipulator(x, y):
        # apply dropout to artefacts
        artefacts_base = [
            y[k] if random.random() >= dropout else "None"
            for k in {"newspaper", "month", "year"} - set([y_filter])
        ]
        artefacts_plus = [
            plus_var_processor(y[k]) if random.random() >= dropout else "None"
            for k in {"subject", "geographic"} - set([y_filter])
        ]

        artefacts = artefacts_base + artefacts_plus
        return {
            **x,
            "craft": ' \n'.join(artefacts) + " \n" + x[x_filter]
        }

    return [
        (
            x_manipulator(x, y),
            {**y, "craft": y[y_filter]}
        )
        for x, y in data
    ]


def craft_1v1(data, x_filter, y_filter_1, y_filter_2):
    """
    1 vs. 1: Prepends y_filter_1 key to the input and makes y_filter_2 key the target class

    Returns a list of tuples with dictionaries.
    Use streamline_data(data, x_filter="craft", y_filter="craft") to get the intended dataset format.
    """
    assert y_filter_1 in Y_KEYS
    assert y_filter_2 in Y_KEYS
    assert x_filter in X_KEYS

    if y_filter_1 in Y_KEYS_LOCAL:
        def x_manipulator(x, y):
            return {
                **x,
                "craft": y[y_filter_1] + " \n" + x[x_filter]
            }
    else:
        def x_manipulator(x, y):
            return {
                **x,
                "craft": plus_var_processor(y[y_filter_1]) + " \n" + x[x_filter]
            }

    return [
        (
            x_manipulator(x, y),
            {**y, "craft": y[y_filter_2]}
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
    X_FILTER = "body"

    # Rv1p
    for dropout in [0, 50]:
        for y_filter in Y_KEYS:
            y_filter_code = Y_KEYS_TO_CODE[y_filter]
            print(f"Crafting Rv1p_{dropout:0>2}_", y_filter, sep="")
            data_new = craft_rv1_plus(
                data, x_filter=X_FILTER,
                y_filter=y_filter, dropout=dropout/100
            )
            save_data(
                args.data_out.replace("{LABEL}", f"Rv1p_{dropout:0>2}_" + y_filter_code),
                data_new
            )

    # Rv1_00
    for dropout in [0, 50]:
        for y_filter in Y_KEYS:
            y_filter_code = Y_KEYS_TO_CODE[y_filter]
            print(f"Crafting Rv1_{dropout:0>2}_", y_filter, sep="")
            data_new = craft_rv1(
                data, x_filter=X_FILTER,
                y_filter=y_filter, dropout=0.0
            )
            save_data(
                args.data_out.replace("{LABEL}", f"Rv1_{dropout:0>2}_" + y_filter_code),
                data_new
            )

    # 1v1
    for y_filter_1 in Y_KEYS:
        y_filter_1_code = Y_KEYS_TO_CODE[y_filter_1]
        for y_filter_2 in Y_KEYS:
            y_filter_2_code = Y_KEYS_TO_CODE[y_filter_2]
            print("Crafting 1v1_", y_filter_1, "-", y_filter_2, sep="")
            data_new = craft_1v1(
                data, x_filter=X_FILTER,
                y_filter_1=y_filter_1,
                y_filter_2=y_filter_2,
            )
            save_data(
                args.data_out.replace(
                    "{LABEL}", f"1v1_{y_filter_1_code}{y_filter_2_code}"),
                data_new
            )
