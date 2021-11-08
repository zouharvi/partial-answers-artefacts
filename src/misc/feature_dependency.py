#!/usr/bin/env python3

"""
Compute p(-,xi|theta) dependency between variables.
"""

import sys
sys.path.append("src")
from utils import *
import argparse
import numpy as np
import sklearn.linear_model
import sklearn.dummy


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--data", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--logfile", default="computed/feature_dependency.out",
        help="Where to store logged values"
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_raw = load_data(args.data)

    logdata = []

    # iterate single-label variables
    for y_filter_1 in Y_KEYS:
        for y_filter_2 in Y_KEYS_LOCAL:
            # prepare data
            _, data = streamline_data(
                data_raw,
                x_filter=y_filter_1,
                y_filter=y_filter_2,
                binarize="input"
            )
            data_x, data_y = zip(*data)

            # define model
            model = sklearn.linear_model.LogisticRegression(
                multi_class="multinomial",
                max_iter=500,
            )

            # unravel data_y which is encapsulated in a list
            data_y = [y[0] for y in data_y]
            model.fit(data_x, data_y)
            acc = model.score(data_x, data_y)

            # get MCCC accuracy
            model_dummy = sklearn.dummy.DummyClassifier(
                strategy="most_frequent"
            )
            model_dummy.fit(data_x, data_y)
            acc_dummy = model_dummy.score(data_x, data_y)

            logdata.append({
                "acc": acc,
                "y_filter_1": y_filter_1,
                "y_filter_2": y_filter_2,
                "dummy": acc_dummy,
            })
            print(
                f"ACC: {acc:.2%} ({y_filter_1} -> {y_filter_2}), dummy: {acc_dummy:.2%}"
            )

    # iterate multi-label variables
    for y_filter_1 in Y_KEYS:
        for y_filter_2 in Y_KEYS - Y_KEYS_LOCAL:
            # prepare data
            _, data = streamline_data(
                data_raw,
                x_filter=y_filter_1,
                y_filter=y_filter_2,
                binarize="all"
            )
            data_x, data_y = zip(*data)
            class_indicies = len(data_y[0])
            matches = []
            # for i in range(5):
            for i in range(class_indicies):
                data_y_i = [y[i] for y in data_y]

                model = sklearn.linear_model.LogisticRegression(
                    multi_class="multinomial",
                    max_iter=500,
                )

                model.fit(data_x, data_y_i)
                # get probability of positive class
                pred_y = [pred[1] for pred in model.predict_proba(data_x)]
                matches.append(pred_y)

            # transpose matches
            matches = list(zip(*matches))

            # compute avg r_prec metric
            rprec_val = rprec(data_y, matches)

            # craft dummy scoring based on item frequency
            dummy_y = np.repeat(
                [np.array(data_y).sum(axis=0)],
                len(data_y), axis=0
            )
            rprec_val_dummy = rprec(data_y, dummy_y)

            logdata.append({
                "rprec": rprec_val,
                "y_filter_1": y_filter_1,
                "y_filter_2": y_filter_2,
                "dummy": rprec_val_dummy,
            })
            print(
                f"RPREC: {rprec_val:.2%} ({y_filter_1} -> {y_filter_2}), dummy: {rprec_val_dummy:.2%}"
            )

    # save output
    save_data(args.logfile, logdata, format="pickle")
