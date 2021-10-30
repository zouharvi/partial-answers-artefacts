#!/usr/bin/env python3

"""
Preparation and cleaning of the data.
"""

import sys
sys.path.append("src")
from utils import *

import argparse
from collections import Counter


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--data-in", default="data/final/all.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--data-out", default="data/final/clean.json",
        help="Location of cleaned data JSON",
    )
    return args.parse_args()


def filter_data(data, cutoff=False):
    data_x = [
        {
            "headline": article["headline"],
            "body": article["body"],
        }
        for article in data
    ]

    def y_filter(x, y_filter_key):
        if x["classification"][y_filter_key] is None:
            return []
        else:
            return [
                item["name"]
                for item in x["classification"][y_filter_key]
            ]

    data_y = [
        {
            "newspaper": article["newspaper"],
            "ncountry": NEWSPAPER_TO_COUNTRY[article["newspaper"]],
            "ncompas": NEWSPAPER_TO_COMPAS[article["newspaper"]],
            "month": article["date"].split()[0],
            "year": article["date"].split()[-1],
            "subject": y_filter(article, "subject"),
            "geographic": y_filter(article, "geographic"),
        }
        for article in data
    ]

    counter_sub = Counter()
    counter_geo = Counter()
    for article_y in data_y:
        counter_sub.update(article_y["subject"])
        counter_geo.update(article_y["geographic"])
    allowed_sub = {x[0] for x in counter_sub.most_common() if x[1] >= 1000}
    allowed_geo = {x[0] for x in counter_geo.most_common() if x[1] >= 250}

    data_y = [
        {
            **article_y,
            "subject": [x for x in article_y["subject"] if x in allowed_sub],
            "geographic": [x for x in article_y["geographic"] if x in allowed_geo],
        }
        for article_y in data_y
    ]

    data = [
        (x, y)
        for x, y in zip(data_x, data_y)
        if (not cutoff) or (len(y["subject"]) != 0 and len(y["geographic"]) != 0)
    ]

    return data


if __name__ == "__main__":
    args = parse_args()
    data = load_data_raw(args.data_in)
    print(len(data), "samples loaded")
    data = filter_data(data, cutoff=True)
    print(len(data), "samples after cleaning")
    print("Fields X:", data[0][0].keys())
    print("Fields Y:", data[0][1].keys())
    save_data(args.data_out, data)
