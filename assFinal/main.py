#!/usr/bin/env python3

import argparse
from utils import *

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data", default="data/final/COP.all.json",
        help="Location of joined data JSON"
    )
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data)
    assert len(data) == 33660
    assert all([
        set(article.keys()) == {'path', 'raw_text', 'newspaper', 'date', 'headline', 'body', 'classification', 'cop_edition'}
        for article in data
    ])
