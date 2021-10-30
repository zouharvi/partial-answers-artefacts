#!/usr/bin/env python3

"""
Processes Bert-based embeddings and projects them into a lower dimension.
"""

import sys
sys.path.append("src")
import utils
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import argparse
import os.path as path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        help="Path to the embeddings file.",
    )
    parser.add_argument(
        "-o", "--output", default='data/embeddings/{i}_reduced_{h}_{p}_{d}.pkl',
        help="Path where to store the dimensionality reduced embeddings.",
    )
    parser.add_argument(
        "-p", "--projection-method", default="tsne",
        help="What projection method to use to implement dimensionality reduction.",
    )
    parser.add_argument(
        "-d", "--dimension", default=2,
        help="How many dimensions to reduce to.",
    )
    parser.add_argument(
        "-em", "--embedding-method", default="avg"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    output_name = args.output.format(
        i=path.basename(args.input[:-4]),
        h=args.embedding_method,
        p=args.projection_method,
        d=args.dimension)


    # Read data
    embeddings = utils.load_data(args.input, format="pickle")

    # Use reduction heuristic
    if args.embedding_method == "cls":
        embeddings = embeddings[:, 0]
    elif args.embedding_method == "avg":
        embeddings = np.mean(embeddings, axis=1)

    if args.projection_method == "tsne":
        assert args.dimension in {1, 2, 3}
        reducer = TSNE(args.dimension, n_jobs=-1)
    elif args.projection_method == "pca":
        reducer = PCA(args.dimension)

    # transform the data
    projected_emb = reducer.fit_transform(embeddings)

    print("Storing in", output_name)
    utils.save_data(output_name, projected_emb, format="pickle")
