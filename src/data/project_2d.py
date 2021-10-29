#!/usr/bin/env python3

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

import argparse
import os.path as path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        help="Path to the embeddings file.")
    parser.add_argument("-o", "--output", default='data/embeddings/{i}_reduced_{h}_{p}_{d}.pkl',
                        help="Path where to store the dimensionality reduced embeddings.")
    parser.add_argument("-p", "--projection-method", default="tsne",
                        help="What projection method to use to implement dimensionality reduction.")
    parser.add_argument("-d", "--dimension", default=2,
                        help="How many dimensions to reduce to.")
    parser.add_argument("-rh", "--reduction-heuristic", default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    output_name = args.output.format(
        i=path.basename(args.input[:-4]),
        h=args.reduction_heuristic,
        p=args.projection_method,
        d=args.dimension)

    print("Output:", output_name)

    # Read data
    with open(args.input, "rb") as f:
        embeddings = pickle.load(f)

    # Use reduction heuristic
    if args.reduction_heuristic == "cls":
        embeddings = embeddings[:, 0]
    elif args.reduction_heuristic == "avg":
        embeddings = np.mean(embeddings, axis=1)

    if args.projection_method == "tsne":
        reducer = TSNE(args.dimension, n_jobs=-1)
    elif args.projection_method == "pca":
        reducer = PCA(args.dimension)

    projected_emb = reducer.fit_transform(embeddings)
    
    with open(output_name, "wb") as f:
        pickle.dump(projected_emb, f)