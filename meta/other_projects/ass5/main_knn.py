#!/usr/bin/env python3

'''Perform 6-way review classification using embeddings based on BERT-like models'''

import random as python_random
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from utils import *
from encoder_zoo import *
from sklearn.neighbors import KNeighborsClassifier

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt',
                        help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", default='dev.txt',
                        help="Separate dev set to read in")
    parser.add_argument("--encoder", default="glove-m",
                        help="Which encoder to use: glove-{a,m}, {bert,sbert,dpr}-{c,t}")
    parser.add_argument("--embeddings", default='glove_reviews.json',
                        help="Path to glove file embedding")

    args = parser.parse_args()
    return args


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)
    Y_dev_m = np.argmax(Y_dev_bin, axis=1)

    # compute the encoding for all reviews
    # note: this could be perhaps hidden behind a lazy lambda to reduce the number of lines
    # but this seems more straightforward
    if args.encoder == "tfidf":
        X_all = encoder_tfidf(X_train + X_dev, max_features=None)
    elif args.encoder == "tfidf-m":
        X_all = encoder_tfidf(X_train + X_dev, max_features=768)
    elif args.encoder == "tfidf-s":
        X_all = encoder_tfidf(X_train + X_dev, max_features=768)
    elif args.encoder == "glove-a":
        X_all = encoder_glove(X_train + X_dev, embeddings, action="avg")
    elif args.encoder == "glove-m":
        X_all = encoder_glove(X_train + X_dev, embeddings, action="max")
    elif args.encoder == "bert-c":
        X_all = encoder_bert(X_train + X_dev, type_out="cls")
    elif args.encoder == "bert-t":
        X_all = encoder_bert(X_train + X_dev, type_out="tokens")
    elif args.encoder == "sbert-c":
        X_all = encoder_sbert(X_train + X_dev, type_out="cls")
    elif args.encoder == "sbert-t":
        X_all = encoder_sbert(X_train + X_dev, type_out="tokens")
    elif args.encoder == "dpr-c-q":
        X_all = encoder_dpr(X_train + X_dev, type_out="cls", version="query")
    elif args.encoder == "dpr-t-q":
        X_all = encoder_dpr(
            X_train + X_dev, type_out="tokens", version="query")
    elif args.encoder == "dpr-c-d":
        X_all = encoder_dpr(X_train + X_dev, type_out="cls", version="doc")
    elif args.encoder == "dpr-t-d":
        X_all = encoder_dpr(X_train + X_dev, type_out="tokens", version="doc")
    else:
        raise Exception("Unknown encoder model")

    # Iterate 3 levels of parameters (centering, voting, n_neighbors)
    for center in [False, True]:
        if center:
            X_cur = center_norm(X_all)
        else:
            X_cur = X_all
        X_train, X_dev = X_cur[:len(X_train)], X_cur[len(X_train):]
        for weights in ["uniform", "distance"]:
            for k in [1, 3, 5, 7, 9, 11, 13, 15, 17]:
                # create, "fit" and evaluate the KNN
                model = KNeighborsClassifier(n_neighbors=k, weights=weights)
                model.fit(X_train, Y_train_bin)
                Y_dev_pred = model.predict(X_dev)
                Y_dev_pred_m = np.argmax(Y_dev_pred, axis=1)

                # compute score and print
                score = accuracy_score(Y_dev_m, Y_dev_pred_m)
                print(
                    ("c" if center else "-") +
                    ("u" if weights == "uniform" else "d") +
                    f"{k:0>2}",
                    f"{score:.2%}",
                )


if __name__ == '__main__':
    main()
