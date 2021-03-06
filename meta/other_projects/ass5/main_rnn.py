#!/usr/bin/env python3

'''Train a customizable RNN model'''

import random as python_random
import argparse
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from utils import *
from model_rnn import *

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-emb", "--embeddings", default='glove_reviews.json', type=str,
                        help="Embedding file we are using (default glove_reviews.json).\n" +
                        "Has no effect when using pretrained language models.")
    parser.add_argument("-ep", "--epochs", default=50, type=int)
    parser.add_argument("-bs", "--batch-size", default=16, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("--optimizer", default="AdamW",
                        help="Which optimizer to use (AdamW, Adam, SGD, RMSProp)")
    parser.add_argument("--embd-dense", action="store_true",
                        help="Use a dense layer after embedding")
    parser.add_argument("--embd-random", action="store_true",
                        help="Initialize embeddings randomly")
    parser.add_argument("--embd-reg", action="store_true",
                        help="Regularize embeddings")
    parser.add_argument("--embd-not-trainable", action="store_true",
                        help="Do not finetune embeddings")
    parser.add_argument("--rnn-unit", default="lstm",
                        help="Which recurrent unit to use (lstm, gru, rnn)")
    parser.add_argument("--rnn-layers", default=2, type=int,
                        help="Number of recurrent layers")
    parser.add_argument("--rnn-not-bi", action="store_true",
                        help="Do not use bidirectional RNN")
    parser.add_argument("--rnn-backwards", action="store_true",
                        help="Process words backward")
    parser.add_argument("--rnn-bimerge", default="concat",
                        help="Which merge operation to use after bidirectional layer")
    parser.add_argument("--dense-dropout", type=float, default=0.1,
                        help="Dropout after the first dense layer after RNN")
    parser.add_argument("--rnn-dropout", type=float, default=0.1,
                        help="Dropout after the recurrent layers (not recurrent dropout)")
    parser.add_argument("--rnn-dropout-rec", type=float, default=0.0,
                        help="Recurrent dropout after the recurrent layers")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="How much to smooth the labels")

    args = parser.parse_args()
    return args


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    # Use encoder.classes_ to find mapping back
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.fit_transform(Y_dev)

    # Create model
    embeddings = read_embeddings(args.embeddings)
    model = ModelRNN(
        embeddings,
        X_all=X_train + X_dev,
        args=args,
    )

    # Train the model
    model.train(X_train, Y_train_bin, X_dev, Y_dev_bin)

    # compute test accuracy if a test file has been passed
    if args.test_file:
        # read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)
        X_test_vect = model.vectorizer(
            np.array([[s] for s in X_test])
        ).numpy()

        # compute predictions and report results
        report_accuracy_score(model.model.predict(X_test_vect), Y_test_bin)


if __name__ == '__main__':
    main()
