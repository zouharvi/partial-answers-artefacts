#!/usr/bin/env python3

'''TODO: add high-level description of this Python script'''

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
    parser.add_argument("--embd-dense", action="store_true",
                        help="Use a dense layer after embedding")
    parser.add_argument("--embd-random", action="store_true",
                        help="Initialize embeddings randomly")
    parser.add_argument("--embd-reg", action="store_true",
                        help="Regularize embeddings")
    parser.add_argument("--embd-not-trainable", action="store_true",
                        help="Do not finetune embeddings")
    parser.add_argument("--embd-unit", default="lstm",
                        help="Which recurrent unit to use (lstm, gru, rnn)")
    parser.add_argument("--embd-layers", default=2, type=int,
                        help="Number of recurrent layers")

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

    # Transform input to vectorized input

    # Train the model
    model.train(X_train, Y_train_bin, X_dev, Y_dev_bin)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test_bin = encoder.fit_transform(Y_test)

        # Finally do the predictions
        report_accuracy_score(model.predict(X_test), Y_test_bin)

if __name__ == '__main__':
    main()
