#!/usr/bin/env python3

'''TODO: add high-level description of this Python script'''

import random as python_random
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from utils import *
from model_bert import *

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
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model to use.\n" +
                        "If not specified will use a LSTM model.")
    parser.add_argument("-ep","--epochs", default=None, type=int,
                        help="Override the default number of epochs to train the model.")
    parser.add_argument("-bs","--batch-size", default=None, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lr","--learning-rate", default=None, type=float,
                        help="Override the default learning rate.")
    parser.add_argument("--max-length", default=None, type=int,
                        help="Override the default maximum length of language model input.\n" +
                        "Only affects when using language models.")
    parser.add_argument("--strategy", default="cls", type=str,
                        help="The strategy to embedd the sentence\n" +
                        "Can be one of: \"cls\",\"avg\",\"lstm\" or \"bilstm\".")
    parser.add_argument("--freeze", action="store_true",
                        help="If this flag is present the weights of the language model\n" +
                        "are frozen (not updated).")

    args = parser.parse_args()
    return args


LM_ALIASES = dict(
    bert="bert-base-uncased",
    roberta="roberta-base",
    albert="albert-base-v2",
    distilroberta="distilroberta-base"
)


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

    # Define general model params
    model_params = dict(embedd_strategy=args.strategy,freeze_lm=args.freeze)
    if args.epochs:
        model_params["epochs"] = args.epochs
    if args.batch_size:
        model_params["batch_size"] = args.batch_size
    if args.learning_rate:
        model_params["learning_rate"] = args.learning_rate
    if args.max_length:
        model_params["max_length"] = args.max_length

    lm = LM_ALIASES[args.language_model]
    model = ModelTransformer(lm=lm, **model_params)

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
