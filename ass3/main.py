#!/usr/bin/env python3

"""Small script to experiment with review classification"""

# User libs
from report_utils import *

# Math/Numeric libraries
import numpy as np
import statistics
import random

# ML library
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator
import sklearn.svm

# Misc
from typing import Union
import pickle
from pprint import pprint
import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    """Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """

    # Argument parsing object
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-i", "--input-file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-t", "--tf-idf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-tp", "--test-percentage", default=0.1, type=float,
                        help="Percentage of the data that is used for the test set (default 0.20)")
    parser.add_argument("--model", default="nb",
                        help="nb - Naive Bayes, lr - logistic regression")
    parser.add_argument("--experiment", default="main",
                        help="Which experiment to run: main, mccc, cv, train_data")
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle data set before splitting in train/test")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used for shuffling")
    parser.add_argument("--data-out", default="tmp.pkl",
                        help="Where to store experiment data")
    parser.add_argument("--table-format", default="default",
                        help="How to format table: default or latex")

    # Parse the args
    args = parser.parse_args()
    return args


def read_corpus(corpus_filepath: str) -> tuple[list[list[str]], Union[list[str], list[bool]]]:
    """Read and parse the corpus from file.

    Parameters
    ==========
        - "corpus_filepath": filepath of the file to be read.

    Returns
    =======
        A 2-tuple containing:
            1. The tokenized sentences.
            2. The sentiment labels for each respective sentence.
    """

    documents = []
    labels_s = []
    with open(corpus_filepath, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            # 2-class problem: positive vs negative
            labels_s.append(tokens[1] == "pos")

    return documents, labels_s

def complete_scoring(Y_test: np.array, Y_pred: np.array) -> dict:
    """Utility function to facilitate computing metrics.

    Parameters
    ==========
        - "Y_test": true labels to compare against
        - "Y_pred": labels predicted by the system that is going to be evaluated

    Returns
    =======
        A dict with each item being one type of scoring computed. So far:
            - "c_mat" is a confusion matrix
            - "report" is a dict containing precission, recall and f1 per class and averaged
    """

    # Compute metrics
    report = classification_report(Y_test, Y_pred, output_dict=True)
    c_mat = confusion_matrix(Y_test, Y_pred)

    # Pack into dict and return
    score = dict(report=report, c_mat=c_mat)
    return score


def report_score(score: dict, labels, args: Namespace):
    """Utility function to facilitate logging results. Prints out nice tables to stdout
    and writes to a pickle file specified by "args.data_out"

    Parameters
    ==========
        - "score": dictionary returned by the "complete_scoring" function
        - "labels": (ordered) set of all possible labels.
        - "args": Namespace object containing the argument values passed in command-line

    """

    # Print tables
    print(format_report(score["report"], format_=args.table_format))
    print(format_auto_matrix(score["c_mat"],
          labels, format_=args.table_format))

    # Add additional information
    score["labels"] = labels
    score["task"] = "sentiment"
    score["model"] = args.model

    # Dump into pickle file
    with open(args.data_out, "wb") as f:
        pickle.dump(score, f)

def experiment_main():
    pass

def experiment_features(X_full, Y_full):
    # use scikit's built-in splitting function to save space
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full,
        test_size=args.test_percentage,
        random_state=args.seed,
        shuffle=args.shuffle
    )

    model = Pipeline([
        ("vec", TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)),
        ("svm", sklearn.svm.SVC()),
    ])
    model.fit(X_train, Y_train)
    score = accuracy_score(Y_test, model.predict(X_test))
    print(f"score: {score:.2%}")
    print(model.get_params())
    print(model.get_params().keys())
    print(model.get_params()["svm"].coef_)

# Script logic
if __name__ == "__main__":
    args = parse_args()

    # load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file)


    if args.experiment == "main":
        experiment_main(X_full, Y_full)

    elif args.experiment == "features":
        experiment_features(X_full, Y_full)