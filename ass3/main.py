#!/usr/bin/env python3

"""Small script to experiment with review classification"""

# User libs
from re import M

from numpy.core.defchararray import isalpha, isnumeric
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
    parser.add_argument("--experiment", default="main",
                        help="Which experiment to run: main, mccc, cv, train_data")
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle data set before splitting in train/test")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used for shuffling")
    parser.add_argument("--max-features", default=10000, type=int,
                        help="Maximum number of features in the vectorizer")
    parser.add_argument("--ngrams", action="store_true")
    parser.add_argument("--data-out", default="tmp.pkl",
                        help="Where to store experiment data")

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


def experiment_features(X_full, Y_full, tf_idf, use_ngrams, max_features, data_out=None):
    ngram_range = (2, 3) if use_ngrams else (1, 1)

    model = Pipeline([
        ("vec",
         TfidfVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
             ngram_range=ngram_range,
         )
         if tf_idf else
         CountVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
             ngram_range=ngram_range,
         ),
         ),
        ("svm", sklearn.svm.SVC(kernel="linear")),
    ])
    model.fit(X_full, Y_full)
    Y_pred = model.predict(X_full)
    score = accuracy_score(Y_full, Y_pred)
    print(f"train acc: {score:.2%}")

    coefs_original = model.get_params()["svm"].coef_.toarray().reshape(-1)
    coefs = sorted(enumerate(coefs_original), key=lambda x: x[1])

    vocab = model.get_params()["vec"].vocabulary_
    vec = {v: k for k, v in vocab.items()}

    # Print
    pivot = (len(coefs) - 8) // 2
    print("positive:\n", "\n".join(
        [f" {vec[ind]} ({v:.2f})" for ind, v in coefs[-8:]]), sep="")
    print("neutral:\n", "\n".join(
        [f" {vec[ind]} ({v:.2f})" for ind, v in coefs[pivot:pivot + 8]]), sep="")
    print("negative:\n", "\n".join(
        [f" {vec[ind]} ({v:.2f})" for ind, v in coefs[:8]]), sep="")

    # Store coefficients if argument is passed
    if data_out is not None:
        with open(data_out, "wb") as f:
            pickle.dump([(vec[ind], v) for ind, v in coefs], f)

    # Compute norms
    X_vec = model["vec"].transform(X_full).toarray()
    print(
        "avg data norm",
        np.average(np.linalg.norm(X_vec, axis=1))
    )
    print(
        "coefs norm",
        np.average(
            np.absolute(
                model.get_params()["svm"].coef_.toarray().reshape(-1)
            )
        )
    )


def color_example(review, vocab, coefs):
    message = ""
    for token in review:
        if token not in vocab:
            message += token + " "
        else:
            index = vocab[token]
            coef = coefs[index]
            if coef < -0.1:
                message += "\\textcolor{DarkRed}{" + token + "} "
            elif coef > 0.1:
                message += "\\textcolor{DarkGreen}{" + token + "} "
            else:
                message += token + " "
    return message


def experiment_confidence(X_full, Y_full, tf_idf, max_features, test_percentage, seed):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full,
        test_size=test_percentage,
        random_state=seed,
        shuffle=True
    )

    model = Pipeline([
        ("vec",
         TfidfVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         )
         if tf_idf else
         CountVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         ),
         ),
        ("svm", sklearn.svm.SVC(kernel="linear")),
    ])
    model.fit(X_train, Y_train)
    conf_test = np.average(np.abs(model.decision_function(X_test)))
    conf_train = np.average(np.abs(model.decision_function(X_train)))
    print("Average confidence test ", f"{conf_test:.2f}")
    print("Average confidence train", f"{conf_train:.2f}")

    Y_pred = model.predict(X_full)
    conf_full = np.abs(model.decision_function(X_full))

    conf_full_tt = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if y_true and y_pred]
    )
    conf_full_tf = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if y_true and not y_pred]
    )
    conf_full_ff = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if not y_true and not y_pred]
    )
    conf_full_ft = np.average(
        [c for c, y_true, y_pred in zip(
            conf_full, Y_full, Y_pred) if not y_true and y_pred]
    )
    print(f"TT {conf_full_tt:.2f}, TF {conf_full_tf:.2f}, FT {conf_full_ft:.2f}, FF {conf_full_ff:.2f}")

def experiment_examples(X_full, Y_full, tf_idf, max_features):
    model = Pipeline([
        ("vec",
         TfidfVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         )
         if tf_idf else
         CountVectorizer(
             preprocessor=lambda x: x,
             tokenizer=lambda x:x, max_features=max_features,
         ),
         ),
        ("svm", sklearn.svm.SVC(kernel="linear")),
    ])
    model.fit(X_full, Y_full)
    Y_pred = model.predict(X_full)

    coefs_original = model.get_params()["svm"].coef_.toarray().reshape(-1)
    vocab = model.get_params()["vec"].vocabulary_

    for review, y_true, y_pred in [(x, y, z) for x, y, z in zip(X_full, Y_full, Y_pred) if len(x) <= 100][:10]:
        message = color_example(review, vocab, coefs_original)
        print(message, (y_true, y_pred),
              model.decision_function([review]), "\n")

    adversial = "this camera works well , except that the shutter speed is a bit slow . the image quality is decent . the use of aa rechargeable batteries is also convenient . the camera is pretty sturdy . i 've dropped it a few times and it still works fine".split()
    small_vocab = [k for k in vocab.keys() if len(k) <= 2]
    print("noise token   coefficient")
    for noise_token in ["..."] + small_vocab:
        if noise_token in vocab and all([not isalpha(x) and not isnumeric(x) for x in noise_token]):
            print(
                f"{noise_token:>11}    {coefs_original[vocab[noise_token]]:.3f}")

    noise_token = "#"
    original_pred = model.predict([adversial])
    hit = False
    for i in range(50):
        adversial_tmp = adversial + [noise_token] * i
        current_pred = model.predict([adversial_tmp])
        print(i, model.predict([adversial_tmp]),
              model.decision_function([adversial_tmp]))
        if current_pred != original_pred and not hit:
            print(color_example(adversial_tmp, vocab, coefs_original))
            hit = True


def experiment_errors(X_full, Y_full):
    pass


# Script logic
if __name__ == "__main__":
    args = parse_args()

    # load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file)

    if args.experiment == "main":
        experiment_main(X_full, Y_full)

    elif args.experiment == "examples":
        experiment_examples(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            max_features=args.max_features,
        )

    elif args.experiment == "confidence":
        experiment_confidence(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            max_features=args.max_features,
            test_percentage=args.test_percentage,
            seed=args.seed,
        )

    elif args.experiment == "features":
        experiment_features(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            use_ngrams=args.ngrams,
            max_features=args.max_features,
            data_out=args.data_out,
        )
