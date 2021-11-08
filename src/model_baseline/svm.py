#!/usr/bin/env python3

"""
Baseline SVM model for classification.
Non-linear models are much more slower.
"""

import sys

sys.path.append("src")
import argparse
import utils
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "-to", "--target-output", default="newspaper",
        help="Target variable",
    )
    args.add_argument(
        "-ti", "--target-input", default="both",
        help="Input variable",
    )
    args.add_argument(
        "--model", default="svc",
        help="Which model to use for running the baseline",
    )
    args.add_argument(
        "--vectorizer", default="tfidf",
        help="Which vectorizer to use for running the baseline",
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = utils.load_data(args.data)
    data = utils.streamline_data(
        data,
        x_filter=args.target_input if args.target_input != "both" else lambda x, y: [
            x["headline"] + " " + x["body"]],
        y_filter=args.target_output, binarize=None
    )

    # select vectorizer
    VECTORIZER_CLASS = {
        "tfidf": TfidfVectorizer(max_features=90000, ngram_range=(1, 2)),
        "bow": CountVectorizer(),
    }[args.vectorizer]

    # select and instantiate model
    if args.target_output in {"subject", "geographic"}:
        args.model = "multi_" + args.model
    MODEL_CLASS = {
        "multi_svc_2": MultiOutputClassifier(SVC(probability=True)),
        "multi_linear_svc_2": MultiOutputClassifier(SVC(probability=True, kernel="linear")),
        "multi_lr": MultiOutputClassifier(LogisticRegression()),
        "svc": SVC(),
        "multi_svc": SVC(),
        "multi_linear_svc": LinearSVC(),
        "linear_svc": LinearSVC(),
        "lr": LogisticRegression(),
        "nb": MultinomialNB(),
    }[args.model]
    model = Pipeline([
        ("tfidf", VECTORIZER_CLASS),
        ("model", MODEL_CLASS)
    ])
    
    # split data
    data_dev, data_test, data_train = utils.make_split(
        (data,),
        (1000,1000,),
        random_state=0,
        simple=True,
    )

    data_x_train, data_y_train = zip(*data_train)
    data_x_test, data_y_test = zip(*data_test)
    data_x_dev, data_y_dev = zip(*data_dev)

    # unravel
    data_x_train = [x[0] for x in data_x_train]
    data_x_test = [x[0] for x in data_x_test]
    data_x_dev = [x[0] for x in data_x_dev]

    if args.target_output in {"subject", "geographic"}:
        # expand items but only for train
        data_y_train_new = []
        data_x_train_new = []
        for x, y in zip(data_x_train, data_y_train):
            for yi in y:
                data_y_train_new.append(yi)
                data_x_train_new.append(x)
        data_x_train, data_y_train = data_x_train_new, data_y_train_new

        # binarize test
        _, data_y_test = utils.binarize_data(data_y_test)

    else:
        # unravel
        data_y_train = [y[0] for y in data_y_train]
        data_y_test = [y[0] for y in data_y_test]

    model.fit(data_x_train, data_y_train)

    if args.target_output in {"subject", "geographic"}:
        # Use probabilities as scores
        # The data needs to be transported for the dimension to match

        pred_y = model.decision_function(data_x_dev)
        rprec_val = utils.rprec(data_y_test, pred_y)
        print(f"Dev  RPrec: {rprec_val:.2%}")

        pred_y = model.decision_function(data_x_test)
        rprec_val = utils.rprec(data_y_test, pred_y)
        print(f"Test RPrec: {rprec_val:.2%}")
        # used for forced multioutput
        # pred_y = np.array(model.predict_proba(data_x_test))[:,:,1].transpose()
    else:
        print("Dev")
        pred_y = model.predict(data_x_dev)
        print(classification_report(data_y_dev, pred_y, zero_division=0))

        print("Test")
        pred_y = model.predict(data_x_test)
        print(classification_report(data_y_test, pred_y, zero_division=0))