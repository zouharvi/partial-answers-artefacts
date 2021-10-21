#!/usr/bin/env python3

import argparse
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
import sklearn.model_selection


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data", default="data/final/clean.json",
        help="Location of joined data JSON",
    )
    args.add_argument(
        "--target-output", "--to", default="newspaper", help="Target variable",
    )
    args.add_argument(
        "--target-input", "--ti", default="both", help="Input variable",
    )
    args.add_argument(
        "--model", default="svc", help="Which model to use for running the baseline",
    )
    args.add_argument(
        "--vectorizer", default="tfidf", help="Which vectorizer to use for running the baseline",
    )
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_data(args.data)
    data = streamline_data(
        data,
        x_filter=args.target_input if args.target_input != "both" else lambda x, y: [
            x["headline"] + " " + x["body"]],
        y_filter=args.target_output, binarize=None
    )

    VECTORIZER_CLASS = {
        "tfidf": TfidfVectorizer(max_features=90000, ngram_range=(1, 2)),
        "bow": CountVectorizer(),
    }[args.vectorizer]

    if args.target_output in {"subject", "geographic"}:
        args.model = "multi_" + args.model
        # data_x, data_y = zip(*data)
        # _, data_y = binarize_data(data_y)
        # data = list(zip(data_x, data_y))

    MODEL_CLASS = {
        "multi_svc_2": MultiOutputClassifier(SVC(probability=True)),
        "multi_linear_svc_2": MultiOutputClassifier(SVC(probability=True, kernel="linear")),
        "multi_linear_lr": MultiOutputClassifier(LogisticRegression()),
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

    data_train, data_test = sklearn.model_selection.train_test_split(
        data,
        test_size=1000,
        random_state=0,
    )

    data_x_train, data_y_train = zip(*data_train)
    data_x_test, data_y_test = zip(*data_test)

    # unravel
    data_x_train = [x[0] for x in data_x_train]
    data_x_test = [x[0] for x in data_x_test]

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
        _, data_y_test = binarize_data(data_y_test)

    else:
        # unravel
        data_y_train = [y[0] for y in data_y_train]
        data_y_test = [y[0] for y in data_y_test]

    print(len(data_x_train))

    model.fit(data_x_train, data_y_train)

    if args.target_output in {"subject", "geographic"}:
        # Use probabilities as scores
        # For some reason the data needs to be transported for the dimension to match
        pred_y = model.decision_function(data_x_test)
        rprec_val = rprec(data_y_test, pred_y)
        print(f"Dev RPrec: {rprec_val:.2%}")
        # used for forced multioutput
        # pred_y = np.array(model.predict_proba(data_x_test))[:,:,1].transpose()
    else:
        acc_val = model.score(data_x_test, data_y_test)
        print(f"Dev ACC: {acc_val:.2%}")
