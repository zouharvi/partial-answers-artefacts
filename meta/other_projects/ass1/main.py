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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator
from sklearn.svm import SVC

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
    parser.add_argument("-d", "--dev_file", default='dev.txt',
                        help="Separate dev set to read in")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tf-idf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("--tf-idf-boost", action="store_true",
                        help="Add extra tf-idf parameters to boost performance")
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


def read_corpus(corpus_filepath: str, use_sentiment: bool = False) -> tuple[list[list[str]], Union[list[str], list[bool]]]:
    """Read and parse the corpus from file.

    Parameters
    ==========
        - "corpus_filepath": filepath of the file to be read.

        - "use_sentiment": Whether to extract the sentiment labels (True) or
                            the topic labels or both (None).

    Returns
    =======
        A 2-tuple containing:
            1. The tokenized sentences.
            2. The labels (corresponding task) for each respective sentence.
    """

    documents = []
    labels_s = []
    labels_m = []
    with open(corpus_filepath, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            # 2-class problem: positive vs negative
            labels_s.append(tokens[1] == "pos")
            # 6-class problem: books, camera, dvd, health, music, software
            labels_m.append(tokens[0])

    if use_sentiment is None:
        return documents, labels_s, labels_m
    elif use_sentiment:
        return documents, labels_s
    else:
        return documents, labels_m


def model_factory(model: str) -> BaseEstimator:
    """Factory function for easy model instantiation.

    Parameters
    ==========
        - "model": abbreviation of the kind of model to instantiate
                    nb - Naive Bayes
                    lr - logistic regression
                    mccc - most frequent class classifier

    Returns
    =======
        An sklearn estimator of the type specified by "model"
    """

    model_lib = {
        "nb": lambda: MultinomialNB(),
        "svc": lambda: SVC(),
        "lr": lambda: LogisticRegression(max_iter=5000),
        "mccc": lambda: DummyClassifier(strategy="most_frequent"),
    }
    if model in model_lib:
        return model_lib[model]
    elif model == "all":
        return model_lib
    else:
        raise Exception(f"Unknown model {model}")


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
    score["task"] = "sentiment" if args.sentiment else "topic"
    score["model"] = args.model

    # Dump into pickle file
    with open(args.data_out, "wb") as f:
        pickle.dump(score, f)


# Script logic
if __name__ == "__main__":
    args = parse_args()

    # load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file, args.sentiment)

    if args.dev_file:
        X_train, Y_train = X_full, Y_full
        X_test, Y_test = read_corpus(args.dev_file, args.sentiment)
        
    else:
        # use scikit's built-in splitting function to save space
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_full, Y_full,
            test_size=args.test_percentage,
            random_state=args.seed,
            shuffle=args.shuffle
        )

    # compute features
    if args.tf_idf:
        if args.tf_idf_boost:
            vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,3), max_features=80*10**3)
        else:
            vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    else:  # Bag of words
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

    model_class = model_factory(args.model)
    # combine the vectorizer with the statistical model
    classifier = Pipeline([('vec', vec), ('cls', model_class())])

    if args.experiment == "main":
        # train the classifier, make inferences and compute metrics

        # Fit classifier and make predictions
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        Y_pred_train = classifier.predict(X_train)

        # Compute metrics
        score = complete_scoring(Y_test, Y_pred)
        score_train = accuracy_score(Y_train, Y_pred_train)
        print(f"Accuracy train: {score_train:.2%}")

        # Report scores
        labels = np.unique(Y_full)
        report_score(score, labels, args)

    elif args.experiment == "cv":
        # same as main but with 10-fold CV

        kf = KFold(n_splits=10)
        X_full, Y_full = map(np.array, (X_full, Y_full))

        scores = []
        # For each fold compute metrics
        for train_i, test_i in kf.split(X_full):

            # Reinstantiate to forget previous fold training
            classifier = Pipeline([('vec', vec), ('cls', model_class())])

            # Perform the split with the corresponding indices
            X_train, Y_train = X_full[train_i], Y_full[train_i]
            X_test, Y_test = X_full[test_i], Y_full[test_i]

            # Fit classifier and make predictions
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)

            # Compute metrics and store them
            score = complete_scoring(Y_test, Y_pred)
            scores.append(score)

        # Aggregate all metrics by averaging
        score = dict_op(avg_dict, *scores)

        # Report scores
        labels = np.unique(Y_full)
        report_score(score, labels, args)

    elif args.experiment == "error_lengths":
        # find some erronerous examples (hard coded) and output average review length per all correctly and incorrectly classified examples
        classifier.fit(X_train, Y_train)

        # find examples
        if args.sentiment:
            Y_pred_prob = classifier.predict_proba(X_train)
            Y_pred = classifier.predict(X_train)
            print([
                (' '.join(doc), gold, pred, pred_prob) for doc, gold, pred, pred_prob in zip(X_train, Y_train, Y_pred, Y_pred_prob)
                if gold != pred and max(pred_prob) >= 0.6 and len(doc) <= 50
            ][:10])

            Y_pred = classifier.predict(X_train)
            avg_incorrect = np.average([len(doc) for doc, gold, pred in zip(
                X_train, Y_train, Y_pred) if gold != pred])
            avg_correct = np.average([len(doc) for doc, gold, pred in zip(
                X_train, Y_train, Y_pred) if gold == pred])
            print(avg_correct, avg_incorrect)
        else:
            Y_pred = classifier.predict(X_train)
            print([' '.join(doc) for doc, gold, pred in zip(X_train, Y_train, Y_pred)
                  if gold == "books" and pred == "dvd" and "watch" in doc][:10])

            avg_incorrect = np.average([len(doc) for doc, gold, pred in zip(
                X_train, Y_train, Y_pred) if gold != pred])
            avg_correct = np.average([len(doc) for doc, gold, pred in zip(
                X_train, Y_train, Y_pred) if gold == pred])
            print(avg_correct, avg_incorrect)

    elif args.experiment == "error_corr":
        # find whether mispredictions correlate along tasks 

        X_full, Y_full_s, Y_full_m = read_corpus(args.input_file, None)
        X_train, _, Y_train, _ = train_test_split(
            X_full, list(zip(Y_full_s, Y_full_m)),
            test_size=args.test_percentage,
            random_state=args.seed,
            shuffle=args.shuffle
        )
        Y_train_m, Y_train_s = zip(*Y_train)

        classifier_m = Pipeline([('vec', vec), ('cls', model_class())])
        classifier_m.fit(X_train, Y_train_m)
        classifier_s = Pipeline([('vec', vec), ('cls', model_class())])
        classifier_s.fit(X_train, Y_train_s)

        mask_m = (Y_train_m != classifier_m.predict(X_train)).astype(int)
        mask_s = (Y_train_s != classifier_s.predict(X_train)).astype(int)
        rev_len = np.array([len(x) for x in X_train])
        print(np.corrcoef([mask_m, mask_s, rev_len]))
        print("same classification", sum(mask_m == mask_s)/len(mask_m))

    elif args.experiment == "train_data":
        # examine the effect of limited data on (all) model performance

        data_out = {}
        # For each relevant model
        for model_name, model_class in model_factory("all").items():
            for tf_idf in [False, True]:  # Using Bow and TF-IDF
                print(model_name, tf_idf)

                if tf_idf:
                    vec = TfidfVectorizer(
                        preprocessor=lambda x: x, tokenizer=lambda x: x)
                else:
                    vec = CountVectorizer(
                        preprocessor=lambda x: x, tokenizer=lambda x: x)
                classifier = Pipeline([('vec', vec), ('cls', model_class())])

                accs = []
                # For different amounts of train data
                for train_index in np.linspace(10, len(X_train), num=20):
                    # compute performance curves (against data) of a specific model configuration
                    train_index = int(train_index)

                    # Fit and predict
                    classifier.fit(X_train[:train_index],
                                   Y_train[:train_index])
                    Y_pred = classifier.predict(X_test)

                    # Store results
                    accs.append((train_index, accuracy_score(Y_test, Y_pred)))
                data_out[(model_name, tf_idf)] = accs

        # Dump results into pickle file
        with open(args.data_out, "wb") as f:
            pickle.dump(data_out, f)

    elif args.experiment == "stability":
        # validate that shuffling training data does not affect performance

        # Fit and predict without shuffling
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        acc1 = accuracy_score(Y_test, Y_pred)

        # Shuffle
        tmp = list(zip(X_train, Y_train))
        random.Random().shuffle(tmp)
        X_train, Y_train = zip(*tmp)

        # Reinstantiate
        classifier = Pipeline([('vec', vec), ('cls', model_class())])

        # Fit and predict again
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        acc2 = accuracy_score(Y_test, Y_pred)

        assert acc1 == acc2  # Check that they perform the same

    elif args.experiment == "train_stability":
        # compute variance in k-fold cross validation runs

        folds = KFold(n_splits=10)
        accs = []
        X_train = np.array(X_train, dtype=object)
        Y_train = np.array(Y_train, dtype=object)

        for train_indicies, _ in folds.split(X_train):  # For each fold
            # Fit and predic
            classifier.fit(X_train[train_indicies], Y_train[train_indicies])
            Y_pred = classifier.predict(X_test)

            # Store accuracy
            accs.append(accuracy_score(Y_test, Y_pred))

        # Compute statistical measurements of variability
        print("average:", np.average(accs))
        print("std:", np.std(accs))
        print("diameter:", max(accs) - min(accs))

    elif args.experiment == "errors":
        # Print out all the errors that the system makes in 10 fold CV
        # And the prediction probabilities for each error
        # Additionally compute the confidence of the system when classifying correctly and incorrectly

        kf = KFold(n_splits=10)

        X_full, Y_full = map(np.array, (X_full, Y_full))
        labels = np.unique(Y_full)

        proba_correct = []  # Probability of the predicted label on correct instances
        proba_incorrect = []  # Probability of the predicted label on incorrect instances

        for i, (train_i, test_i) in enumerate(kf.split(X_full)):  # For each fold
            classifier = Pipeline([('vec', vec), ('cls', model_class())])

            # Compute split
            X_train, Y_train = X_full[train_i], Y_full[train_i]
            X_test, Y_test = X_full[test_i], Y_full[test_i]

            # Fit and predict
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)

            # Get prediction probabilities and indices of the predicted label
            Y_pred_proba = classifier.predict_proba(X_test)
            Y_pred_idx = np.argmax(Y_pred_proba, axis=1)

            # ids for correct and incorrect instances
            error_mask = Y_pred != np.array(Y_test)
            error_idx = error_mask.nonzero()[0]
            correct_idx = (~error_mask).nonzero()[0]

            # store confidences
            proba_correct.extend(
                Y_pred_proba[correct_idx, Y_pred_idx[correct_idx]])
            proba_incorrect.extend(
                Y_pred_proba[error_idx, Y_pred_idx[error_idx]])

            print("\n\n", "#" * 10, "Fold", i, "#" * 10, "\n\n")

            # Print incorrectly labeled instances to screen and information about the classification of these
            errors = [dict(zip(labels, err))
                      for err in Y_pred_proba[error_idx]]
            for j, error_id in enumerate(error_idx):
                print("{:.200}".format(" ".join(X_test[error_id])))
                print("Predicted: {}, Gold: {}".format(
                    labels[Y_pred_idx[error_id]], Y_test[error_id]))
                pprint(errors[j])

        # Compute the mean for the confidence
        correct_conf = statistics.mean(proba_correct)
        incorrect_conf = statistics.mean(proba_incorrect)

        # Print to stdout
        print("Average confidence on correct instances: {:.4}".format(
            correct_conf * 100))
        print("Average confidence on incorrect instances: {:.4}".format(
            incorrect_conf * 100))
