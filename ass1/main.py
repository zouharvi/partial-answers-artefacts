#!/usr/bin/env python3

'''Small script to experiment with review classification'''

import argparse
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from report_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tf-idf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-tp", "--test_percentage", default=0.20, type=float,
                        help="Percentage of the data that is used for the test set (default 0.20)")
    parser.add_argument("--model", default="nb",
                        help="nb - Naive Bayes, lr - logistic regression")
    parser.add_argument("--experiment", default="main",
                        help="Which experiment to run: main, mccc, cv, train_data")
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle data set before splitting in train/test")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used for shuffling")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    '''Load and parse corpus structure'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1] == "pos")
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


if __name__ == "__main__":
    args = parse_args()

    # Load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file, args.sentiment)

    # use scikit's built-in splitting function to save space
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full,
        test_size=args.test_percentage,
        random_state=args.seed,
        shuffle=args.shuffle
    )

    # Convert the texts to vectors
    if args.tf_idf:
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

    # Combine the vectorizer with a Naive Bayes classifier
    if args.model == "nb":
        model_class = lambda: MultinomialNB()
    elif args.model == "lr":
        model_class = lambda: LogisticRegression(max_iter=5000)
    elif args.model == "mccc":
        model_class = lambda: DummyClassifier(strategy="most_frequent")
    else:
        raise Exception(f"Unknown model {args.model}")

    classifier = Pipeline([('vec', vec), ('cls', model_class())])

    if args.experiment == "main":
        # train the classifier
        classifier.fit(X_train, Y_train)
        # make inferences
        Y_pred = classifier.predict(X_test)
        # compute evaluation metrics
        acc = accuracy_score(Y_test, Y_pred)
        print("Final accuracy: {}".format(acc))
    elif args.experiment == "cv":
        # TODO: how to pass random state to CV?
        score = cross_validate(
            classifier, X_full, Y_full, cv=10, n_jobs=5,
            scoring=["accuracy"], return_train_score=False,
        )
        print(f'acc: {np.average(score["test_accuracy"]):.2%}')
        print(f'std: {np.std(score["test_accuracy"]):.5f}')
    elif args.experiment == "train_data":
        pass
    elif args.experiment == "train_stability":
        folds = KFold(n_splits=10)
        accs = []
        X_train = np.array(X_train, dtype=object)
        Y_train = np.array(Y_train, dtype=object)
        for train_indicies, _ in folds.split(X_train):
            # train the classifier
            classifier.fit(X_train[train_indicies], Y_train[train_indicies])
            # make inferences
            Y_pred = classifier.predict(X_test)
            # compute evaluation metrics
            accs.append(accuracy_score(Y_test, Y_pred))
        print("average:", np.average(accs))
        print("std:", np.std(accs))
        print("diameter:", max(accs)-min(accs))
    elif args.experiment == "error_classes":
        # train the classifier
        classifier.fit(X_train, Y_train)

        # make inferences
        Y_pred = classifier.predict(X_test)

        # compute evaluation metrics
        acc = accuracy_score(Y_test, Y_pred)
        c_report = classification_report(Y_test, Y_pred, output_dict=True)

        # Compute confusion matrix
        c_mat = confusion_matrix(Y_test, Y_pred)

        print("Final accuracy: {}".format(acc))
        print(format_report(c_report))
        print(format_auto_matrix(c_mat,np.unique(Y_train)))
