#!/usr/bin/env python3

'''Small script to experiment with review classification'''

import argparse
import numpy as np
import statistics

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer

from pprint import pprint

from report_utils import *
import pickle
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
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


def model_factory(model):
    '''return a model given the description string (or the whole dictionary)'''

    model_lib = {
        "nb": lambda: MultinomialNB(),
        "lr": lambda: LogisticRegression(max_iter=5000),
        "mccc": lambda: DummyClassifier(strategy="most_frequent"),
    }
    if model in model_lib:
        return model_lib[model]
    elif model == "all":
        return model_lib
    else:
        raise Exception(f"Unknown model {model}")


def complete_scoring(Y_test, Y_pred):
    # TODO: comment

    report = classification_report(Y_test, Y_pred, output_dict=True)
    c_mat = confusion_matrix(Y_test, Y_pred)
    score = dict(report=report, c_mat=c_mat)

    return score


def report_score(score, labels, args):
    # TODO: comment

    print(format_report(score["report"], format_=args.table_format))
    print(format_auto_matrix(score["c_mat"],
          labels, format_=args.table_format))

    score["labels"] = labels
    score["task"] = "sentiment" if args.sentiment else "topic"
    score["model"] = args.model

    with open(args.data_out, "wb") as f:
        pickle.dump(score, f)


if __name__ == "__main__":
    args = parse_args()

    # load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file, args.sentiment)

    # use scikit's built-in splitting function to save space
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_full, Y_full,
        test_size=args.test_percentage,
        random_state=args.seed,
        shuffle=args.shuffle
    )

    # compute features
    if args.tf_idf:
        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    else:
        # bag of Words vectorizer
        vec = CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)

    model_class = model_factory(args.model)
    # combine the vectorizer with the statistical model
    classifier = Pipeline([('vec', vec), ('cls', model_class())])

    if args.experiment == "main":
        # train the classifier, make inferences and compute metrics

        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        score = complete_scoring(Y_test, Y_pred)
        labels = np.unique(Y_full)
        report_score(score, labels, args)

    elif args.experiment == "cv_errors":
        # TODO: comment

        kf = KFold(n_splits=10)
        
        X_full, Y_full = map(np.array,(X_full,Y_full))
        
        scores = []
        for train_i, test_i in kf.split(X_full):
            classifier = Pipeline([('vec', vec), ('cls', model_class())])
            
            X_train,Y_train = X_full[train_i],Y_full[train_i]
            X_test,Y_test = X_full[test_i],Y_full[test_i]
            
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)

            score = complete_scoring(Y_test, Y_pred)
            scores.append(score)
        score = dict_op(avg_dict, *scores)

        labels = np.unique(Y_full)
        report_score(score, labels, args)

    elif args.experiment == "error_lengths":
        classifier.fit(X_train, Y_train)

        # find examples
        if args.sentiment:
            Y_pred = classifier.predict(X_train)
            avg_incorrect = np.average([len(doc) for doc, gold, pred  in zip(X_train, Y_train, Y_pred) if gold != pred])
            avg_correct = np.average([len(doc) for doc, gold, pred  in zip(X_train, Y_train, Y_pred) if gold == pred])
            print(avg_correct, avg_incorrect)
        else:
            Y_pred = classifier.predict(X_train)
            print([' '.join(doc) for doc, gold, pred  in zip(X_train, Y_train, Y_pred) if gold == "books" and pred == "dvd" and "watch" in doc][:10])

            avg_incorrect = np.average([len(doc) for doc, gold, pred  in zip(X_train, Y_train, Y_pred) if gold != pred])
            avg_correct = np.average([len(doc) for doc, gold, pred  in zip(X_train, Y_train, Y_pred) if gold == pred])
            print(avg_correct, avg_incorrect)

    elif args.experiment == "train_data":
        # examine the effect of limited data on (all) model performance

        data_out = {}
        for model_name, model_class in model_factory("all").items():
            for tf_idf in [False, True]:
                print(model_name, tf_idf)

                if tf_idf:
                    vec = TfidfVectorizer(
                        preprocessor=lambda x: x, tokenizer=lambda x: x)
                else:
                    vec = CountVectorizer(
                        preprocessor=lambda x: x, tokenizer=lambda x: x)
                classifier = Pipeline([('vec', vec), ('cls', model_class())])

                accs = []
                for train_index in np.linspace(10, len(X_train), num=20):
                    # compute performance curves (against data) of a specific model configuration
                    train_index = int(train_index)
                    classifier.fit(X_train[:train_index],
                                   Y_train[:train_index])
                    Y_pred = classifier.predict(X_test)
                    accs.append((train_index, accuracy_score(Y_test, Y_pred)))

                data_out[(model_name, tf_idf)] = accs

        with open(args.data_out, "wb") as f:
            pickle.dump(data_out, f)

    elif args.experiment == "stability":
        # validate that shuffling training data does not affect performance

        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        acc1 = accuracy_score(Y_test, Y_pred)

        tmp = list(zip(X_train, Y_train))
        random.Random().shuffle(tmp)
        X_train, Y_train = zip(*tmp)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        acc2 = accuracy_score(Y_test, Y_pred)

        assert acc1 == acc2

    elif args.experiment == "train_stability":
        # compute variance in k-fold cross validation runs

        folds = KFold(n_splits=10)
        accs = []
        X_train = np.array(X_train, dtype=object)
        Y_train = np.array(Y_train, dtype=object)

        for train_indicies, _ in folds.split(X_train):
            classifier.fit(X_train[train_indicies], Y_train[train_indicies])
            Y_pred = classifier.predict(X_test)
            accs.append(accuracy_score(Y_test, Y_pred))

        print("average:", np.average(accs))
        print("std:", np.std(accs))
        print("diameter:", max(accs) - min(accs))
        
    elif args.experiment == "errors":
        kf = KFold(n_splits=10)
        
        X_full, Y_full = map(np.array,(X_full,Y_full))
        labels = np.unique(Y_full)
        
        proba_correct = []
        proba_incorrect = []
        for i,(train_i, test_i) in enumerate(kf.split(X_full)):
            classifier = Pipeline([('vec', vec), ('cls', model_class())])
            
            X_train,Y_train = X_full[train_i],Y_full[train_i]
            X_test,Y_test = X_full[test_i],Y_full[test_i]
            
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)
            
            Y_pred_proba = classifier.predict_proba(X_test)
            Y_pred_idx = np.argmax(Y_pred_proba,axis=1)
            
            error_mask = Y_pred != np.array(Y_test)
            error_idx = error_mask.nonzero()[0]
            correct_idx = (~error_mask).nonzero()[0]
            
            proba_correct.extend(Y_pred_proba[correct_idx,Y_pred_idx[correct_idx]])
            proba_incorrect.extend(Y_pred_proba[error_idx,Y_pred_idx[error_idx]])
            
            print("\n\n", "#"*10, "Fold", i, "#"*10, "\n\n")
            
            errors = [dict(zip(labels,err)) for err in Y_pred_proba[error_idx]]
            for j,error_id in enumerate(error_idx):
                print("{:.200}".format(" ".join(X_test[error_id])))
                pprint(errors[j])
                print()
                
        correct_conf = statistics.mean(proba_correct)
        incorrect_conf = statistics.mean(proba_incorrect)
        
                
        print("Average confidence on correct instances: {:.4}".format(correct_conf*100))
        print("Average confidence on incorrect instances: {:.4}".format(incorrect_conf*100))        