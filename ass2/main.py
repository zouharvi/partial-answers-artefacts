#!/usr/bin/env python3

"""Small script to experiment with review classification"""

# Math/Numeric libraries
import numpy as np
from scipy.sparse.construct import rand

# ML library
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn.tree, sklearn.naive_bayes, sklearn.ensemble, sklearn.neighbors, sklearn.base
import sklearn.preprocessing

# Misc
from typing import Union
import argparse
from argparse import Namespace


def parse_args() -> Namespace:
    # Argument parsing object
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-i", "--input-file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-t", "--tf-idf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("--model", default="nb",
                        help="nb - Naive Bayes, lr - logistic regression")
    parser.add_argument("--experiment", default="main",
                        help="Which experiment to run: main, mccc, cv, train_data")
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
            2. The labels (corresponding task) for each respective sentence.
    """

    documents = []
    labels_m = []
    with open(corpus_filepath, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(' '.join(tokens[3:]))
            # 6-class problem: books, camera, dvd, health, music, software
            labels_m.append(tokens[0])

    return documents, labels_m


from nltk.stem import WordNetLemmatizer
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, article):
        return article.split()
        # print(article, type(article))
        # print(article, self.wnl.lemmatize(article))
        return [self.wnl.lemmatize(x) for x in article.split()]

# Notes:
# - WorNetLemmatizer didn't help
# - Neither did feature union

class CustomFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y = None ):
        return self

    def transform(self, X, y=None):
        out = np.array([len(line.split()) for line in X]).reshape(-1, 1)
        return out

# Script logic
if __name__ == "__main__":
    args = parse_args()

    vec = FeatureUnion([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=200)),
        # ("custom", CustomFeatures()),
        # ("bow", CountVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)),
    ])

    model_nb = sklearn.naive_bayes.MultinomialNB()
    model_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model_knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=200, weights="distance")

    model_ensemble = sklearn.ensemble.VotingClassifier(
        estimators=[
            ('nb', model_nb), ('rf', model_rf), ('knn', model_knn)
        ],
        voting='soft',
        weights=[1.5, 1, 1],
        n_jobs=-1
    )

    # load the corpus
    X_full, Y_full = read_corpus(args.input_file)

    kf = KFold(n_splits=10)
    X_full = np.array(X_full, dtype=object)
    Y_full = np.array(Y_full, dtype=object)

    scores = []
    # For each fold compute metrics
    for train_i, test_i in list(kf.split(X_full))[:1]:
        # combine the vectorizer with the statistical model
        classifier = Pipeline([
            ('vec', vec),
            ('kernel', sklearn.preprocessing.PolynomialFeatures()),
            # ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
            ('cls', model_ensemble)
        ])

        # Perform the split with the corresponding indices
        X_train, Y_train = X_full[train_i], Y_full[train_i]
        X_test, Y_test = X_full[test_i], Y_full[test_i]

        # Fit classifier and make predictions
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)

        score = accuracy_score(Y_test, Y_pred)
        scores.append(score)

    print(np.average(scores))
