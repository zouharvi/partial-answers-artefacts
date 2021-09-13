#!/usr/bin/env python3

"""
Train a model for six-way review classification
"""

# Math/Numeric libraries
import numpy as np
from scipy.sparse.construct import rand

# ML library
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn.tree, sklearn.naive_bayes, sklearn.ensemble, sklearn.neighbors, sklearn.base
import sklearn.preprocessing, sklearn.feature_selection

# Misc
import argparse
import itertools


def parse_args():
    # Argument parsing object
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-i", "--input-file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("--experiment", default="main",
                        help="Which experiment to run: main, mccc, cv, train_data")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used for shuffling")

    args, _ = parser.parse_known_args()

    return args


def read_corpus(corpus_filepath):
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
            labels_m.append(tokens[0])

    return documents[:250], labels_m[:250]

# Notes:
# - WorNetLemmatizer didn't help
# - Automatic feature selection
# - Custom feature (review length)

if __name__ == "__main__":
    args = parse_args()

    # load the corpus
    X_full, Y_full = read_corpus(args.input_file)
    print(len(X_full))

    kf = KFold(n_splits=10)
    X_full = np.array(X_full, dtype=object)
    Y_full = np.array(Y_full, dtype=object)

    scores_all = []
    
    # for weights in itertools.product(*([[1,2]]*4)):
    for weights in [(1, 1, 1, 0.5, 0.5)]:
        scores = []
        # For each fold compute metrics
        for train_i, test_i in list(kf.split(X_full))[:1]:
            # combine the vectorizer with the statistical model
            classifier = Pipeline([
                ('vec', FeatureUnion([
                    ("tfidf", TfidfVectorizer(stop_words="english")),
                    ("tfidf_kernel", Pipeline([      
                        ("tfidf_crop", TfidfVectorizer(stop_words="english", max_features=1000)),
                        ('kernel', sklearn.preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)),
                    ])),
                ])),
                ('cls', sklearn.ensemble.VotingClassifier(
                    estimators=[
                        ('gauss_nb', sklearn.naive_bayes.GaussianNB()),
                        ('multin_nb', sklearn.naive_bayes.MultinomialNB()),
                        ('rf', sklearn.ensemble.RandomForestClassifier(random_state=0, n_estimators=200, n_jobs=-1)),
                        ('knn_1', Pipeline([
                            ("scaler", sklearn.preprocessing.Normalizer()),
                            ("knn_model", sklearn.neighbors.KNeighborsClassifier(n_neighbors=200, weights="distance"))
                        ])),
                        ('knn_2', Pipeline([
                            ("scaler", sklearn.preprocessing.Normalizer()),
                            ("knn_model", sklearn.neighbors.KNeighborsClassifier(n_neighbors=200, weights="distance", metric="cosine"))
                        ])),
                    ],
                    voting='soft',
                    weights=weights,
                    n_jobs=-1
                ))
            ])

            # Perform the split with the corresponding indices
            X_train, Y_train = X_full[train_i], Y_full[train_i]
            X_test, Y_test = X_full[test_i], Y_full[test_i]

            # Fit classifier and make predictions
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)

            score = accuracy_score(Y_test, Y_pred)
            scores.append(score)

        print(weights, f"avg: {np.average(scores):.2%}")
        scores_all.append((weights, np.average(scores)))

    print(sorted(scores_all, key=lambda x: x[1])[-1])