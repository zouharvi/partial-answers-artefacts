#!/usr/bin/env python3

"""
This is a small script for topic classification using ensemble of ComplementNB, MultinomialNB, RandomForests and KNN.
The model hyperparameters were first optimized individually and then gridsearch was performed on the vote weights.
Simple LogisticRegression is better than the whole ensemble (see assignment 1) but its use is not allowed.

If one wished to overenginner this, they could do error analysis on the individual models
and select model based on the input (e.g. MultinomialNB is good only on longer reviews).

The 10-fold CV estimates the accuracy of 92.5%. This will, however, be probably lower for the test set, because
I've made decisions based on these results and hence "overfitted" the hyperparameters to the training data.
The runtime for single ensemble run is 20s. Because the ensemble is paralelized, it needs to fit 5 times the feature
matrix into memory <4GB.

Notes what didn't help:
- WorNetLemmatizer
- Custom features (e.g. review length)
- Feature union with BoW
- This elaborate feature kernel that was supposed to model term interactions (combined with standard TF-IDF):
```
("tfidf_kernel", Pipeline([      
    ("tfidf_crop", TfidfVectorizer(stop_words="english", max_features=1000)),
    ('kernel', sklearn.preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)),
])),
```
"""

import argparse
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import sklearn.tree
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.neighbors
import sklearn.base
import sklearn.preprocessing
import sklearn.feature_selection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--train-set", default='reviews.txt', help="Train set"
    )
    parser.add_argument("-ts", "--test-set", help="Test set")
    parser.add_argument(
        "--experiment", default="main", help="Options: main, search"
    )
    args, _ = parser.parse_known_args()
    return args


def read_corpus(corpus_filepath):
    """
    Parse corpus lines and return a tuple of (texts, classes)
    """
    documents = []
    labels_m = []
    with open(corpus_filepath, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(' '.join(tokens[3:]))
            labels_m.append(tokens[0])

    return documents, labels_m


# model definition
classifier_complnb = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words=None, max_features=90 * 1000,
        max_df=0.6, ngram_range=(1, 2), sublinear_tf=False
    )),
    ('complnb', sklearn.naive_bayes.ComplementNB()),
])
classifier_multinb = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", max_features=70 * 1000,
        max_df=0.2, ngram_range=(1, 7), sublinear_tf=False
    )),
    ('multinb', sklearn.naive_bayes.MultinomialNB()),
])
classifier_rforest = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", max_df=0.6, max_features=100 * 1000, ngram_range=(1, 2)
    )),
    ('rforest', sklearn.ensemble.RandomForestClassifier(
        random_state=0, n_jobs=-1, n_estimators=200, min_samples_split=3)),
])
classifier_knneuc = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", max_df=0.4, max_features=90 * 1000, ngram_range=(1, 2)
    )),
    ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
    ("normalizer", sklearn.preprocessing.Normalizer()),
    ("knn", sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=200, weights="distance"))
])
classifier_knncos = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english", max_df=0.4, max_features=90 * 1000, ngram_range=(1, 3)
    )),
    ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
    ("normalizer", sklearn.preprocessing.Normalizer()),
    # the ordering out of knn with cosine distance is probably the same as with L2
    # because the vectors are normalized but the weights may be different
    ("knn_model", sklearn.neighbors.KNeighborsClassifier(
        n_neighbors=200, weights="distance", metric="cosine"))
])
# it would be possible to first create the vectorizer and feed the output to every model
# but every model performs optimally with different vectorizer parameters
classifier_ensemble = sklearn.ensemble.VotingClassifier(
    estimators=[
        ("pip_complnb", classifier_complnb),
        ("pip_multinb", classifier_multinb),
        ("pip_rforest", classifier_rforest),
        ("pip_knneuc", classifier_knneuc),
        ("pip_knncos", classifier_knncos),
    ],
    voting='soft',
    weights=(1.3, 0.8, 1.0, 0.3, 0.3),
    n_jobs=-1,
)

if __name__ == "__main__":
    args = parse_args()

    # load the corpus
    X_train, Y_train = read_corpus(args.train_set)
    if args.test_set:
        X_test, Y_test = read_corpus(args.test_set)

    if args.experiment == "main":
        # fit classifier and make predictions
        classifier_ensemble.fit(X_train, Y_train)

        if args.test_set:
            Y_pred = classifier_ensemble.predict(X_test)
        else:
            print("test_set is not available, evaluating on train")
            Y_pred = classifier_ensemble.predict(X_train)

        score = accuracy_score(Y_train, Y_pred)
        print(f"score: {score:.2%}")

    elif args.experiment == "search":
        scores = []

        # parameter gridsearch
        # (most parameters are commented out because they were used for individual model hyperparameter optimalization)
        clf = sklearn.model_selection.GridSearchCV(
            classifier_ensemble,
            param_grid={
                # "tfidf__ngram_range": [(1, 2), (1, 3), (1, 4)],
                # "tfidf__max_df": [0.4, 0.5, 0.6, 0.7],
                # "tfidf__max_features": [90 * 1000, 100 * 1000, 110 * 1000],
                # "knn__weights": ["uniform", "distance"],
                # "knn__n_neighbors": [5, 50, 100, 150, 200, 250],
                # "knn__p": [1, 2, 3],
                # "knn__metric": ["minkowski", "cosine"],
                # "rforest__min_samples_split": [2, 3, 4],
                # "rforest__n_estimators": [100],
                # "tfidf__sublinear_tf": [True, False]
                # "tfidf__stop_words": [None, "english"],
                "weights": [
                    # baseline
                    (1.3, 0.8, 1.0, 0.3, 0.3),

                    # individual increase
                    (1.4, 0.8, 1.0, 0.3, 0.3),
                    (1.3, 0.9, 1.0, 0.3, 0.3),
                    (1.3, 0.8, 1.1, 0.3, 0.3),
                    (1.3, 0.8, 1.0, 0.4, 0.3),
                    (1.3, 0.8, 1.0, 0.3, 0.4),

                    # individual decrease
                    (1.2, 0.8, 1.0, 0.3, 0.3),
                    (1.3, 0.7, 1.0, 0.3, 0.3),
                    (1.3, 0.8, 0.9, 0.3, 0.3),
                    (1.3, 0.8, 1.0, 0.2, 0.3),
                    (1.3, 0.8, 1.0, 0.3, 0.2),
                ]
            },
            verbose=10,  # monitor progress
            n_jobs=4,
            cv=KFold(n_splits=10),
        )
        # run grisearch and print best result
        results = clf.fit(X_train, Y_train)
        print(results.best_params_, results.best_score_)

    elif args.experiment == "time":
        """
        Measure train and inference times for used model families
        """

        classifiers = {
            "complnb": sklearn.naive_bayes.ComplementNB(),
            "multinb": sklearn.naive_bayes.MultinomialNB(),
            "knn_200": sklearn.neighbors.KNeighborsClassifier(n_neighbors=200, weights="distance"),
            "knn_5": sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights="distance"),
            "rforest": sklearn.ensemble.RandomForestClassifier(n_estimators=200),
            "dt": sklearn.tree.DecisionTreeClassifier(),
        }

        for model_name, model in classifiers.items():
            model = Pipeline([
                ("vec", CountVectorizer()),
                ("model", model)
            ])

            start_fit = time.time()
            model.fit(X_train, Y_train)
            time_fit = time.time() - start_fit

            start_pred = time.time()
            model.predict(X_train)
            time_pred = time.time() - start_pred
    
            print(model_name, "FIT:", time_fit, "PRED:", time_pred)