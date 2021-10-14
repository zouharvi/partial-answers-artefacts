"""
This script computes and plots the dependency of K in KNN on performance (ACC, macro-F1)
"""

import matplotlib.pyplot as plt
import argparse
import numpy as np
import sklearn.preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


def parse_args():
    """
    Return instantiated Namespace object with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--train-set", default='reviews.txt', help="Train set"
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


if __name__ == "__main__":
    args = parse_args()
    data_x, data_y = read_corpus(args.train_set)
    data_x = np.array(data_x, dtype=object)
    data_y = np.array(data_y, dtype=object)
    logdata_train_acc = []
    logdata_train_f1 = []
    logdata_loocv_acc = []
    logdata_loocv_f1 = []
    N_NEIGHBOURS = [1, 3, 5, 7, 9, 11, 25, 50, 75,
                    100, 150, 200, 250, 500, 1000, 1500, 2000]
    LABELS = list(set(data_y))

    print("Train")
    for weights in ["uniform", "distance"]:
        for k in N_NEIGHBOURS:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    stop_words="english", max_df=0.4, max_features=90 * 1000, ngram_range=(1, 2)
                )),
                ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
                ("normalizer", sklearn.preprocessing.Normalizer()),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights)),
            ])
            model.fit(data_x, data_y)
            pred = model.predict(data_x)
            score_acc = accuracy_score(pred, data_y)
            score_f1 = f1_score(pred, data_y, labels=LABELS, average="macro")
            print(k, score_acc, score_f1, "\n", confusion_matrix(pred, data_y))
            logdata_train_acc.append((k, weights, score_acc))
            logdata_train_f1.append((k, weights, score_f1))

    print("CV")
    for weights in ["uniform", "distance"]:
        for k in N_NEIGHBOURS:
            model = Pipeline([
                ("tfidf", TfidfVectorizer(
                    stop_words="english", max_df=0.4, max_features=90 * 1000, ngram_range=(1, 2)
                )),
                ("scaler", sklearn.preprocessing.StandardScaler(with_mean=False)),
                ("normalizer", sklearn.preprocessing.Normalizer()),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights)),
            ])

            intermediate_acc = []
            intermediate_pred = []
            intermediate_f1 = []
            for train_index, test_index in KFold(n_splits=10).split(data_x):
                model.fit(data_x[train_index], data_y[train_index])
                pred = model.predict(data_x[test_index])
                intermediate_acc.append(accuracy_score(
                    y_true=data_y[test_index], y_pred=pred))
                intermediate_f1.append(f1_score(
                    y_true=data_y[test_index], y_pred=pred, labels=LABELS, average="macro"))

            score_acc = np.average(intermediate_acc)
            score_f1 = np.average(intermediate_f1)
            print(k, score_acc, score_f1)
            logdata_loocv_acc.append((k, weights, score_acc))
            logdata_loocv_f1.append((k, weights, score_f1))

    plt.figure(figsize=(9, 4))
    fig1 = plt.subplot(121)

    plt.plot(
        [x[2] for x in logdata_train_acc if x[1] == "uniform"],
        label="Train (ACC)", c="tab:blue", linestyle=":"
    )
    plt.plot(
        [x[2] for x in logdata_train_f1 if x[1] == "uniform"],
        label="Train (F1)", c="tab:blue", linestyle="-", alpha=0.7,
    )
    plt.plot(
        [x[2] for x in logdata_loocv_acc if x[1] == "uniform"],
        label="LOOCV (ACC)", c="tab:red", linestyle=":"
    )
    plt.plot(
        [x[2] for x in logdata_loocv_f1 if x[1] == "uniform"],
        label="LOOCV (F1)", c="tab:red", linestyle="-", alpha=0.7,
    )
    plt.xticks(list(range(len(N_NEIGHBOURS))), N_NEIGHBOURS, rotation=45)
    plt.title("KNN Topic Classification (Uniform)")
    plt.ylabel("Train/LOOCV accuracy/macro-F1 score")
    plt.xlabel("n_neighbours")
    plt.legend()

    fig2 = plt.subplot(122)
    plt.plot(
        [x[2] for x in logdata_train_acc if x[1] == "distance"],
        label="Train (ACC)", c="tab:blue", linestyle=":"
    )
    plt.plot(
        [x[2] for x in logdata_train_f1 if x[1] == "distance"],
        label="Train (F1)", c="tab:blue", linestyle="-", alpha=0.7,
    )
    plt.plot(
        [x[2] for x in logdata_loocv_acc if x[1] == "distance"],
        label="LOOCV (ACC)", c="tab:red", linestyle=":"
    )
    plt.plot(
        [x[2] for x in logdata_loocv_f1 if x[1] == "distance"],
        label="LOOCV (F1)", c="tab:red", linestyle="-", alpha=0.7,
    )
    plt.xticks(list(range(len(N_NEIGHBOURS))), N_NEIGHBOURS, rotation=45)
    plt.title("KNN Topic Classification (Distance)")
    plt.xlabel("n_neighbours")
    plt.legend()

    plt.tight_layout()
    plt.show()
