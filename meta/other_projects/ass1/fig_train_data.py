#!/usr/bin/env python3

"""
Vizualization of data size effect on model performance
"""

# Visualization libs
import matplotlib.pyplot as plt

# Misc
import argparse
from argparse import Namespace
import pickle


def parse_args() -> Namespace:
    """Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """

    parser = argparse.ArgumentParser()  # Argument parsing object

    # Arguments
    parser.add_argument("--data", default="tmp.pkl",
                        help="Where to find experiment data")
    parser.add_argument("-s", "--sentiment", action="store_true")
    parser.add_argument("-l", "--legend", action="store_true")

    # Parse the args
    args = parser.parse_args()
    return args


# Utility maps
PRETTY_NAME_MODEL = {
    "nb": "Naive Bayes",
    "lr": "Logistic Regression",
    "mccc": "Baseline",
}
PRETTY_NAME_VEC = {
    True: "TF-IDF",
    False: "BoW",
}
COLOR_NAME_MODEL = {
    "nb": "tab:red",
    "lr": "tab:blue",
    "mccc": "tab:green",
}
STYLE_VEC = {
    True: "-",
    False: ":",
}

if __name__ == "__main__":
    args = parse_args()

    # Load pickled data
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    # Create figure
    plt.figure(figsize=(4.7, 3.7))

    # plot each model configuration
    for (model_name, tfidf), accs in data.items():
        plt.plot(
            [x[0] for x in accs],
            [x[1] for x in accs],
            label=PRETTY_NAME_MODEL[model_name] + " " + PRETTY_NAME_VEC[tfidf],
            color=COLOR_NAME_MODEL[model_name],
            linestyle=STYLE_VEC[tfidf],
        )

    plt.xlabel("Data size")
    plt.ylabel("Accuracy")

    # Add textual info
    if args.legend:
        plt.legend(bbox_to_anchor=(0.5, 1.25), loc="upper center", ncol=3)
    else:
        if args.sentiment:
            plt.title("Sentiment task")
        else:
            plt.title("Topic task")
        plt.tight_layout()

    # Show the figure
    plt.show()
