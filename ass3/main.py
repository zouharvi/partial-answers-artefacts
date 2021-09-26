#!/usr/bin/env python3

"""
Small script to experiment with review classification using SVM.
See README.md for further information
"""

# Misc
from typing import Union
import argparse
from argparse import Namespace

# Experiments
from experiments.exp_size import experiment_size
from experiments.exp_features import experiment_features
from experiments.exp_confidence import experiment_confidence
from experiments.exp_errors import experiment_errors
from experiments.exp_examples import experiment_examples
from experiments.exp_gridsearch import experiment_gridsearch
from experiments.exp_main import experiment_main
from experiments.exp_cv_eval import experiment_cv_eval 

def parse_args() -> Namespace:
    """
    Function containing all the argument parsing logic. Parses command line arguments and
    handles exceptions and help queries. 

    Returns
    =======
        Namespace object that has an attribute per command line parameter.
    """

    # Argument parsing object
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-i", "--input-file", default='reviews.txt', type=str,
                        help="Input file with all data")
    parser.add_argument("-ts", "--test-set", type=str,
                        help="Input file with data for test")
    parser.add_argument("-t", "--tf-idf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of Bag of Words")
    parser.add_argument("-tp", "--test-percentage", default=0.1, type=float,
                        help="Percentage of the data that is used for the test set")
    parser.add_argument("--experiment", default="main",
                        help="Which experiment to run: main, examples, errors, confidence, size, features")
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle data set before splitting in train/test")
    parser.add_argument("--seed", default=0, type=int,
                        help="Seed used for shuffling")
    parser.add_argument("--max-features", default=10000, type=int,
                        help="Maximum number of features in the vectorizer")
    parser.add_argument("--ngrams", action="store_true",
                        help="Use ngrams for feature exploration")
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
            1. The tokenized sentences (each a list)
            2. The sentiment labels for each respective sentence.
    """

    documents = []
    labels_s = []
    with open(corpus_filepath, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            # 2-class problem: positive vs negative
            labels_s.append(tokens[1] == "pos")

    return documents, labels_s

# Script logic
if __name__ == "__main__":
    args = parse_args()

    # load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file)

    # choose which experiment to run
    if args.experiment == "main":
        if args.test_set:
            X_test, Y_test = read_corpus(args.test_set)
        else:
            print("Test set not provided, evaluating with train set.")
            X_test, Y_test = X_full, Y_full
            
        experiment_main(X_full, Y_full, 
                        X_test, Y_test,
                        table_format=args.table_format)
        
    if args.experiment == "cv_eval":
        experiment_cv_eval(X_full, Y_full, table_format=args.table_format)
        
    elif args.experiment == "gridsearch":
        experiment_gridsearch(
            X_full, Y_full,
            data_out=args.data_out)

    elif args.experiment == "examples":
        experiment_examples(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            max_features=args.max_features,
        )

    elif args.experiment == "confidence":
        experiment_confidence(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            max_features=args.max_features,
            test_percentage=args.test_percentage,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        
    elif args.experiment == "errors":
        experiment_errors(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            max_features=args.max_features,
            data_out=args.data_out,
            test_percentage=args.test_percentage,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    elif args.experiment == "size":
        experiment_size(
            X_full, Y_full,
            data_out=args.data_out,
            test_percentage=args.test_percentage,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    elif args.experiment == "features":
        experiment_features(
            X_full, Y_full,
            tf_idf=args.tf_idf,
            use_ngrams=args.ngrams,
            max_features=args.max_features,
            data_out=args.data_out,
        )
