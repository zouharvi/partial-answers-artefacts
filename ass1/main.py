#!/usr/bin/env python3

'''This code is aimed to process the labeled file and to predict to which class the text belongs to'''

import sys
import argparse
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='reviews.txt', type=str,
                        help="Input file to learn from (default reviews.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-t", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-tp", "--test_percentage", default=0.20, type=float,
                        help="Percentage of the data that is used for the test set (default 0.20)")
    parser.add_argument("-sh", "--shuffle", action="store_true",
                        help="Shuffle data set before splitting in train/test")
    parser.add_argument("--seed", default=123, type=int, help="Seed used for shuffling")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file, use_sentiment):
    '''TODO: add function description'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def shuffle_dependent_lists(l1, l2, seed):
    '''Shuffle two lists, but keep the dependency between them'''
    tmp = list(zip(l1, l2))
    # Seed the random generator so results are consistent between runs
    random.Random(seed).shuffle(tmp)
    return zip(*tmp)


def split_data(X_full, Y_full, test_percentage, shuffle, seed):
    '''This function is aimed to split the data to the train and test set'''
    split_point = int(test_percentage*len(X_full))
    # TODO: comment
    if shuffle:
        X_full, Y_full = shuffle_dependent_lists(X_full, Y_full, seed)
    X_train = X_full[:split_point]
    Y_train = Y_full[:split_point]
    X_test = X_full[split_point:]
    Y_test = Y_full[split_point:]
    return X_train, Y_train, X_test, Y_test


def identity(x):
    '''Dummy function that just returns the input'''
    return x


if __name__ == "__main__":
    args = create_arg_parser()

    # Here we load the corpus and split the data
    X_full, Y_full = read_corpus(args.input_file, args.sentiment)
    X_train, Y_train, X_test, Y_test = split_data(
            X_full, Y_full, args.test_percentage, args.shuffle, args.seed
    )

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # Combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec), ('cls', MultinomialNB())])

    # train the classifier
    classifier.fit(X_train, Y_train)

    # test the classifier
    Y_pred = classifier.predict(X_test)

    # check the accuracy
    acc = accuracy_score(Y_test, Y_pred)
    print("Final accuracy: {}".format(acc))
