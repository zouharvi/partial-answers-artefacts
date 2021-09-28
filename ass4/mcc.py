#!/usr/bin/env python3

import random as python_random
import json
import argparse
import numpy
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelBinarizer

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", default='train_NE.txt', type=str,
                        help="Input file to learn from (default train_NE.txt)")
    parser.add_argument("-e", "--embeddings", default='glove_filtered.json', type=str,
                        help="Embedding file we are using (default glove_filtered.json)")
    parser.add_argument("--seed", default=1234, type=int)

    return parser.parse_args()

def read_corpus(corpus_file):
    '''Read in the named entity data from a file'''
    names = []
    labels = []
    for line in open(corpus_file, 'r'):
        name, label = line.strip().split()
        names.append(name)
        labels.append(label)
    return names, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: numpy.array(embeddings[word]) for word in embeddings}

def vectorizer(words, embeddings):
    '''Turn words into embeddings, i.e. replace words by their corresponding embeddings'''
    return numpy.array([embeddings[word] for word in words])


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Make reproducible as much as possible
    print("Using seed", args.seed)
    numpy.random.seed(args.seed)

    # Read in the data and embeddings
    X_full, Y_full = read_corpus(args.input_file)
    embeddings = read_embeddings(args.embeddings)

    # Transform words to embeddings
    X_full = vectorizer(X_full, embeddings)

    # Transform string labels to one-hot encodings
    # Use encoder.classes_ to find mapping back
    encoder = LabelBinarizer()
    _ = encoder.fit_transform(Y_full)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_full, Y_full)
    for i, prior in enumerate(model.class_prior_):
        print(encoder.classes_[i], f"{prior:.2%}")
    
    print(f"ACC: {model.score(X_full, Y_full):.2%}")

if __name__ == '__main__':
    main()
