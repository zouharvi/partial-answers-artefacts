#!/usr/bin/env python3

'''TODO: add high-level description of this Python script'''

import random
import argparse
import numpy as np
from utils import *

# Make reproducible as much as possible
np.random.seed(1234)
random.seed(1234)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train-file", default='train.txt',
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev-file", default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-o", "--output-dir", default="../pertrubed_data",
                        help="Where to store produced data")
    args = parser.parse_args()
    return args

def line_shuffle(line):
    tokens = line.split()
    random.shuffle(tokens)
    return ' '.join(tokens)

def line_drop(line, perc):
    tokens = line.split()
    count = int(perc*len(tokens))

    # this is not a very efficient way but is quite safe
    for _ in range(count):
        tokens.pop(random.randrange(len(tokens)))

    return ' '.join(tokens)

def line_add(line, perc, tokens_all):
    tokens = line.split()
    count = int(perc*len(tokens))

    # this is not a very efficient way but is quite safe
    for _ in range(count):
        tokens.insert(random.randrange(len(tokens)), random.choice(tokens_all))

    return ' '.join(tokens)


def line_change(line, perc, tokens_all):
    tokens = line.split()
    count = int(perc*len(tokens))

    # this is not a very efficient way but is quite safe
    for _ in range(count):
        tokens[random.randrange(len(tokens))] = random.choice(tokens_all)

    return ' '.join(tokens)

def save_data(filename, X, Y):
    with open(filename, "w") as f:
        f.writelines([f"{y} - - {x}\n" for x, y in zip(X, Y)])

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    SMALL_PCT = 25
    LARGE_PCT = 50

    for filename, X, Y in [("train", X_train, Y_train), ("dev", X_dev, Y_dev)]:
        save_data(
            args.output_dir + f"/{filename}_s.txt",
            [line_shuffle(line) for line in X],
            Y
        )
        save_data(
            args.output_dir + f"/{filename}_d{SMALL_PCT}.txt",
            [line_drop(line, SMALL_PCT/100) for line in X],
            Y
        )
        save_data(
            args.output_dir + f"/{filename}_d{LARGE_PCT}.txt",
            [line_drop(line, LARGE_PCT/100) for line in X],
            Y
        )

        tokens_all = list(set([item for subl in X_train+X_dev for item in subl]))

        save_data(
            args.output_dir + f"/{filename}_a{SMALL_PCT}.txt",
            [line_add(line, SMALL_PCT/100, tokens_all) for line in X],
            Y
        )
        save_data(
            args.output_dir + f"/{filename}_a{LARGE_PCT}.txt",
            [line_add(line, LARGE_PCT/100, tokens_all) for line in X],
            Y
        )
        save_data(
            args.output_dir + f"/{filename}_c{SMALL_PCT}.txt",
            [line_change(line, SMALL_PCT/100, tokens_all) for line in X],
            Y
        )
        save_data(
            args.output_dir + f"/{filename}_c{LARGE_PCT}.txt",
            [line_change(line, LARGE_PCT/100, tokens_all) for line in X],
            Y
        )


if __name__ == '__main__':
    main()
