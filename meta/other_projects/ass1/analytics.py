#!/usr/bin/python3

'''
Rudimentary data distribution statistics
'''

from collections import Counter
import numpy as np
import argparse


def read_corpus(corpus_file):
    documents = []
    labels_s = []
    labels_m = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            # 2-class problem: positive vs negative
            labels_s.append(tokens[1])
            # 6-class problem: books, camera, dvd, health, music, software
            labels_m.append(tokens[0])
    return documents, labels_s, labels_m


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-file", default='reviews.txt', type=str,
                    help="Input file to learn from (default reviews.txt)")
args = parser.parse_args()

X_all, Ys_all, Ym_all = read_corpus(args.input_file)

print("count:", len(X_all))
print("class M distribution", Counter(Ym_all))
print("class S distribution", Counter(Ys_all))
print("class MS distribution", Counter(zip(Ys_all, Ym_all)))

print(f"Avereage token count: {np.average([len(x) for x in X_all]):.2f}")
print(
    f"Avereage sentence count: {np.average([x.count('.') for x in X_all]):.2f}")
