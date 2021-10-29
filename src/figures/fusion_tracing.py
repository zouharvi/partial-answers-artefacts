#!/usr/bin/env python3

import sys
sys.path.append("src")
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", default='data/misc/tracing_month_small.pkl',
        help="Path of the data file."
    )

    args = parser.parse_args()
    return args


def l2_dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def dists(sampleA, sampleB):
    softmax_dist = l2_dist(sampleA[1], sampleB[1])
    cls_dists = [l2_dist(a, b) for a, b in zip(sampleA[0], sampleA[1])]
    return [
        *cls_dists,
        softmax_dist,
    ]


if __name__ == "__main__":
    args = parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    # collate
    assert len(data) % 16 == 0
    bucket = []
    data_new = []
    for x in data:
        bucket.append(x)
        if len(bucket) == 16:
            data_new.append(bucket)
            bucket = []
    assert len(bucket) == 0

    dist_t_t = []
    dist_f_t = []
    dist_f_f = []
    dist_t_f = []

    for bucket_i, bucket in enumerate(data_new):
        print([sum(x[0][2]) for x in bucket])
        # zeroth sample should be without artefacts
        sample_zero = bucket[0]
        assert sum(sample_zero[0][2]) == 0
        # use input from zeroth sample
        for sample_i, sample in enumerate(bucket[1:]):
            dist = dists(sample_zero[0], sample[0])
            if sample_zero[1] and sample[1]:
                dist_t_t.append(dist)
            elif not sample_zero[1] and sample[1]:
                dist_f_t.append(dist)
            elif not sample_zero[1] and not sample[1]:
                dist_f_f.append(dist)
            elif sample_zero[1] and not sample[1]:
                dist_t_f.append(dist)

    dist_t_t = np.average(np.array(dist_t_t), axis=0)
    dist_f_t = np.average(np.array(dist_f_t), axis=0)
    dist_f_f = np.average(np.array(dist_f_f), axis=0)
    dist_t_f = np.average(np.array(dist_t_f), axis=0)
    print(dist_t_t.shape)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(
        dist_t_t,
        marker=".",
        linestyle="-",
        color="tab:green",
        label=r"$\checkmark$ $\rightarrow$ $\checkmark$",
    )
    plt.plot(
        dist_f_t,
        marker=".",
        linestyle=":",
        color="tab:green",
        label=r"$\times$ $\rightarrow$ $\checkmark$",
    )

    plt.plot(
        dist_t_f,
        marker=".",
        linestyle="-",
        color="tab:red",
        label=r"$\checkmark$ $\rightarrow$ $\times$",
    )
    plt.plot(
        dist_f_f,
        marker=".",
        linestyle=":",
        color="tab:red",
        label=r"$\checkmark$ $\rightarrow$ $\checkmark$",
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
