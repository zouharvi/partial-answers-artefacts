#!/usr/bin/env python3

import sys

sys.path.append("src")
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional

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

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

def dists(sampleA, sampleB):
    indexA = np.argmax(sampleA[1])
    onehotA = np.array([float(i == indexA) for i, _ in enumerate(sampleA[1])])
    indexB = np.argmax(sampleB[1])
    onehotB = np.array([float(i == indexB) for i, _ in enumerate(sampleB[1])])
    hard_dist = l2_dist(onehotA, onehotB)
    last_layer_dist = l2_dist(sampleA[1], sampleB[1])
    softmax_dist = l2_dist(
        softmax(sampleA[1]),
        softmax(sampleB[1])
    )
    cls_dists = [l2_dist(a, b) for a, b in zip(sampleA[0], sampleB[0])]
    return [
        *cls_dists,
        last_layer_dist,
        softmax_dist,
        hard_dist,
    ]


if __name__ == "__main__":
    args = parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    data = data[:8000]

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

    fig = plt.figure(figsize=(5, 4))
    plt.plot(
        dist_t_t,
        marker=".",
        linestyle="-",
        color="tab:green",
        label=r"$\checkmark \rightarrow \checkmark$",
    )
    plt.plot(
        dist_f_t,
        marker=".",
        linestyle=":",
        color="tab:green",
        label=r"$\times \rightarrow \checkmark$",
    )
    plt.plot(
        dist_f_f,
        marker=".",
        linestyle="-",
        color="tab:red",
        label=r"$\times \rightarrow \times$",
    )
    plt.plot(
        dist_t_f,
        marker=".",
        linestyle=":",
        color="tab:red",
        label=r"$\checkmark \rightarrow \times$",
    )
    plt.ylabel("$L^2$ distance", labelpad=-50)
    plt.xticks(
        ticks=list(range(16)),
        labels=["CLS$_{" + str(i) + "}$" for i in range(13)] +
        ["Last layer", "Softmax", "Prediction"],
        rotation=45,
    )
    plt.legend(ncol=2)
    plt.tight_layout(rect=(-0.02, -0.04, 1.04, 1.02))
    plt.savefig("fusion_tracing.png")
    plt.show()
