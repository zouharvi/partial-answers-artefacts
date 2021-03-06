#!/usr/bin/env python3

"""
Train a language model based classifier to predict output-targets given certain input-target.
This script allows control over many aspects of the model / training. The output is a 
model checkpoint stored in the filesystem.
"""

from lm_model import LMModel
import sys
sys.path.append("src")
import utils
import utils.eval

import os.path as path
import argparse
import collections as col
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/models/{m}_{es}_{pm}_{ls}_{ep}_{ht}_{ml}_{ti}_{to}_{inp}.pt', type=str,
                        help="Path where to store the model.")
    parser.add_argument("-ti", "--target-input", default='body', type=str,
                        help="Input of the model.")
    parser.add_argument("-to", "--target-output", default=['newspaper'], type=str, nargs="+",
                        help="Target output of the model")
    parser.add_argument("-ts", "--train-samples", default=-2000, type=int,
                        help="Number of samples for the training (may be negative to get all except a given number)")
    parser.add_argument("-ds", "--dev-samples", default=1000, type=int,
                        help="Number of samples for the validation.")
    parser.add_argument("-ht", "--head-thickness", default='shallow',
                        help="Architecture of the classification head (shallow/mid)")
    parser.add_argument("--embed-strategy", default='cls',
                        help="Strategy to embed sentence with last layer hidden states.")
    parser.add_argument("--loss-powermean-degree", default=1, type=int,
                        help="Degree of powermean to take when reducing losses")
    parser.add_argument("--loss-scaling",default="uniform",type=str,
                        help="Type of scaling to use for loss averaging. Possibilities are \"uniform\" and \"scaled\".")
    parser.add_argument("-ep", "--epochs", default=2, type=int,
                        help="Override the default number of epochs.")
    parser.add_argument("-bs", "--batch-size", default=8, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model.")
    parser.add_argument("--max-length", default=512, type=int,
                        help="Maximum length of language model input.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # format output name
    output_name = args.output.format(
        m=args.language_model,
        ti=args.target_input,
        to="_".join(args.target_output),
        ml=args.max_length,
        inp=path.basename(args.input)[:-5],
        ht=args.head_thickness,
        es=args.embed_strategy,
        ep=args.epochs,
        pm=args.loss_powermean_degree,
        ls=args.loss_scaling
    )

    # read data
    data = utils.load_data(args.input)

    # prepare target labels
    targets = args.target_output
    if len(targets) == 1:
        if targets[0] == "all":
            targets = utils.Y_KEYS_LIST
        elif targets[0] == "craft":
            code = path.basename(args.input)[-6]
            targets = [utils.CODE_TO_Y_KEYS[code]]

    # process data
    target_input = utils.get_x(data, args.target_input)
    target_outputs, label_names, labels = utils.get_y(data, targets)

    # split data
    train_size = (len(data) + args.train_samples) % len(data)
    dev_size = args.dev_samples
    test_size = len(data) - train_size - dev_size
    (x_test, y_test), (x_dev, y_dev), (x_train, y_train) = utils.make_split(
        (target_input, labels),
        splits=(test_size, dev_size,),
        random_state=0
    )

    # compute label dimension
    dimensions = list(map(len, label_names))
    
    # compute scaling weights
    if args.loss_scaling == "uniform":
        weights = None
    elif args.loss_scaling == "scaled":
        count_targets = col.Counter(target_outputs)
        weights = 1 / np.array([count_targets[x] for x in target_outputs])
    else:
        raise ValueError("Could not find value ")
    
    # instantiate model
    lm = LMModel(
        cls_target_dimensions=dimensions,
        loss_weights=weights,
        lm=args.language_model,
        embed_strategy=args.embed_strategy,
        loss_powermean_degree=args.loss_powermean_degree,
        head_thickness=args.head_thickness,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # train model
    lm.fit(x_train, y_train, x_dev, y_dev)

    # save model
    lm.save_to_file(output_name)