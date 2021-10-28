#!/usr/bin/env python3

import pickle
import random
import sys
sys.path.append("src")

import numpy as np
import utils
import json
import argparse
from collections import Counter
from lm_model import LMModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/misc/meta_{}.pkl',
                        help="Path of the data file.")
    parser.add_argument("-mp", "--model_path", required=True, type=str,
                        help="Path where to load the model from.")
    parser.add_argument("-ti", "--target-input", default='body', type=str,
                        help="Input of the model.")
    parser.add_argument("-to", "--target-output", default=['month'], type=str, nargs="+",
                        help="Target output of the model")
    parser.add_argument("-ts", "--test-samples", default=1000, type=int,
                        help="Amount of samples with which to test.")
    parser.add_argument("-ht", "--head-thickness", default='shallow',
                        help="Architecture of the classification head (shallow/mid)")
    parser.add_argument("-bs", "--batch-size", default=128, type=int,
                        help="Evaluation batch size.")
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model.")
    parser.add_argument("--max-length", default=256, type=int,
                        help="Maximum length of language model input.")

    args = parser.parse_args()
    return args


LM_ALIASES = dict(
    bert="bert-base-uncased",
    roberta="roberta-base",
    albert="albert-base-v2",
    distilroberta="distilroberta-base"
)


def artefacts_signature(artefacts, y_filter):
    return [1 if k in artefacts else 0 for k in utils.Y_KEYS_LOCAL - {y_filter}]


def x_manipulator_all(x, y, x_filter, y_filter):
    output = []
    for subset in utils.powerset(utils.Y_KEYS_LOCAL - {y_filter}, nonempty=False):
        artefacts = [
            y[k] if k in subset else "None"
            for k in utils.Y_KEYS_LOCAL - {y_filter}
        ]
        artefacts_str = '\n'.join(artefacts) + "\n" + x[x_filter]
        output.append((
            artefacts_str, artefacts_signature(subset, y_filter)
        ))
    return output

def hack(data, labels, lm):
    print("Predicting all subsets of artefacts")
    data_x_all = [
        (
            x_manipulator_all(x, y, "body", "month"),
            y_true[0]
        )
        for (x, y), y_true in zip(data, labels)
    ]
    # flatten data
    data_x_all = [
        (z, true_y)
        for artefacts, true_y in data_x_all
        for z in artefacts
    ]
    
    preds_all = lm.predict2([x[0] for x, y in data_x_all])

    # flatten results
    preds_all = [
        (r,x) 
        for rs,rx in preds_all
        for r,x in zip(rs, rx)
    ]

    data_out = []
    hits_full = []
    hits = []
    for (rep, y_pred_posterior), ((x, signature), y_true) in zip(preds_all, data_x_all):
        y_pred = np.argmax(y_pred_posterior)
        if sum(signature) == 4:
            hits_full.append(y_pred == y_true)
        hits.append(y_pred == y_true)
        data_out.append(((rep, y_pred_posterior, signature), y_pred == y_true))
        print(signature, y_pred, y_true)

    print("Acc (4 artefacts):", format(np.average(hits_full), ".2%"))
    print("Acc (any):", format(np.average(hits), ".2%"))

    with open("data/misc/meta_month.pkl", "wb") as f:
        pickle.dump(data_out, f)

if __name__ == "__main__":
    args = parse_args()

    assert len(args.target_output) == 1

    # Read data
    data = utils.load_data(args.input)

    # This may be an overkill because we're only interested in single class
    # label_names and labels but it asserts consistency across scripts
    _, label_names, labels = utils.get_y(data, args.target_output)

    # Instantiate transformer
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    lm = LMModel(
        cls_target_dimensions=[len(x) for x in label_names],
        lm=lm_name,
        head_thickness=args.head_thickness,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    # load the weights
    lm.load_from_file(args.model_path)

    print("Predicting all subsets of artefacts")
    data_x_all = [
        (
            x_manipulator_all(x, y, args.target_input, args.target_output[0]),
            y_true[0]
        )
        for (x, y), y_true in random.choices(list(zip(data, labels)), k=256)
    ]
    # flatten data
    data_x_all = [
        (z, true_y)
        for artefacts, true_y in data_x_all
        for z in artefacts
    ]

    preds_all = lm.predict2([x[0] for x, y in data_x_all])

    # flatten results
    preds_all = [
        (r,x) 
        for rs,rx in preds_all
        for r,x in zip(rs, rx)
    ]

    data_out = []
    hits_full = []
    hits = []
    for (rep, y_pred_posterior), ((x, signature), y_true) in zip(preds_all, data_x_all):
        y_pred = np.argmax(y_pred_posterior)
        if sum(signature) == 4:
            hits_full.append(y_pred == y_true)
        hits.append(y_pred == y_true)
        data_out.append(((rep, y_pred_posterior, signature), y_pred == y_true))
        print(signature, y_pred, y_true)

    print("Acc (4 artefacts):", format(np.average(hits_full), ".2%"))
    print("Acc (any):", format(np.average(hits), ".2%"))

    with open(args.output.format(args.target_output[0]), "wb") as f:
        pickle.dump(data_out, f)
