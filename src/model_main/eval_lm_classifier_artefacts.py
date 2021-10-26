#!/usr/bin/env python3

import pickle
import sys

import numpy as np
sys.path.append("src")
import utils
import utils_eval
import json
import argparse
import os.path as path
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
    parser.add_argument("-to", "--target-output", default=['newspaper'], type=str, nargs="+",
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

def x_manipulator_base(x, y, x_filter, y_filter):
    artefacts = [
        "None"
        for k in utils.Y_KEYS_LOCAL
    ]
    return ' | '.join(artefacts) + " | " + x[x_filter]


def x_manipulator_all(x, y, x_filter, y_filter):
    y = {**y, y_filter: "None"}
    artefacts = [
        y[k]
        for k in utils.Y_KEYS_LOCAL
    ]
    return ' | '.join(artefacts) + " | " + x[x_filter]


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
    print("Predicting base")
    data_x_base = [x_manipulator_base(x, y, args.target_input, args.target_output[0]) for x, y in data[:10]]
    preds_base = lm.predict(data_x_base)[0]
    print("Predicting all artefacts")
    data_x_all = [x_manipulator_all(x, y, args.target_input, args.target_output[0]) for x, y in data[:10]]
    preds_all = lm.predict(data_x_all)[0]

    data_out = []

    for pred_base, pred_all, gold_label in zip(preds_base, preds_all, labels):
        y_base = np.argmax(pred_base)
        y_all = np.argmax(pred_all)
        gold_label = gold_label[0]
        data_out.append({
            "base": (pred_base, y_base==gold_label),
            "all": (pred_all, y_all==gold_label)
        })    
        print(y_base, gold_label)

    with open(args.output.format(args.target_output[0]), "wb") as f:
        pickle.dump(data_out, f)
