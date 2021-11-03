#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
import utils.eval
import json
import argparse
import os.path as path
from lm_model import LMModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/eval/{}_eval.json', type=str,
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


if __name__ == "__main__":
    args = parse_args()

    # Format output name
    output_name = args.output.format(path.basename(args.model_path)[:-3])

    # Read data
    data = utils.load_data(args.input)

    target_input = utils.get_x(data, args.target_input)
    target_outputs, label_names, labels = utils.get_y(data, args.target_output)

    (x_test, y_test), _ = utils.make_split(
        (target_input, labels),
        splits=(args.test_samples,),
        random_state=0
    )
    # Instantiate model
    lm = LMModel(
        cls_target_dimensions=list(map(len, label_names)),
        lm=args.language_model,
        head_thickness=args.head_thickness,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    lm.load_from_file(args.model_path)
    y_logits = lm.predict(x_test)

    evaluation = utils.eval.complete_evaluation(
        y_true=y_test, y_logits=y_logits,
        evaluation_targets=target_outputs,
        target_names=label_names
    )
    print(utils.pretty_json(evaluation))

    with open(output_name, "w") as f:
        json.dump(evaluation, f)
