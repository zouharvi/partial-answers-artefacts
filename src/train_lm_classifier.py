#!/usr/bin/env python3

# TODO segregate into fucntions

from lm_model import LMModel
import utils

import sklearn.model_selection
import numpy as np

import argparse
import operator as op
import itertools as it

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/models/{m}_{ml}_{ti}_{to}.pt', type=str,
                        help="Path where to store the model.")
    parser.add_argument("-ti", "--target-input", default='headline',type=str,
                        help="Input of the model.")
    parser.add_argument("-to", "--target-output", default=['newspaper'], type=str, nargs="+",
                        help="Target output of the model")
    parser.add_argument("-ts", "--train-samples", default=-2000, type=int, nargs="+",
                        help="Target output of the model")
    parser.add_argument("-ds", "--dev-samples", default=1000, type=int, nargs="+",
                        help="Target output of the model")
    parser.add_argument("-bs","--batch-size", default=16, type=int,
                        help="Override the default batch size.")
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

if __name__ == "__main__":
    args = parse_args()
    
    # Format output name
    output_name = args.output.format(
        m=args.language_model,
        ti=args.target_input,
        to="_".join(args.target_output),
        ml=args.max_length
        )
    
    # Read data
    data = utils.load_data(args.input)
    print("Number of articles: ", len(data))
    
    target_input = utils.get_x(data,args.target_input)
    target_outputs, label_names, labels = utils.get_y(data,args.target_output)    
    
    # Split
    test_size = abs(args.train_samples + args.dev_samples) % len(data)
    data_train, _ = sklearn.model_selection.train_test_split(
        list(zip(target_input,labels)),
        test_size=test_size,
        random_state=0,
    )
    
    data_train, data_dev = sklearn.model_selection.train_test_split(
        data_train,
        test_size=args.dev_samples,
        random_state=0,
    )

    x_train, y_train = zip(*data_train)
    x_dev, y_dev = zip(*data_dev)
    
    ## Instantiate transformer
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    lm = LMModel(
        cls_target_dimensions=list(map(len,label_names)),
        lm=lm_name,
        batch_size=args.batch_size,
        max_length=args.max_length)
    
    lm.fit(x_train,y_train,x_dev,y_dev)
    lm.save_to_file(output_name)
    