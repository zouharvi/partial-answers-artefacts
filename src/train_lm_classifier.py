#!/usr/bin/env python3

# TODO segregate into fucntions

from lm_model import LMModel
import utils

import numpy as np

import argparse
import operator as op

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/models/{m}_{ml}_{ti}_{to}.pt', type=str,
                        help="Path where to store the model.")
    parser.add_argument("-ti", "--target-input", default='headline',type=str,
                        help="Input of the model.")
    # TODO: does this really work if I pass this via the command line?
    parser.add_argument("-to", "--target-output", default=['newspaper'], type=str,
                        help="Target output of the model")
    parser.add_argument("-bs","--batch-size", default=16, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model to use for the embeddings.")
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
        to=args.target_output,
        ml=args.max_length
        )
    
    # Read data
    data = utils.load_data(args.input)
    data_x, data_y = zip(*data)
    
    print("Number of articles: ", len(data_y))
    
    target_input = list(map(op.itemgetter(args.target_input),data_x))
    
    # idk how to rewrite this to functional format sorry
    target_outputs = {to:[[x[to]] for x in data_y] for to in args.target_output}
    target_outputs = {k:utils.binarize_data(v)[1] for k,v in target_outputs.items()}
    # flatten, this makes "newspaper" first index
    # I'd advise to keep using dictionary to make it easier to navigate though
    target_outputs = list(target_outputs.values())
    # idk what this does?
    # target_outputs = list(map(op.itemgetter(-1),target_outputs))
    print(target_outputs[0])
    # sanity check
    assert all(
        [sum(x) == 1 for x in target_outputs[0]]
    )
    
    targets = map(op.itemgetter(0),target_outputs)
    targets = list(map(len,targets))
    
    ## Instantiate transformer
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    lm = LMModel(
        classification_targets=targets,
        lm=lm_name,
        batch_size=args.batch_size,
        max_length=args.max_length)
    
    lm.fit(target_input,target_outputs)
    