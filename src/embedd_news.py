#!/usr/bin/env python3

import utils
from lm_model import LMModel

import numpy as np

import argparse
import operator as op

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/embeddings/embeddings_{m}_{t}_{ml}.npz', type=str,
                        help="Path where to store the embeddings.")
    parser.add_argument("-t", "--target", default='headline', type=str,
                        help="Target field of the new to use for embedding.")
    parser.add_argument("-bs","--batch-size", default=128, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lm", "--language-model", default="bert", type=str,
                        help="Name of pretrained language model to use for the embeddings.")
    parser.add_argument("--max-length", default=128, type=int,
                        help="Override the default maximum length of language model input.")
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
        t=args.target,
        ml=args.max_length)
    
    # Read data
    data = utils.load_data(args.input)
    data, _ = zip(*data)
    
    print("Number of articles: ", len(data))
    target = list(map(op.itemgetter(args.target),data))
    
    ## Instantiate model
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    lm = LMModel(
            embed_strategy="all",
            lm=lm_name, 
            batch_size=args.batch_size,
            max_length=args.max_length)
    
    embeddings = lm.predict(target)
        
    np.savez_compressed(output_name,data=embeddings)