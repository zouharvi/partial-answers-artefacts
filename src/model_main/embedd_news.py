#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
from lm_model import LMModel
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json',
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/embeddings/embeddings_{m}_{t}_{ml}.pkl',
                        help="Path where to store the embeddings.")
    parser.add_argument("-t", "--target", default='headline',
                        help="Target field of the new to use for embedding.")
    parser.add_argument("-bs", "--batch-size", default=128, type=int,
                        help="Override the default batch size.")
    parser.add_argument("-lm", "--language-model", default="bert",
                        help="Name of pretrained language model to use for the embeddings.")
    parser.add_argument("-st", "--embed-strategy", default="avg",
                        help="Which embedding strategy from the model to use (cls, avg, all)")
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
        ml=args.max_length
    )

    # Read data
    data = utils.load_data(args.input)

    print("Number of articles: ", len(data))
    target = [x[args.target] for x, y in data]

    # Instantiate model
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    lm = LMModel(
        embed_strategy=args.embed_strategy,
        lm=lm_name,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    embeddings = lm.predict(target)

    print("Output dim:", embeddings[0].shape)
    with open(output_name, "wb") as f:
        pickle.dump(embeddings, f)
