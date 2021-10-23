#!/usr/bin/env python3

# TODO segregate into fucntions

from lm_model import LMModel
import utils
import utils_eval

import sklearn.model_selection
import numpy as np

import random
import os.path as path
import json
import argparse
import operator as op
import itertools as it
import collections as col

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path of the data file.")
    parser.add_argument("-o", "--output", default='data/models/{m}_{ml}_{ti}_{to}_{inp}.pt', type=str,
                        help="Path where to store the model.")
    parser.add_argument("-ti", "--target-input", default='body',type=str,
                        help="Input of the model.")
    parser.add_argument("-to", "--target-output", default=['newspaper'], type=str, nargs="+",
                        help="Target output of the model")
    parser.add_argument("-ts", "--train-samples", default=-2000, type=int,
                        help="Number of samples for the training (may be negative to get all except a given number)")
    parser.add_argument("-ds", "--dev-samples", default=1000, type=int,
                        help="Number of samples for the validation.")
    parser.add_argument("-ep","--epochs", default=2, type=int,
                        help="Override the default number of epochs.")
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
        ml=args.max_length,
        inp=path.basename(args.input)[:-5]
        )
    
    # Read data
    data = utils.load_data(args.input)
    
    targets = args.target_output
    if len(targets) == 1:
        if targets[0] == "all":
            targets = list(utils.Y_KEYS)
        elif targets[0] == "craft":
            code = path.basename(args.input)[-6]
            targets = [utils.CODE_TO_Y_KEYS[code]]
       
    
    target_input = utils.get_x(data,args.target_input)
    target_outputs, label_names, labels = utils.get_y(data,targets)    
    
    train_size = (len(data) + args.train_samples) % len(data)
    dev_size = args.dev_samples
    test_size = len(data) - train_size - dev_size
    (x_test, y_test), (x_dev, y_dev), (x_train, y_train) = utils.make_split(
                                                (target_input,labels),
                                                splits=(test_size,dev_size,),
                                                random_state=0)
    
    print(label_names)
    ## Instantiate transformer
    lm_name = LM_ALIASES[args.language_model] if args.language_model in LM_ALIASES else args.language_model
    
    dimensions = list(map(len,label_names))
    count_targets = col.Counter(target_outputs)
    weights = 1 / np.array(list(map(count_targets.__getitem__,target_outputs)))
    
    lm = LMModel(
        cls_target_dimensions=dimensions,
        loss_weights=weights,
        lm=lm_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length)
    
    lm.fit(x_train,y_train,x_dev,y_dev)
    lm.save_to_file(output_name)
    
    ## TODO put these in eval script
    ## Evaluations
    evals = dict()
    
    # Development eval
    y_pred_dev = lm.predict(x_dev)
    evals["dev"] = utils_eval.complete_evaluation(target_outputs,
                                   y_dev,y_pred_dev,
                                   target_names=label_names)
    
    # Test eval
    y_pred_test = lm.predict(x_test)
    evals["test"] = utils_eval.complete_evaluation(target_outputs,
                                   y_test,y_pred_test,
                                   target_names=label_names)
    
    
    save_path = "data/eval/{}_eval.json".format(path.basename(output_name)[:-3])
    with open(save_path,"w") as f:
        json.dump(evals,f)
        