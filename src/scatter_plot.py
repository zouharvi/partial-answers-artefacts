import utils
import utils_data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import pickle
import operator as op

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input", default='data/final/COP.clean.json', type=str,
                        help="Path to the news dataset.")
    parser.add_argument("-e", "--embeddings", type=str,
                        help="Path to the embeddings file to plot.")
    parser.add_argument("-o", "--output", default='assFinal/figures/scatter.png', type=str,
                        help="Path where to store the figure.")
    parser.add_argument("-l","--label-key",default="newspaper")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
        
    # Read data
    data = utils.load_data(args.input)
    with open(args.embeddings,"rb") as f:
        embeddings = pickle.load(f)
    df = pd.DataFrame(embeddings["embeddings"],index=embeddings["ids"])
    
    data_ids = map(op.itemgetter("path"),data)
    df = df.loc[data_ids]
    
    print(data[0].keys())
    
    df["newspaper"] = list(map(op.itemgetter("newspaper"),data))
    df["newspaper_compas"] = list(map(utils_data.NEWSPAPER_TO_COMPAS.__getitem__,df["newspaper"]))

    plt.figure(figsize=(16,9))
    for label in df[args.label_key].unique():
        t = df[df[args.label_key] == label]
        
        plt.scatter(t[0],t[1],alpha=0.6,s=2,label=label)
    
    plt.legend(markerscale=4)
    plt.savefig(args.output)
    