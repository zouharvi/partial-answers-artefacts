#!/usr/bin/env python3

import utils
import utils_data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse
import operator as op
import os.path as path

def add_plot_options(parser: argparse.ArgumentParser):
    # Plot options
    p_group = parser.add_argument_group("Plot","Arguments to control the details of the plot.")
    p_group.add_argument("-fs", "--figsize", nargs=2, default=[16,9], type=float, metavar=("WIDTH","HEIGHT"),
                        help="Dimensions of output figure.")
    p_group.add_argument("-t", "--title", type=str,
                        help="Title of the figure.")
    p_group.add_argument("-xl", "--xlabel", type=str,
                        help="X-axis label of the figure.")
    p_group.add_argument("-yl", "--ylabel", type=str,
                        help="Y-axis label of the figure.")
    
    # Legend options
    l_group = parser.add_argument_group("Legend","Arguments to control the details of the legend.")
    l_group.add_argument("--legend-markerscale", default=4, type=float,
                        help="Size of marker in legend.")
    
    # Scatter options
    s_group = parser.add_argument_group("Scatter","Arguments to control the details of the scatter plot.")
    s_group.add_argument("--scatter-alpha",default=0.6,type=float,
                       help="Alpha of the dots of the scatterplot.")
    s_group.add_argument("--scatter-s",default=2,type=float,
                       help="Size of the dots of the scatterplot.")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input", default='data/final/clean.json', type=str,
                        help="Path to the news dataset.")
    parser.add_argument("-e", "--embeddings", type=str, required=True,
                        help="Path to the embeddings file to plot.")
    parser.add_argument("-o", "--output", default='data/figures/scatter_{e}_label_{l}.png', type=str,
                        help="Path where to store the figure.")
    parser.add_argument("-l","--label-key",default="newspaper")
    
    add_plot_options(parser)
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
        
    output_name = args.output.format(
        e=path.basename(args.embeddings[:-4]),
        l=args.label_key)
        
    # Read data
    data = utils.load_data(args.input)
    _, data = zip(*data)
    
    embeddings = np.load(args.embeddings)["data"]
    
    df = pd.DataFrame(embeddings)
    
    df["newspaper"] = list(map(op.itemgetter("newspaper"),data))
    df["newspaper_compas"] = list(map(utils_data.NEWSPAPER_TO_COMPAS.__getitem__,df["newspaper"]))

    plt.figure(figsize=args.figsize)
    for label in df[args.label_key].unique():
        t = df[df[args.label_key] == label]
        
        sc = plt.scatter(t[0],t[1],
                         alpha=args.scatter_alpha,
                         s=args.scatter_s,
                         label=label)
    
    lg = plt.legend(markerscale=args.legend_markerscale)
    for h in lg.legendHandles:
        h.set_alpha(1)
    
    if "title" in args: plt.title(args.title)
    if "xlabel" in args: plt.xlabel(args.xlabel)
    if "ylabel" in args: plt.ylabel(args.ylabel)
    
    plt.savefig(output_name)
    