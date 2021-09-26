#!/usr/bin/env python3

"""
Vizualization of grid search results. Performance depending on kernel
and regularization coefficient, and train time distributions.
"""

# Data manipulation
import numpy as np
import pandas as pd

# Visualization libs
import matplotlib.pyplot as plt

# Misc
import argparse
from argparse import Namespace

def parse_args() -> Namespace:
    """
    Returns the parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--data-in", default="ass3/grid_search_results2.csv")

    args = parser.parse_args()
    return args


COLORS_PLOT = dict(
    linear="tab:blue",
    rbf="tab:orange",
    sigmoid="tab:red",
    poly_2=plt.get_cmap("Greens")(0.6),
    poly_3=plt.get_cmap("Greens")(0.7),
    poly_4=plt.get_cmap("Greens")(0.8),
    poly_5=plt.get_cmap("Greens")(0.9)) 

def plot_non_linear_perf(df):
    """
    Plots the C vs accuracy graph for non-linear kernels.
    """
    
    non_linear = df[df.param_kernel.notnull()]
    non_linear = non_linear.sort_values("param_C")
    
    Cs = non_linear.param_C.unique()
    
    # Define plot dimensions
    plt.figure(figsize=(4, 3),dpi=200)
    
    # Plot graphs
    for kernel in non_linear.param_kernel.unique():
        df_kernel = non_linear[non_linear.param_kernel == kernel]
        
        if kernel == "poly": # There is more than one polynomial kernel
            for degree in df_kernel.param_degree.unique():
                df_kernel_degree = df_kernel[df_kernel.param_degree == degree]
                
                p_label = "poly_{}".format(int(degree))
                
                plt.plot(
                    Cs,
                    df_kernel_degree.mean_test_score,
                    marker=".",
                    markeredgecolor="black",
                    alpha=0.9,
                    label=p_label,
                    color=COLORS_PLOT[p_label])
                
        else:
            plt.plot(
                Cs,
                df_kernel.mean_test_score,
                marker=".",
                markeredgecolor="black",
                alpha=0.9,
                label=kernel,
                color=COLORS_PLOT[kernel])
            
    
    # Finishing plot annotations
    plt.xscale("log")
    plt.xticks(Cs,
               list(map("$2^{{{}}}$".format,np.asarray(np.log2(Cs),dtype=np.int32))))
    plt.legend(fontsize=8)
    plt.xlabel("$C$")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    
    # Save plot
    plt.savefig("C_perf_kernel.png")
    
def plot_linear_perf(df):
    """
    Plots the C vs accuracy graph for the linear kernel with different
    loss and regularizaton params.
    """
    linear_df = df[df.param_kernel.isnull()]
    linear_df = linear_df.sort_values("param_C")
    
    Cs = linear_df.param_C.unique()
    
    # Define plot dimensions
    plt.figure(figsize=(4, 3),dpi=200)
    
    # Plot graphs
    plt.plot( # L1 squared_hinge
        Cs,
        linear_df[linear_df.param_penalty == "l1"].mean_test_score,
        marker=".",
        markeredgecolor="black",
        alpha=0.9,
        label="L1 + squared hinge",
        color=plt.get_cmap("Blues")(0.9))
    
    plt.plot( # L2 hinge
        Cs,
        linear_df[(linear_df.param_penalty == "l2") & (linear_df.param_loss == "hinge")].mean_test_score,
        marker=".",
        markeredgecolor="black",
        alpha=0.9,
        label="L2 + hinge",
        color=plt.get_cmap("Blues")(0.7))
    
    plt.plot( # L2 squared_hinge
        Cs,
        linear_df[(linear_df.param_penalty == "l2") & (linear_df.param_loss == "squared_hinge")].mean_test_score,
        marker=".",
        markeredgecolor="black",
        alpha=0.9,
        label="L2 + squared hinge",
        color=plt.get_cmap("Blues")(0.5))
            
    # Finishing plot annotations
    plt.xscale("log")
    plt.xticks(Cs,
               list(map("$2^{{{}}}$".format,np.asarray(np.log2(Cs),dtype=np.int32))))
    plt.legend()
    plt.xlabel("$C$")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    
    # Save plot
    plt.savefig("C_perf_kernel_lin.png")


def box_times(df):
    """
    Plots a boxplot with the train time distributions of each SVM configuration.
    """
    
    df = df.sort_values("param_C")
    
    linear_df = df[df.param_kernel.isnull()]
    non_linear = df[df.param_kernel.notnull()]
    
    Cs = non_linear.param_C.unique()
    
    # Gather the distribution for each configuration
    data = {}
    # Non linear
    for kernel in non_linear.param_kernel.unique():
        df_kernel = non_linear[non_linear.param_kernel == kernel]
        
        if kernel == "poly": # More than one polynomial kernel
            for degree in df_kernel.param_degree.unique():
                df_kernel_degree = df_kernel[df_kernel.param_degree == degree]
                
                p_label = "poly_{}".format(int(degree))
                
                data[p_label] = df_kernel_degree.mean_fit_time
                
        else:
            data[kernel] = df_kernel.mean_fit_time
            
    # Linear
    data["L1+sh"] = linear_df[linear_df.param_penalty == "l1"].mean_fit_time
    data["L2+h"] = linear_df[(linear_df.param_penalty == "l2") & (linear_df.param_loss == "hinge")].mean_fit_time
    data["L2+sh"] = linear_df[(linear_df.param_penalty == "l2") & (linear_df.param_loss == "squared_hinge")].mean_fit_time
    
    # Make the plot
    plt.figure(figsize=(7, 3),dpi=200)
    plt.boxplot(data.values(),labels=data.keys())
    
    # Make annotations
    plt.vlines(6.5,0,30) # Separate SVM and LinearSVM
    plt.ylabel("Train time (sec)")
    plt.text(5,15,"SVC",horizontalalignment="center",fontsize=16)
    plt.text(8,15,"LinearSVC",horizontalalignment="center",fontsize=16)
    plt.tight_layout()
    
    # Save plot
    plt.savefig("bplot_speed.png")
    
    
def rename_columns(df):
    """
    Rename the columns that correspond to GS parameters to save on characters later.
    """
    
    if "param_cls__kernel" in df.columns: df["param_kernel"] = df.param_cls__kernel
    if "param_cls__C" in df.columns: df["param_C"] = df.param_cls__C
    if "param_cls__degree" in df.columns: df["param_degree"] = df.param_cls__degree
    if "param_cls__penalty" in df.columns: df["param_penalty"] = df.param_cls__penalty
    if "param_cls__loss" in df.columns: df["param_loss"] = df.param_cls__loss
    
if __name__ == "__main__":
    args = parse_args()

    # Read csv file
    df = pd.read_csv(args.data_in)
    
    # Get SVC rows
    df = df[df.mean_test_score.notnull()]
    
    # Rename the columns of the dataframe
    rename_columns(df)
    
    # Plot
    plot_non_linear_perf(df)
    plot_linear_perf(df)
    box_times(df)