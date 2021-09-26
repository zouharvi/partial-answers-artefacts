#!/usr/bin/env python3

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
    parser.add_argument("--data-in", default="ass3/grid_search_results.csv")

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
    non_linear = df[df.param_kernel.notnull()]
    non_linear = non_linear.sort_values("param_C")
    
    Cs = non_linear.param_C.unique()
    
    #plt.figure(figsize=(6, 4.5))
    plt.figure(figsize=(4, 3),dpi=200)
    for kernel in non_linear.param_kernel.unique():
        df_kernel = non_linear[non_linear.param_kernel == kernel]
        
        if kernel == "poly":
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
            
    
            
    plt.xscale("log")
    plt.xticks(Cs,
               list(map("$2^{{{}}}$".format,np.asarray(np.log2(Cs),dtype=np.int32))))
    plt.legend(fontsize=8)
    plt.xlabel("$C$")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("C_perf_kernel.png")
    
def plot_linear_perf(df):
    linear_df = df[df.param_kernel.isnull()]
    linear_df = linear_df.sort_values("param_C")
    
    Cs = linear_df.param_C.unique()
    
    #plt.figure(figsize=(6, 4.5))
    plt.figure(figsize=(4, 3),dpi=200)
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
            
        
    plt.xscale("log")
    plt.xticks(Cs,
               list(map("$2^{{{}}}$".format,np.asarray(np.log2(Cs),dtype=np.int32))))
    plt.legend()
    plt.xlabel("$C$")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("C_perf_kernel_lin.png")


def box_times(df):
    df = df.sort_values("param_C")
    
    linear_df = df[df.param_kernel.isnull()]
    non_linear = df[df.param_kernel.notnull()]
    
    Cs = non_linear.param_C.unique()
    
    data = {}
    
    # Non linear
    for kernel in non_linear.param_kernel.unique():
        df_kernel = non_linear[non_linear.param_kernel == kernel]
        
        if kernel == "poly":
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
    
    
    #plt.figure(figsize=(8, 4))
    plt.figure(figsize=(7, 3),dpi=200)
    plt.boxplot(data.values(),labels=data.keys())
    plt.vlines(7.5,0,30)
    
    plt.ylabel("Train time (sec)")
    plt.text(6,15,"SVC",horizontalalignment="center",fontsize=16)
    plt.text(9,15,"LinearSVC",horizontalalignment="center",fontsize=16)
    plt.tight_layout()
    plt.savefig("bplot_speed.png")
    
if __name__ == "__main__":
    args = parse_args()

    # Read csv file
    df = pd.read_csv(args.data_in)
    
    # Plot
    plot_non_linear_perf(df)
    plot_linear_perf(df)
    box_times(df)