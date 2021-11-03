#!/usr/bin/env python3

"""
Mock results for fusion tracing (presentation purposes only)
"""

import argparse
import sys
sys.path.append("src")
import matplotlib.pyplot as plt


def parse_args():
    args = argparse.ArgumentParser()
    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.figure(figsize=(6, 4))

    plt.plot(
        [0.5, 0.7, 0.6, 0.4, 0.1],
        marker=".",
        linestyle="-",
        color="tab:green",
        label=r"pred. $\checkmark$ $\rightarrow_{\xi \checkmark}$ pred. $\checkmark$",
    )
    plt.plot(
        [0.6, 0.9, 1.3, 1.4, 0.75],
        marker=".",
        linestyle=":",
        color="tab:green",
        label=r"pred. $\times$ $\rightarrow_{\xi \checkmark}$ pred. $\checkmark$",
    )
    plt.plot(
        [0.65, 0.9, 1.35, 1.5, 0.9],
        marker=".",
        linestyle="-",
        color="tab:red",
        label=r"pred. $\checkmark$ $\rightarrow_{\xi {\times}}$ pred. $\times$",
    )
    plt.plot(
        [0.58, 1.0, 1.2, 1.2, 0.4],
        marker=".",
        linestyle=":",
        color="tab:red",
        label=r"pred. $\times$ $\rightarrow_{\xi \checkmark}$ pred. $\times$",
    )

    plt.ylabel("$L^2$ distance")
    plt.ylim(0, 2)
    XTICK_LABELS = [
        "Input", "Layer 1\n(avg)", "Layer 2\n(avg)", "Layer 3\n(avg)", "Softmax Input"]
    plt.xticks(list(range(len(XTICK_LABELS))), XTICK_LABELS, rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.show()
