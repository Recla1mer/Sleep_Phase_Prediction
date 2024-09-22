"""
Author: Johannes Peter Knoll

Collection of functions for plotting the results of the project.
"""

# IMPORTS
import copy
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

# LOCAL IMPORTS
from dataset_processing import load_from_pickle

"""
----------------------
Guide To bitsandbobs:
----------------------
(Using it primarily for the color cycle and the default colors)
Source: https://github.com/pSpitzner/bitsandbobs

In Console:
conda install numpy matplotlib h5py
python -m pip install git+https://github.com/pSpitzner/bitsandbobs
"""
import bitsandbobs as bnb

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler( # type: ignore
    "color", bnb.plt.get_default_colors()
) 
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 2  # padding between text and the tick
matplotlib.rcParams["ytick.major.pad"] = 2  # default 3.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)  # transparent figure bg
matplotlib.rcParams["axes.facecolor"] = (1.0, 0.0, 0.0, 0.0)



def plot_accuracy_results(
        paths_to_pkl_files: list,
        results_key: str,
        labels: list,
        figsize: list = [3.4, 2.7],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        **kwargs
    ):
    """
    """
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    # kwargs.setdefault("marker", "o") # or "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    # kwargs.setdefault("markersize", 4)
    # kwargs.setdefault("markeredgewidth", 1)
    # kwargs.setdefault("markeredgecolor", "white")
    # kwargs.setdefault("markerfacecolor", "white")
    kwargs.setdefault("zorder", 3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i, path in enumerate(paths_to_pkl_files):
        data_generator = load_from_pickle(path)
        print(next(load_from_pickle(path)))
        ax.plot(
            results_key,
            data = next(data_generator),
            label = labels[i],
        )
    
    ax.legend(labels)
    plt.show()


if __name__ == "__main__":
    accuracy_paths = ["Accuracy/Neural_Network.pkl"]
    plot_accuracy_results(
        paths_to_pkl_files = accuracy_paths,
        results_key = "train_accuracy",
        labels = ["Neural Network"],
        title = "Accuracy of Neural Network",
        xlabel = "Epoch",
        ylabel = "Training Accuracy",
    )