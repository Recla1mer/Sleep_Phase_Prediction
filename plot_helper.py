"""
Author: Johannes Peter Knoll

Collection of functions for plotting the results of the project.
"""

# IMPORTS
import copy
import numpy as np
import pandas as pd

from sklearn import metrics

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


def plot_accuracy_per_epoch(
        paths_to_pkl_files: list,
        result_keys: list,
        **kwargs
    ):
    """
    Plot the accuracy/loss per epoch for the given keys and files.

    ATTENTION:  Assume that you have: paths_to_pkl_files = ["file1.pkl", "file2.pkl"] and
                result_keys = ["key1", "key2"]. To assign the labels correctly to the data,
                the order must be as follows: label = ["file1_key1", "file1_key2", "file2_key1", "file2_key2"]
    
    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    paths_to_pkl_files: list
        the paths to the pickle files containing the data
    result_keys: list
        the keys that access the data in the pickle files
    
    KEYWORD ARGUMENTS:
    ------------------------------
    figsize: list
        the size of the figure
    title: str
        the title of the plot
    xlabel: str
        the label of the x-axis
    ylabel: str
        the label of the y-axis
    label: list
        the labels of the data
    loc: str
        the location of the legend
    grid: bool
        whether to show the grid
    linewidth: float
        the width of the lines
    alpha: float
        the transparency of the lines
    linestyle: str
        the style of the lines
    marker: str
        the style of the markers
    markersize: float
        the size of the markers
    markeredgewidth: float
        the width of the marker edges
    markeredgecolor: str   
        the color of the marker edges
    ylim: list 
        the limits of the y-axis
    xlim: list
        the limits of the x-axis
    """
    
    # Default values
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "count")
    kwargs.setdefault("label", [])
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    labels = kwargs["label"]
    if len(kwargs["label"]) == 0:
        labels = ["" for _ in range(len(paths_to_pkl_files) * len(result_keys))]

    for i, path in enumerate(paths_to_pkl_files):
        data_generator = load_from_pickle(path)
        data = next(data_generator)
        for key in result_keys:
            ax.plot(
                key,
                data = data,
                label = labels[i],
                **plot_args
            )
    
    if len(kwargs["label"]) > 0:
        ax.legend(loc=kwargs["loc"])

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


def plot_distribution_of_score(
        paths_to_pkl_files: list,
        prediction_result_keys: list,
        actual_result_keys: list,
        score_function: callable, # type: ignore
        additional_function_args: dict = {},
        **kwargs
        ):
    """
    Calculate the score values using score_function(predicted_results, actual_results) and plot the
    distribution of the scores for the given keys and files.

    ATTENTION:  Assume that you have: paths_to_pkl_files = ["file1.pkl", "file2.pkl"] and 
                prediction_result_keys/actual_results_keys = ["key1", "key2"].
                To assign the labels correctly to the data, the order must be as follows:
                label = ["file1_key1", "file1_key2", "file2_key1", "file2_key2"]
    
    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    paths_to_pkl_files: list
        the paths to the pickle files containing the data
    prediction_result_keys: list
        the keys that access predicted results in the data (for example: "train_predicted_results", "test_predicted_results")
    actual_result_keys: list
        the keys that access actual results in the data (for example: "train_actual_results", "test_actual_results")
    score_function: callable
        the function that calculates the score (must take two arguments: predicted_results, actual_results)
    additional_function_args: dict
        additional arguments for the score function
    
    KEYWORD ARGUMENTS:
    ------------------------------
    figsize: list
        the size of the figure
    title: str
        the title of the plot
    xlabel: str
        the label of the x-axis
    ylabel: str
        the label of the y-axis
    label: list
        the labels of the data
    loc: str
        the location of the legend
    grid: bool
        whether to show the grid
    kde: bool
        whether to show the kernel density estimate
    binwidth: float
        the width of the bins
    edgecolor: str
        the color of the edges of the bins
    common_bins: bool
        whether to use the same bins for all data
    multiple: str
        how to display multiple data
    alpha: float
        the transparency of the data
    yscale: str
        the scale of the y-axis
    ylim: list
        the limits of the y-axis
    xlim: list
        the limits of the x-axis
    """

    # Default values
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "count")
    kwargs.setdefault("label", [])
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("kde", True)
    kwargs.setdefault("binwidth", 0.1)
    kwargs.setdefault("binrange", None)
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer")
    kwargs.setdefault("alpha", 0.5)
    
    kwargs.setdefault("yscale", "linear")

    hist_args = dict(
        kde = kwargs["kde"],
        binwidth = kwargs["binwidth"],
        binrange = kwargs["binrange"],
        edgecolor = kwargs["edgecolor"],
        common_bins = kwargs["common_bins"],
        multiple = kwargs["multiple"],
        alpha = kwargs["alpha"]
    )

    # Variables to store the score values
    score_values = []

    # iterate over files
    for path_index in range(len(paths_to_pkl_files)):
        # Load the data
        data_generator = load_from_pickle(paths_to_pkl_files[path_index])
        data = next(data_generator)

        # iterate over keys
        for key_index in range(len(prediction_result_keys)):
            this_keys_score_values = []
            predicted_results = data[prediction_result_keys[key_index]]
            actual_results = data[actual_result_keys[key_index]]

            # Calculate the score
            for i in range(len(predicted_results)):
                this_keys_score_values.append(score_function(actual_results[i], predicted_results[i], **additional_function_args))
            
            score_values.append(this_keys_score_values)
    
    # Create a dataframe
    dataframe = pd.DataFrame(score_values).T
    if len(kwargs["label"]) > 0:
        dataframe.columns = kwargs["label"]
    else:
        hist_args["legend"] = False

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])
    if len(kwargs["label"]) > 0:
        ax.legend(kwargs["label"], loc=kwargs["loc"])

    ax = sns.histplot(data = dataframe, **hist_args)
    ax.set_yscale(kwargs["yscale"])

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    plt.show()


def plot_confusion_matrix(
        path_to_pkl_file: str,
        prediction_result_key: str,
        actual_result_keys: str,
        display_labels: list = ["Wake", "LS", "DS", "REM"],
        all_known_labels = None,
        **kwargs
    ):
    """
    Plot the confusion matrix for the given data.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_pkl_file: str
        the path to the pickle file containing the data
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "test_predicted_results")
    actual_result_keys: str
        the key that accesses the actual results in the data (for example: "test_actual_results")
    display_labels: list
        the labels for the confusion matrix
    all_known_labels: list
        all known labels (if not all labels occur in the data)
        if None, labels are guessed from display_labels
    
    KEYWORD ARGUMENTS:
    ------------------------------
    figsize: list
        the size of the figure
    title: str
        the title of the plot
    xlabel: str
        the label of the x-axis
    ylabel: str
        the label of the y-axis
    cmap: str
        the color map
    values_format: str
        the format of the values
    colorbar: bool
        whether to show the color bar
    normalize: str
        how to normalize the values
    """

    # Default values
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "predicted stage")
    kwargs.setdefault("ylabel", "real stage")

    kwargs.setdefault("cmap", "Blues")
    kwargs.setdefault("values_format", ".1%")
    kwargs.setdefault("colorbar", False)
    kwargs.setdefault("normalize", "true")

    # Load the data
    data_generator = load_from_pickle(path_to_pkl_file)
    data = next(data_generator)

    # Get the predicted and actual results
    predicted_results = data[prediction_result_key]
    actual_results = data[actual_result_keys]

    # Flatten the arrays
    predicted_results = predicted_results.flatten()
    actual_results = actual_results.flatten()

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=kwargs["figsize"])

    try:
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_true = actual_results,
            y_pred = predicted_results,
            ax = ax,
            display_labels = display_labels,
            cmap = kwargs["cmap"],
            values_format = kwargs["values_format"],
            colorbar = kwargs["colorbar"],
            normalize = kwargs["normalize"]
            )
    except:
        # If not all labels occur in the data:
        if all_known_labels is None:
            all_known_labels = [i for i in range(len(display_labels))]
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_true = actual_results,
            y_pred = predicted_results,
            ax = ax,
            display_labels = display_labels,
            labels = [i for i in range(len(display_labels))],
            cmap = kwargs["cmap"],
            values_format = kwargs["values_format"],
            colorbar = kwargs["colorbar"],
            normalize = kwargs["normalize"]
            )
    
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])

    plt.show()


if __name__ == "__main__":

    """
    =============================
    Plot Accuracy/Loss per Epoch
    =============================
    """

    # plot_accuracy_per_epoch(
    #     paths_to_pkl_files = ["Model_Accuracy/Neural_Network.pkl"],
    #     result_keys = ["train_accuracy", "test_accuracy"],
    #     # label = ["train_accuracy", "test_accuracy"],
    #     title = "Accuracy of Neural Network",
    #     xlabel = "Epoch",
    #     ylabel = "Accuracy",
    # )

    """
    ================================================
    Plot Distribution for different Score Functions
    ================================================
    """

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Model_Accuracy/Neural_Network.pkl"],
    #     score_function = metrics.accuracy_score,
    #     prediction_result_keys = ["train_predicted_results", "test_predicted_results"],
    #     actual_result_keys = ["train_actual_results", "test_actual_results"],
    #     title = "Distribution of Accuracy",
    #     xlabel = "Accuracy",
    #     label = ["Train", "Test"],
    #     binrange = (0, 1)
    # )

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Model_Accuracy/Neural_Network.pkl"],
    #     score_function = metrics.cohen_kappa_score,
    #     prediction_result_keys = ["train_predicted_results", "test_predicted_results"],
    #     actual_result_keys = ["train_actual_results", "test_actual_results"],
    #     title = "Distribution of Kappa Score",
    #     xlabel = r"$\kappa$ Score",
    #     label = ["Train", "Test"],
    # )

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Model_Accuracy/Neural_Network.pkl"],
    #     score_function = metrics.f1_score,
    #     additional_function_args={"average": "macro"}, # or: None, 'micro', 'macro', 'weighted'
    #     prediction_result_keys = ["train_predicted_results", "test_predicted_results"],
    #     actual_result_keys = ["train_actual_results", "test_actual_results"],
    #     title = "Distribution of f1 Score",
    #     xlabel = "f1 Score",
    #     label = ["Train", "Test"],
    # )

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Model_Accuracy/Neural_Network.pkl"],
    #     score_function = metrics.precision_score,
    #     additional_function_args={"average": "micro"}, # or: None, 'micro', 'macro', 'weighted' ('binary', 'samples')
    #     prediction_result_keys = ["train_predicted_results", "test_predicted_results"],
    #     actual_result_keys = ["train_actual_results", "test_actual_results"],
    #     title = "Distribution of Precision",
    #     xlabel = "Precision",
    #     label = ["Train", "Test"],
    # )

    """
    ======================
    Plot Confusion Matrix
    ======================
    """

    plot_confusion_matrix(
        path_to_pkl_file = "Model_Accuracy/Yao_SHHS_GIF.pkl",
        prediction_result_key = "test_predicted_results",
        actual_result_keys = "test_actual_results",
        display_labels = ["Wake", "LS", "DS", "REM"],
        title = "Confusion Matrix of Neural Network",
    )