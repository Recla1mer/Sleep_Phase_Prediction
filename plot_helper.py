"""
Author: Johannes Peter Knoll

Collection of functions for plotting the results of the project.
"""

# IMPORTS
import copy
import numpy as np
import pandas as pd
import pickle

from sklearn import metrics

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

# LOCAL IMPORTS
from dataset_processing import load_from_pickle # type: ignore
from main import loss_per_epoch_shhs_file, loss_per_epoch_gif_file, project_configuration_file, model_performance_file

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


def accuracy_from_confusion_matrix(
        confusion_matrix: np.ndarray,
    ):
    """
    # return macro accuracy from confusion matrix
    """
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()


def precision_from_confusion_matrix(
        confusion_matrix: np.ndarray,
    ):
    """
    # return macro precision from confusion matrix
    """
    precision = []
    for i in range(len(confusion_matrix)):
        true_positive = confusion_matrix[i][i]
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        if true_positive + false_positive == 0:
            precision.append(0.0)
        else:
            precision.append(true_positive / (true_positive + false_positive))
    return np.mean(precision)


def recall_from_confusion_matrix(
        confusion_matrix: np.ndarray,
    ):
    """
    # return macro recall from confusion matrix
    """
    recall = []
    for i in range(len(confusion_matrix)):
        true_positive = confusion_matrix[i][i]
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        if true_positive + false_negative == 0:
            recall.append(0.0)
        else:
            recall.append(true_positive / (true_positive + false_negative))
    return np.mean(recall)


def f1_score_from_confusion_matrix(
        confusion_matrix: np.ndarray,
    ):
    """
    # return macro F1 score from confusion matrix
    """
    precision = precision_from_confusion_matrix(confusion_matrix)
    recall = recall_from_confusion_matrix(confusion_matrix)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def plot_performance_per_epoch(
        model_directory_path: str,
        use_metric: str = "loss", # "accuracy", "f1_score", "precision", "recall", "loss"
        **kwargs
    ):
    """
    Plot the accuracy/loss per epoch for the given keys and files.

    ATTENTION:  You can either provide multiple files and one result key or one file and multiple result keys.
    
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
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "count")
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

    # load data
    shhs_file = model_directory_path + loss_per_epoch_shhs_file

    with open(shhs_file, "rb") as f:
        results = pickle.load(f)
    
    shhs_train_avg_loss = results["train_avg_loss"]
    shhs_train_confusion_matrices = results["train_confusion_matrix"]
    shhs_test_shhs_loss = results["SHHS_avg_loss"]
    shhs_test_shhs_confusion_matrices = results["SHHS_confusion_matrix"]
    shhs_test_gif_loss = results["GIF_avg_loss"]
    shhs_test_gif_confusion_matrices = results["GIF_confusion_matrix"]
    
    gif_file = model_directory_path + loss_per_epoch_gif_file

    with open(gif_file, "rb") as f:
        results = pickle.load(f)

    gif_train_avg_loss = results["train_avg_loss"]
    gif_train_confusion_matrices = results["train_confusion_matrix"]
    gif_test_shhs_loss = results["SHHS_avg_loss"]
    gif_test_shhs_confusion_matrices = results["SHHS_confusion_matrix"]
    gif_test_gif_loss = results["GIF_avg_loss"]
    gif_test_gif_confusion_matrices = results["GIF_confusion_matrix"]

    if use_metric == "accuracy":
        train_performance = [accuracy_from_confusion_matrix(cm) for cm in shhs_train_confusion_matrices] + [accuracy_from_confusion_matrix(cm) for cm in gif_train_confusion_matrices]
        test_shhs_performance = [accuracy_from_confusion_matrix(cm) for cm in shhs_test_shhs_confusion_matrices] + [accuracy_from_confusion_matrix(cm) for cm in gif_test_shhs_confusion_matrices]
        test_gif_performance = [accuracy_from_confusion_matrix(cm) for cm in shhs_test_gif_confusion_matrices] + [accuracy_from_confusion_matrix(cm) for cm in gif_test_gif_confusion_matrices]
    elif use_metric == "f1_score":
        train_performance = [f1_score_from_confusion_matrix(cm) for cm in shhs_train_confusion_matrices] + [f1_score_from_confusion_matrix(cm) for cm in gif_train_confusion_matrices]
        test_shhs_performance = [f1_score_from_confusion_matrix(cm) for cm in shhs_test_shhs_confusion_matrices] + [f1_score_from_confusion_matrix(cm) for cm in gif_test_shhs_confusion_matrices]
        test_gif_performance = [f1_score_from_confusion_matrix(cm) for cm in shhs_test_gif_confusion_matrices] + [f1_score_from_confusion_matrix(cm) for cm in gif_test_gif_confusion_matrices]
    elif use_metric == "precision":
        train_performance = [precision_from_confusion_matrix(cm) for cm in shhs_train_confusion_matrices] + [precision_from_confusion_matrix(cm) for cm in gif_train_confusion_matrices]
        test_shhs_performance = [precision_from_confusion_matrix(cm) for cm in shhs_test_shhs_confusion_matrices] + [precision_from_confusion_matrix(cm) for cm in gif_test_shhs_confusion_matrices]
        test_gif_performance = [precision_from_confusion_matrix(cm) for cm in shhs_test_gif_confusion_matrices] + [precision_from_confusion_matrix(cm) for cm in gif_test_gif_confusion_matrices]
    elif use_metric == "recall":
        train_performance = [recall_from_confusion_matrix(cm) for cm in shhs_train_confusion_matrices] + [recall_from_confusion_matrix(cm) for cm in gif_train_confusion_matrices]
        test_shhs_performance = [recall_from_confusion_matrix(cm) for cm in shhs_test_shhs_confusion_matrices] + [recall_from_confusion_matrix(cm) for cm in gif_test_shhs_confusion_matrices]
        test_gif_performance = [recall_from_confusion_matrix(cm) for cm in shhs_test_gif_confusion_matrices] + [recall_from_confusion_matrix(cm) for cm in gif_test_gif_confusion_matrices]
    elif use_metric == "loss":
        train_performance = shhs_train_avg_loss + gif_train_avg_loss
        test_shhs_performance = shhs_test_shhs_loss + gif_test_shhs_loss
        test_gif_performance = shhs_test_gif_loss + gif_test_gif_loss

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    # epochs = [i for i in range(1, len(shhs_train_avg_loss) + 1)] + [i for i in range(1, len(gif_train_avg_loss) + 1)]# type: ignore
    epochs = [i for i in range(1, len(train_performance) + 1)] # type: ignore
    ax.plot(
        epochs,
        train_performance,
        label = "Train",
        **plot_args
    )
    ax.plot(
        epochs,
        test_shhs_performance,
        label = "Test SHHS",
        **plot_args
    )
    ax.plot(
        epochs,
        test_gif_performance,
        label = "Test GIF",
        **plot_args
    )
    
    ax.legend(loc=kwargs["loc"])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


def plot_distribution_of_score(
        paths_to_pkl_files: list,
        path_to_project_configuration: str,
        prediction_result_key: str,
        actual_result_key: str,
        score_function: callable, # type: ignore
        additional_score_function_args: dict = {},
        combine_file_predictions: bool = False,
        **kwargs
    ):
    """
    Calculate the score values using score_function(predicted_results, actual_results) and plot the
    distribution of the scores for the given keys and files.

    If the score_function returns multiple values for each sleep stage, the resulting plot will show the 
    distribution of the sleep stages and not the files. In this case, you can only provide one file.
    
    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    paths_to_pkl_files: list
        the paths to the pickle files containing the data
    path_to_project_configuration: str
        the path to all signal processing parameters
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "Predicted" or "Predicted_in_windows")
    actual_result_key: str
        the key that accesses the actual results in the data (for example: "Actual" or "Actual_in_windows")
    score_function: callable
        the function that calculates the score (must take two arguments: predicted_results, actual_results)
    additional_score_function_args: dict
        additional arguments for some of the score functions (precision_score, recall_score, f1_score), e.g.:
            - average: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
                average parameter
            - zero_division: {"warn", 0.0, 1.0, np.nan}
                zero division parameter
    combine_file_predictions: bool
        whether to combine the predictions of all files into one distribution
    
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
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
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

    # Interrupt if user wants to overkill the plot
    if "average" in additional_score_function_args:
        if additional_score_function_args["average"] is None and len(paths_to_pkl_files) > 1:
            raise ValueError("Your current setting would lead to number_sleep_stages * number_files different scores. This is overkill. Either change 'average' in the 'additional_score_function_args' parameter to None or use only one file.")
        
        # load signal processing parameters
        with open(path_to_project_configuration, "rb") as f:
            project_configuration = pickle.load(f)
        
        # access dictionary that maps sleep stages (display labels) to integers
        sleep_stage_to_label = project_configuration["sleep_stage_label"]

        # Create a list of the integer labels, sorted
        integer_labels = np.array([value for value in sleep_stage_to_label.values()])
        integer_labels = np.unique(integer_labels)
        integer_labels.sort()

        # Create a list of the display labels
        display_labels = []
        for integer_label in integer_labels:
            for key, value in sleep_stage_to_label.items():
                if value == integer_label:
                    display_labels.append(key)
                    break
        
        # append labels to additional_score_function_args
        additional_score_function_args["labels"] = integer_labels

        if additional_score_function_args["average"] is None:
            kwargs["label"] = display_labels

    # Variables to store the score values
    score_values = []

    for file_path in paths_to_pkl_files:
        # Variables to store this files score values
        this_score_values = []

        # Load the data
        data_generator = load_from_pickle(file_path)
        
        for data in data_generator:

            # Get the predicted and actual results
            predicted_results = data[prediction_result_key]
            actual_results = data[actual_result_key]

            # Flatten the arrays
            predicted_results = predicted_results.flatten()
            actual_results = actual_results.flatten()

            # Calculate the score
            this_score_values.append(score_function(actual_results, predicted_results, **additional_score_function_args))
        
        score_values.append(this_score_values)
    
    if len(paths_to_pkl_files) > 1:
        # Combine the score values from multiple paths into one, if wanted
        if combine_file_predictions:
            final_score_values = np.empty(0)
            for file_score_values in score_values:
                final_score_values = np.append(final_score_values, file_score_values)
            final_score_values = np.array([final_score_values])
        else:
            final_score_values = score_values

        # Create a dataframe
        dataframe = pd.DataFrame(final_score_values).T
    else:
        final_score_values = score_values[0]

        # Create a dataframe
        dataframe = pd.DataFrame(final_score_values)

    # Try using provided labels, if length mismatches, generate labels
    if len(kwargs["label"]) == len(dataframe.columns):
        dataframe.columns = kwargs["label"]
    else:
        if len(kwargs["label"]) > 0:
            print("The number of labels does not match the number of different \'signals\'. Using empty labels.")
            
        if len(dataframe.columns) == 1:
            hist_args["legend"] = False
            kwargs["label"] = []
        else:
            if len(paths_to_pkl_files) > 1:
                kwargs["label"] = [f"File {i}" for i in range(len(dataframe.columns))]
            else:
                kwargs["label"] = [f"Class {i}" for i in range(len(dataframe.columns))]
            dataframe.columns = kwargs["label"]


    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
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
        path_to_model_directory: str,
        dataset: str, # "SHHS_Complete", "SHHS_Splitted", "GIF_Complete" or "GIF_Splitted"
        prediction_result_key: str,
        actual_result_key: str,
        remove_classes = [],
        map_classes = [],
        save_path = "",
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
    path_to_project_configuration: str
        the path to all signal processing parameters
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "Predicted" or "Predicted_in_windows")
    actual_result_key: str
        the key that accesses the actual results in the data (for example: "Actual" or "Actual_in_windows")
    
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
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Predicted Class")
    kwargs.setdefault("ylabel", "Actual Class")

    kwargs.setdefault("cmap", "Blues")
    kwargs.setdefault("values_format", ".1%") # or None, 'd', '.2g'
    kwargs.setdefault("colorbar", False)
    kwargs.setdefault("normalize", "true") # or "pred", "all", None

    # load signal processing parameters
    path_to_project_configuration = path_to_model_directory + project_configuration_file
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access dictionary that maps sleep stages (display labels) to integers
    sleep_stage_to_label = project_configuration["target_classes"]

    integer_labels = []
    display_labels = []
    for key, value in sleep_stage_to_label.items():
        if key not in remove_classes and value not in integer_labels:
            integer_labels.append(value)
            display_labels.append(key)
    
    for label_index in range(len(display_labels)):
        if display_labels[label_index] == "artifact":
            display_labels[label_index] = "Art"
        elif display_labels[label_index] == "wake":
            display_labels[label_index] = "Wake"
    
    for label_index in range(len(display_labels)):
        for map_index in range(len(map_classes)):
            if display_labels[label_index] == map_classes[map_index][0]:
                display_labels[label_index] = map_classes[map_index][1]

    # variables to store results
    all_predicted_results = np.empty(0)
    all_actual_results = np.empty(0)

    # Load the data
    data_generator = load_from_pickle(path_to_model_directory + model_performance_file[:-4] + "_" + dataset + "_Validation_Pid.pkl")

    for data in data_generator:
        
        # Get the predicted and actual results
        predicted_results = data[prediction_result_key]
        actual_results = data[actual_result_key]

        # Flatten the arrays
        predicted_results = predicted_results.flatten()
        actual_results = actual_results.flatten()

        # Add the results to the arrays
        all_predicted_results = np.append(all_predicted_results, predicted_results)
        all_actual_results = np.append(all_actual_results, actual_results)
    
    if len(remove_classes) > 0:
        remove_values = []
        for rem_clas in remove_classes:
            if rem_clas in sleep_stage_to_label:
                remove_values.append(sleep_stage_to_label[rem_clas])
    
        remove_result_indices = []
        for result_index in range(len(all_actual_results)):
            if all_actual_results[result_index] in remove_values or all_predicted_results[result_index] in remove_values:
                remove_result_indices.append(result_index)
        
        all_predicted_results = np.delete(all_predicted_results, remove_result_indices)
        all_actual_results = np.delete(all_actual_results, remove_result_indices)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)

    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true = all_actual_results,
        y_pred = all_predicted_results,
        ax = ax,
        display_labels = display_labels,
        labels = integer_labels,
        cmap = kwargs["cmap"],
        values_format = kwargs["values_format"],
        colorbar = kwargs["colorbar"],
        normalize = kwargs["normalize"]
        )
    
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])

    if len(save_path) == 0:
        plt.show()
    else:
        plt.savefig(save_path, format="pdf", dpi=200)#, bbox_inches="tight")
        plt.close()


def plot_actual_predicted(
        path_to_pkl_file: str,
        path_to_project_configuration: str,
        prediction_result_key: str,
        actual_result_key: str,
        data_position = None,
        **kwargs
    ):
    """
    Plot the actual and predicted results for some datapoint in the file.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_pkl_file: str
        the path to the pickle file containing the data
    path_to_project_configuration: str
        the path to all signal processing parameters
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "Predicted" or "Predicted_in_windows")
    actual_result_key: str
        the key that accesses the actual results in the data (for example: "Actual" or "Actual_in_windows")
    data_position: int or None
        the position of the data in the file (if None, a random position is chosen)

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
    linestyle: str
        the style of the lines
    ylim: list 
        the limits of the y-axis
    xlim: list
        the limits of the x-axis
    """

    # Default values
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")
    kwargs.setdefault("label", ["Actual", "Predicted"])
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
        # alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        # marker = kwargs["marker"],
        # markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # create dictionary to map labels to sleep stages
    sleep_stage_to_label = project_configuration["sleep_stage_label"]
    label_to_sleep_stage = {}
    for key, value in sleep_stage_to_label.items():
        value = str(value)
        if value not in label_to_sleep_stage:
            label_to_sleep_stage[value] = key

    # count entries in the file
    data_generator = load_from_pickle(path_to_pkl_file)
    number_entries = 0
    for data in data_generator:
        number_entries += 1
    
    # check if data_position is out of range
    if data_position is not None:
        if data_position >= number_entries:
            print(f"Data position {data_position} is not in the range of the data. Choosing position at random.")
            data_position = None
    
    # choose random data position if not given
    if data_position is None:
        data_position = np.random.randint(0, number_entries)
        print(f"Choosing random data position: {data_position}")

    # load the data at the given position
    data_generator = load_from_pickle(path_to_pkl_file)
    for i in range(data_position + 1):
        data = next(data_generator)
    
    predicted_results = data[prediction_result_key]
    actual_results = data[actual_result_key]

    # Calculate the accuracy values
    accuracy = metrics.accuracy_score(actual_results, predicted_results)
    kappa = metrics.cohen_kappa_score(actual_results, predicted_results)
    print(f"Accuracy: {accuracy}, Kappa: {kappa}")

    # Retrieve the unique classification results
    # unique_predicted_results = np.unique(predicted_results)
    # unique_actual_results = np.unique(actual_results)
    # unique_results = np.unique(np.concatenate((unique_predicted_results, unique_actual_results)))
    
    # Create a dictionary to map the unique results to the display labels
    data[prediction_result_key] = [label_to_sleep_stage[str(i)] for i in predicted_results]
    data[actual_result_key] = [label_to_sleep_stage[str(i)] for i in actual_results]

    # Plot the actual and predicted results
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    keys = [actual_result_key, prediction_result_key]
    for key_pos in range(len(keys)):
        ax.plot(
            keys[key_pos],
            data = data,
            label = kwargs["label"][key_pos],
            **plot_args
        )
    
    ax.legend(loc=kwargs["loc"])

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


tex_correction = 0.5
tex_look = {
    "text.usetex": True,
    # "text.latex.preamble": \usepackage{amsmath}\usepackage{amssymb},
    "font.family": "serif",
    "font.serif": "Computer Modern",
    #
    "legend.fontsize": 10-tex_correction,
    "xtick.labelsize": 10-tex_correction,
    "ytick.labelsize": 10-tex_correction,
    # "font.size": 12-tex_correction,
    "font.size": 10-tex_correction,
    "axes.titlesize": 12-tex_correction,
    "axes.labelsize": 12-tex_correction,
    #
    "savefig.format": "pdf",
    #
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.dpi": 600,
}

python_correction = 0
python_look = {
    "legend.fontsize": 8+python_correction,
    "xtick.labelsize": 8+python_correction,
    "ytick.labelsize": 8+python_correction,
    "font.size": 10+python_correction,
    "axes.titlesize": 10+python_correction,
    "axes.labelsize": 10+python_correction,
    #
    "savefig.format": "pdf",
    #
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.dpi": 600,
}

pt_to_inch = 1./72.27
cm_to_inch = 1/2.54

# linewidth = 16.2*cm_to_inch
# linewidth = 459.6215*pt_to_inch
linewidth = 429.1688*pt_to_inch

# fig_ratio = 3.4 / 2.7

if __name__ == "__main__":
    matplotlib.rcParams.update(tex_look)
    
    # multi-plots
    fig_ratio = 4 / 3
    linewidth *= 0.45 # 0.48, 0.5, 0.3, 0.322
    linewidth = 193.1246 * pt_to_inch

    # standalone plots
    # fig_ratio = 3 / 2
    # fig_ratio = 2 / 1
    # linewidth *= 0.8
    matplotlib.rcParams["figure.figsize"] = [linewidth, linewidth / fig_ratio]

    model_directory_path = "SSG_Local_120s_FullClass_RAW/"

    plot_confusion_matrix(
        path_to_model_directory = model_directory_path,
        dataset = "GIF_Complete", # "SHHS" or "GIF"
        # normalize = None,
        # values_format = None,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        remove_classes = ["artifact"],
        values_format = None, #'d', '.2g'
        normalize = None,
        # values_format = ".1%", #'d', '.2g'
        # normalize = "all"
    )

    model_directory_path = "SAE_Local_30s_AH_RAW/"

    plot_confusion_matrix(
        path_to_model_directory = model_directory_path,
        dataset = "GIF_Splitted", # "SHHS" or "GIF"
        # normalize = None,
        # values_format = None,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        remove_classes = ["artifact"],
        values_format = None, #'d', '.2g'
        normalize = None,
        # values_format = ".1%", #'d', '.2g'
        # normalize = "all"
    )

    raise SystemExit

    model_directory_path = "Neural_Network/"
    # model_directory_path = "Yao_no_overlap/"
    model_directory_path = "test/"

    plot_performance_per_epoch(
        model_directory_path = model_directory_path,
        use_metric = "accuracy", # "accuracy", "f1_score", "precision",
    )
    raise SystemExit

    """
    =============================
    Plot Accuracy/Loss per Epoch
    =============================
    """

    plot_accuracy_per_epoch(
        paths_to_pkl_files = [model_directory_path + loss_per_epoch_shhs_file],
        result_keys = ["train_accuracy", "train_avg_loss", "test_accuracy", "test_avg_loss"],
        label = ["train_accuracy", "train_avg_loss", "test_accuracy", "test_avg_loss"],
        title = "Training Neural Network on SHHS Data",
        xlabel = "Epoch",
        ylabel = "Accuracy / Loss",
    )

    plot_accuracy_per_epoch(
        paths_to_pkl_files = [model_directory_path + loss_per_epoch_shhs_file, model_directory_path + loss_per_epoch_gif_file],
        result_keys = ["test_accuracy"],
        label = ["SHHS", "GIF"],
        title = "History of Neural Network Accuracy",
        xlabel = "Epoch",
        ylabel = "Validation Accuracy",
    )

    """
    ================================================
    Plot Distribution for different Score Functions
    ================================================
    """

    path_to_save_gif_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl"
    gif_training_pid_results_path = path_to_save_gif_results[:-4] + "_Training_Pid.pkl"
    gif_validation_pid_results_path = path_to_save_gif_results[:-4] + "_Validation_Pid.pkl"

    plot_distribution_of_score(
        paths_to_pkl_files = [gif_training_pid_results_path, gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        score_function = metrics.accuracy_score, # metrics.cohen_kappa_score
        combine_file_predictions = False,
        title = "Distribution of Accuracy",
        xlabel = "Accuracy",
        label = ["Training Data", "Validation Data"],
        binrange = (0, 1),
        binwidth = 0.05,
        xlim = (0.6, 1.01),
    )

    plot_distribution_of_score(
        paths_to_pkl_files = [gif_training_pid_results_path, gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        score_function = metrics.accuracy_score, # metrics.cohen_kappa_score
        combine_file_predictions = True,
        title = "Combined Training and Validation Accuracy Distribution",
        xlabel = "Accuracy",
        binrange = (0, 1),
        binwidth = 0.05,
        xlim = (0.6, 1.01),
    )

    plot_distribution_of_score(
        paths_to_pkl_files = [gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        score_function = metrics.precision_score, # metrics.f1_score
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        title = "Distribution of Precision for GIF Validation Data",
        xlabel = "Precision",
        label = ["Wake", "LS", "DS", "REM"],
        binrange = (0, 1),
        binwidth = 0.05,
        xlim = (0.0, 1.01),
    )

    plot_distribution_of_score(
        paths_to_pkl_files = [gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        score_function = metrics.recall_score, # metrics.f1_score
        additional_score_function_args = {"average": 'weighted', "zero_division": np.nan},
        title = "Distribution of Recall for GIF Validation Data",
        xlabel = "Weighted Recall",
        binrange = (0, 1),
        binwidth = 0.05,
        xlim = (0.5, 1.01),
    )

    """
    ======================
    Plot Confusion Matrix
    ======================
    """

    plot_confusion_matrix(
        path_to_pkl_file = gif_validation_pid_results_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        title = "Confusion Matrix of Neural Network",
        xlabel = "predicted stage",
        ylabel = "actual stage",
        normalize = None, # 'true', 'pred', 'all'
        values_format = None, # 'd', 'f', '.1%'
    )

    """
    ==========================
    Plot Actual vs. Predicted
    ==========================
    """

    with open("Yao_no_overlap/Project_Configuration.pkl", "rb") as f:
        project_configuration = pickle.load(f)
    
    reciprocal_slp_frequency = round(1 / project_configuration['SLP_frequency'])

    plot_actual_predicted(
        path_to_pkl_file = gif_validation_pid_results_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        data_position = None,
        title = "Sleep Stages",
        xlabel = r"Time $\left(\text{in } %i \text{s}\right)$" % reciprocal_slp_frequency,
        ylabel = "Sleep Stage",
    )