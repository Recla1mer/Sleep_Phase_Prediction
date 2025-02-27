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


def plot_accuracy_per_epoch(
        paths_to_pkl_files: list,
        result_keys: list,
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

    # check if user wants to overkill the plot
    if len(paths_to_pkl_files) > 1 and len(result_keys) > 1:
        raise ValueError("You can either provide multiple files and one result key or one file and multiple result keys.")
    
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

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    labels = kwargs["label"]
    if len(kwargs["label"]) != len(paths_to_pkl_files) * len(result_keys):
        print("The number of labels does not match the number of data. Using empty labels.")
        labels = ["" for _ in range(len(paths_to_pkl_files) * len(result_keys))]
        kwargs["label"] = []

    if len(paths_to_pkl_files) > 1:
        for i, path in enumerate(paths_to_pkl_files):
            data_generator = load_from_pickle(path)
            data = next(data_generator)
            ax.plot(
                result_keys[0],
                data = data,
                label = labels[i],
                **plot_args
            )
    elif len(result_keys) > 1:
        data_generator = load_from_pickle(paths_to_pkl_files[0])
        data = next(data_generator)
        for i, key in enumerate(result_keys):
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
        path_to_pkl_file: str,
        path_to_project_configuration: str,
        prediction_result_key: str,
        actual_result_key: str,
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
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "predicted stage")
    kwargs.setdefault("ylabel", "real stage")

    kwargs.setdefault("cmap", "Blues")
    kwargs.setdefault("values_format", ".1%") # or None, 'd', '.2g'
    kwargs.setdefault("colorbar", False)
    kwargs.setdefault("normalize", "true") # or "pred", "all", None

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

    # Load the data
    data_generator = load_from_pickle(path_to_pkl_file)
    data = next(data_generator)

    # Get the predicted and actual results
    predicted_results = data[prediction_result_key]
    actual_results = data[actual_result_key]

    # Flatten the arrays
    predicted_results = predicted_results.flatten()
    actual_results = actual_results.flatten()

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)

    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true = actual_results,
        y_pred = predicted_results,
        ax = ax,
        display_labels = display_labels,
        labels = integer_labels,
        cmap = kwargs["cmap"],
        values_format = kwargs["values_format"],
        colorbar = kwargs["colorbar"],
        normalize = kwargs["normalize"]
        )
    
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])

    plt.show()


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


if __name__ == "__main__":

    model_directory_path = "Neural_Network/"
    # model_directory_path = "Yao_no_overlap/"

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