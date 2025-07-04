"""
Author: Johannes Peter Knoll

Python File for implementing a custom PyTorch Dataset and a Deep Convolutional Neural Network for Sleep Stage 
Prediction.
"""

# IMPORTS:
import numpy as np
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor

# LOCAL IMPORTS:
from dataset_processing import *
from side_functions import *


"""
==============================
Implementing a Custom Dataset
==============================
"""


def final_data_preprocessing(
        signal: np.ndarray,
        signal_id: str,
        target_frequency: int,
        signal_length_seconds: int,
        pad_with,
        reshape_to_overlapping_windows: bool,
        normalize: bool,
        datatype_mappings: list,
        transform,
        **kwargs
    ):
    """
    Final preprocessing transformations (Normalization, Reshaping) applied to the data before it is being
    passed to the neural network.

    Attention:  Most keyword arguments are not assigned a default value here because there existence is only
                required if the corresponding preprocessing transformation should be applied in the first place.
    """

    if signal_id == "SLP":
        signal_type = "target"
    elif signal_id in ["RRI", "MAD"]:
        signal_type = "feature"
    else:
        raise ValueError(f"Unknown signal_id '{signal_id}'. Expected 'SLP', 'RRI' or 'MAD'.")

    if signal_id == "SLP":
        # transform sleep stage labels to uniform labels:
        signal = map_slp_labels(
            slp_labels = signal, # type: ignore
            slp_label_mapping = kwargs["slp_label_mapping"] # type: ignore
        )
    else:
        # remove outliers from signal:
        signal = remove_outliers(
            signal = signal, # type: ignore
            inlier_interval = kwargs["inlier_interval"]
        )

    # set default values for necessary keyword arguments:
    kwargs.setdefault("normalization_mode", "local")

    # collect normalization parameters:
    normalization_params = dict()
    for key in ["normalization_mode", "normalization_technique", "normalization_max", "normalization_min"]:
        if key in kwargs:
            normalization_params[key] = kwargs[key]

    # apply normalization prior to window segmentation if global normalization is requested:
    if kwargs["normalization_mode"] == "global" and normalize:
        signal = signal_normalization(
            signal = copy.deepcopy(signal), # type: ignore
            **normalization_params,
        )

    # reshape signal into overlapping windows if requested:
    if reshape_to_overlapping_windows:
        signal = reshape_signal_to_overlapping_windows(
            signal = copy.deepcopy(signal), # type: ignore
            target_frequency = target_frequency,
            nn_signal_duration_seconds = signal_length_seconds,
            pad_with = pad_with,
            number_windows = kwargs["windows_per_signal"],
            window_duration_seconds = kwargs["window_duration_seconds"],
            overlap_seconds = kwargs["overlap_seconds"],
            signal_type = signal_type,
            priority_order = kwargs["priority_order"],
        )
    else:
        # pad signal if it is shorter than the expected length (automatically done by reshape_signal_to_overlapping_windows):
        if len(signal) < int(signal_length_seconds * target_frequency):
            signal = np.append(signal, np.full_like(np.empty(int(signal_length_seconds * target_frequency) - signal.shape[0]), pad_with, dtype=signal.dtype), axis=0)

        # ensure only one label per signal (automatically done by reshape_signal_to_overlapping_windows):
        if signal_type == "target":
            # collect unique labels and their counts
            different_labels, label_counts = np.unique(copy.deepcopy(signal), return_counts=True)
            # take label with highest count
            signal = np.array([different_labels[np.argmax(label_counts)]])
        

    # apply normalization after window segmentation if local normalization is requested:
    if kwargs["normalization_mode"] == "local" and normalize:
        signal = signal_normalization(
            signal = copy.deepcopy(signal), # type: ignore
            **normalization_params,
        )
    
    # convert signal to correct data type:
    for mapping in datatype_mappings:
        if signal.dtype == mapping[0]:
            signal = signal.astype(mapping[1])

    # apply additional transformation (e.g. ToTensor()) if requested:
    if transform:
        signal = transform(copy.deepcopy(signal))
    
    return signal


class CustomSleepDataset(Dataset):
    """
    Custom Dataset class for our Sleep Stage Data. The class is used to load data from a file and
    prepare it for training a neural network (reshape signal into windows).

    Created in analogy to the PyTorch Tutorial: 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(
            self, 
            path_to_data_directory: str,
            pid: str,
            slp_label_mapping: dict,
            rri_inlier_interval: tuple,
            mad_inlier_interval: tuple,
            signal_length_seconds: int,
            pad_feature_with,
            pad_target_with,
            reshape_to_overlapping_windows: bool,
            normalize_rri: bool = False,
            normalize_mad: bool = False,
            feature_transform = None,
            target_transform = None,
            **kwargs
        ):
        """
        ARGUMENTS:
        ------------------------------
        data_manager_class : type
            The class used to access the data
        reshape_to_overlapping_windows : bool
            Whether to reshape the signal into overlapping windows
        normalize_rri : bool
            Whether to normalize the R-R Interval (RRI) signal, by default False
        normalize_mad : bool
            Whether to normalize the Mean Absolute Deviation (MAD) signal, by default False
        feature_transform : callable or None
            Optional transform to be applied on a feature sample, by default None
        target_transform : callable or None
            Optional transform to be applied on a target, by default None
        
        Keyword Arguments:
        ------------------------------
        
        ### Parameters for reshape_signal_to_overlapping_windows function in dataset_processing.py ###

        pad_feature_with : int
            Value to pad feature (RRI and MAD) with if signal too short, by default 0
        pad_target_with : int
            Value to pad target (SLP) with if signal too short, by default 0
        signal_length_seconds: int
            The length of the signal in seconds. This is used to determine the number of windows to create.
        windows_per_signal: int
            The number of windows to split the signal into.
        window_duration_seconds: int
            The window length in seconds.
        overlap_seconds: int
            The overlap between windows in seconds.
        priority_order: list
            The order in which labels should be prioritized in case of a tie. Only relevant if signal_type = 'target
        
        ### Parameters for signal_normalization function in dataset_processing.py ###

        normalization_technique: str
            The normalization technique to be used.
            if "z-score":   Standardizes the signal to have a mean of 0 and a standard deviation of 1.
            if "min-max":   Scales the signal to a specified range, typically [0, 1] or [-1, 1], based on the
                            maximum and minimum values. (Range adjustable via normalization_max and normalization_min)
        normalization_mode: str
            The normalization mode.
            if "global":    Scales all elements in the entire multi-dimensional array relative to the global
                            maximum and minimum values across all arrays.
            if "local":     Normalizes each sub-array independently, scaling the elements within relative to its
                            own maximum and minimum values.
        normalization_max: float
            The new maximum value for "min-max" normalization.
            If not specified, defaults to 1.0.
        normalization_min: float
            The new minimum value for "min-max" normalization.
            If not specified, defaults to 0.0.
        """
        
        # access data file and sampling frequencies:
        self.data_manager = SleepDataManager(
            directory_path = path_to_data_directory,
            pid = pid,
        )
        
        self.rri_frequency = self.data_manager.database_configuration["RRI_frequency"]
        self.mad_frequency = self.data_manager.database_configuration["MAD_frequency"]
        self.slp_frequency = self.data_manager.database_configuration["SLP_frequency"]

        # access target and feature value mapping parameters:
        self.slp_label_mapping = slp_label_mapping
        self.rri_inlier_interval = rri_inlier_interval
        self.mad_inlier_interval = mad_inlier_interval
        
        # parameters needed for ensuring uniform signal shape
        self.signal_length_seconds = signal_length_seconds
        self.pad_feature_with = pad_feature_with
        self.pad_target_with = pad_target_with

        # access common window_reshape_parameters
        self.reshape_to_overlapping_windows = reshape_to_overlapping_windows
        self.common_window_reshape_parameters = dict()

        if reshape_to_overlapping_windows:
            for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]:
                self.common_window_reshape_parameters[key] = kwargs[key]

        # access common signal_normalization_parameters
        self.normalize_rri = normalize_rri
        self.normalize_mad = normalize_mad
        self.common_signal_normalization_parameters = dict()

        if normalize_mad or normalize_rri:
            self.common_signal_normalization_parameters = {key: kwargs[key] for key in kwargs if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

        # access settings for additional transformations (if required):
        self.feature_transform = feature_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_manager)


    def __getitem__(self, idx):
        # load dictionary with data from file using data_manager
        data_sample = self.data_manager.load(idx)

        # extract feature (RRI) from dictionary and perform final preprocessing:
        rri_sample = final_data_preprocessing(
            signal = data_sample["RRI"], # type: ignore
            signal_id = "RRI",
            inlier_interval = self.rri_inlier_interval,
            target_frequency = self.rri_frequency,
            signal_length_seconds = self.signal_length_seconds,
            pad_with = self.pad_feature_with,
            reshape_to_overlapping_windows = self.reshape_to_overlapping_windows,
            **self.common_window_reshape_parameters,
            normalize = self.normalize_rri,
            **self.common_signal_normalization_parameters,
            datatype_mappings = [(np.float64, np.float32)],
            transform = self.feature_transform
        )

        if "MAD" in data_sample: # type: ignore
            # extract feature (MAD) from dictionary and perform final preprocessing:
            mad_sample = final_data_preprocessing(
                signal = data_sample["MAD"], # type: ignore
                signal_id = "MAD",
                inlier_interval = self.mad_inlier_interval,
                target_frequency = self.mad_frequency,
                signal_length_seconds = self.signal_length_seconds,
                pad_with = self.pad_feature_with,
                reshape_to_overlapping_windows = self.reshape_to_overlapping_windows,
                **self.common_window_reshape_parameters,
                normalize = self.normalize_mad,
                **self.common_signal_normalization_parameters,
                datatype_mappings = [(np.float64, np.float32)],
                transform = self.feature_transform,
            )
        
        else:
            mad_sample = "None"

        # extract labels (Sleep Phase) from dictionary and perform final preprocessing:
        slp_labels = final_data_preprocessing(
            signal = data_sample["SLP"], # type: ignore
            signal_id = "SLP",
            slp_label_mapping = self.slp_label_mapping,
            target_frequency = self.slp_frequency,
            signal_length_seconds = self.signal_length_seconds,
            pad_with = self.pad_target_with,
            reshape_to_overlapping_windows = self.reshape_to_overlapping_windows,
            **self.common_window_reshape_parameters,
            normalize = False,  # SLP labels should not be normalized
            datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
            transform = self.target_transform
        )
        
        return rri_sample, mad_sample, slp_labels


"""
=============================
Implementing Neural Networks
=============================
"""


def calculate_pooling_layer_start(
        rri_datapoints: int,
        mad_datapoints: int,
        rri_convolutional_channels: list,
        mad_convolutional_channels: list,
        max_pooling_layers: int
    ):
    """
    Function designed for the very specific case of our network structure and data.
    Only works if rri_datapoints = 2**n * mad_datapoints (n being an integer), convolution with padding "same"
    and pooling chosen so that it halves the number of datapoints.
    
    At the beginning of the network, the RRI branch and MAD branch are processed separately, both however with
    the same structure (convolutional layers, activation functions, pooling layers).

    To ensure both branches result in the same structured output, the rri structure must include more pooling layers.
    This function calculates the number of needed pooling layers and when the pooling layers are integrated into
    the structure mentioned above for RRI and MAD branches.
    """

    rri_mad_ratio = rri_datapoints / mad_datapoints / 2
    if int(rri_mad_ratio) != rri_mad_ratio:
        raise ValueError("Ratio of RRI to MAD datapoints divided by two must be an integer.")
    rri_mad_ratio = int(rri_mad_ratio)

    if rri_convolutional_channels[0] != 1:
        rri_convolutional_channels.insert(0, 1) # Ensure first channel is 1
    if mad_convolutional_channels[0] != 1:
        mad_convolutional_channels.insert(0, 1) # Ensure first channel is 1

    if rri_convolutional_channels[-1] != mad_convolutional_channels[-1]:
        raise ValueError(f"Number of channels in last convolutional layer of RRI and MAD branch must be equal. Last number of RRI channels: {rri_convolutional_channels[-1]}, Last number of MAD channels: {mad_convolutional_channels[-1]}.")
    if len(mad_convolutional_channels) < 2 or len(rri_convolutional_channels) < 2:
        raise ValueError(f"Number of convolutional layers (corresponds to channels disregarding first '1' at the start) in RRI and MAD branch must be at least 1. Current number of RRI channels: {len(rri_convolutional_channels)-1}, MAD channels: {len(mad_convolutional_channels)-1}.")
    if len(rri_convolutional_channels) < rri_mad_ratio + 1:
        raise ValueError(f"Number of convolutional layers (corresponds to channels disregarding first '1' at the start) in RRI branch must be at least {rri_mad_ratio + 1} (RRI to MAD datapoints ratio divided by two) to ensure that the number of datapoints after Signal Learning is equal for RRI and MAD branch. Current number of RRI convolutional layers: {len(rri_convolutional_channels)-1}.")
    if max_pooling_layers < rri_mad_ratio + 1:
        raise ValueError(f"Number of pooling layers must be at least {rri_mad_ratio} (RRI to MAD datapoints ratio divided by two) to ensure that the number of datapoints after Signal Learning is equal for RRI and MAD branch. Current number of maximum pooling layers: {max_pooling_layers}.")
    
    # check how many times the number of datapoints can be divided by two:
    mad_divisable_by_two = 0
    rri_divisable_by_two = 0

    while True:
        this_mad_ratio = mad_datapoints / (2 ** mad_divisable_by_two)
        if int(this_mad_ratio) != this_mad_ratio:
            mad_divisable_by_two -= 1
            break
        mad_divisable_by_two += 1

    while True:
        this_rri_ratio = rri_datapoints / (2 ** rri_divisable_by_two)
        if int(this_rri_ratio) != this_rri_ratio:
            rri_divisable_by_two -= 1
            break
        rri_divisable_by_two += 1
    
    if rri_divisable_by_two < rri_mad_ratio + 1:
        raise ValueError(f"RRI datapoints dividable by two must be at least {rri_mad_ratio + 1} (RRI to MAD datapoints ratio) to ensure that the number of datapoints after Signal Learning is equal for RRI and MAD branch. Current RRI datapoints dividable by two: {rri_divisable_by_two}.")
    
    # calculate the maximum number of pooling layers that can be applied:
    rri_poolings = min(mad_divisable_by_two + rri_mad_ratio, rri_divisable_by_two, len(rri_convolutional_channels) - 1, len(mad_convolutional_channels) - 1 + rri_mad_ratio)
    rri_poolings = min(rri_poolings, max_pooling_layers)
    mad_start_pooling = len(mad_convolutional_channels) - (rri_poolings - rri_mad_ratio)
    rri_start_pooling = len(rri_convolutional_channels) - rri_poolings

    # print(f"\nThe given settings allow for {rri_poolings} pooling layers in RRI branch and {rri_poolings - rri_mad_ratio} pooling layers in MAD branch.\nPooling starts at layer {rri_start_pooling} in RRI branch and at layer {mad_start_pooling} in MAD branch, corresponding to the transition from {rri_convolutional_channels[rri_start_pooling-1]} input to {rri_convolutional_channels[rri_start_pooling]} output channels and {mad_convolutional_channels[mad_start_pooling-1]} input to {mad_convolutional_channels[mad_start_pooling]} output channels, respectively.\n")
    mad_poolings = rri_poolings - rri_mad_ratio

    return rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling


class Window_Learning(nn.Module):
    """
    Window Learning part for YaoModel. 
    Consists of a series of dilated convolutional layers with residual connections.

    Same structure as ResBlock in: https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/
    """
    def __init__(
            self, 
            number_window_learning_features: int, 
            window_branch_convolutional_kernel_size: int,
            window_learning_dilations: list,
            negative_slope_leaky_relu: float,
            dropout_probability: float
            ):
        """
        ARGUMENTS:
        ------------------------------
        number_window_learning_features : int
            Number of features learned from Signal Learning, by default 128
        window_branch_convolutional_kernel_size : int
            Kernel size for convolutional layers during Window Learning, by default 7
        window_learning_dilations : list
            dilations for convolutional layers during Window Learning, by default [2, 4, 8, 16, 32]
        negative_slope_leaky_relu : float
            Negative slope for LeakyReLU activation function, by default 0.15
        dropout_probability : float
            Probability for dropout, by default 0.2
        """

        super(Window_Learning, self).__init__()

        window_layers = []
        for d in window_learning_dilations:
            window_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features,
                out_channels = number_window_learning_features,
                kernel_size = window_branch_convolutional_kernel_size,
                dilation = d,
                padding = 'same'
            ))
            window_layers.append(nn.Dropout(dropout_probability))
        
        self.window_branch = nn.Sequential(
            *window_layers
        )

    def forward(self, x):
        out = self.window_branch(x)
        return x + out


class Window_Learning_New(nn.Module):
    """
    Window Learning part for YaoModel. 
    Consists of a series of dilated convolutional layers with residual connections.

    Same structure as ResBlock in: https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/
    """
    def __init__(
            self, 
            number_window_learning_features: int, 
            window_branch_convolutional_kernel_size: int,
            window_learning_dilations: list,
            negative_slope_leaky_relu: float,
            dropout_probability: float
            ):
        """
        ARGUMENTS:
        ------------------------------
        number_window_learning_features : int
            Number of features learned from Signal Learning, by default 128
        window_branch_convolutional_kernel_size : int
            Kernel size for convolutional layers during Window Learning, by default 7
        window_learning_dilations : list
            dilations for convolutional layers during Window Learning, by default [2, 4, 8, 16, 32]
        negative_slope_leaky_relu : float
            Negative slope for LeakyReLU activation function, by default 0.15
        dropout_probability : float
            Probability for dropout, by default 0.2
        """

        super(Window_Learning_New, self).__init__()

        window_layers = []
        window_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
        for d in window_learning_dilations:
            window_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features,
                out_channels = number_window_learning_features,
                kernel_size = window_branch_convolutional_kernel_size,
                dilation = d,
                padding = 'same'
            ))
            window_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_layers.append(nn.Dropout(dropout_probability))
        
        self.window_branch = nn.Sequential(
            *window_layers
        )

    def forward(self, x):
        out = self.window_branch(x)
        return x + out


class YaoModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Tried to reproduce the architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/ for our way of data preprocessing.
    
    Differences to the original architecture:
    - Number of datapoints per window does not equal 2^x (x being an integer)
        - Reason:   SLP stage was sampled with 1/30 Hz, which made it impossible to have a window size of 
                    2^x which fits the sleep stage labels perfectly
        - Advantage:    Every window better represents the actual sleep stage
        - Disadvantage: Less repetitions of original structure possible 
                        (because each step requires to be dividable by 2)
    - Signal Feature Learning of MAD has different structure:
        - Before:       Same structure for RRI and MAD: Conv, LeakyReLU, MaxPool, ...
        - Now:          Different structure for MAD: Conv, LeakyReLU, Conv, LeakyReLU, MaxPool, ...
        - Reason:       RRI length = 4 * MAD length, so MAD can not be divided by 2 as often as RRI
                        (due to difference above, not many repetitions of original structure possible)
        - Advantage:    MAD signal can be processed more effectively, similar to original structure
    - Window Feature Learning has different structure:
        - Before:   2x ResBlock [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Now:      1x Window_Learning [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Note:     Window_Learning = ResBlock (Residual Block ?)
        - Reason:   Simplification of structure to reduce number of parameters
    
    ATTENTION:  It is advisable to choose the number of convolutional channels so that:
                2^(len_RRI_Conv_Channels - 1) / 2^[(len_MAD_Conv_Channels - 1)/2] = len_RRI_signal / len_MAD_signal

                This ensures that from RRI and MAD the same number of values remain after Signal Learning.
    """
    def __init__(
            self, 
            datapoints_per_rri_window: int,
            datapoints_per_mad_window: int,
            windows_per_signal: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        windows_per_signal: int
            Number of windows in each signal
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal
        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = datapoints_per_rri_window,
            mad_datapoints = datapoints_per_mad_window,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        super(YaoModel, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        window_branch_convolutional_kernel_size = 7

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_window_learning_features, 
                kernel_size = window_branch_convolutional_kernel_size, 
                dilation = dilation,
                padding='same'
                ))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        
        self.window_feature_learning = nn.Sequential(
            Window_Learning(
                number_window_learning_features = number_window_learning_features, 
                window_branch_convolutional_kernel_size = window_branch_convolutional_kernel_size, 
                window_learning_dilations = window_learning_dilations, 
                negative_slope_leaky_relu = negative_slope_leaky_relu, 
                dropout_probability = dropout_probability
                ),
            nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_sleep_stages, 
                kernel_size = 1
                )
            )

        """
        ======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        ======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, num_windows_rri, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."
        assert num_windows_rri == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_rri}."

        # Reshape RRI signal
        rri_signal = rri_signal.view(batch_size * num_windows_rri, 1, samples_in_window_rri)  # Combine batch and windows dimensions
        # rri_signal = rri_signal.reshape(-1, 1, samples_in_window_rri) # analogous to the above line

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, num_windows_mad, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."
            assert num_windows_mad == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_mad}."

            # Reshape MAD signal
            mad_signal = mad_signal.view(batch_size * num_windows_mad, 1, samples_in_window_mad)  # Combine batch and windows dimensions

        """
        ========================
        Signal Feature Learning
        ========================
        """

        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)
        #ecg_features = ecg_features.view(batch_size, num_windows, -1)  # Separate batch and windows dimensions

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            num_windows_mad = self.windows_per_signal
            mad_features = torch.zeros(batch_size * num_windows_mad, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)

        # Flatten features
        window_features = self.flatten(window_features)

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)

        # Reshape for convolutional layers
        output = output.reshape(batch_size, self.windows_per_signal, -1)
        output = output.transpose(1, 2).contiguous()

        # Convolutional layers
        output = self.window_feature_learning(output)

        # Reshape for output
        output = output.transpose(1, 2).contiguous().reshape(batch_size * self.windows_per_signal, -1)

        return output


class YaoModelNew(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Tried to reproduce the architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/ for our way of data preprocessing.
    
    Differences to the original architecture:
    - Number of datapoints per window does not equal 2^x (x being an integer)
        - Reason:   SLP stage was sampled with 1/30 Hz, which made it impossible to have a window size of 
                    2^x which fits the sleep stage labels perfectly
        - Advantage:    Every window better represents the actual sleep stage
        - Disadvantage: Less repetitions of original structure possible 
                        (because each step requires to be dividable by 2)
    - Signal Feature Learning of MAD has different structure:
        - Before:       Same structure for RRI and MAD: Conv, LeakyReLU, MaxPool, ...
        - Now:          Different structure for MAD: Conv, LeakyReLU, Conv, LeakyReLU, MaxPool, ...
        - Reason:       RRI length = 4 * MAD length, so MAD can not be divided by 2 as often as RRI
                        (due to difference above, not many repetitions of original structure possible)
        - Advantage:    MAD signal can be processed more effectively, similar to original structure
    - Window Feature Learning has different structure:
        - Before:   2x ResBlock [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Now:      1x Window_Learning [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Note:     Window_Learning = ResBlock (Residual Block ?)
        - Reason:   Simplification of structure to reduce number of parameters
    
    ATTENTION:  It is advisable to choose the number of convolutional channels so that:
                2^(len_RRI_Conv_Channels - 1) / 2^[(len_MAD_Conv_Channels - 1)/2] = len_RRI_signal / len_MAD_signal

                This ensures that from RRI and MAD the same number of values remain after Signal Learning.
    """
    def __init__(
            self, 
            datapoints_per_rri_window: int,
            datapoints_per_mad_window: int,
            windows_per_signal: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        windows_per_signal: int
            Number of windows in each signal
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = datapoints_per_rri_window,
            mad_datapoints = datapoints_per_mad_window,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        super(YaoModelNew, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        window_branch_convolutional_kernel_size = 7

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        """
        window_feature_learning_layers = []
        window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_window_learning_features, 
                kernel_size = window_branch_convolutional_kernel_size, 
                dilation = dilation,
                padding='same'
                ))
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        """
        
        self.window_feature_learning = nn.Sequential(
            Window_Learning_New(
                number_window_learning_features = number_window_learning_features, 
                window_branch_convolutional_kernel_size = window_branch_convolutional_kernel_size, 
                window_learning_dilations = window_learning_dilations, 
                negative_slope_leaky_relu = negative_slope_leaky_relu, 
                dropout_probability = dropout_probability
                ),
            nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_sleep_stages, 
                kernel_size = 1
                )
            )

        """
        ======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        ======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, num_windows_rri, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."
        assert num_windows_rri == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_rri}."

        # Reshape RRI signal
        rri_signal = rri_signal.view(batch_size * num_windows_rri, 1, samples_in_window_rri)  # Combine batch and windows dimensions
        # rri_signal = rri_signal.reshape(-1, 1, samples_in_window_rri) # analogous to the above line

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, num_windows_mad, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."
            assert num_windows_mad == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_mad}."

            # Reshape MAD signal
            mad_signal = mad_signal.view(batch_size * num_windows_mad, 1, samples_in_window_mad)  # Combine batch and windows dimensions

        """
        ========================
        Signal Feature Learning
        ========================
        """

        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)
        #ecg_features = ecg_features.view(batch_size, num_windows, -1)  # Separate batch and windows dimensions

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            num_windows_mad = self.windows_per_signal
            mad_features = torch.zeros(batch_size * num_windows_mad, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)

        # Flatten features
        window_features = self.flatten(window_features)

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)

        # Reshape for convolutional layers
        output = output.reshape(batch_size, self.windows_per_signal, -1)
        output = output.transpose(1, 2).contiguous()

        # Convolutional layers
        output = self.window_feature_learning(output)

        # Reshape for output
        output = output.transpose(1, 2).contiguous().reshape(batch_size * self.windows_per_signal, -1)

        return output


# conv, relu, conv, relu, pool
class SleepStageModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Inspired by architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/

    While YaoModel tried to recreate the architecture as good as possible, this time the architecture is 
    modified subtly (not only to fit to the way the data is preprocessed).
    
    Differences to the original architecture:
    - Window Feature Learning has different structure:
        - Before:       1x Window_Learning [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Now:          2x (LeakyRelu, Conv, Dropout, ...), Conv
        - Reason:       Adding the input after applying structure on input seemed shady
        - Disadvantage: Loss is higher at beginning (by about 23%)
        - Advantage:    Loss decreases quicker
    """
    def __init__(
            self, 
            datapoints_per_rri_window: int,
            datapoints_per_mad_window: int,
            windows_per_signal: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        windows_per_signal: int
            Number of windows in each signal
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = datapoints_per_rri_window,
            mad_datapoints = datapoints_per_mad_window,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        super(SleepStageModel, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        window_branch_convolutional_kernel_size = 7

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_window_learning_features, 
                kernel_size = window_branch_convolutional_kernel_size, 
                dilation = dilation,
                padding ='same'
                ))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        
        self.window_feature_learning = nn.Sequential(
            *window_feature_learning_layers,
            *window_feature_learning_layers,
            nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_sleep_stages, 
                kernel_size = 1
                )
            )

        """
        =======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        =======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, num_windows_rri, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."
        assert num_windows_rri == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_rri}."

        # Reshape RRI signal
        rri_signal = rri_signal.view(batch_size * num_windows_rri, 1, samples_in_window_rri)  # Combine batch and windows dimensions
        # rri_signal = rri_signal.reshape(-1, 1, samples_in_window_rri) # analogous to the above line

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, num_windows_mad, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."
            assert num_windows_mad == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_mad}."

            # Reshape MAD signal
            mad_signal = mad_signal.view(batch_size * num_windows_mad, 1, samples_in_window_mad)  # Combine batch and windows dimensions

        """
        ========================
        Signal Feature Learning
        ========================
        """

        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            num_windows_mad = self.windows_per_signal
            mad_features = torch.zeros(batch_size * num_windows_mad, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)

        # Flatten features
        window_features = self.flatten(window_features)

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)

        # Reshape for convolutional layers
        output = output.reshape(batch_size, self.windows_per_signal, -1)
        output = output.transpose(1, 2).contiguous()

        # Convolutional layers
        output = self.window_feature_learning(output)

        # Reshape for output
        output = output.transpose(1, 2).contiguous().reshape(batch_size * self.windows_per_signal, -1)

        return output

        """
        # Reshape for fully connected layers
        combined_features = combined_features.view(batch_size, -1)  # Combine windows and features dimensions

        # Fully connected layers
        output = self.fc(combined_features)
        return output
        """


class SleepStageModelNew(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Inspired by architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/

    While YaoModel tried to recreate the architecture as good as possible, this time the architecture is 
    modified subtly (not only to fit to the way the data is preprocessed).
    
    Differences to the original architecture:
    - Window Feature Learning has different structure:
        - Before:       1x Window_Learning [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Now:          2x (LeakyRelu, Conv, Dropout, ...), Conv
        - Reason:       Adding the input after applying structure on input seemed shady
        - Disadvantage: Loss is higher at beginning (by about 23%)
        - Advantage:    Loss decreases quicker
    """
    def __init__(
            self, 
            datapoints_per_rri_window: int,
            datapoints_per_mad_window: int,
            windows_per_signal: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        windows_per_signal: int
            Number of windows in each signal
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = datapoints_per_rri_window,
            mad_datapoints = datapoints_per_mad_window,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        super(SleepStageModelNew, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        window_branch_convolutional_kernel_size = 7

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_window_learning_features, 
                kernel_size = window_branch_convolutional_kernel_size, 
                dilation = dilation,
                padding ='same'
                ))
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        
        self.window_feature_learning = nn.Sequential(
            *window_feature_learning_layers,
            *window_feature_learning_layers,
            nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_sleep_stages, 
                kernel_size = 1
                )
            )

        """
        =======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        =======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, num_windows_rri, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."
        assert num_windows_rri == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_rri}."

        # Reshape RRI signal
        rri_signal = rri_signal.view(batch_size * num_windows_rri, 1, samples_in_window_rri)  # Combine batch and windows dimensions
        # rri_signal = rri_signal.reshape(-1, 1, samples_in_window_rri) # analogous to the above line

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, num_windows_mad, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."
            assert num_windows_mad == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_mad}."

            # Reshape MAD signal
            mad_signal = mad_signal.view(batch_size * num_windows_mad, 1, samples_in_window_mad)  # Combine batch and windows dimensions

        """
        ========================
        Signal Feature Learning
        ========================
        """

        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            num_windows_mad = self.windows_per_signal
            mad_features = torch.zeros(batch_size * num_windows_mad, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)

        # Flatten features
        window_features = self.flatten(window_features)

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)

        # Reshape for convolutional layers
        output = output.reshape(batch_size, self.windows_per_signal, -1)
        output = output.transpose(1, 2).contiguous()

        # Convolutional layers
        output = self.window_feature_learning(output)

        # Reshape for output
        output = output.transpose(1, 2).contiguous().reshape(batch_size * self.windows_per_signal, -1)

        return output


# conv, relu, conv, relu, pool
class DemoWholeNightModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Inspired by architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/

    ATTENTION:  This model should not be used. It continuosly prints out the shape of the data during the
                forward pass. Therefore, it is only useful for debugging purposes. The model will reshape the
                data similarly to the SleepStageModel, YaoModel and SleepStageModelNew.

    """
    def __init__(
            self, 
            datapoints_per_rri_window: int,
            datapoints_per_mad_window: int,
            windows_per_signal: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        windows_per_signal: int
            Number of windows in each signal
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = datapoints_per_rri_window,
            mad_datapoints = datapoints_per_mad_window,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        print(f"RRI Poolings: {rri_poolings}, RRI Start Pooling: {rri_start_pooling}, MAD Poolings: {mad_poolings}, MAD Start Pooling: {mad_start_pooling}")

        super(DemoWholeNightModel, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        window_branch_convolutional_kernel_size = 7

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_window_learning_features, 
                kernel_size = window_branch_convolutional_kernel_size, 
                dilation = dilation,
                padding ='same'
                ))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        
        self.window_feature_learning = nn.Sequential(
            *window_feature_learning_layers,
            *window_feature_learning_layers,
            nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_sleep_stages, 
                kernel_size = 1
                )
            )

        """
        =======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        =======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, num_windows_rri, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."
        assert num_windows_rri == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_rri}."

        print(f"RRI input signal: {rri_signal.size()}")  # Debugging line

        # Reshape RRI signal
        rri_signal = rri_signal.view(batch_size * num_windows_rri, 1, samples_in_window_rri)  # Combine batch and windows dimensions
        # rri_signal = rri_signal.reshape(-1, 1, samples_in_window_rri) # analogous to the above line
        print(f"RRI reshaped signal: {rri_signal.size()}")  # Debugging line

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, num_windows_mad, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."
            assert num_windows_mad == self.windows_per_signal, f"Expected {self.windows_per_signal} windows in each batch, but got {num_windows_mad}."
            print(f"MAD input signal: {mad_signal.size()}")  # Debugging line
            
            # Reshape MAD signal
            mad_signal = mad_signal.view(batch_size * num_windows_mad, 1, samples_in_window_mad)  # Combine batch and windows dimensions
            print(f"MAD reshaped signal: {mad_signal.size()}")  # Debugging line

        """
        ========================
        Signal Feature Learning
        ========================
        """

        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)
        print(f"RRI features: {rri_features.size()}")  # Debugging line

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            num_windows_mad = self.windows_per_signal
            mad_features = torch.zeros(batch_size * num_windows_mad, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)

        print(f"MAD features: {mad_features.size()}")  # Debugging line
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)
        print(f"Window features (RRI and MAD concatenated): {window_features.size()}")  # Debugging line

        # Flatten features
        window_features = self.flatten(window_features)
        print(f"Flattened window features: {window_features.size()}")  # Debugging line

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)
        print(f"Output after linear layer: {output.size()}")  # Debugging line

        # Reshape for convolutional layers
        output = output.reshape(batch_size, self.windows_per_signal, -1)
        print(f"Output reshaped: {output.size()}")  # Debugging line
        
        output = output.transpose(1, 2).contiguous()
        print(f"Output transposed: {output.size()}")

        # Convolutional layers
        output = self.window_feature_learning(output)
        print(f"Output after window feature learning: {output.size()}")  # Debugging line

        # Reshape for output
        output = output.transpose(1, 2).contiguous().reshape(batch_size * self.windows_per_signal, -1)
        print(f"Output reshaped (final result): {output.size()}")  # Debugging line

        return output


class DemoLocalIntervalModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Inspired by architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/

    While YaoModel tried to recreate the architecture as good as possible, this time the architecture is 
    modified subtly (not only to fit to the way the data is preprocessed).
    
    Differences to the original architecture:
    - Window Feature Learning has different structure:
        - Before:       1x Window_Learning [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Now:          2x (LeakyRelu, Conv, Dropout, ...), Conv
        - Reason:       Adding the input after applying structure on input seemed shady
        - Disadvantage: Loss is higher at beginning (by about 23%)
        - Advantage:    Loss decreases quicker
    """
    def __init__(
            self, 
            rri_datapoints: int,
            mad_datapoints: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = rri_datapoints
        self.datapoints_per_mad_window = mad_datapoints

        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = rri_datapoints,
            mad_datapoints = mad_datapoints,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        print(f"RRI Poolings: {rri_poolings}, RRI Start Pooling: {rri_start_pooling}, MAD Poolings: {mad_poolings}, MAD Start Pooling: {mad_start_pooling}")

        super(DemoLocalIntervalModel, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:  # Last two layers have pooling
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = rri_datapoints * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = 1, 
                out_channels = 1, 
                kernel_size = number_window_learning_features, 
                dilation = dilation,
                padding ='same'
                ))
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        
        self.window_feature_learning = nn.Sequential(
            *window_feature_learning_layers,
            *window_feature_learning_layers,
            )
        
        # Final Fully connected layer
        self.final = nn.Linear(number_window_learning_features, number_sleep_stages)

        """
        =======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        =======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = mad_datapoints // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."

        """
        ========================
        Signal Feature Learning
        ========================
        """

        print(f"RRI input signal: {rri_signal.size()}")  # Debugging line
        
        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)
        print(f"RRI features: {rri_features.size()}")  # Debugging line

        print(f"MAD input signal: {mad_signal.size() if mad_signal is not None else 'None'}")  # Debugging line

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            mad_features = torch.zeros(batch_size, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)

        print(f"MAD features: {mad_features.size()}")  # Debugging line
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)
        print(f"Window features (RRI and MAD concatenated): {window_features.size()}")  # Debugging line

        # Flatten features
        window_features = self.flatten(window_features)
        print(f"Flattened window features: {window_features.size()}")  # Debugging line

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)
        print(f"Output after linear layer: {output.size()}")

        # Reshape for convolutional layers
        output = output.unsqueeze(1)
        print(f"Output reshaped for convolutional layers: {output.size()}")  # Debugging line

        # Convolutional layers
        output = self.window_feature_learning(output)
        print(f"Output after window feature learning: {output.size()}")  # Debugging line

        # Reshape for final fully connected layer
        output = output.squeeze(1)
        print(f"Output reshaped for final layer: {output.size()}")
        
        output = self.final(output)
        print(f"Output after final layer: {output.size()}")

        return output


class LocalIntervalModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction. Inspired by architecture of:
    https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/

    While YaoModel tried to recreate the architecture as good as possible, this time the architecture is 
    modified subtly (not only to fit to the way the data is preprocessed).
    
    Differences to the original architecture:
    - Window Feature Learning has different structure:
        - Before:       1x Window_Learning [input + (LeakyReLU, Conv, Dropout, ...) applied on input], Conv
        - Now:          2x (LeakyRelu, Conv, Dropout, ...), Conv
        - Reason:       Adding the input after applying structure on input seemed shady
        - Disadvantage: Loss is higher at beginning (by about 23%)
        - Advantage:    Loss decreases quicker
    """
    def __init__(
            self, 
            rri_datapoints: int,
            mad_datapoints: int,
            rri_convolutional_channels: list,
            mad_convolutional_channels: list,
            max_pooling_layers: int,
            number_window_learning_features: int,
            window_learning_dilations: list,
            number_sleep_stages: int
            ):
        """
        ARGUMENTS:
        ------------------------------
        rri_datapoints: int
            Number of RRI data points
        mad_datapoints: int
            Number of MAD data points
        rri_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to RRI signal
        mad_convolutional_channels: list
            Number of output channels in subsequent 1D-convolutional layers applied to MAD signal
        max_pooling_layers: int
            Number of maximum pooling layers applied to RRI signal inbetween beforementioned convolutional layers
        number_window_learning_features: int
            Number of features learned from Signal Learning
        window_learning_dilations: list
            dilations for subsequent convolutional layers during Window Learning
        number_sleep_stages: int
            Number of predictable sleep stages
        """

        self.datapoints_per_rri_window = rri_datapoints
        self.datapoints_per_mad_window = mad_datapoints

        rri_poolings, rri_start_pooling, mad_poolings, mad_start_pooling = calculate_pooling_layer_start(
            rri_datapoints = rri_datapoints,
            mad_datapoints = mad_datapoints,
            rri_convolutional_channels = rri_convolutional_channels,
            mad_convolutional_channels = mad_convolutional_channels,
            max_pooling_layers = max_pooling_layers
        )

        super(LocalIntervalModel, self).__init__()

        # Parameters
        negative_slope_leaky_relu = 0.15
        dropout_probability = 0.2

        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        """
        ========================
        Signal Feature Learning
        ========================
        """

        """
        -----------------
        RRI Branch
        -----------------
        """

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(1, len(rri_convolutional_channels)):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos - 1], 
                out_channels = rri_convolutional_channels[num_channel_pos], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos]))
            # Activation function:
            rri_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= rri_start_pooling:  # Last two layers have pooling
                rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        """
        -----------------
        MAD Branch
        -----------------
        """

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(1, len(mad_convolutional_channels)):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos - 1], 
                out_channels = mad_convolutional_channels[num_channel_pos], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos]))
            # Activation function:
            mad_branch_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            # Pooling layer:
            if num_channel_pos >= mad_start_pooling:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied to rri branch)
        # -> number of pooling operations for mad branch were chosen so that result matches the shape of the rri branchs result

        remaining_feature_branch_values = rri_datapoints * rri_convolutional_channels[-1] // (2 ** (rri_poolings))

        if int(remaining_feature_branch_values) != remaining_feature_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")

        remaining_values_after_signal_learning = 2 * int(remaining_feature_branch_values)

        self.flatten = nn.Flatten()

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = 1, 
                out_channels = 1, 
                kernel_size = number_window_learning_features, 
                dilation = dilation,
                padding ='same'
                ))
            window_feature_learning_layers.append(nn.LeakyReLU(negative_slope_leaky_relu))
            window_feature_learning_layers.append(nn.Dropout(dropout_probability))
        
        self.window_feature_learning = nn.Sequential(
            *window_feature_learning_layers,
            *window_feature_learning_layers,
            )
        
        # Final Fully connected layer
        self.final = nn.Linear(number_window_learning_features, number_sleep_stages)

        """
        =======================================================
        Save Output Shape Of MAD Branch (for data without MAD)
        =======================================================
        """

        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = mad_datapoints // (2 ** mad_poolings)

        if int(self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        self.mad_values_after_signal_learning = int(self.mad_values_after_signal_learning)


    def forward(self, rri_signal, mad_signal = None):
        """
        =============================================
        Checking And Preparing Data For Forward Pass
        =============================================
        """

        # Check Dimensions of RRI signal
        batch_size, _, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."

        """
        ========================
        Signal Feature Learning
        ========================
        """
        
        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            mad_features = torch.zeros(batch_size, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device) # type: ignore
        else:
            mad_features = self.mad_signal_learning(mad_signal)
        
        """
        =======================
        Create Window Features
        =======================
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)

        # Flatten features
        window_features = self.flatten(window_features)

        """
        ========================
        Window Feature Learning
        ========================
        """

        # Fully connected layer
        output = self.linear(window_features)

        # Reshape for convolutional layers
        output = output.unsqueeze(1)

        # Convolutional layers
        output = self.window_feature_learning(output)

        # Reshape for final fully connected layer
        output = output.squeeze(1)
        
        output = self.final(output)

        return output


"""
=========================
Learning Rate Scheduling
=========================
"""

class CosineScheduler:
    """
    Source: https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/

    The Learning Rate defines how much to update the models parameters at each batch/epoch. 
    Smaller values yield slow learning speed, while large values may result in unpredictable behavior during
    training (https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html).

    That's why we want to start with a larger learning rate and decrease it over time. A useful approach is to
    decrease it in a cosine manner.

    Using this class we can also define the parameters so, that the learning rate increases linearly in the beginning
    and then decreases in a cosine manner.
    """
    def __init__(
            self, 
            number_updates_total, 
            number_updates_to_max_lr,
            start_learning_rate, 
            max_learning_rate,
            end_learning_rate, 
        ):
        """
        ARGUMENTS:
        ------------------------------
        number_updates_total : int
            Total number of learning rate updates for new epochs
        number_updates_to_max_lr : int
            Number of updates to reach max learning rate
        start_learning_rate : float
            Initial learning rate
        max_learning_rate : float
            Maximum learning rate
        end_learning_rate : float
            Final learning rate
        """

        self.number_updates_total = number_updates_total
        self.number_increase_lr = number_updates_to_max_lr
        self.number_decrease_lr = number_updates_total - number_updates_to_max_lr
        self.start_learning_rate = start_learning_rate
        self.max_learning_rate = max_learning_rate
        self.end_learning_rate = end_learning_rate
    
    def linear_lr_increase(self, epoch):
        """
        Calculates linear increase of learning rate depending on the epoch
        """

        increase = self.max_learning_rate - self.start_learning_rate
        increase *= epoch / self.number_increase_lr
        return increase

    def cosine_lr_decay(self, epoch):
        """
        Calculates cosine decrease of learning rate depending on the epoch
        """

        decay = self.max_learning_rate - self.end_learning_rate
        decay *= (1 + math.cos(math.pi * (epoch-self.number_increase_lr) / self.number_decrease_lr)) / 2
        return decay

    def __call__(self, epoch):
        """
        Returns the learning rate for the given epoch
        """

        if epoch < self.number_increase_lr:
            return self.start_learning_rate + self.linear_lr_increase(epoch)
        if epoch <= self.number_updates_total:
            return self.end_learning_rate + self.cosine_lr_decay(epoch)


"""
=========================
Looping Over The Dataset
=========================
"""

# TRAINING LOOP
def train_loop(dataloader, model, device, loss_fn, optimizer_fn, lr_scheduler, current_epoch, batch_size, number_classes):
    """
    Iterate over the training dataset and try to converge to optimal parameters.

    Source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    RETURNS:
    ------------------------------
    train_loss : float
        Average loss value of the training dataset
    correct : float
        Ratio of correctly predicted values of the training dataset
    predicted_results : list
        Predicted sleep stages
    actual_results : list
        Actual sleep stages

    ARGUMENTS:
    ------------------------------
    dataloader : DataLoader
        DataLoader object containing the training dataset
    model : nn.Module
        Neural Network model to train
    device : str
        Device to train the model on
    loss_fn : nn.Module
        Loss function to be minimized
    optimizer_fn : torch.optim
        Optimizer to update the model parameters
    lr_scheduler :
        Scheduler for the learning rate
    current_epoch : int
        Current epoch number
    batch_size : int
        Number of samples in each batch
    """

    # set optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr_scheduler(current_epoch))

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    # variables to save accuracy progress
    train_loss = 0
    train_confusion_matrix = np.zeros((number_classes, number_classes))

    # variables to track progress
    num_batches = len(dataloader)
    print("\nTraining Neural Network Model:")
    progress_bar = DynamicProgressBar(total = len(dataloader.dataset), batch_size = batch_size)

    # Iterate over the training dataset
    for batch, (rri, mad, slp) in enumerate(dataloader):

        # check if MAD signal was not provided
        if mad[0] == "None":
            mad = None
        else:
            mad = mad.to(device)
        
        # Send data to device
        rri, slp = rri.to(device), slp.to(device)

        # reshape slp to fit the model output
        slp = slp.view(-1) # Combine batch and windows dimensions

        # Compute prediction and loss
        pred = model(rri, mad)
        slp = slp.long()
        loss = loss_fn(pred, slp)

        # Backpropagation
        loss.backward()
        optimizer.step() # updates the model parameters based on the gradients computed during the backward pass
        optimizer.zero_grad()

        # update confusion matrix
        pred = pred.argmax(1).cpu().numpy()
        slp = slp.cpu().numpy()
        for i in range(len(slp)):
            train_confusion_matrix[slp[i], pred[i]] += 1

        train_loss += loss.item()

        # print progress bar
        accuracy = train_confusion_matrix.diagonal().sum() / train_confusion_matrix.sum()
        progress_bar.update(
            additional_info = f'Loss: {format_float(loss.item(), 3)} | Acc: {(100*accuracy):>0.1f}%',
            )
    
    train_loss /= num_batches

    return train_loss, train_confusion_matrix


# TESTING LOOP
def test_loop(dataloader, model, device, loss_fn, batch_size, number_classes):
    """
    Iterate over the test dataset to check if model performance is improving

    Source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    RETURNS:
    ------------------------------
    test_loss : float
        Average loss value of the test dataset
    correct : float
        Ratio of correctly predicted values of the test dataset
    predicted_results : list
        Predicted sleep stages
    actual_results : list
        Actual sleep stages


    ARGUMENTS:
    ------------------------------
    dataloader : DataLoader
        DataLoader object containing the test dataset
    model : nn.Module
        Neural Network model to test
    device : str
        Device to test the model on
    loss_fn : nn.Module
        Loss function to be minimized
    batch_size : int
        Number of samples in each batch
    collect_results : bool
        If True, predicted and actual results are collected
    """

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    # variables to track progress
    num_batches = len(dataloader)
    print("\nCalculating Prediction Accuracy on Test Data:")
    progress_bar = DynamicProgressBar(total = len(dataloader.dataset), batch_size = batch_size)

    # variables to save performance progress
    test_loss = 0
    test_confusion_matrix = np.zeros((number_classes, number_classes))

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        # Iterate over the test dataset
        for batch, (rri, mad, slp) in enumerate(dataloader):
            # check if MAD signal was not provided
            if mad[0] == "None":
                mad = None
            else:
                mad = mad.to(device)

            # Send data to device
            rri, slp = rri.to(device), slp.to(device)

            # reshape slp to fit the model output
            slp = slp.view(-1) # Combine batch and windows dimensions

            # Compute prediction and loss
            pred = model(rri, mad)
            slp = slp.long()
            test_loss += loss_fn(pred, slp).item()

            # update confusion matrix
            pred = pred.argmax(1).cpu().numpy()
            slp = slp.cpu().numpy()

            for i in range(len(slp)):
                test_confusion_matrix[slp[i], pred[i]] += 1

            # print progress bar
            progress_bar.update()

    test_loss /= num_batches
    accuracy = test_confusion_matrix.diagonal().sum() / test_confusion_matrix.sum()

    print(f"\nTest Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")

    return test_loss, test_confusion_matrix


# Example usage
if __name__ == "__main__":

    """
    --------------------------------------
    Preparing Random Data File For Testing
    --------------------------------------
    """
    print("\n\nPreparing random data file for testing...")
    print("="*80)
    # creating dataset file and data manager instance on it
    random_directory_path = "Testing_NNM/"
    random_data_manager = SleepDataManager(directory_path = random_directory_path)
    random_sleep_stage_labels = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifact": ["other"]}

    # creating and saving random data to file
    for index in range(10):
        random_datapoint = {
            "ID": str(index),
            "RRI": np.random.rand(36000*4), # 10 hours with 4 Hz sampling rate
            "RRI_frequency": 4,
            "MAD": np.random.rand(36000), # 10 hours with 1 Hz sampling rate
            "MAD_frequency": 1,
            "SLP": np.random.randint(5, size=1200), # 10 hours with 1/30 Hz sampling rate
            "SLP_frequency": 1/30,
            "sleep_stage_label": random_sleep_stage_labels
        }
        random_datapoint_without_mad = {
            "ID": str(index),
            "RRI": np.random.rand(36000*4), # 10 hours with 4 Hz sampling rate
            "RRI_frequency": 4,
            "SLP": np.random.randint(5, size=1200), # 10 hours with 1/30 Hz sampling rate
            "SLP_frequency": 1/30,
            "sleep_stage_label": random_sleep_stage_labels
        }
        random_data_manager.save(random_datapoint, unique_id=True) # comment to test data without MAD signal
        #random_data_manager.save(random_datapoint_without_mad) # uncomment to test data without MAD signal
    
    # retrieve slp label mapping
    slp_label_mapping = get_slp_label_mapping(
        current_labels = random_sleep_stage_labels,
        desired_labels = {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0}
    )

    # print some data
    some_datapoint = random_data_manager.load(0)

    print("Shape of Signals:")
    print(f"RRI Signal: {some_datapoint["RRI"].shape}") # type: ignore
    try:
        print(f"MAD Signal: {some_datapoint["MAD"].shape}") # type: ignore
    except:
        pass
    print(f"SLP Signal: {some_datapoint["SLP"].shape}") # type: ignore
    
    del random_data_manager, random_sleep_stage_labels, some_datapoint

    print("="*80)


    """
    ---------------------------------
    Testing The Custom Dataset Class
    ---------------------------------
    """

    print("\n\nTesting the Custom Dataset Class...")
    print("="*80)
    print("")

    # Create dataset
    dataset = CustomSleepDataset(
        path_to_data_directory = random_directory_path,
        pid = "main",
        slp_label_mapping = slp_label_mapping,
        rri_inlier_interval = (None, None), # no inlier interval for RRI
        mad_inlier_interval = (None, None), # no inlier interval for MAD
        reshape_to_overlapping_windows = True,
        normalize_rri = True,
        normalize_mad = True,
        # kwargs for CustomSleepDataset:
        transform=None,
        target_transform = None,
        pad_feature_with = 0,
        pad_target_with = 0,
        signal_length_seconds = 36000,
        windows_per_signal = 1197,
        window_duration_seconds = 120, 
        overlap_seconds = 90,
        priority_order = [3, 2, 1, 0],
        normalization_max = 1,
        normalization_min = 0,
        normalization_mode = "global",
        )

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # Iterate over dataloader and print shape of features and labels
    for batch, (rri, mad, slp) in enumerate(dataloader):
        # print shape of data and labels:
        print(f"Batch {batch}:")
        print("-"*80)
        print("RRI shape:", rri.shape)
        if mad[0] == "None":
            mad = None
            print(f"MAD shape: {mad}")
        else:
            print("MAD shape:", mad.shape)
        print("SLP shape:", slp.shape)
        print("")
    
    # delete directory
    for file in os.listdir(random_directory_path):
        file_path = os.path.join(random_directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(random_directory_path)

    print("="*80)


    """
    ---------------------------------
    Testing The Neural Network Model
    ---------------------------------
    """

    print("\n\nTesting the Neural Network Model...")
    print("="*80)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    """
    Full Night Sleep Stage Models:
    """
    print("\nFull Night Sleep Stage Models:")
    print("-"*80)

    # Define the Neural Network
    DCNN = DemoWholeNightModel(
        datapoints_per_rri_window = 480, 
        datapoints_per_mad_window = 120,
        windows_per_signal = 1197,
        rri_convolutional_channels = [1, 8, 16, 32, 64],
        mad_convolutional_channels = [1, 8, 16, 32, 64],
        max_pooling_layers = 5,
        number_window_learning_features = 128,
        window_learning_dilations = [2, 4, 8, 16, 32],
        number_sleep_stages = 4
        ) # SleepStageModel, YaoModel, SleepStageModelNew
    DCNN.to(device)

    # Create example data
    rri_example = torch.rand((2, 1, 1197, 480), device=device)
    mad_example = torch.rand((2, 1, 1197, 120), device=device)
    # mad_example = None # uncomment to test data without MAD signal

    # Send data to device
    rri_example = rri_example.to(device)
    if mad_example is not None:
        mad_example = mad_example.to(device)

    # Pass data through the model
    output = DCNN(rri_example, mad_example)
    print("-"*80)
    print(output.shape)

    """
    Local (Short Time) Sleep Stage Models:
    """

    print("\nLocal (Short Time) Sleep Stage Models:")
    print("-"*80)

    seconds = 120

    # Define the Neural Network
    DCNN = DemoLocalIntervalModel(
        rri_datapoints = seconds * 4,  # 4 Hz sampling rate
        mad_datapoints = seconds,  # 1 Hz sampling rate
        rri_convolutional_channels = [1, 8, 16, 32, 64],
        mad_convolutional_channels = [1, 8, 16, 32, 64],
        max_pooling_layers = 5,
        number_window_learning_features = 128,
        window_learning_dilations = [2, 4, 8, 16, 32],
        number_sleep_stages = 4
        )
    DCNN.to(device)

    # Create example data
    rri_example = torch.rand((2, 1, seconds * 4), device=device)
    mad_example = torch.rand((2, 1, seconds), device=device)
    # mad_example = None # uncomment to test data without MAD signal

    # Send data to device
    rri_example = rri_example.to(device)
    if mad_example is not None:
        mad_example = mad_example.to(device)

    # Pass data through the model
    output = DCNN(rri_example, mad_example)
    print("-"*80)
    print(output.shape)

    print("="*80)