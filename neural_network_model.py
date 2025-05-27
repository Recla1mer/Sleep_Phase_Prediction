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
from torchvision.transforms import ToTensor

# LOCAL IMPORTS:
from dataset_processing import *
from side_functions import *


"""
==============================
Implementing a Custom Dataset
==============================
"""

class CustomSleepDataset(Dataset):
    """
    Custom Dataset class for our Sleep Stage Data. The class is used to load data from a file and
    prepare it for training a neural network (reshape signal into windows).

    Created in analogy to the PyTorch Tutorial: 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """
    def __init__(
            self, 
            path_to_data: str, 
            transform = None,
            target_transform = None,
            pad_feature_with = 0,
            pad_target_with = 0,
            number_windows: int = 1197, 
            window_duration_seconds: int = 120, 
            overlap_seconds: int = 90,
            priority_order: list = [3, 2, 1, 0],
            normalize_rri: bool = False,
            normalize_mad: bool = False,
            normalization_max: float = 1,
            normalization_min: float = 0,
            normalization_mode = "global"
        ):
        """
        ARGUMENTS:
        ------------------------------
        path_to_data : str
            Path to the data file
        transform : callable
            Optional transform to be applied on a sample, by default None
        target_transform : callable
            Optional transform to be applied on a target, by default None
        
        ### Parameters for reshape_signal_to_overlapping_windows function in dataset_processing.py ###

        pad_feature_with : int
            Value to pad feature (RRI and MAD) with if signal too short, by default 0
        pad_target_with : int
            Value to pad target (SLP) with if signal too short, by default 0
        number_windows: int
            The number of windows to split the signal into.
        window_duration_seconds: int
            The window length in seconds.
        overlap_seconds: int
            The overlap between windows in seconds.
        priority_order: list
            The order in which labels should be prioritized in case of a tie. Only relevant if signal_type = 'target
        
        ### Parameters for unity_based_normalization function in dataset_processing.py ###

        normalization_max: float
            The new maximum value.
        normalization_min: float
            The new minimum value.
        normalization_mode: str
            The normalization mode.
            if "global":    Scales all elements in the entire multi-dimensional array relative to the global
                            maximum and minimum values across all arrays.
            if "local":     Normalizes each sub-array independently, scaling the elements within relative to its
                            own maximum and minimum values.
        """

        self.transform = transform
        self.target_transform = target_transform
        
        self.data_manager = SleepDataManager(path_to_data)
        self.rri_frequency = self.data_manager.file_info["RRI_frequency"]
        self.mad_frequency = self.data_manager.file_info["MAD_frequency"]
        self.slp_frequency = self.data_manager.file_info["SLP_frequency"]
        
        self.pad_feature_with = pad_feature_with
        self.pad_target_with = pad_target_with

        self.window_reshape_parameters = {
            "nn_signal_duration_seconds": self.data_manager.file_info["signal_length_seconds"],
            "number_windows": number_windows, 
            "window_duration_seconds": window_duration_seconds, 
            "overlap_seconds": overlap_seconds,
            "priority_order": priority_order
        }

        self.normalize_rri = normalize_rri
        self.normalize_mad = normalize_mad

        self.unity_based_normalization_parameters = {
            "normalization_max": normalization_max,
            "normalization_min": normalization_min,
            "normalization_mode": normalization_mode
        }
        

    def __len__(self):
        return len(self.data_manager)


    def __getitem__(self, idx):
        # load dictionary with data from file using data_manager
        data_sample = self.data_manager.load(idx)

        self.window_reshape_parameters["signal_type"] = "feature"
        self.window_reshape_parameters["pad_with"] = self.pad_feature_with

        # extract feature (RRI) from dictionary and reshape it into windows:
        rri_sample = reshape_signal_to_overlapping_windows(
            signal = data_sample["RRI"], # type: ignore
            target_frequency = self.rri_frequency,
            **self.window_reshape_parameters
        )

        # normalize RRI signal if requested:
        if self.normalize_rri:
            rri_sample = unity_based_normalization(
                signal = copy.deepcopy(rri_sample), # type: ignore
                **self.unity_based_normalization_parameters
            )

        if rri_sample.dtype == np.float64:
            rri_sample = rri_sample.astype(np.float32)

        try:
            # extract feature (MAD) from dictionary and reshape it into windows:
            mad_sample = reshape_signal_to_overlapping_windows(
                signal = data_sample["MAD"], # type: ignore
                target_frequency = self.mad_frequency,
                **self.window_reshape_parameters
            )

            # normalize MAD signal if requested:
            if self.normalize_mad:
                mad_sample = unity_based_normalization(
                    signal = copy.deepcopy(mad_sample), # type: ignore
                    **self.unity_based_normalization_parameters
                )

            if mad_sample.dtype == np.float64:
                mad_sample = mad_sample.astype(np.float32)
        except:
            mad_sample = "None"

        self.window_reshape_parameters["signal_type"] = "target"
        self.window_reshape_parameters["pad_with"] = self.pad_target_with

        # extract labels (Sleep Phase) from dictionary:
        slp_labels = reshape_signal_to_overlapping_windows(
            signal = data_sample["SLP"], # type: ignore 
            target_frequency = self.slp_frequency,
            **self.window_reshape_parameters
        )
        if slp_labels.dtype == np.int64:
            slp_labels = slp_labels.astype(np.int32)
        if slp_labels.dtype == np.float64:
            slp_labels = slp_labels.astype(np.float32)

        if self.transform:
            rri_sample = self.transform(rri_sample)
            try:
                mad_sample = self.transform(mad_sample)
            except:
                pass
            
        if self.target_transform:
            slp_labels = self.target_transform(slp_labels)
        
        return rri_sample, mad_sample, slp_labels


"""
=============================
Implementing Neural Networks
=============================
"""


class Window_Learning(nn.Module):
    """
    Window Learning part for YaoModel. 
    Consists of a series of dilated convolutional layers with residual connections.

    Same structure as ResBlock in: https://github.com/AlexMa123/DCNN-SHHS/blob/main/DCNN_SHHS/
    """
    def __init__(
            self, 
            number_window_learning_features = 128, 
            window_branch_convolutional_kernel_size = 7,
            window_learning_dilations = [2, 4, 8, 16, 32],
            negative_slope_leaky_relu = 0.15,
            dropout_probability = 0.2
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
            number_window_learning_features = 128, 
            window_branch_convolutional_kernel_size = 7,
            window_learning_dilations = [2, 4, 8, 16, 32],
            negative_slope_leaky_relu = 0.15,
            dropout_probability = 0.2
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
            datapoints_per_rri_window = 480, 
            datapoints_per_mad_window = 120,
            windows_per_signal = 1197,
            number_window_learning_features = 128,
            rri_convolutional_channels = [1, 8, 16, 32, 64],
            mad_convolutional_channels = [1, 8, 16, 32, 64],
            window_learning_dilations = [2, 4, 8, 16, 32],
            number_sleep_stages = 4
            ):
        """
        ARGUMENTS:
        ------------------------------
        datapoints_per_rri_window : int, optional
            Number of data points in each RRI window, by default 480
        datapoints_per_mad_window : int, optional
            Number of data points in each MAD window, by default 120
        windows_per_signal : int, optional
            Number of windows in each batch, by default 1197
        number_window_learning_features : int, optional
            Number of features learned from Signal Learning, by default 128
        rri_convolutional_channels : list, optional
            Number of channels to process RRI signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        mad_convolutional_channels : list, optional
            Number of channels to process MAD signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        window_learning_dilations : list, optional
            dilations for convolutional layers during Window Learning, by default [2, 4, 6, 8]
        number_sleep_stages : int, optional
            Number of predictable sleep stages, by default 4
        """

        # check parameters:
        if len(mad_convolutional_channels) % 2 != 1 or len(mad_convolutional_channels) < 3:
            raise ValueError("Number of convolutional channels in MAD branch must be odd and more than 2.")
        if datapoints_per_rri_window % 2**(len(rri_convolutional_channels)-1) != 0:
            raise ValueError("Number of RRI datapoints per window must be dividable by 2^(number of RRI convolutional layers - 1) without rest.")
        if datapoints_per_mad_window % 2 ** ((len(rri_convolutional_channels)-1)/2) != 0:
            raise ValueError("Number of MAD datapoints per window must be dividable by 2^((number of MAD convolutional layers - 1)/2) without rest.")
        if rri_convolutional_channels[-1] != mad_convolutional_channels[-1]:
            raise ValueError("Number of channels in last convolutional layer of RRI and MAD branch must be equal.")
        if 2**(len(rri_convolutional_channels) - 1) / 2**((len(mad_convolutional_channels) - 1) / 2) != datapoints_per_rri_window / datapoints_per_mad_window:
            raise ValueError("Number of remaining values after Signal Learning must be equal for RRI and MAD branch. Adjust number of convolutional channels accordingly.")

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

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
            if num_channel_pos % 2 == 0:
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
        # -> datapoints before branch must be multiplied by the number of channels of the 
        # last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # MaxPooling is applied after each convolutional layer in RRI branch and after every second 
        # convolutional layer in MAD branch
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied)
        # -> datapoints after mad branch must be divided by (2 ** (number of pooling layers applied / 2))

        remaining_rri_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (len(rri_convolutional_channels)-1))
        remaining_mad_branch_values = datapoints_per_mad_window * mad_convolutional_channels[-1] // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int(remaining_rri_branch_values) != remaining_rri_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")
        if int(remaining_mad_branch_values) != remaining_mad_branch_values:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        
        remaining_rri_branch_values = int(remaining_rri_branch_values)
        remaining_mad_branch_values = int(remaining_mad_branch_values)
        
        remaining_values_after_signal_learning = remaining_rri_branch_values + remaining_mad_branch_values

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
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int( self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
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
            datapoints_per_rri_window = 480, 
            datapoints_per_mad_window = 120,
            windows_per_signal = 1197,
            number_window_learning_features = 128,
            rri_convolutional_channels = [1, 8, 16, 32, 64],
            mad_convolutional_channels = [1, 8, 16, 32, 64],
            window_learning_dilations = [2, 4, 8, 16, 32],
            number_sleep_stages = 4
            ):
        """
        ARGUMENTS:
        ------------------------------
        datapoints_per_rri_window : int, optional
            Number of data points in each RRI window, by default 480
        datapoints_per_mad_window : int, optional
            Number of data points in each MAD window, by default 120
        windows_per_signal : int, optional
            Number of windows in each batch, by default 1197
        number_window_learning_features : int, optional
            Number of features learned from Signal Learning, by default 128
        rri_convolutional_channels : list, optional
            Number of channels to process RRI signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        mad_convolutional_channels : list, optional
            Number of channels to process MAD signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        window_learning_dilations : list, optional
            dilations for convolutional layers during Window Learning, by default [2, 4, 6, 8]
        number_sleep_stages : int, optional
            Number of predictable sleep stages, by default 4
        """

        # check parameters:
        if len(mad_convolutional_channels) % 2 != 1 or len(mad_convolutional_channels) < 3:
            raise ValueError("Number of convolutional channels in MAD branch must be odd and more than 2.")
        if datapoints_per_rri_window % 2**(len(rri_convolutional_channels)-1) != 0:
            raise ValueError("Number of RRI datapoints per window must be dividable by 2^(number of RRI convolutional layers - 1) without rest.")
        if datapoints_per_mad_window % 2 ** ((len(rri_convolutional_channels)-1)/2) != 0:
            raise ValueError("Number of MAD datapoints per window must be dividable by 2^((number of MAD convolutional layers - 1)/2) without rest.")
        if rri_convolutional_channels[-1] != mad_convolutional_channels[-1]:
            raise ValueError("Number of channels in last convolutional layer of RRI and MAD branch must be equal.")
        if 2**(len(rri_convolutional_channels) - 1) / 2**((len(mad_convolutional_channels) - 1) / 2) != datapoints_per_rri_window / datapoints_per_mad_window:
            raise ValueError("Number of remaining values after Signal Learning must be equal for RRI and MAD branch. Adjust number of convolutional channels accordingly.")

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

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
            if num_channel_pos % 2 == 0:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the 
        # last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # MaxPooling is applied after each convolutional layer in RRI branch and after every second 
        # convolutional layer in MAD branch
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied)
        # -> datapoints after mad branch must be divided by (2 ** (number of pooling layers applied / 2))

        remaining_rri_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (len(rri_convolutional_channels)-1))
        remaining_mad_branch_values = datapoints_per_mad_window * mad_convolutional_channels[-1] // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int(remaining_rri_branch_values) != remaining_rri_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")
        if int(remaining_mad_branch_values) != remaining_mad_branch_values:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        
        remaining_rri_branch_values = int(remaining_rri_branch_values)
        remaining_mad_branch_values = int(remaining_mad_branch_values)
        
        remaining_values_after_signal_learning = remaining_rri_branch_values + remaining_mad_branch_values

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
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int( self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
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
            datapoints_per_rri_window = 480, 
            datapoints_per_mad_window = 120,
            windows_per_signal = 1197,
            number_window_learning_features = 128,
            rri_convolutional_channels = [1, 8, 16, 32, 64],
            mad_convolutional_channels = [1, 8, 16, 32, 64],
            window_learning_dilations = [2, 4, 8, 16, 32],
            number_sleep_stages = 4
            ):
        """
        ARGUMENTS:
        ------------------------------
        datapoints_per_rri_window : int, optional
            Number of data points in each RRI window, by default 480
        datapoints_per_mad_window : int, optional
            Number of data points in each MAD window, by default 120
        windows_per_signal : int, optional
            Number of windows in each signal, by default 1197
        number_window_learning_features : int, optional
            Number of features learned from Signal Learning, by default 128
        rri_convolutional_channels : list, optional
            Number of channels to process RRI signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        mad_convolutional_channels : list, optional
            Number of channels to process MAD signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        window_learning_dilations : list, optional
            dilations for convolutional layers during Window Learning, by default [2, 4, 6, 8]
        number_sleep_stages : int, optional
            Number of predictable sleep stages, by default 4
        """

        # check parameters:
        if len(mad_convolutional_channels) % 2 != 1 or len(mad_convolutional_channels) < 3:
            raise ValueError("Number of convolutional channels in MAD branch must be odd and more than 2.")
        if datapoints_per_rri_window % 2**(len(rri_convolutional_channels)-1) != 0:
            raise ValueError("Number of RRI datapoints per window must be dividable by 2^(number of RRI convolutional layers - 1) without rest.")
        if datapoints_per_mad_window % 2 ** ((len(rri_convolutional_channels)-1)/2) != 0:
            raise ValueError("Number of MAD datapoints per window must be dividable by 2^((number of MAD convolutional layers - 1)/2) without rest.")
        if rri_convolutional_channels[-1] != mad_convolutional_channels[-1]:
            raise ValueError("Number of channels in last convolutional layer of RRI and MAD branch must be equal.")
        if 2**(len(rri_convolutional_channels) - 1) / 2**((len(mad_convolutional_channels) - 1) / 2) != datapoints_per_rri_window / datapoints_per_mad_window:
            raise ValueError("Number of remaining values after Signal Learning must be equal for RRI and MAD branch. Adjust number of convolutional channels accordingly.")

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

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
            if num_channel_pos % 2 == 0:
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
        # -> datapoints before branch must be multiplied by the number of channels of the 
        # last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # MaxPooling is applied after each convolutional layer in RRI branch and after every second 
        # convolutional layer in MAD branch
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied)
        # -> datapoints after mad branch must be divided by (2 ** (number of pooling layers applied / 2))

        remaining_rri_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (len(rri_convolutional_channels)-1))
        remaining_mad_branch_values = datapoints_per_mad_window * mad_convolutional_channels[-1] // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int(remaining_rri_branch_values) != remaining_rri_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")
        if int(remaining_mad_branch_values) != remaining_mad_branch_values:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        
        remaining_rri_branch_values = int(remaining_rri_branch_values)
        remaining_mad_branch_values = int(remaining_mad_branch_values)
        
        remaining_values_after_signal_learning = remaining_rri_branch_values + remaining_mad_branch_values

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
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int( self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
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
            datapoints_per_rri_window = 480, 
            datapoints_per_mad_window = 120,
            windows_per_signal = 1197,
            number_window_learning_features = 128,
            rri_convolutional_channels = [1, 8, 16, 32, 64],
            mad_convolutional_channels = [1, 8, 16, 32, 64],
            window_learning_dilations = [2, 4, 8, 16, 32],
            number_sleep_stages = 4
            ):
        """
        ARGUMENTS:
        ------------------------------
        datapoints_per_rri_window : int, optional
            Number of data points in each RRI window, by default 480
        datapoints_per_mad_window : int, optional
            Number of data points in each MAD window, by default 120
        windows_per_signal : int, optional
            Number of windows in each signal, by default 1197
        number_window_learning_features : int, optional
            Number of features learned from Signal Learning, by default 128
        rri_convolutional_channels : list, optional
            Number of channels to process RRI signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        mad_convolutional_channels : list, optional
            Number of channels to process MAD signal by 1D-convolution, by default [1, 8, 16, 32, 64]
        window_learning_dilations : list, optional
            dilations for convolutional layers during Window Learning, by default [2, 4, 6, 8]
        number_sleep_stages : int, optional
            Number of predictable sleep stages, by default 4
        """

        # check parameters:
        if len(mad_convolutional_channels) % 2 != 1 or len(mad_convolutional_channels) < 3:
            raise ValueError("Number of convolutional channels in MAD branch must be odd and more than 2.")
        if datapoints_per_rri_window % 2**(len(rri_convolutional_channels)-1) != 0:
            raise ValueError("Number of RRI datapoints per window must be dividable by 2^(number of RRI convolutional layers - 1) without rest.")
        if datapoints_per_mad_window % 2 ** ((len(rri_convolutional_channels)-1)/2) != 0:
            raise ValueError("Number of MAD datapoints per window must be dividable by 2^((number of MAD convolutional layers - 1)/2) without rest.")
        if rri_convolutional_channels[-1] != mad_convolutional_channels[-1]:
            raise ValueError("Number of channels in last convolutional layer of RRI and MAD branch must be equal.")
        if 2**(len(rri_convolutional_channels) - 1) / 2**((len(mad_convolutional_channels) - 1) / 2) != datapoints_per_rri_window / datapoints_per_mad_window:
            raise ValueError("Number of remaining values after Signal Learning must be equal for RRI and MAD branch. Adjust number of convolutional channels accordingly.")

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_signal = windows_per_signal

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
            if num_channel_pos % 2 == 0:
                mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        =================================================
        Combining Features Obtained From Signal Learning
        =================================================
        """

        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the 
        # last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # MaxPooling is applied after each convolutional layer in RRI branch and after every second 
        # convolutional layer in MAD branch
        # -> datapoints after rri branch must be divided by (2 ** number of pooling layers applied)
        # -> datapoints after mad branch must be divided by (2 ** (number of pooling layers applied / 2))

        remaining_rri_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (len(rri_convolutional_channels)-1))
        remaining_mad_branch_values = datapoints_per_mad_window * mad_convolutional_channels[-1] // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int(remaining_rri_branch_values) != remaining_rri_branch_values:
            raise ValueError("Number of remaining values after RRI branch must be an integer. Something went wrong.")
        if int(remaining_mad_branch_values) != remaining_mad_branch_values:
            raise ValueError("Number of remaining values after MAD branch must be an integer. Something went wrong.")
        
        remaining_rri_branch_values = int(remaining_rri_branch_values)
        remaining_mad_branch_values = int(remaining_mad_branch_values)
        
        remaining_values_after_signal_learning = remaining_rri_branch_values + remaining_mad_branch_values

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
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** ((len(rri_convolutional_channels)-1)/2))

        if int( self.mad_values_after_signal_learning) != self.mad_values_after_signal_learning:
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
def train_loop(dataloader, model, device, loss_fn, optimizer_fn, lr_scheduler, current_epoch, batch_size):
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

    # get number of windows the signals are reshaped to
    windows_per_signal = model.windows_per_signal

    # set optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr_scheduler(current_epoch))

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    # variables to save accuracy progress
    train_loss, correct = 0, 0

    # variables to track progress
    num_batches = len(dataloader)
    total_number_predictions = 0
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

        # collect accuracy progress values
        this_correct_predicted = (pred.argmax(1) == slp).type(torch.float).sum().item()
        this_number_predictions = slp.shape[0]

        train_loss += loss.item()
        correct += this_correct_predicted
        total_number_predictions += this_number_predictions

        # print progress bar
        progress_bar.update(
            additional_info = f'Loss: {format_float(loss.item(), 3)} | Acc: {round(this_correct_predicted / this_number_predictions*100, 2)}%',
            )

        del this_correct_predicted
    
    train_loss /= num_batches
    correct /= total_number_predictions
    
    return train_loss, correct


# TESTING LOOP
def test_loop(dataloader, model, device, loss_fn, batch_size):
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

    # get number of windows the signals are reshaped to
    windows_per_signal = model.windows_per_signal

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    # variables to track progress
    num_batches = len(dataloader)
    total_number_predictions = 0
    print("\nCalculating Prediction Accuracy on Test Data:")
    progress_bar = DynamicProgressBar(total = len(dataloader.dataset), batch_size = batch_size)

    # variables to save accuracy progress
    test_loss, correct = 0, 0

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

            # collect accuracy values
            correct += (pred.argmax(1) == slp).type(torch.float).sum().item()
            total_number_predictions += slp.shape[0]

            # print progress bar
            progress_bar.update()

    test_loss /= num_batches
    correct /= total_number_predictions
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


# Example usage
if __name__ == "__main__":

    """
    --------------------------------------
    Preparing Random Data File For Testing
    --------------------------------------
    """
    print("\n\nPreparing random data file for testing...")
    print("="*40)
    # creating dataset file and data manager instance on it
    random_file_path = "Testing_NNM/random_data.pkl"
    random_data_manager = SleepDataManager(file_path = random_file_path)
    random_sleep_stage_labels = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifect": ["other"]}

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
    
    some_datapoint = random_data_manager.load(0)

    print("Shape of Signals:")
    print(f"RRI Signal: {some_datapoint["RRI"].shape}") # type: ignore
    try:
        print(f"MAD Signal: {some_datapoint["MAD"].shape}") # type: ignore
    except:
        pass
    print(f"SLP Signal: {some_datapoint["SLP"].shape}") # type: ignore
    
    del random_data_manager, random_sleep_stage_labels, some_datapoint

    print("="*40)


    """
    ---------------------------------
    Testing The Custom Dataset Class
    ---------------------------------
    """

    print("\n\nTesting the Custom Dataset Class...")
    print("="*40)
    print("")

    # Create dataset
    dataset = CustomSleepDataset(path_to_data = random_file_path, transform=ToTensor())

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # Iterate over dataloader and print shape of features and labels
    for batch, (rri, mad, slp) in enumerate(dataloader):
        # print shape of data and labels:
        print(f"Batch {batch}:")
        print("-"*40)
        print("RRI shape:", rri.shape)
        if mad[0] == "None":
            mad = None
            print(f"MAD shape: {mad}")
        else:
            print("MAD shape:", mad.shape)
        print("SLP shape:", slp.shape)
        print("")
    
    # delete data file
    os.remove(random_file_path)
    os.rmdir(os.path.split(random_file_path)[0])

    print("="*40)


    """
    ---------------------------------
    Testing The Neural Network Model
    ---------------------------------
    """

    print("\n\nTesting the Neural Network Model...")
    print("="*40)

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
    My network:
    """
    print("\nSleepStageModel:")

    # Define the Neural Network
    DCNN = SleepStageModel()
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
    print(output.shape)

    """
    Yaopeng-like network:
    """
    
    print("\nYao:")

    # Define the Neural Network
    DCNN = YaoModel()
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
    print(output.shape)

    print("="*40)