"""
Author: Johannes Peter Knoll

Python File for implementing a custom PyTorch Dataset and a Deep Convolutional Neural Network for Sleep Stage 
Prediction.
"""

# IMPORTS:
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# LOCAL IMPORTS:
from dataset_processing import *


"""
------------------------------
Implementing a Custom Dataset
------------------------------
"""

class CustomSleepDataset(Dataset):
    def __init__(
            self, 
            path_to_data: str, 
            transform=None
        ):

        self.data_manager = SleepDataManager(path_to_data)
        self.transform = transform

    def __len__(self):
        return len(self.data_manager)

    def __getitem__(self, idx):
        # load dictionary with data from file using data_manager
        data_sample = self.data_manager.load(idx)

        # extract features from dictionary:
        rri_sample = data_sample["RRI_windows"] # type: ignore

        # mad not present in all files:
        try:
            mad_sample = data_sample["MAD_windows"] # type: ignore
        except:
            mad_sample = "None"

        # extract labels from dictionary:
        slp_labels = data_sample["SLP_windows"] # type: ignore

        if self.transform:
            rri_sample = self.transform(rri_sample)
            try:
                mad_sample = self.transform(mad_sample)
            except:
                pass

        return rri_sample, mad_sample, slp_labels


# conv, relu, conv, relu, pool
class SleepStageModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction

    Attention:  Number of convolutional channels (-1) must be smaller equals than the number of times 
                datapoints_per_mad_window and datapoints_per_rri_window are dividable by 2 without rest.
    """
    def __init__(
            self, 
            datapoints_per_rri_window = 480, 
            datapoints_per_mad_window = 120,
            windows_per_batch = 1197,
            number_window_learning_features = 128,
            rri_convolutional_channels = [1, 2, 4, 8],
            mad_convolutional_channels = [1, 2, 4, 8],
            window_learning_dilations = [2, 4, 6, 8],
            number_sleep_stages = 5
            ):
        """
        Parameters
        ----------
        datapoints_per_rri_window : int, optional
            Number of data points in each RRI window, by default 480
        datapoints_per_mad_window : int, optional
            Number of data points in each MAD window, by default 120
        windows_per_batch : int, optional
            Number of windows in each batch, by default 1197
        number_window_learning_features : int, optional
            Number of features learned from Signal Learning, by default 128
        rri_convolutional_channels : list, optional
            Number of channels to process RRI signal by 1D-convolution, by default [2, 4, 8, 16, 32, 64]
        mad_convolutional_channels : list, optional
            Number of channels to process MAD signal by 1D-convolution, by default [2, 4, 8, 16, 32, 64]
        window_learning_dilations : list, optional
            dilations for convolutional layers during Window Learning, by default [2, 4, 6, 8]
        number_sleep_stages : int, optional
            Number of predictable sleep stages, by default 5
        
        """
        # check parameters:
        if datapoints_per_rri_window % 2**(len(rri_convolutional_channels)-1) != 0:
            raise ValueError("Number of RRI datapoints per window must be dividable by 2^(number of RRI convolutional layers - 1) without rest.")
        if datapoints_per_mad_window % 2**(len(mad_convolutional_channels)-1) != 0:
            raise ValueError("Number of MAD datapoints per window must be dividable by 2^(number of MAD convolutional layers - 1) without rest.")
        if rri_convolutional_channels[-1] != mad_convolutional_channels[-1]:
            raise ValueError("Number of channels in last convolutional layer of RRI and MAD branch must be equal.")

        self.datapoints_per_rri_window = datapoints_per_rri_window
        self.datapoints_per_mad_window = datapoints_per_mad_window
        self.windows_per_batch = windows_per_batch

        super(SleepStageModel, self).__init__()

        # Parameters
        rri_branch_convolutional_kernel_size = 3
        rri_branch_max_pooling_kernel_size = 2

        mad_branch_convolutional_kernel_size = 3
        mad_branch_max_pooling_kernel_size = 2

        window_branch_convolutional_kernel_size = 7

        """
        Signal Feature Learning
        """

        # RRI branch:

        # Create layer structure for RRI branch
        rri_branch_layers = []
        for num_channel_pos in range(0, len(rri_convolutional_channels) - 1):
            # Convolutional layer:
            rri_branch_layers.append(nn.Conv1d(
                in_channels = rri_convolutional_channels[num_channel_pos], 
                out_channels = rri_convolutional_channels[num_channel_pos + 1], 
                kernel_size = rri_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            rri_branch_layers.append(nn.ReLU())
            # Pooling layer:
            rri_branch_layers.append(nn.MaxPool1d(kernel_size=rri_branch_max_pooling_kernel_size))
            # Batch normalization:
            rri_branch_layers.append(nn.BatchNorm1d(rri_convolutional_channels[num_channel_pos + 1]))

        self.rri_signal_learning = nn.Sequential(*rri_branch_layers)

        # MAD branch:

        # Create layer structure for MAD branch
        mad_branch_layers = []
        for num_channel_pos in range(0, len(mad_convolutional_channels) - 1):
            # Convolutional layer:
            mad_branch_layers.append(nn.Conv1d(
                in_channels = mad_convolutional_channels[num_channel_pos], 
                out_channels = mad_convolutional_channels[num_channel_pos + 1], 
                kernel_size = mad_branch_convolutional_kernel_size, 
                padding='same'
                ))
            # Activation function:
            mad_branch_layers.append(nn.ReLU())
            # Pooling layer:
            mad_branch_layers.append(nn.MaxPool1d(kernel_size=mad_branch_max_pooling_kernel_size))
            # Batch normalization:
            mad_branch_layers.append(nn.BatchNorm1d(mad_convolutional_channels[num_channel_pos + 1]))
        
        self.mad_signal_learning = nn.Sequential(*mad_branch_layers)

        """
        Combining features obtained from Signal Learning
        """
        # Calculating number of remaining values after each branch: 

        # Padding is chosen so that conv layer does not change size 
        # -> datapoints before branch must be multiplied by the number of channels of the 
        # last conv layer

        # MaxPooling is chosen so that the size of the data is halved
        # -> datapoints after branch must be divided by (2 ** number of pooling layers applied)

        remaining_rri_branch_values = datapoints_per_rri_window * rri_convolutional_channels[-1] // (2 ** (len(rri_convolutional_channels) - 1))
        remaining_mad_branch_values = datapoints_per_mad_window * mad_convolutional_channels[-1] // (2 ** (len(mad_convolutional_channels) - 1))
        
        remaining_values_after_signal_learning = remaining_rri_branch_values + remaining_mad_branch_values

        self.flatten = nn.Flatten()

        """
        Window Feature Learning
        """
        # Fully connected layer after concatenation
        self.linear = nn.Linear(remaining_values_after_signal_learning, number_window_learning_features)
        
        # Create layer structure for Window Feature Learning
        window_feature_learning_layers = []
        for dilation in window_learning_dilations:
            # Residual block:
            window_feature_learning_layers.append(nn.ReLU())
            window_feature_learning_layers.append(nn.Conv1d(
                in_channels = number_window_learning_features, 
                out_channels = number_window_learning_features, 
                kernel_size = window_branch_convolutional_kernel_size, 
                dilation = dilation,
                padding='same'
                ))
            window_feature_learning_layers.append(nn.Dropout(0.2))
        
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
        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 + 32 * 5, 128),  # Adjust input size based on concatenated features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Assuming 5 sleep stages
        )
        """

        """
        Save MAD shape for data where MAD is not provided
        """
        self.mad_channels_after_signal_learning = mad_convolutional_channels[-1]
        self.mad_values_after_signal_learning = datapoints_per_mad_window // (2 ** (len(mad_convolutional_channels) - 1))


    def forward(self, rri_signal, mad_signal = None):
        """
        Checking and preparing data for forward pass
        """

        # Check Dimensions of RRI signal
        batch_size, _, num_windows_rri, samples_in_window_rri = rri_signal.size()
        assert samples_in_window_rri == self.datapoints_per_rri_window, f"Expected {self.datapoints_per_rri_window} data points in each RRI window, but got {samples_in_window_rri}."
        assert num_windows_rri == self.windows_per_batch, f"Expected {self.windows_per_batch} windows in each batch, but got {num_windows_rri}."

        # Reshape RRI signal
        rri_signal = rri_signal.view(batch_size * num_windows_rri, 1, samples_in_window_rri)  # Combine batch and windows dimensions
        # rri_signal = rri_signal.reshape(-1, 1, samples_in_window_rri) # analogous to the above line

        if mad_signal is not None:
            # Check Dimensions of MAD signal
            _, _, num_windows_mad, samples_in_window_mad = mad_signal.size()
            assert samples_in_window_mad == self.datapoints_per_mad_window, f"Expected {self.datapoints_per_mad_window} data points in each MAD window, but got {samples_in_window_mad}."
            assert num_windows_mad == self.windows_per_batch, f"Expected {self.windows_per_batch} windows in each batch, but got {num_windows_mad}."

            # Reshape MAD signal
            mad_signal = mad_signal.view(batch_size * num_windows_mad, 1, samples_in_window_mad)  # Combine batch and windows dimensions

        """
        Signal Feature Learning
        """

        # Process RRI Signal
        rri_features = self.rri_signal_learning(rri_signal)
        #ecg_features = ecg_features.view(batch_size, num_windows, -1)  # Separate batch and windows dimensions

        # Process MAD Signal or create 0 tensor if MAD signal is not provided
        if mad_signal is None:
            mad_features = torch.zeros(batch_size * num_windows_mad, self.mad_channels_after_signal_learning, self.mad_values_after_signal_learning, device=rri_signal.device)
        else:
            mad_features = self.mad_signal_learning(mad_signal)
        
        """
        Create Window Features
        """

        # Concatenate features
        window_features = torch.cat((rri_features, mad_features), dim=-1)

        # Flatten features
        window_features = self.flatten(window_features)

        """
        Window Feature Learning
        """

        # Fully connected layer
        output = self.linear(window_features)

        # Reshape for convolutional layers
        output = output.reshape(batch_size, self.windows_per_batch, -1)
        output = output.transpose(1, 2).contiguous()

        # Convolutional layers
        output = self.window_feature_learning(output)

        # Reshape for output
        output = output.transpose(1, 2).contiguous().reshape(batch_size * self.windows_per_batch, -1)

        return output

        """
        # Reshape for fully connected layers
        combined_features = combined_features.view(batch_size, -1)  # Combine windows and features dimensions

        # Fully connected layers
        output = self.fc(combined_features)
        return output
        """


# Example usage
if __name__ == "__main__":

    """
    --------------------------------------
    preparing random data file for testing
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
        random_data_manager.save(random_datapoint) # comment to test data without MAD signal
        #random_data_manager.save(random_datapoint_without_mad) # uncomment to test data without MAD signal
    
    # transforming all data in file to windows
    random_data_manager.transform_signals_to_windows(
        number_windows = 1197, 
        window_duration_seconds = 120, 
        overlap_seconds = 90, 
        priority_order = [0, 1, 2, 3, 5, -1]
        )
    
    some_datapoint = random_data_manager.load(0)

    print("Shape of Signals (before / after):")
    print(f"RRI Signal: {some_datapoint["RRI"].shape} / {some_datapoint["RRI_windows"].shape}") # type: ignore
    try:
        print(f"MAD Signal: {some_datapoint["MAD"].shape} / {some_datapoint["MAD_windows"].shape}") # type: ignore
    except:
        pass
    print(f"SLP Signal: {some_datapoint["SLP"].shape} / {some_datapoint["SLP_windows"].shape}") # type: ignore
    
    del random_data_manager, random_sleep_stage_labels, some_datapoint

    print("="*40)


    """
    ---------------------------------
    Testing the Custom Dataset Class
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
    Testing the Neural Network Model
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

    # Define the Neural Network
    DCNN = SleepStageModel()
    DCNN.to(device)

    # Create example data
    rri_example = torch.rand((2, 1, 1197, 480), device=device)
    mad_example = torch.rand((2, 1, 1197, 120), device=device)

    # Pass data through the model
    output = DCNN(rri_example, mad_example)
    print(output.shape)

    print("="*40)