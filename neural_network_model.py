"""
Author: Johannes Peter Knoll

Python File for implementing a custom PyTorch Dataset and a Deep Convolutional Neural Network for Sleep Stage 
Prediction.
"""

# IMPORTS:
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# LOCAL IMPORTS:
from dataset_processing import *


class CustomArrayDataset(Dataset):
    def __init__(self, ecg_data, mad_data, labels, transform=None):
        self.ecg_data = ecg_data
        self.mad_data = mad_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_sample = self.ecg_data[idx]
        mad_sample = self.mad_data[idx]
        labels = self.labels[idx]

        if self.transform:
            ecg_sample = self.transform(ecg_sample)
            mad_sample = self.transform(mad_sample)

        return ecg_sample, mad_sample, labels


# Example usage
if __name__ == "__main__":
    # Sample ECG data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (6,))
    ecg_data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                          [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                          [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
                         [[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                          [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                          [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                          [49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                          [55.0, 56.0, 57.0, 58.0, 59.0, 60.0]]])

    # Sample MAD data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (3,))
    mad_data = np.array([[[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9],
                          [1.0, 1.1, 1.2],
                          [1.3, 1.4, 1.5]],
                         [[1.6, 1.7, 1.8],
                          [1.9, 2.0, 2.1],
                          [2.2, 2.3, 2.4],
                          [2.5, 2.6, 2.7],
                          [2.8, 2.9, 3.0]]])

    # Corresponding labels for each window (e.g., 2 nights, each with 5 windows, each window with 1 label)
    labels = torch.tensor([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1]])

    # Create dataset
    dataset = CustomArrayDataset(ecg_data, mad_data, labels, transform=ToTensor())

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # for batch, (ecg, mad, y) in enumerate(dataloader):
    #     # print shape of data and labels:
    #     print(f"Batch {batch}:")
    #     print("ECG Data:", ecg)
    #     print("MAD Data:", mad)
    #     print("Labels:", y)
    #     print("ECG Data shape:", ecg.shape)
    #     print("MAD Data shape:", mad.shape)
    #     print("Labels shape:", y.shape)




    # for batch, (ecg, mad, y) in enumerate(dataloader):
    #     print(f"Batch {batch}:")
    #     print("ECG Data:", ecg)
    #     print("MAD Data:", mad)
    #     print("Labels:", y)
    #     print("ECG Data shape:", ecg.shape)
    #     print("MAD Data shape:", mad.shape)
    #     print("Labels shape:", y.shape)

    #     output = model(ecg, mad)
    #     print("Output shape:", output.shape)


        
    # # Sample ECG and MAD data
    # ecg_data = np.array([[[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
    #                       [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
    #                       [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
    #                       [49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
    #                       [55.0, 56.0, 57.0, 58.0, 59.0, 60.0]],
    #                      [[61.0, 62.0, 63.0, 64.0, 65.0, 66.0],
    #                       [67.0, 68.0, 69.0, 70.0, 71.0, 72.0],
    #                       [73.0, 74.0, 75.0, 76.0, 77.0, 78.0],
    #                       [79.0, 80.0, 81.0, 82.0, 83.0, 84.0],
    #                       [85.0, 86.0, 87.0, 88.0, 89.0, 90.0]]])

    # mad_data = np.array([[[0.1, 0.2, 0.3],
    #                       [0.4, 0.5, 0.6],
    #                       [0.7, 0.8, 0.9],
    #                       [1.0, 1.1, 1.2],
    #                       [1.3, 1.4, 1.5]],
    #                      [[1.6, 1.7, 1.8],
    #                       [1.9, 2.0, 2.1],
    #                       [2.2, 2.3, 2.4],
    #                       [2.5, 2.6, 2.7],
    #                       [2.8, 2.9, 3.0]]])

    # labels = torch.tensor([[0, 1, 0, 1, 0],
    #                        [1, 0, 1, 0, 1]])

    # class CustomArrayDataset(Dataset):
    #     def __init__(self, ecg_data, mad_data, labels, transform=None):
    #         self.ecg_data = ecg_data
    #         self.mad_data = mad_data
    #         self.labels = labels
    #         self.transform = transform

    #     def __len__(self):
    #         return len(self.ecg_data)

    #     def __getitem__(self, idx):
    #         ecg = self.ecg_data[idx]
    #         mad = self.mad_data[idx]
    #         label = self.labels[idx]
    #         if self.transform:
    #             ecg = self.transform(ecg)
    #             mad = self.transform(mad)
    #         return ecg, mad, label

    # class ToTensor:
    #     def __call__(self, sample):
    #         return torch.tensor(sample, dtype=torch.float32)

    # dataset = CustomArrayDataset(ecg_data, mad_data, labels, transform=ToTensor())

    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # model = SleepStageModel()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # for batch, (ecg, mad, y) in enumerate(dataloader):
    #     print(f"Batch {batch}:")
    #     print("ECG Data:", ecg)
    #     print("MAD Data:", mad)
    #     print("Labels:", y)
    #     print("ECG Data shape:", ecg.shape)
    #     print("MAD Data shape:", mad.shape)
    #     print("Labels shape:", y.shape)

    #     output = model(ecg, mad)
    #     print("Output shape:", output.shape)

"""
length = 1000
frequency = 1 / 30
target_freq = 1 / 30

import random

random_array = [random.randint(0, 5) for _ in range(length)]
reshaped_array = reshape_signal(
    signal = random_array, # type: ignore
    sampling_frequency = frequency,
    target_frequency = target_freq, 
    number_windows = 1197, 
    window_duration_seconds = 120, 
    overlap_seconds = 90,
    signal_type = "target",
    nn_signal_duration_seconds = 10*3600,
    )

print(f"Shape of new array: {reshaped_array.shape}")
print(f" Datapoints in new array: {reshaped_array.shape[0]}")
print(f"Unique Datapoints in new array: {120 * target_freq + (reshaped_array.shape[0] - 1) * (120 - 90) * target_freq}")
print(f" Datapoints in scaled original array: {length/frequency*target_freq}")
"""


# conv, relu, conv, relu, pool
class SleepStageModel(nn.Module):
    """
    Deep Convolutional Neural Network for Sleep Stage Prediction
    """
    def __init__(
            self, 
            datapoints_per_rri_window = 512, 
            datapoints_per_mad_window = 128,
            windows_per_batch = 1200,
            number_window_learning_features = 128,
            rri_convolutional_channels = [1, 2, 4, 8, 16, 32, 64],
            mad_convolutional_channels = [1, 2, 4, 8, 16, 32, 64],
            window_learning_dilations = [2, 4, 6, 8],
            number_sleep_stages = 5
            ):
        """
        Parameters
        ----------
        datapoints_per_rri_window : int, optional
            Number of data points in each RRI window, by default 512
        datapoints_per_mad_window : int, optional
            Number of data points in each MAD window, by default 128
        windows_per_batch : int, optional
            Number of windows in each batch, by default 1200
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
    rri_example = torch.rand((2, 1, 1200, 512), device=device)
    mad_example = torch.rand((2, 1, 1200, 128), device=device)

    # Pass data through the model
    output = DCNN(rri_example, mad_example)
    print(output.shape)