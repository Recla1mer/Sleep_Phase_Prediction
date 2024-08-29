import torch
import torch.nn as nn


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