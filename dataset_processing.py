import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np


def calculate_overlap(signal_length: int, number_windows: int, datapoints_per_window: int) -> float:
    """
    Calculate the overlap between windows of length 'datapoints_per_window' in a signal of
    length 'signal_length'.

    Assuming number_windows * datapoints_per_window > signal_length:
    ----------------------------------------------------------------
    new_datapoints_per_window = datapoints_per_window - window_overlap
    signal_length   = datapoints_per_window + (number_windows - 1) * new_datapoints_per_window
                    = datapoints_per_window + (number_windows - 1) * (datapoints_per_window - window_overlap)

    -> window_overlap = datapoints_per_window - (signal_length - datapoints_per_window) / (number_windows - 1)

    Returns:
    --------
    window_overlap: float
        The needed overlap between windows of size datapoints_per_window to fit into signal_length.

    Parameters:
    -----------
    signal_length: int
        The length of the signal.
    number_windows: int
        The number of windows.
    datapoints_per_window: int
        The number of datapoints in each window.
    """

    return datapoints_per_window - ((signal_length - datapoints_per_window) / (number_windows - 1))


def find_suitable_window_parameters(
        signal_length: int = 36000,
        number_windows_range: tuple = (1000, 1400),
        window_size_range: tuple = (120, 180),
        minimum_window_size_overlap_difference: int = 30
    ) -> None:
    """
    The data passed to the neural network will be split into windows of equal length that
    overlap each other.

    This function searches for suitable window parameters (number_windows, datapoints_per_window, 
    window_overlap) in a given range for a signal of length 'signal_length' and prints them to console.

    Returns:
    --------
    None, but prints suitable window parameters to console.

    Parameters:
    -----------
    signal_length: int
        The length of the signal.
    number_windows_range: tuple
        The range of number of windows to consider.
    window_size_range: tuple
        The range of window sizes to consider.
    minimum_window_size_overlap_difference: int
        The minimum difference between window_size and window_overlap.

    """

    window_parameters_found = False

    for num_win in range(number_windows_range[0], number_windows_range[1]):
        for win_size in range(window_size_range[0], window_size_range[1]):
            window_overlap = calculate_overlap(signal_length, num_win, win_size)
            if int(window_overlap) == window_overlap and win_size - window_overlap >= minimum_window_size_overlap_difference:
                if not window_parameters_found:
                    window_parameters_found = True
                    print(f"Suitable window parameters for signal of length: {signal_length}:")
                    print("-" * 55)
                print(f"Number of windows: {num_win}, Window size: {win_size}, Overlap: {window_overlap}")
    
    if not window_parameters_found:
        print("No suitable window parameters found. Expand search range.")


def interpolate_signal(signal: list, signal_frequency: float, target_frequency: float) -> np.ndarray:
    """
    This function uses linear interpolation to resample a signal to a target frequency.
    If the signal is of type int, the interpolated signal will be rounded to the nearest integer.

    Returns:
    --------
    interpolated_signal: list
        The resampled signal.

    Example:
    --------
    signal = [1., 2., 3., 4., 5.]
    signal_frequency = 1
    target_frequency = 2
    interpolated_signal = [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.]

    Parameters:
    -----------
    signal: list
        The signal to be resampled.
    signal_frequency: int
        The frequency of the input signal.
    target_frequency: int
        The frequency to resample the signal to.
    """

    signal = np.array(signal) # type: ignore
    signal_data_type = signal.dtype # type: ignore

    scale_factor = signal_frequency / target_frequency
    
    interpolated_signal =  np.interp(
        np.arange(0, len(signal), scale_factor),
        np.arange(0, len(signal)),
        signal
        )

    if signal_data_type == int:
        interpolated_signal = np.round(interpolated_signal).astype(int)
    
    return interpolated_signal


def calculate_optimal_shift_length(
        signal_length: int, 
        desired_length: int, 
        wanted_shift_length: float,
        absolute_shift_deviation: float,
    ) -> float:
    """
    The neural network must always be applied with a signal of the same length. If the signal is longer
    than the desired length, it will be split into multiple signals of the desired length. To create more data, 
    the signal will only be shifted by a certain amount. 
    
    This function calculates the optimal shift length. It tries to find the shift length that is closest to the
    wanted shift length, but still within the allowed deviation. If a shift length within the deviation is an
    integer, it will be preferred over a non-integer shift length.

    Returns:
    --------
    best_possible_shift_length: float
        The optimal shift length.
    
    Parameters:
    -----------
    signal_length: int
        The length of the signal.
    desired_length: int
        The desired length of the signal.
    wanted_shift_length: float
        The shift length that is desired.
    absolute_shift_deviation: float
        The allowed deviation from the wanted shift length.
    """

    # Calculate min and max shift length
    min_shift = wanted_shift_length - absolute_shift_deviation
    max_shift = wanted_shift_length + absolute_shift_deviation

    # Initialize variables
    number_shifts = 1

    collect_shift_length = []
    collect_shift_deviation = []

    integer_shift_lengths = []
    integer_shift_length_deviation = []

    # Find all possible shift lengths:
    # The idea is that the number of shifts is always increased by 1, until the shift length is 
    # smaller than the minimum shift length.
    while True:
        current_shift_length = (signal_length-desired_length) / number_shifts
        current_shift_deviation = abs(current_shift_length - wanted_shift_length)

        if current_shift_length < min_shift:
            break

        if current_shift_length <= max_shift:
            collect_shift_length.append(current_shift_length)
            collect_shift_deviation.append(current_shift_deviation)
            if int(current_shift_length) == current_shift_length:
                integer_shift_lengths.append(current_shift_length)
                integer_shift_length_deviation.append(current_shift_deviation)

        number_shifts += 1
    
    if len(collect_shift_deviation) == 0:
        return 0
    
    # Find the shift length with the smallest deviation, prioritize integer shift lengths
    if len(integer_shift_lengths) > 0:
        best_possible_shift_length = integer_shift_lengths[integer_shift_length_deviation.index(min(integer_shift_length_deviation))]
    else:
        best_possible_shift_length = collect_shift_length[collect_shift_deviation.index(min(collect_shift_deviation))]

    return best_possible_shift_length


def split_long_signal(
        signal: list, 
        sampling_frequency: int,
        target_frequency: int,
        nn_signal_duration_seconds: int = 10*3600,
        wanted_shift_length_seconds: int = 3600,
        absolute_shift_deviation_seconds: int = 1800
    ) -> tuple[np.ndarray, int]:
    """
    If the signal is longer than nn_signal_duration_seconds, the signal will be split into multiple signals of 
    length 'nn_signal_duration_seconds'. The signal will only be shifted by a certain amount though, to
    create more data.

    Attention:  If sampling_frequency is not equal to target_frequency, the signal will be resampled to the
                target frequency, before splitting.
    
    Returns:
    --------
    splitted_signals: np.ndarray
        The signal split into multiple signals.
    optimal_shift_length: int
        The optimal shift length that was used to split the signal.
    
    Parameters:
    -----------
    signal: list
        The signal to be split into multiple signals.

    sampling_frequency: int
        The frequency of the input signal.
    target_frequency: int
        The frequency to resample the signal to. Frequency of signal in the neural network.
    
    nn_signal_duration_seconds: int
        The duration of the signal that will be passed to the neural network.
    wanted_shift_length_seconds: int
        The shift length that is desired by user.
    absolute_shift_deviation_seconds: int
        The allowed deviation from the wanted shift length.
    """
    
    # Check parameters
    if absolute_shift_deviation_seconds > wanted_shift_length_seconds:
        raise ValueError("The absolute shift deviation must be smaller than the wanted shift length.")
    if absolute_shift_deviation_seconds < 0:
        raise ValueError("The absolute shift deviation must be a positive number.")

    signal = np.array(signal) # type: ignore

    # Scale number of datapoints in signal if sampling frequency is not equal to target frequency
    if sampling_frequency != target_frequency:
        signal = interpolate_signal(
            signal = signal, # type: ignore
            signal_frequency = sampling_frequency, 
            target_frequency = target_frequency
            )
        
    # Calculate number of datapoints from signal length in seconds
    number_nn_datapoints = int(nn_signal_duration_seconds * target_frequency)

    # Check if signal is shorter than those that will be passed to the neural network
    if len(signal) <= number_nn_datapoints:
        return np.array([signal]), 0
    
    splitted_signals = np.empty((0, number_nn_datapoints), signal.dtype) # type: ignore

    # Calculate optimal shift length
    optimal_shift_length = calculate_optimal_shift_length(
        signal_length = len(signal),
        desired_length = number_nn_datapoints,
        wanted_shift_length = wanted_shift_length_seconds * target_frequency,
        absolute_shift_deviation = absolute_shift_deviation_seconds * target_frequency
        )
    
    # to ensure that we not miss any datapoints by reducing the shift length to nearest integer,
    # we will not perform the last shift within the loop, but instead use the last 'number_nn_datapoints'
    # of the signal
    optimal_shift_length = int(optimal_shift_length)
    
    # Split signal into multiple signals by shifting, transform to windows and append to reshaped_signals
    for start_index in range(0, len(signal)-number_nn_datapoints-optimal_shift_length+1, optimal_shift_length):
        splitted_signals = np.append(splitted_signals, [signal[start_index:start_index+number_nn_datapoints]], axis=0)
    
    # Append last 'number_nn_datapoints' of signal to reshaped_signals
    splitted_signals = np.append(splitted_signals, [signal[-number_nn_datapoints:]], axis=0)
    
    return splitted_signals, optimal_shift_length


def signal_to_windows(
        signal: list,   
        datapoints_per_window: int,
        window_overlap: int,
        signal_type: str = "feature",
        priority_order: list = [0, 1, 2, 3, 4, 5]
    ) -> np.ndarray:
    """
    This function splits a signal into windows of length 'datapoints_per_window' that overlap by 
    'window_overlap'.

    Returns:
    --------
    windows: np.ndarray
        The signal split into overlapping windows.
    
    Parameters:
    -----------
    signal: np.ndarray
        The signal to be split into windows.
    datapoints_per_window: int
        The number of datapoints in each window.
    window_overlap: int
        The number of overlapping datapoints between windows.
    signal_type: str
        The type of signal. Either 'feature' or 'target'.
            If 'feature':   The windows will be 2D arrays that will contain the same datapoints as the signal,
                            works like described above.
            If 'target':    The signal is assumed to contain classification labels. Each window will be reduced
                            to represent a single value (i.e., the 'label' that was most common in the window).
                            The returned windows will therefore be 1D arrays.
    priority_order: list
        The order in which labels should be prioritized in case of a tie. Only relevant if signal_type = 'target'.
    
    Attention: signal_type = 'target' is a very specific transformation, only useful for our classification task.
    """

    signal = np.array(signal) # type: ignore
    step_size = datapoints_per_window - window_overlap

    # Check if signal_type is valid
    if signal_type not in ["feature", "target"]:
        raise ValueError("Parameter 'mode' must be either 'signal' or 'target'.")

    # Initialize windows
    if signal_type == "feature":
        windows = np.empty((0, datapoints_per_window), signal.dtype) # type: ignore
    elif signal_type == "target":
        windows = np.empty((0), signal.dtype) # type: ignore
    
    # Split signal into windows
    for i in range(0, len(signal)-datapoints_per_window+1, step_size):
        this_window = signal[i:i+datapoints_per_window]
        
        if signal_type == "feature":
            windows = np.append(windows, [this_window], axis=0)

        elif signal_type == "target":
            # collect unique labels and their counts
            different_labels, label_counts = np.unique(this_window, return_counts=True)

            # remove labels that did not appear the most
            max_count = max(label_counts)
            most_common_labels = different_labels[label_counts == max_count]

            # prioritize labels in priority_order
            not_appended = True
            for class_label in priority_order:
                if class_label in most_common_labels:
                    not_appended = False
                    windows = np.append(windows, class_label)
                    break
            
            if not_appended:
                print(f"\nWARNING: No label found in priority order. Appending first label. Better terminate and recheck priority_order.\n Labels: {most_common_labels}")
                windows = np.append(windows, most_common_labels[0])
    
    return windows


def reshape_signal(
        signal: list, 
        sampling_frequency: float,
        target_frequency: float, 
        number_windows: int = 1197, 
        window_duration_seconds: int = 120, 
        overlap_seconds: int = 90,
        signal_type: str = "feature",
        nn_signal_duration_seconds: int = 10*3600,
    ) -> np.ndarray:
    """
    Reshape a signal with shape (n < nn_signal_duration_seconds * target_frequency) to 
    (number_windows, window_size), where windows overlap by 'overlap_seconds' and adjust the signal to the
    neural network's requirements.

    Returns:
    --------
    reshaped_signals: list
        The signal split into overlapping windows.
    
    Parameters:
    -----------
    signal: list
        The signal to be split into windows.
    
    If signal was recorded with a different frequency than the neural network expects, the signal will be
    resampled to the target frequency, using following parameters:

    sampling_frequency: int
        The frequency of the input signal.
    target_frequency: int
        The frequency to resample the signal to. Frequency of signal in the neural network.
    
    The signal will be split into 'number_windows' windows of length 'window_duration_seconds' that 
    overlap by 'overlap_seconds', needing the following parameters:

    number_windows: int
        The number of windows to split the signal into.
    window_duration_seconds: int
        The window length in seconds.
    overlap_seconds: int
        The overlap between windows in seconds.
    signal_type: str
        The type of signal. Either 'feature' or 'target'.
            If 'feature':   The windows will be 2D arrays that will contain the same datapoints as the signal,
                            works like described above.
            If 'target':    The signal is assumed to contain classification labels. Each window will be reduced
                            to represent a single value (i.e., the 'label' that was most common in the window).
                            The returned windows will therefore be 1D arrays.
    """
    
    signal = np.array(signal) # type: ignore

    # Scale number of datapoints in signal if sampling frequency is not equal to target frequency
    if sampling_frequency != target_frequency:
        signal = interpolate_signal(
            signal = signal, # type: ignore
            signal_frequency = sampling_frequency, 
            target_frequency = target_frequency
            )
        
    # Calculate number of datapoints from signal length in seconds
    number_nn_datapoints = nn_signal_duration_seconds * target_frequency
    datapoints_per_window = window_duration_seconds * target_frequency
    window_overlap = overlap_seconds * target_frequency
    
    # Check parameters
    if int(number_nn_datapoints) != number_nn_datapoints:
        raise ValueError("Number of datapoints must be an integer. Choose 'nn_signal_duration_seconds' and 'target_frequency' accordingly.")
    number_nn_datapoints = int(number_nn_datapoints)

    if int(datapoints_per_window) != datapoints_per_window:
        raise ValueError("Datapoints per window must be an integer. Choose 'window_duration_seconds' and 'target_frequency' accordingly.")
    datapoints_per_window = int(datapoints_per_window)

    if int(window_overlap) != window_overlap:
        raise ValueError("Window overlap must be an integer. Choose 'overlap_seconds' and 'target_frequency' accordingly.")
    window_overlap = int(window_overlap)

    if len(signal) > number_nn_datapoints:
        print("\nWARNING: Signal is longer than required for the neural network. Better terminate program and rerun dataprocessing to split signals that are too long. For now, continuing with cropped signal.")
        signal = signal[:number_nn_datapoints]
    
    # Pad signal with zeros if signal is shorter than 'signal_duration_seconds'
    number_missing_datapoints = number_nn_datapoints - len(signal)
    signal = np.append(signal, [0 for i in range(number_missing_datapoints)]) # type: ignore

    # Reshape signal to windows
    signal_windows = signal_to_windows(
            signal = signal, # type: ignore
            datapoints_per_window = datapoints_per_window,
            window_overlap = window_overlap, 
            signal_type = signal_type
            )
    
    # check if signal_windows has the correct shape
    if signal_type == "feature":
        if signal_windows.shape != (number_windows, datapoints_per_window):
            raise ValueError("Signal windows have wrong shape. Check parameters.")
    elif signal_type == "target":
        if len(signal_windows) != number_windows:
            raise ValueError("Signal windows have wrong shape. Check parameters.")
    
    return signal_windows


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
