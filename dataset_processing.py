import numpy as np


"""

Section 1: Operations on Signals

"""


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


def scale_classification_signal(
        signal: list, 
        signal_frequency: float,
        target_frequency: float
    ) -> np.ndarray:
    """
    This function resamples a classification signal to a target frequency.

    Returns:
    --------
    target_signal: np.ndarray
        The resampled signal.
    
    Parameters:
    -----------
    signal: list
        The signal to be resampled.
    signal_frequency: float
        The frequency of the input signal.
    target_frequency: float
        The frequency to resample the signal to.
    """

    signal = np.array(signal) # type: ignore

    # Calculate number of datapoints in target signal
    signal_duration_seconds = len(signal) / signal_frequency
    datapoints_in_target_signal = round(signal_duration_seconds * target_frequency)

    # Assign time points to each datapoint
    original_signal_time = [i / signal_frequency for i in range(len(signal))]
    target_signal_time = [i / target_frequency for i in range(datapoints_in_target_signal)]

    # Initialize target signal
    target_signal = np.empty((0), signal.dtype) # type: ignore

    # Assign values to target signal (use nearest value in original signal)
    for time_point in target_signal_time:
        nearest_index = np.argmin(np.abs(np.array(original_signal_time) - time_point))
        target_signal = np.append(target_signal, signal[nearest_index])
    
    return target_signal


def scale_continuous_signal(
        signal: list, 
        signal_frequency: float,
        target_frequency: float
    ) -> np.ndarray:
    """
    This function resamples a continuous signal to a target frequency.

    Fun Fact: After testing it, I realized that this function is the same as 'interpolate_signal'.

    Returns:
    --------
    target_signal: np.ndarray
        The resampled signal.
    
    Parameters:
    -----------
    signal: list
        The signal to be resampled.
    signal_frequency: float
        The frequency of the input signal.
    target_frequency: float
        The frequency to resample the signal to.
    """

    signal = np.array(signal) # type: ignore
    signal_data_type = signal.dtype # type: ignore

    # Calculate number of datapoints in target signal
    signal_duration_seconds = len(signal) / signal_frequency
    datapoints_in_target_signal = round(signal_duration_seconds * target_frequency)

    # Assign time points to each datapoint
    original_signal_time = [i / signal_frequency for i in range(len(signal))]
    target_signal_time = [i / target_frequency for i in range(datapoints_in_target_signal)]

    # Initialize target signal
    target_signal = np.empty((0), signal.dtype) # type: ignore

    for time in target_signal_time:
        index_before = -1
        index_after = -1
        same_value_found = False

        for i in range(len(original_signal_time)):
            if original_signal_time[i] < time:
                index_before = i
            if original_signal_time[i] == time:
                target_signal = np.append(target_signal, signal[i])
                same_value_found = True
                break
            if original_signal_time[i] > time:
                index_after = i
                break
        
        if same_value_found:
            continue
        
        if index_before == -1:
            # Case: any signal_time > target_time -> target_time is before start of signal
            target_signal = np.append(target_signal, signal[0])
        elif index_after == -1:
            # Case: any signal_time < target_time -> target_time is after end of signal
            target_signal = np.append(target_signal, signal[-1])
        else:
            # Case: target_time is between two signal_time points
            time_before = original_signal_time[index_before]
            time_after = original_signal_time[index_after]
            value_before = signal[index_before]
            value_after = signal[index_after]

            target_value = value_before + (value_after - value_before) * ((time - time_before) / (time_after - time_before))
            target_signal = np.append(target_signal, target_value)
        
        if signal_data_type == int:
            target_signal = np.round(target_signal).astype(int)
    
    return target_signal


def scale_signal(
        signal: list, 
        signal_frequency: float,
        target_frequency: float,
        signal_type: str = "continuous"
    ) -> np.ndarray: # type: ignore
    """
    This function resamples a signal to a target frequency.

    Returns:
    --------
    target_signal: np.ndarray
        The resampled signal.
    
    Parameters:
    -----------
    signal: list
        The signal to be resampled.
    signal_frequency: float
        The frequency of the input signal.
    target_frequency: float
        The frequency to resample the signal to.
    signal_type: str
        The type of signal. Either 'continuous' or 'classification'.
    """

    signal = np.array(signal) # type: ignore

    if signal_frequency == target_frequency:
        return signal # type: ignore
    
    if signal_type == "continuous":
        return interpolate_signal(signal, signal_frequency, target_frequency)
    elif signal_type == "classification":
        return scale_classification_signal(signal, signal_frequency, target_frequency)


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

    Attention:  'sampling_frequency' must be equal to 'target_frequency'
    
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
    signal_type: str
        The type of signal. Either 'feature' or 'target'.
    
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
    if sampling_frequency != target_frequency:
        raise ValueError("Signal must be resampled to target frequency before splitting.")

    signal = np.array(signal) # type: ignore
        
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


def split_signals_within_dictionary(
        data_dict: dict,
        id_key: str,
        valid_signal_keys: list,
        signal_frequencies: list,
        signal_target_frequencies: list,
        nn_signal_duration_seconds: int,
        wanted_shift_length_seconds: int,
        absolute_shift_deviation_seconds: int):
    """
    You might handle with data where multiple signals are stored in a dictionary. If those signals are too long,
    they all need to be split, using 'split_long_signal'. After splitting, you have more datapoints than before, 
    for which you need to create new dictionaries with different ID's.

    This function splits all signals in a dictionary that are too long and creates new dictionaries for each
    splitted signal with a unique ID.

    ATTENTION: The order of the signals in 'valid_signal_keys' must be the same as the order of the frequencies
    in 'signal_frequencies' and 'signal_target_frequencies'.

    Returns:
    --------
    new_dictionaries_for_splitted_signals: list
        The new dictionaries for the splitted signals.
    
    Parameters:
    -----------
    data_dict: dict
        The dictionary containing the signals.
    id_key: str
        The key of the ID in the dictionary.
    valid_signal_keys: list
        The keys of the signals that should be split.
    signal_frequencies: list
        The frequencies of the above signals.
    signal_target_frequencies: list
        The target frequencies of the above signals.
    nn_signal_duration_seconds: int
        The duration of the signal that will be passed to the neural network.
    wanted_shift_length_seconds: int
        The shift length that is desired by user.
    absolute_shift_deviation_seconds: int
        The allowed deviation from the wanted shift length.
    """
    splitted_signals = list()

    # split signals if they are too long
    for signal_key_index in range(0, len(valid_signal_keys)):
        signal_key = valid_signal_keys[signal_key_index]

        this_splitted_signal, this_shift_length = split_long_signal(
            signal = data_dict[signal_key], # type: ignore
            sampling_frequency = signal_frequencies[signal_key_index],
            target_frequency = signal_target_frequencies[signal_key_index],
            nn_signal_duration_seconds = nn_signal_duration_seconds,
            wanted_shift_length_seconds = wanted_shift_length_seconds,
            absolute_shift_deviation_seconds = absolute_shift_deviation_seconds
            )

        splitted_signals.append(this_splitted_signal)
        this_shift_length_seconds = int(this_shift_length / signal_target_frequencies[signal_key_index])
        del this_splitted_signal

    # create new dictionaries for splitted signals
    new_dictionaries_for_splitted_signals = list()
    for i in range(0, len(splitted_signals[0])):
        splitted_data_dict = dict()
        for key in data_dict:
            if key in valid_signal_keys:
                splitted_data_dict[key] = splitted_signals[valid_signal_keys.index(key)][i]
            else:
                splitted_data_dict[key] = data_dict[key]
        if i > 0:
            splitted_data_dict[id_key] = f"{data_dict[id_key]}_shift_{this_shift_length_seconds}s_x{i}"
        new_dictionaries_for_splitted_signals.append(splitted_data_dict)
    
    return new_dictionaries_for_splitted_signals


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


def reshape_signal_to_overlapping_windows(
        signal: list, 
        sampling_frequency: float,
        target_frequency: float, 
        nn_signal_duration_seconds: int = 10*3600,
        pad_with = 0,
        number_windows: int = 1197, 
        window_duration_seconds: int = 120, 
        overlap_seconds: int = 90,
        signal_type: str = "feature",
        priority_order: list = [0, 1, 2, 3, 4, 5],
    ) -> np.ndarray:
    """
    Reshape a signal with shape (n <= nn_signal_duration_seconds * target_frequency) to 
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
    
    -----
    Attention: 'sampling_frequency' must match 'target_frequency' before transforming is useful. 
    Check uses following parameters:
    -----

    sampling_frequency: int
        The frequency of the input signal.
    target_frequency: int
        The frequency to resample the signal to. Frequency of signal in the neural network.
    
    -----
    If the signal is shorter than 'nn_signal_duration_seconds', it will be padded with 'pad_with' to match
    the required length, needing the following parameters:
    -----

    nn_signal_duration_seconds: int
        The duration of the signal that will be passed to the neural network.
    pad_with: int
        The value to pad the signal with, if it is shorter than 'nn_signal_duration_seconds'.
    
    -----
    The signal will be split into 'number_windows' windows of length 'window_duration_seconds' that 
    overlap by 'overlap_seconds', needing the following parameters:
    -----

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
    priority_order: list
        The order in which labels should be prioritized in case of a tie. Only relevant if signal_type = 'target
    """
    
    signal = np.array(signal) # type: ignore
        
    # Calculate number of datapoints from signal length in seconds
    number_nn_datapoints = nn_signal_duration_seconds * target_frequency
    datapoints_per_window = window_duration_seconds * target_frequency
    window_overlap = overlap_seconds * target_frequency
    
    # Check parameters
    if sampling_frequency != target_frequency:
        raise ValueError("Signal must be resampled to target frequency before reshaping.")
    
    if int(number_nn_datapoints) != number_nn_datapoints:
        raise ValueError("Number of datapoints must be an integer. Choose 'nn_signal_duration_seconds' and 'target_frequency' accordingly.")
    number_nn_datapoints = int(number_nn_datapoints)

    if int(datapoints_per_window) != datapoints_per_window:
        raise ValueError("Datapoints per window must be an integer. Choose 'window_duration_seconds' and 'target_frequency' accordingly.")
    datapoints_per_window = int(datapoints_per_window)

    if int(window_overlap) != window_overlap:
        raise ValueError("Window overlap must be an integer. Choose 'overlap_seconds' and 'target_frequency' accordingly.")
    window_overlap = int(window_overlap)

    check_overlap = calculate_overlap(
        signal_length = number_nn_datapoints, 
        number_windows = number_windows, 
        datapoints_per_window = datapoints_per_window
        )
    
    if window_overlap != check_overlap:
        raise ValueError("Overlap does not match the number of windows and datapoints per window. Check parameters.")
    
    del check_overlap

    if len(signal) > number_nn_datapoints:
        print("\nWARNING: Signal is longer than required for the neural network. Better terminate program and rerun dataprocessing to split signals that are too long. For now, continuing with cropped signal.")
        signal = signal[:number_nn_datapoints]
    
    # Pad signal with zeros if signal is shorter than 'signal_duration_seconds'
    number_missing_datapoints = number_nn_datapoints - len(signal)
    signal = np.append(signal, [pad_with for i in range(number_missing_datapoints)]) # type: ignore

    # Reshape signal to windows
    signal_windows = signal_to_windows(
            signal = signal, # type: ignore
            datapoints_per_window = datapoints_per_window,
            window_overlap = window_overlap, 
            signal_type = signal_type,
            priority_order = priority_order
            )
    
    # check if signal_windows has the correct shape
    if signal_type == "feature":
        if signal_windows.shape != (number_windows, datapoints_per_window):
            raise ValueError("Signal windows have wrong shape. Check parameters.")
    elif signal_type == "target":
        if len(signal_windows) != number_windows:
            raise ValueError("Signal windows have wrong shape. Check parameters.")
    
    return signal_windows


def alter_slp_labels(
        shhs_labels: list,
        transformation: dict
    ) -> np.ndarray:
    """
    Alter the labels that classify the sleep stage.

    Returns:
    --------
    altered_labels: list
        The altered labels.
    
    Parameters:
    -----------
    labels: list
        The labels to be altered.
    transformation: dict
        The transformation that should be applied to the labels. The keys are the original labels and the values
        are the new labels. Both keys and values must be strings. If the original labels are integers, the altered
        labels will be converted back
    """
    shhs_labels = np.array(shhs_labels) # type: ignore
    signal_data_type = shhs_labels.dtype # type: ignore
    altered_labels = np.empty((0), str) # type: ignore

    for label in shhs_labels:
        altered_labels = np.append(altered_labels, transformation[str(label)])
    
    if signal_data_type == int:
        altered_labels = altered_labels.astype(int)
    
    return altered_labels


"""

Section 2: Managing Data

"""

import pickle
import os
import inspect

def save_to_pickle(data, file_name):
    """
    Save data to a pickle file, overwriting the file if it already exists.

    ARGUMENTS:
    --------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    None
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def append_to_pickle(data, file_name):
    """
    Append data to a pickle file, without deleting previous data.

    ARGUMENTS:
    --------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    None
    """
    with open(file_name, "ab") as f:
        pickle.dump(data, f)


def load_from_pickle(file_name: str):
    """
    Load data from a pickle file as a generator.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the pickle file
    key: str
        key of the data to be loaded
    
    RETURNS:
    --------------------------------
    any
        data from the pickle file
    """
    # with open(file_name, "rb") as f:
    #     data = pickle.load(f)
    # return data
    with open(file_name, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def find_non_existing_path(path_without_file_type: str, file_type: str = "pkl"):
    """
    Find a path that does not exist yet by adding a number to the end of the path.

    ARGUMENTS:
    --------------------------------
    path_without_file_type: str
        path without the file type
    file_type: str
        file type of the file to be saved
    
    RETURNS:
    --------------------------------
    str
        path that does not exist yet
    """
    if not os.path.exists(f"{path_without_file_type}.{file_type}"):
        return f"{path_without_file_type}.{file_type}"
    i = 0
    while os.path.exists(f"{path_without_file_type}_{i}.{file_type}"):
        i += 1
    return f"{path_without_file_type}_{i}.{file_type}"


class SleepDataManager:
    # Define class variables
    signal_keys = ["RRI", "MAD", "SLP"]
    signal_frequency_keys = ["RRI_frequency", "MAD_frequency", "SLP_frequency"] # same order is important
    
    valid_datapoint_keys = ["ID"]
    valid_datapoint_keys.extend(signal_keys)
    valid_datapoint_keys.extend(signal_frequency_keys)
    
    default_file_info = dict()
    default_file_info["RRI_frequency"] = 4
    default_file_info["MAD_frequency"] = 1
    default_file_info["SLP_frequency"] = 1/30
    default_file_info["signal_length_seconds"] = 36000
    default_file_info["wanted_shift_length_seconds"] = 3600
    default_file_info["absolute_shift_deviation_seconds"] = 1800
    default_file_info["contained_data"] = "main" # main (means data was not divided yet), train, validation, test
    default_file_info["main_file_path"] = "unassigned"
    default_file_info["signal_in_windows"] = False
    default_file_info["number_windows"] = 1197
    default_file_info["window_duration_seconds"] = 120
    default_file_info["overlap_seconds"] = 90


    def __init__(self, file_path):
        self.file_path = file_path

        # load general information from file
        file_generator = load_from_pickle(self.file_path)
        self.file_info = next(file_generator)
        del file_generator

    # Getter and setter for file_path
    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        self._file_path = value

        # Check if file exists, if not create it and save default file information
        if not os.path.exists(self._file_path):
            save_to_pickle(data = self.default_file_info, file_name = self._file_path)
    

    def _correct_datapoint(self, new_data):
        """
        The datapoint will be altered to match the file's signal frequencies and signal length. If the signal
        is too long, it will be split into multiple signals.
        """

        # INSPECT NEW DATAPOINT:
        # ----------------------

        # Check if new_data is a dictionary
        if not isinstance(new_data, dict):
            raise ValueError("The datapoint must be a dictionary!")
        
        # Check ID key was provided
        if "ID" not in new_data:
            raise ValueError("The ID key: \"ID\" must be provided in the datapoint.")
        
        # Check if ID key is misleading
        if new_data["ID"] in self.valid_datapoint_keys:
            raise ValueError("Value for ID key: \"ID\" must not be the same as a key in the datapoint dictionary!")
        
        # Check if key in new_data is unknown
        for new_data_key in new_data:
            if new_data_key not in self.valid_datapoint_keys:
                raise ValueError(f"Unknown key in datapoint: {new_data_key}")
        
        # Check if frequency keys are provided if signal keys are provided
        for signal_key_index in range(0, len(self.signal_keys)):
            signal_key = self.signal_keys[signal_key_index]
            signal_frequency_key = self.signal_frequency_keys[signal_key_index]
            if signal_key in new_data and signal_frequency_key not in new_data:
                raise ValueError(f"If you want to add a {signal_key} Signal (key: \"{signal_key}\"), then you must also provide the sampling frequency: \"{signal_frequency_key}\" !")
            
        
        # ALTER ENTRIES IN NEW DATAPOINT:
        # -------------------------------

        # make sure sampling frequency matches the one in the file, rescale signal if necessary
        for signal_key_index in range(0, len(self.signal_keys)):
            signal_key = self.signal_keys[signal_key_index]
            signal_frequency_key = self.signal_frequency_keys[signal_key_index]
            if signal_key in new_data and new_data[signal_frequency_key] != self.file_info[signal_frequency_key]:
                this_signal_type = "classification" if signal_key == "SLP" else "continuous"
                new_data[signal_key] = scale_signal(
                    signal = new_data[signal_key],
                    signal_frequency = new_data[signal_frequency_key],
                    target_frequency = self.file_info[signal_frequency_key],
                    signal_type = this_signal_type
                    )
                del this_signal_type
                new_data[signal_frequency_key] = self.file_info[signal_frequency_key]


        # make sure signal length is not longer than the one in the file
            # check if signal is too long
        split_signals_needed = False
        for signal_key_index in range(0, len(self.signal_keys)):
            signal_key = self.signal_keys[signal_key_index]
            signal_frequency_key = self.signal_frequency_keys[signal_key_index]
            if signal_key in new_data:   
                if len(new_data[signal_key]) > self.file_info["signal_length_seconds"] * new_data[signal_frequency_key]:
                    split_signals_needed = True
                    break
                
            # split signals in dictionary and create new data dictionaries with unique ID, pass each 
            # dictionary again to save_data
        if split_signals_needed:
            signal_keys_in_new_data = [key for key in self.signal_keys if key in new_data]
            signal_frequency_keys_in_new_data = [key for key in self.signal_frequency_keys if key in new_data]
            corresponding_frequencies = [new_data[key] for key in signal_frequency_keys_in_new_data]
            corresponding_target_frequencies = [self.file_info[key] for key in signal_frequency_keys_in_new_data]

            splitted_data_dictionaries = split_signals_within_dictionary(
                data_dict = new_data,
                id_key = "ID",
                valid_signal_keys = signal_keys_in_new_data,
                signal_frequencies = corresponding_frequencies,
                signal_target_frequencies = corresponding_target_frequencies,
                nn_signal_duration_seconds = self.file_info["signal_length_seconds"],
                wanted_shift_length_seconds = self.file_info["wanted_shift_length_seconds"],
                absolute_shift_deviation_seconds = self.file_info["absolute_shift_deviation_seconds"]
                ) # returns a list of dictionaries, len == 1 if signal was not split

            return splitted_data_dictionaries
        else:
            return [new_data]
    

    def _save_datapoint(self, new_data, overwrite_id = True):
        """
        Save single datapoint to the file. If the ID already exists in the file, the existing data will be overwritten
        with new values, if allowed.
        """
        
        # Remove frequency keys from new_data (frequency should match the one in the file, saving it is unnecessary)
        for signal_frequency_key in self.signal_frequency_keys:
            if signal_frequency_key in new_data:
                del new_data[signal_frequency_key]
        
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)
        
        # Create temporary file to save data in progress
        working_file_path = find_non_existing_path(path_without_file_type = "save_in_progress", file_type = "pkl")

        # save file information to working file
        save_to_pickle(data = next(file_generator), file_name = working_file_path)

        overwrite_denied = False
        not_appended = True

        # Check if ID already exists in the data file, then overwrite keys if allowed
        for data_point in file_generator:
            if data_point["ID"] == new_data["ID"]:
                not_appended = False
                if overwrite_id:
                    new_data_point = dict()
                    for possible_key in self.valid_datapoint_keys:
                        if possible_key in new_data:
                            new_data_point[possible_key] = new_data[possible_key]
                        elif possible_key in data_point:
                            new_data_point[possible_key] = data_point[possible_key]
                else:
                    new_data_point = data_point
                    overwrite_denied = True
            else:
                new_data_point = data_point
            
            # Append data point to the working file
            append_to_pickle(data = new_data_point, file_name = working_file_path)
        
        # Append new data point if ID was not found
        if not_appended:
            append_to_pickle(data = new_data, file_name = working_file_path)
        
        # Remove the old file and rename the working file
        try:
            os.remove(self.file_path)
        except:
            pass

        os.rename(working_file_path, self.file_path)

        if overwrite_denied:
            raise ValueError("ID already existed in the data file and Overwrite was denied. Data was not saved.")
    

    def save(self, data_dict, overwrite_id = True):
        """
        Save data to the file. If the ID already exists in the file, the existing data will be overwritten
        with new values, if allowed.

        New datapoint will be altered to match the file's signal frequencies and signal length. If the signal
        is too long, it will be split into multiple signals.
        """
        corrected_data_dicts = self._correct_datapoint(data_dict)
        for corrected_data_dict in corrected_data_dicts:
            self._save_datapoint(corrected_data_dict, overwrite_id)
        


    def load_data(self, key_id_index):
        """
        Loads data from file path. The data can be loaded by ID, key, or index.
        """
        # check if key_id_index is an id, a key, or an index
        load_keys = False
        load_id = False
        load_index = False

        if isinstance(key_id_index, str):
            if key_id_index in self.valid_datapoint_keys:
                load_keys = True
            else:
                load_id = True
        elif isinstance(key_id_index, int):
            load_index = True
        else:
            raise ValueError("\'key_id_index\' must be a string, integer, or a key from the valid_datapoint_keys (also a string).")

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Skip file information
        next(file_generator)

        if load_id:
            id_found = False
            for data_point in file_generator:
                if data_point["ID"] == key_id_index:
                    id_found = True
                    return data_point
            
            if not id_found:
                raise ValueError(f"ID {key_id_index} not found in the data file.")
        
        elif load_keys:
            values_for_key_from_all_data_points = list()
            count_data_points_missing_key = 0

            for data_point in file_generator:
                if key_id_index in data_point:
                    values_for_key_from_all_data_points.append(data_point[key_id_index])
                else:
                    count_data_points_missing_key += 1
            
            if count_data_points_missing_key > 0:
                print(f"Attention: {count_data_points_missing_key} data points are missing the key {key_id_index}")
            
            return values_for_key_from_all_data_points

        elif load_index:
            count = 0
            for data_point in file_generator:
                if count == key_id_index:
                    return data_point
                count += 1
            
            raise ValueError(f"Index {key_id_index} out of bounds in the data file.")

        del file_generator
    

    def remove_data(self, key_id_index):
        """
        Remove data from the file path. The data can be removed by ID, key, or index.
        """
        # check if key_id_index is an id, a key, or an index
        remove_keys = False
        remove_id = False
        remove_index = False

        if isinstance(key_id_index, str):
            if key_id_index in self.valid_datapoint_keys:
                remove_keys = True
            else:
                remove_id = True
        elif isinstance(key_id_index, int):
            remove_index = True
        else:
            raise ValueError("\'key_id_index\' must be a string, integer, or a key from the valid_datapoint_keys (also a string).")

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Create temporary file to save data in progress
        working_file_path = find_non_existing_path(path_without_file_type = "save_in_progress", file_type = "pkl")

        # save file information to working file
        save_to_pickle(data = next(file_generator), file_name = working_file_path)

        # Remove data point from the working file
        if remove_id:
            id_found = False
            for data_point in file_generator:
                if data_point["ID"] == key_id_index:
                    id_found = True
                else:
                    append_to_pickle(data = data_point, file_name = working_file_path)
            
            if not id_found:
                raise ValueError(f"ID {key_id_index} not found in the data file.")
        
        elif remove_keys:
            for data_point in file_generator:
                if key_id_index in data_point:
                    del data_point[key_id_index]
                append_to_pickle(data = data_point, file_name = working_file_path)
        
        elif remove_index:
            count = 0
            index_out_of_bounds = True
            for data_point in file_generator:
                if count != key_id_index:
                    append_to_pickle(data = data_point, file_name = working_file_path)
                else:
                    index_out_of_bounds = False
                count += 1
            
            if index_out_of_bounds:
                raise ValueError(f"Index {key_id_index} out of bounds in the data file.")

        # Remove the old file and rename the working file
        try:
            os.remove(self.file_path)
        except:
            pass

        os.rename(working_file_path, self.file_path)

        del file_generator
    

    def __len__(self):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Skip file information
        next(file_generator)

        count = 0
        for _ in file_generator:
            count += 1
        
        del file_generator

        return count
    

    def __contains__(self, id):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Skip file information
        next(file_generator)

        id_found = False
        for data_point in file_generator:
            if data_point["ID"] == id:
                id_found = True
                break
        
        del file_generator

        return id_found
    

    def __iter__(self):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Skip file information
        next(file_generator)

        for data_point in file_generator:
            yield data_point
        
        del file_generator
    

    def __getitem__(self, key):        

        if key in self.valid_datapoint_keys:
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            # Skip file information
            next(file_generator)

            values_for_key_from_all_data_points = list()
            count_data_points_missing_key = 0

            for data_point in file_generator:
                if key in data_point:
                    values_for_key_from_all_data_points.append(data_point[key])
                else:
                    count_data_points_missing_key += 1
            
            del file_generator

            if count_data_points_missing_key > 0:
                print(f"Attention: {count_data_points_missing_key} data points are missing the key {key}")
            
            return values_for_key_from_all_data_points


# shhs_data_manager = DataManager(file_path = "messing_around.pkl")
# shhs_data_manager = DataManager(file_path = "messing_around_2.pkl")

#raise SystemExit

# shhs_data_manager.save_data(new_data = {"ID": 1, "RRI": [1, 2, 3], "MAD": [5, 6, 7]}, overwrite_id = True)
# shhs_data_manager.save_data(new_data = {"ID": 2, "RRI": [2, 3, 4], "MAD": [6, 7, 8]}, overwrite_id = True)
# shhs_data_manager.save_data(new_data = {"ID": 3, "RRI": [3, 4, 5], "MAD": [7, 8, 9]}, overwrite_id = True)

"""
print(len(shhs_data_manager))

for data_point in shhs_data_manager:
    print(data_point)

print(shhs_data_manager.load_data(id = 2))

print(shhs_data_manager["RRI"])

print(shhs_data_manager['1'])

# let getitem retrieve also ids
# change so that sampling frequency must always be the same
# maybe create first dictionary that holds information like sampling frequency, etc.
# which is not transformable by the user
import inspect

class DataManager:
    _instance_registry = {}

    def __init__(self, file_path):
        self._file_path = file_path
        self._register_instance()

    def _register_instance(self):
        # Get the name of the variable assigned to this instance
        frame = inspect.currentframe().f_back
        variable_name = None
        for name, value in frame.f_locals.items():
            if value is self:
                variable_name = name
                break

        if variable_name:
            if variable_name in DataManager._instance_registry:
                raise ValueError(f"Variable '{variable_name}' is already assigned to an instance of DataManager.")
            DataManager._instance_registry[variable_name] = self

# Example usage:
try:
    dm1 = DataManager(file_path="file1.pkl")
    dm1 = DataManager(file_path="file2.pkl")  # This will raise a ValueError
except ValueError as e:
    print(e)

try:
    dm2 = DataManager(file_path="file3.pkl")
except ValueError as e:
    print(e)
"""