"""
Author: Johannes Peter Knoll

This file contains functions we need to preprocess the data for the neural network.

The most important part is the class 'SleepDataManager', which will help us to store the data from multiple
sources in one place. It will make sure that the data is uniform and that we can access it in a 
memory-efficient way.

All of this files functions and the class are thoroughly described and tested in 'Processing_Demo.ipynb'.

If you trust me on the code below and just want to get a quick start into Sleep Stage Prediction, 
then head to 'Classification_Demo.ipynb' and follow the instructions there.
"""

# IMPORTS:
import numpy as np
import os
import copy


"""
=====================
Operating On Signals
=====================
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

    RETURNS:
    ------------------------------
    window_overlap: float
        The needed overlap between windows of size datapoints_per_window to fit into signal_length.

    ARGUMENTS:
    ------------------------------
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

    RETURNS:
    ------------------------------
    None, but prints suitable window parameters to console.

    ARGUMENTS:
    ------------------------------
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

    RETURNS:
    ------------------------------
    interpolated_signal: list
        The resampled signal.

    EXAMPLE:
    ------------------------------
    signal = [1., 2., 3., 4., 5.]
    signal_frequency = 1
    target_frequency = 2
    interpolated_signal = [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.]

    ARGUMENTS:
    ------------------------------
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

    RETURNS:
    ------------------------------
    target_signal: np.ndarray
        The resampled signal.
    
    ARGUMENTS:
    ------------------------------
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

    RETURNS:
    ------------------------------
    target_signal: np.ndarray
        The resampled signal.
    
    ARGUMENTS:
    ------------------------------
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

    RETURNS:
    ------------------------------
    target_signal: np.ndarray
        The resampled signal.
    
    ARGUMENTS:
    ------------------------------
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

    RETURNS:
    ------------------------------
    best_possible_shift_length: float
        The optimal shift length.
    
    ARGUMENTS:
    ------------------------------
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
        absolute_shift_deviation_seconds: int = 1800,
        use_shift_length_seconds: float = 0
    ) -> tuple[np.ndarray, float]:
    """
    If the signal is longer than nn_signal_duration_seconds, the signal will be split into multiple signals of 
    length 'nn_signal_duration_seconds'. The signal will only be shifted by a certain amount though, to
    create more data.

    Attention:  'sampling_frequency' must be equal to 'target_frequency'
    
    RETURNS:
    ------------------------------
    splitted_signals: np.ndarray
        The signal split into multiple signals.
    optimal_shift_length: int
        The optimal shift length that was used to split the signal.
    
    ARGUMENTS:
    ------------------------------
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
    use_shift_length_seconds: int
        If this parameter is set to a value > 0, the function will use this shift length to split the signal.
        This can be useful if you want to split multiple signals with the same shift length.
        
        HIGHLY IMPORTANT for "bad" data with a signal (RRI, MAD, SLP... probably in most cases the latter) 
        so that: (signal_seconds * sampling_frequency != integer)
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

    # evaluate shift length
    if use_shift_length_seconds > 0:
        optimal_shift_length = int(use_shift_length_seconds * target_frequency)
        shift_length_seconds = use_shift_length_seconds
    else:
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
        shift_length_seconds = optimal_shift_length / target_frequency
        optimal_shift_length = int(optimal_shift_length)
    
    # Split signal into multiple signals by shifting, transform to windows and append to reshaped_signals
    if optimal_shift_length > 0:
        for start_index in range(0, len(signal)-number_nn_datapoints-optimal_shift_length+1, optimal_shift_length):
            splitted_signals = np.append(splitted_signals, [signal[start_index:start_index+number_nn_datapoints]], axis=0)
    else:
        splitted_signals = np.append(splitted_signals, [signal[0:number_nn_datapoints]], axis=0)

    
    # Append last 'number_nn_datapoints' of signal to reshaped_signals
    splitted_signals = np.append(splitted_signals, [signal[-number_nn_datapoints:]], axis=0)
    
    return splitted_signals, shift_length_seconds


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

    RETURNS:
    ------------------------------
    new_dictionaries_for_splitted_signals: list
        The new dictionaries for the splitted signals.
    
    ARGUMENTS:
    ------------------------------
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
    previous_shift_length_seconds = 0

    # split signals if they are too long
    for signal_key_index in range(0, len(valid_signal_keys)):
        signal_key = valid_signal_keys[signal_key_index]

        this_splitted_signal, previous_shift_length_seconds = split_long_signal(
            signal = data_dict[signal_key], # type: ignore
            sampling_frequency = signal_frequencies[signal_key_index],
            target_frequency = signal_target_frequencies[signal_key_index],
            nn_signal_duration_seconds = nn_signal_duration_seconds,
            wanted_shift_length_seconds = wanted_shift_length_seconds,
            absolute_shift_deviation_seconds = absolute_shift_deviation_seconds,
            use_shift_length_seconds = previous_shift_length_seconds
            )

        splitted_signals.append(this_splitted_signal)
        this_shift_length_seconds = int(previous_shift_length_seconds)
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
        priority_order: list = [3, 2, 1, 0]
    ) -> np.ndarray:
    """
    This function splits a signal into windows of length 'datapoints_per_window' that overlap by 
    'window_overlap'.

    RETURNS:
    ------------------------------
    windows: np.ndarray
        The signal split into overlapping windows.
    
    ARGUMENTS:
    ------------------------------
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
    priority_order = np.array(priority_order) # type: ignore

    # check if signal data type equals priority_order data type
    signal_data_type = signal.dtype # type: ignore
    priority_order_data_type = priority_order.dtype # type: ignore
    if signal_data_type != priority_order_data_type:
        priority_order = priority_order.astype(signal_data_type) # type: ignore

    # Check if signal_type is valid
    if signal_type not in ["feature", "target"]:
        raise ValueError("Parameter 'mode' must be either 'feature' or 'target'.")

    # Initialize windows
    if signal_type == "feature":
        windows = np.empty((0, datapoints_per_window), signal_data_type) # type: ignore
    elif signal_type == "target":
        windows = np.empty((0), signal_data_type) # type: ignore
    
    step_size = datapoints_per_window - window_overlap
    
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
        target_frequency: float, 
        nn_signal_duration_seconds: int = 10*3600,
        pad_with = 0,
        number_windows: int = 1197, 
        window_duration_seconds: int = 120, 
        overlap_seconds: int = 90,
        signal_type: str = "feature",
        priority_order: list = [3, 2, 1, 0],
    ) -> np.ndarray:
    """
    Reshape a signal with shape (n <= nn_signal_duration_seconds * target_frequency) to 
    (number_windows, window_size), where windows overlap by 'overlap_seconds' and adjust the signal to the
    neural network's requirements.

    RETURNS:
    ------------------------------
    reshaped_signals: list
        The signal split into overlapping windows.
    
    ARGUMENTS:
    ------------------------------
    signal: list
        The signal to be split into windows.
    
    -----
    Attention: 'sampling_frequency' must match 'target_frequency' before transforming is useful. 
    Check uses following parameters:
    -----

    target_frequency: int
        Frequency of signal in the neural network.
    
    -----
    If the signal is shorter than 'nn_signal_duration_seconds', it will be padded with 'pad_with' to match
    the required length, needing the following parameters:
    -----

    nn_signal_duration_seconds: int
        The duration of the signal that will be passed to the neural network.
    pad_with:
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
    signal_data_type = signal.dtype # type: ignore

    number_missing_datapoints = number_nn_datapoints - len(signal)
    signal = np.append(signal, [pad_with for i in range(number_missing_datapoints)]) # type: ignore
    signal = signal.astype(signal_data_type) # type: ignore

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


def slp_label_transformation(
        current_labels: dict,
        desired_labels: dict = {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": -1},
    ) -> dict:
    """
    Create transformation dictionary to alter the labels that classify the sleep stage.

    RETURNS:
    ------------------------------
    transformation_dict: dict
        The transformation dictionary.
    
    ARGUMENTS:
    ------------------------------
    current_labels: dict
        The current labels that classify the sleep stage. Values must be lists of strings or integers.
    desired_labels: dict
        The desired labels that classify the sleep stage. Values must be strings or integers.
    """

    # make sure values are strings.
    for key in desired_labels:
        desired_labels[key] = str(desired_labels[key])
    for key in current_labels:
        for i in range(len(current_labels[key])):
            current_labels[key][i] = str(current_labels[key][i])

    transformation_dict = dict()
    for key in current_labels:
        for entry in current_labels[key]:
            if entry in transformation_dict:
                raise ValueError(f"Label '{entry}' already found in transformation dictionary. Be sure not to assign multiple values to the same classification.")
            transformation_dict[entry] = desired_labels[key]
    
    return transformation_dict


def alter_slp_labels(
        slp_labels: list,
        current_labels: dict,
        desired_labels: dict = {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": -1},
    ) -> np.ndarray:
    """
    Alter the labels that classify the sleep stage.

    RETURNS:
    ------------------------------
    altered_labels: list
        The altered labels.
    
    ARGUMENTS:
    ------------------------------
    labels: list
        The labels to be altered.
    current_labels: dict
        The current labels that classify the sleep stage.
    desired_labels: dict
        The desired labels that classify the sleep stage.
    """

    transformation = slp_label_transformation(current_labels, desired_labels)

    slp_labels = np.array(slp_labels) # type: ignore
    signal_data_type = slp_labels.dtype # type: ignore
    slp_labels = slp_labels.astype(str) # type: ignore

    altered_labels = np.empty((0), str) # type: ignore

    transform_all_other = False
    if "other" in transformation:
        transform_all_other = True
    
    if transform_all_other:
        for label in slp_labels:
            if label in transformation:
                altered_labels = np.append(altered_labels, transformation[label])
            else:
                altered_labels = np.append(altered_labels, transformation["other"])
    else:
        for label in slp_labels:
            if label in transformation:
                altered_labels = np.append(altered_labels, transformation[label])
            else:
                raise ValueError(f"Label '{label}' not found in transformation dictionary.")
        
    altered_labels = altered_labels.astype(signal_data_type)
    
    return altered_labels


"""
================================
Handling Pickle Files And Paths
================================
"""

import pickle
import os
from sklearn.model_selection import train_test_split
import copy


def save_to_pickle(data, file_name):
    """
    Save data to a pickle file, overwriting the file if it already exists.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    """

    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def append_to_pickle(data, file_name):
    """
    Append data to a pickle file, without deleting previous data.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    """

    with open(file_name, "ab") as f:
        pickle.dump(data, f)


def load_from_pickle(file_name: str):
    """
    Load data from a pickle file as a generator.

    RETURNS:
    ------------------------------
    any
        data from the pickle file

    ARGUMENTS:
    ------------------------------
    file_name: str
        path to the pickle file
    key: str
        key of the data to be loaded
    """

    with open(file_name, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def find_non_existing_path(path_without_file_type: str, file_type: str = "pkl"):
    """
    Find a path that does not exist yet by adding a number to the end of the path.

    RETURNS:
    ------------------------------
    str
        path that does not exist yet

    ARGUMENTS:
    ------------------------------
    path_without_file_type: str
        path without the file type
    file_type: str
        file type of the file to be saved
    """

    if not os.path.exists(f"{path_without_file_type}.{file_type}"):
        return f"{path_without_file_type}.{file_type}"
    i = 0
    while os.path.exists(f"{path_without_file_type}_{i}.{file_type}"):
        i += 1
    return f"{path_without_file_type}_{i}.{file_type}"


def create_directories_along_path(file_path: str):
    """
    Create all directories along a given path that do not exist yet.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    file_path: str
        path to a file
    """

    if "/" in file_path:
        path_parts = file_path.split("/")
        for i in range(1, len(path_parts)):
            path = "/".join(path_parts[:i])
            if not os.path.exists(path):
                os.mkdir(path)


def check_if_splitted_signals_align_with_data(
        signal_to_split: list,
        splitted_signals: list,
    ):
    """
    Function was used to check something. Isn't implemented currently. Still leaving it here for now.
    """

    collect_indices = list()
    multiplier = 1
    if len(splitted_signals[0]) < 36000:
        multiplier = 120
    for splitted_signal in splitted_signals:
        splitted_signal = np.array(splitted_signal) # type: ignore
        fits_for_this_signal = list()
        for j in range(0, len(signal_to_split)-len(splitted_signal)+1):
            if np.array_equal(signal_to_split[j:j+len(splitted_signal)], splitted_signal):
                fits_for_this_signal.append(j*multiplier)
        collect_indices.append(fits_for_this_signal)
    
    return collect_indices
    

"""
===================
Data Manager Class
===================
"""

class SleepDataManager:
    # Define class variables
    signal_keys = ["RRI", "MAD", "SLP"]
    signal_frequency_keys = ["RRI_frequency", "MAD_frequency", "SLP_frequency"] # same order is important
    
    valid_datapoint_keys = ["ID", "sleep_stage_label", "RRI", "MAD", "SLP", "RRI_frequency", "MAD_frequency", "SLP_frequency"]
    
    default_file_info = dict()
    default_file_info["RRI_frequency"] = 4
    default_file_info["MAD_frequency"] = 1
    default_file_info["SLP_frequency"] = 1/30

    default_file_info["sleep_stage_label"] = {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": 0}

    default_file_info["signal_length_seconds"] = 36000
    default_file_info["wanted_shift_length_seconds"] = 5400
    default_file_info["absolute_shift_deviation_seconds"] = 1800

    default_file_info["train_val_test_split_applied"] = False
    default_file_info["main_file_path"] = "unassigned"
    default_file_info["train_file_path"] = "unassigned"
    default_file_info["validation_file_path"] = "unassigned"
    default_file_info["test_file_path"] = "unassigned"


    def __init__(self, file_path):
        """
        ARGUMENTS:
        ------------------------------
        file_path: str
            path to the pickle file
        """

        self.file_path = file_path

        # load general information from file
        file_generator = load_from_pickle(self.file_path)
        self.file_info = next(file_generator)
        del file_generator

        # edit some general information
        if self.file_info["main_file_path"] == "unassigned":
            self.file_info["main_file_path"] = self.file_path
            self.file_info["train_file_path"] = self.file_path[:-4] + "_training_pid.pkl"
            self.file_info["validation_file_path"] = self.file_path[:-4] + "_validation_pid.pkl"
            self.file_info["test_file_path"] = self.file_path[:-4] + "_test_pid.pkl"


    # Getter and setter for file_path
    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value):
        # check if file is a pickle file
        if value[-4:] != ".pkl":
            raise ValueError("File must be a pickle file.")

        create_directories_along_path(value)

        self._file_path = value

        # Check if file exists, if not create it and save default file information
        if not os.path.exists(self._file_path):
            save_to_pickle(data = self.default_file_info, file_name = self._file_path)
    

    def _correct_datapoint(self, new_data):
        """
        The datapoint will be altered to match the file's signal frequencies and signal length. If the signal
        is too long, it will be split into multiple signals. It will also correct the sleep labels if necessary.

        ATTENTION:  Never call this function manually. It will not do any damage, but it is not useful.

        RETURNS:
        ------------------------------
        splitted_data_dictionaries: list
            The new dictionaries for the splitted signals.
        
        ARGUMENTS:
        ------------------------------
        new_data: dict
            The datapoint to be corrected.
        """

        """
        ----------------------
        Inspect New Datapoint
        ----------------------
        """

        # Check if new_data is a dictionary
        if not isinstance(new_data, dict):
            raise ValueError("The datapoint must be a dictionary!")
        
        # Check ID key was provided
        if "ID" not in new_data:
            raise ValueError("The ID key: \"ID\" must be provided in the datapoint.")
        
        # Check if ID key is misleading
        if new_data["ID"] in self.valid_datapoint_keys:
            raise ValueError("Value for ID key: \"ID\" must not be the same as a key in the datapoint dictionary!")
        if "shift" in new_data["ID"]:
            raise ValueError("Value for ID key: \"ID\" must not contain the string: \"shift\"!")
        
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
            if signal_frequency_key in new_data and signal_key not in new_data:
                raise ValueError(f"What sense does it make to provide a sampling frequency \"{signal_frequency_key}\" without the corresponding signal \"{signal_key}\" ?")
        
        # Check if declaration of sleep stage labels was provided when SLP signal is provided
        if "SLP" in new_data and "sleep_stage_label" not in new_data:
            raise ValueError("If you provide a SLP signal, you must also provide the sleep stage labels!")
        if "sleep_stage_label" in new_data and "SLP" not in new_data:
            raise ValueError("What sense does it make to provide sleep stage labels without the corresponding SLP signal?")
            
        """
        -------------------------------
        Alter Entries In New Datapoint
        -------------------------------
        """

        # make sure ID is a string
        new_data["ID"] = str(new_data["ID"])

        # transform signals to np.ndarray
        for signal_key in self.signal_keys:
            if signal_key in new_data:
                new_data[signal_key] = np.array(new_data[signal_key])

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
        
        # make sure sleep stage labels match the ones in the file
        if "SLP" in new_data:
            new_data["SLP"] = alter_slp_labels(
                slp_labels = new_data["SLP"],
                current_labels = new_data["sleep_stage_label"],
                desired_labels = self.file_info["sleep_stage_label"]
                )

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
    

    def _save_datapoint(self, new_data, unique_id = False, overwrite_id = True):
        """
        Save single datapoint to the file. If the ID already exists in the file, the existing data will be overwritten
        with new values, if allowed.

        If user requested transformation to windows prior to saving, the signal will also be transformed to 
        windows.

        ATTENTION:  NEVER CALL THIS FUNCTION MANUALLY. USE save() INSTEAD.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        new_data: dict
            The datapoint to be saved.
        unique_id: bool
            If True, the ID will be expected to be unique and directly appended to the file.
            if False, current files will be checked to see if the ID already exists.
        overwrite_id: bool
            If True, the existing data will be overwritten with new values if the ID already exists in the file.
            If False, the data will not be saved if the ID already exists in the file.
        """
        
        # Remove frequency keys from new_data (frequency should match the one in the file, saving it is unnecessary)
        for signal_frequency_key in self.signal_frequency_keys:
            if signal_frequency_key in new_data:
                del new_data[signal_frequency_key]
        
        # Remove sleep stage label key from new_data (label should match the one in the file, saving it is unnecessary)
        if "sleep_stage_label" in new_data:
            del new_data["sleep_stage_label"]
        
        if unique_id:
            # Append new data point to the file
            append_to_pickle(data = new_data, file_name = self.file_path)
        else:
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)
            
            # Create temporary file to save data in progress
            working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
            working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

            # save file information to working file
            save_to_pickle(data = next(file_generator), file_name = working_file_path)

            overwrite_denied = False
            not_appended = True

            # Check if ID already exists in the data file, then overwrite keys if allowed
            for data_point in file_generator:
                if data_point["ID"] == new_data["ID"]:
                    not_appended = False
                    if overwrite_id:
                        for key in new_data:
                            data_point[key] = new_data[key]
                    else:
                        overwrite_denied = True
                
                # Append data point to the working file
                append_to_pickle(data = data_point, file_name = working_file_path)
            
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
    

    def save(self, data_dict, unique_id = False, overwrite_id = True):
        """
        Save data to the file. If the ID already exists in the file, the existing data will be overwritten
        with new values, if allowed.

        New datapoint will be altered to match the file's signal frequencies and signal length. If the signal
        is too long, it will be split into multiple signals.

        If user requested transformation to windows prior to saving, the signal will also be transformed to
        windows.

        data_dict must have the following structure and can contain the following keys:
        {
            "ID": str,                  # always required
            "RRI": np.ndarray,
            "MAD": np.ndarray,
            "SLP": np.ndarray,
            "RRI_frequency": int,       # required if RRI signal is provided
            "MAD_frequency": int,       # required if MAD signal is provided
            "SLP_frequency": int,       # required if SLP signal is provided
            "sleep_stage_label": list   # required if SLP signal is provided
        }

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        data_dict: dict
            The datapoint to be saved.
        overwrite_id: bool
            If True, the existing data will be overwritten with new values if the ID already exists in the file.
            If False, the data will not be saved if the ID already exists in the file.
        unique_id: bool
            If True, the ID will be expected to be unique and directly appended to the file.
            if False, current files will be checked to see if the ID already exists.
        """

        # prevent runnning this function from secondary files (train, validation, test)
        if self.file_path != self.file_info["main_file_path"]:
            raise ValueError("This function can only be called from the main file. Training-, Validation-, or Test- file data manager instances can only load data.")

        # Warn user that data will remain in main file and won't forwarded into training, validation, or test file automatically
        if self.file_info["train_val_test_split_applied"]:
            print("Attention: Data will remain in the main file and won't be forwarded into training, validation, or test file automatically. If you want to include this data, fuse files using 'fuse_train_test_validation' and resaparate again.")

        corrected_data_dicts = self._correct_datapoint(copy.deepcopy(data_dict))
        for corrected_data_dict in corrected_data_dicts:
            self._save_datapoint(corrected_data_dict, unique_id, overwrite_id)
    

    def check_if_ids_are_unique(self, ids: list):
        """
        Check if the IDs in the list are unique. 
        
        RETURNS:
        ------------------------------
        None, but prints information to console

        ARGUMENTS:
        ------------------------------
        ids: list
            List of IDs to be checked.
        """

        ids_in_database = self.load("ID")
        new_ids = list()
        multiple_ids = list()
        multiple_weird = list()

        for id in ids:
            if id in ids_in_database:
                multiple_ids.append(id)
            if id in new_ids:
                multiple_weird.append(id)
            new_ids.append(id)
        
        if len(multiple_ids) > 0:
            print("Following IDs are not unique between your new ids and the database:")
            print(multiple_ids)
            raise ValueError("IDs are not unique.")
        if len(multiple_weird) > 0:
            print("Following IDs are not unique in your new ids:")
            print(multiple_weird)
            raise ValueError("IDs are not unique.")
        if len(multiple_ids) == 0 and len(multiple_weird) == 0:
            print("All IDs are unique.")


    def load(self, key_id_index):
        """
        Loads data from file path. The data can be loaded by ID, key, or index.

        RETURNS:
        ------------------------------
        data_point: dict
            The data point that was loaded.
        
                        OR

        values_for_key_from_all_data_points: list
            The values for the key from all data points.
        
        ARGUMENTS:
        ------------------------------
        key_id_index: str, int
            The ID, key, or index of the data to be loaded.
        """

        # check if key_id_index is an id, a key, or an index
        load_keys = False
        valid_keys = ["ID", "RRI", "MAD", "SLP"]
        load_id = False
        load_index = False

        if isinstance(key_id_index, str):
            if key_id_index in valid_keys:
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
    

    def remove(self, key_id_index):
        """
        Remove data from the file path. The data can be removed by ID, key, or index.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        key_id_index: str, int
            The ID, key, or index of the data to be removed.
        """

        # check if key_id_index is an id, a key, or an index
        remove_keys = False
        valid_keys = ["ID", "RRI", "MAD", "SLP", "RRI_windows", "MAD_windows", "SLP_windows"]
        remove_id = False
        remove_index = False

        if isinstance(key_id_index, str):
            if key_id_index in valid_keys:
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
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

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
    

    def order_datapoints(self, custom_order = ["ID", "RRI", "RRI_windows", "MAD", "MAD_windows", "SLP", "SLP_windows"]):
        """
        Order the keys in the datapoints of the database for better readability when printing.
        Quite useless I know... But you know, maybe there is someone out there who appreciates this feature.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        custom_order: list
            The order in which the keys should be ordered.
        """

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)
        
        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # save file information to working file
        save_to_pickle(data = next(file_generator), file_name = working_file_path)

        # order keys in datapoints
        for datapoint in file_generator:
            ordered_datapoint = dict()
            ordered_datapoint["ID"] = datapoint["ID"]
            for key in custom_order:
                if key in datapoint:
                    ordered_datapoint[key] = datapoint[key]
            append_to_pickle(data = ordered_datapoint, file_name = working_file_path)
        
        # delete old file and rename working file
        try:
            os.remove(self.file_path)
        except:
            pass
        os.rename(working_file_path, self.file_path)
    

    def separate_train_test_validation(
            self, 
            train_size = 0.8, 
            validation_size = 0.1, 
            test_size = None, 
            random_state = None, 
            shuffle = True
        ):
        """
        Depending whether "test_size" = None/float: Separate the data in the file into training and validation, 
        data / training, validation, and test data. New files will be created in the same directory as the 
        main file. The file information will be saved to each file.

        Data that can not be used to train the network (i.e. missing "RRI" and "SLP") will be left in the
        main file. 
        
        As we can manage data with "RRI" and "MAD" and data with "RRI" only, the algorithm makes sure
        that only one of the two types of data is used (the one with more samples). The other type will 
        be left in the main file.

        The individual files can be accessed by another instance of this class. 

        ATTENTION:  The instances on all files will have reduced functionality from now on. As the data should
                    be fully prepared for the network now, the instances are designed to only load data and
                    not save or edit it.

                    The functionality of the instance on the main file is not as restricted as the ones on the
                    training, validation, and test files. The main file instance can additionally save data
                    (only to main file, won't be forwarded to training, validation, or test files), reshuffle 
                    the data in the secondary files or pull them back into the main file for further processing.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        train_size: float
            The ratio of the training data.
        validation_size: float
            The ratio of the validation data.
        test_size: float
            The ratio of the test data.
        random_state: int
            The random state for the train_test_split function.
        shuffle: bool
            If True, the data will be shuffled before splitting.
        """

        # prevent runnning this function from secondary files (train, validation, test)
        if self.file_path != self.file_info["main_file_path"]:
            raise ValueError("This function can only be called from the main file. Training-, Validation-, or Test- file data manager instances can only load data.")

        # Fuse data back together if train_val_test_split_applied is True
        if self.file_info["train_val_test_split_applied"]:
            self.fuse_train_test_validation()

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # skip file information
        next(file_generator)

        # collect ID's of all valid datapoints (needs to contain "RRI" and "SLP"), divide into ones 
        # that contain "RRI" and ones that contain "RRI" and "MAD"
        id_with_rri_and_mad = list()
        id_with_rri = list()

        for data_point in file_generator:
            if "SLP" in data_point and "RRI" in data_point:
                if "MAD" in data_point:
                    id_with_rri_and_mad.append(data_point["ID"])
                else:
                    id_with_rri.append(data_point["ID"])
        
        del file_generator

        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # Change file information
        self.file_info["train_val_test_split_applied"] = True

        # save file information to working file
        save_to_pickle(data = self.file_info, file_name = working_file_path)

        if test_size is None:
            """
            split into training and validation data
            """
            # check arguments:
            if train_size + validation_size != 1:
                self.file_info["train_val_test_split_applied"] = False
                raise ValueError("The sum of train_size and validation_size must be 1.")

            # choose which data to keep in the main file
            if len(id_with_rri_and_mad) > len(id_with_rri):
                if len(id_with_rri) != 0:
                    print(f"\nAttention: {len(id_with_rri)} datapoints without MAD signal will be left in the main file.")
                train_data_ids, validation_data_ids = train_test_split(copy.deepcopy(id_with_rri_and_mad), train_size = train_size, random_state = random_state, shuffle = shuffle)
            else:
                if len(id_with_rri_and_mad) != 0:
                    print(f"\nAttention: {len(id_with_rri_and_mad)} datapoints with MAD signal will be left in the main file.")
                train_data_ids, validation_data_ids = train_test_split(copy.deepcopy(id_with_rri), train_size = train_size, random_state = random_state, shuffle = shuffle)

            # Create files and save file information to it
            for file_path in [self.file_info["train_file_path"], self.file_info["validation_file_path"]]:
                try:
                    os.remove(file_path)
                except:
                    pass
                
                save_to_pickle(data = self.file_info, file_name = file_path)
            
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            # skip file information
            next(file_generator)

            for data_point in file_generator:
                if data_point["ID"] in train_data_ids:
                    append_to_pickle(data = data_point, file_name = self.file_info["train_file_path"])
                elif data_point["ID"] in validation_data_ids:
                    append_to_pickle(data = data_point, file_name = self.file_info["validation_file_path"])
                else:
                    append_to_pickle(data = data_point, file_name = working_file_path)
        
        else:
            """
            split into training validation and test data
            """

            # check arguments:
            if train_size + validation_size + test_size != 1: # type: ignore
                self.file_info["train_val_test_split_applied"] = False
                raise ValueError("The sum of train_size, validation_size, and test_size must be 1.")

            # choose which data to keep in the main file
            if len(id_with_rri_and_mad) > len(id_with_rri):
                if len(id_with_rri) != 0:
                    print(f"\nAttention: {len(id_with_rri)} datapoints without MAD signal will be left in the main file.")
                train_data_ids, rest_data_ids = train_test_split(copy.deepcopy(id_with_rri_and_mad), train_size = train_size, random_state = random_state, shuffle = shuffle)
                validation_data_ids, test_data_ids = train_test_split(rest_data_ids, train_size = validation_size / (1 - train_size), random_state = random_state, shuffle = shuffle)
            else:
                if len(id_with_rri_and_mad) != 0:
                    print(f"\nAttention: {len(id_with_rri_and_mad)} datapoints with MAD signal will be left in the main file.")
                train_data_ids, rest_data_ids = train_test_split(copy.deepcopy(id_with_rri), train_size = train_size, random_state = random_state, shuffle = shuffle)
                validation_data_ids, test_data_ids = train_test_split(rest_data_ids, train_size = validation_size / (1 - train_size), random_state = random_state, shuffle = shuffle)

            # Create files and save file information to it
            for file_path in [self.file_info["train_file_path"], self.file_info["validation_file_path"], self.file_info["test_file_path"]]:
                try:
                    os.remove(file_path)
                except:
                    pass
                
                save_to_pickle(data = self.file_info, file_name = file_path)
            
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            # skip file information
            next(file_generator)

            for data_point in file_generator:
                if data_point["ID"] in train_data_ids:
                    append_to_pickle(data = data_point, file_name = self.file_info["train_file_path"])
                elif data_point["ID"] in validation_data_ids:
                    append_to_pickle(data = data_point, file_name = self.file_info["validation_file_path"])
                elif data_point["ID"] in test_data_ids:
                    append_to_pickle(data = data_point, file_name = self.file_info["test_file_path"])
                else:
                    append_to_pickle(data = data_point, file_name = working_file_path)
        
        # Remove the old file and rename the working file
        try:
            os.remove(self.file_path)
        except:
            pass
            
        os.rename(working_file_path, self.file_path)
    

    def fuse_train_test_validation(self):
        """
        Fuses the training, validation, and test data back into the main file.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        None
        """

        # Check if train_val_test_split_applied is True
        if not self.file_info["train_val_test_split_applied"]:
            raise ValueError("Train-Validation-Test split was not applied to the data file. Therefore, it cannot be fused back.")
        
        # edit file information
        self.file_info["train_val_test_split_applied"] = False

        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # save file information to working file
        save_to_pickle(data = self.file_info, file_name = working_file_path)

        # Append data points from main file
        main_file_generator = load_from_pickle(self.file_info["main_file_path"])
        next(main_file_generator)
        for data_point in main_file_generator:
            append_to_pickle(data = data_point, file_name = working_file_path)
        
        # Remove main file
        try:
            os.remove(self.file_info["main_file_path"])
        except:
            pass

        # Append data points from training file
        training_file_generator = load_from_pickle(self.file_info["train_file_path"])
        next(training_file_generator)
        for data_point in training_file_generator:
            append_to_pickle(data = data_point, file_name = working_file_path)
        
        # Remove training file
        try:
            os.remove(self.file_info["train_file_path"])
        except:
            pass
        
        # Append data points from validation file
        validation_file_generator = load_from_pickle(self.file_info["validation_file_path"])
        next(validation_file_generator)
        for data_point in validation_file_generator:
            append_to_pickle(data = data_point, file_name = working_file_path)
        
        # Remove validation file
        try:
            os.remove(self.file_info["validation_file_path"])
        except:
            pass
        
        # Append data points from test file if it exists
        try:
            test_file_generator = load_from_pickle(self.file_info["test_file_path"])
            next(test_file_generator)
            for data_point in test_file_generator:
                append_to_pickle(data = data_point, file_name = working_file_path)
            
            # Remove test file
            try:
                os.remove(self.file_info["test_file_path"])
            except:
                pass
        except:
            pass
        
        # Rename the working file
        os.rename(working_file_path, self.file_info["main_file_path"])
    

    def change_file_information(self, new_file_info: dict):
        """
        Change the file information of the file. Only possible if no datapoints are in the file.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        new_file_info: dict
            The new file information.
        """

        # check if there are datapoints in the file
        if len(self) > 0:
            raise ValueError("File information can only be changed if no data points are in the file.")

        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # update file information
        for key in new_file_info:
            if key not in self.file_info:
                print(f"Attention: Key {key} not recognized. It will be skipped.")
                continue
            if key in ["train_val_test_split_applied", "main_file_path", "train_file_path", "validation_file_path", "test_file_path"]:
                print(f"Attention: Key {key} is a reserved key and cannot be changed.")
                continue
            self.file_info[key] = new_file_info[key]

        # save file information to working file
        save_to_pickle(data = self.file_info, file_name = working_file_path)
        
        # Remove the old file and rename the working file
        try:
            os.remove(self.file_path)
        except:
            pass

        os.rename(working_file_path, self.file_path)
    

    def __len__(self):
        """
        Returns the number of data points in the file. Usage: len(data_manager_instance)

        RETURNS:
        ------------------------------
        count: int
            The number of data points in the file.
        
        ARGUMENTS:
        ------------------------------
        None
        """

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
        """
        Check if the ID is in the file. Usage: id in data_manager_instance:

        RETURNS:
        ------------------------------
        id_found: bool
            True if the ID is in the file, False otherwise.
        
        ARGUMENTS:
        ------------------------------
        id: str
            The ID to be checked.
        """

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
        """
        Iterate over all data points in the file. Usage: for data_point in data_manager_instance:

        RETURNS:
        ------------------------------
        data_point: dict
            The data point that was loaded.
        
        ARGUMENTS:
        ------------------------------
        None
        """

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Skip file information
        next(file_generator)

        for data_point in file_generator:
            yield data_point
        
        del file_generator
    

    def __getitem__(self, key):
        """
        Get all data with the given key from the file. Usage: data_manager_instance[key]

        Identical to data_manager_instance.load(key), but more convenient.

        RETURNS:
        ------------------------------
        values_for_key_from_all_data_points: list
            The values for the key from all data points.
        
        ARGUMENTS:
        ------------------------------
        key: str
            The key to be loaded.
        """    

        valid_keys = ["ID", "RRI", "MAD", "SLP"]

        if key in valid_keys:
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
    

    def __str__(self):
        """
        Returns a string representation of the file path and file information.

        Usage: print(data_manager_instance) or str(data_manager_instance)

        RETURNS:
        ------------------------------
        str
            A string representation of the file path and file information.
        
        ARGUMENTS:
        ------------------------------
        None
        """

        return f"file_path: {self.file_path}\nfile_info: {self.file_info}"