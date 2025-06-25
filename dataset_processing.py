"""
Author: Johannes Peter Knoll

This file contains functions we need to preprocess the data for the neural network.

The most important part is the class 'SleepDataManager'. 
It will make sure that the data is uniform and that we can access it in a memory-efficient way.

All of this files functions and the class are thoroughly described and tested in 'Processing_Demo.ipynb'.

If you trust me on the code below and just want to get a quick start into Sleep Stage Prediction, 
then head to 'Classification_Demo.ipynb' and follow the instructions there.
"""

# IMPORTS:
import os
import pickle
import copy

import numpy as np
from sklearn.model_selection import train_test_split


# LOCAL IMPORTS:
from side_functions import *


"""
=====================
Operating On Signals
=====================
"""


def remove_outliers(
        signal: list, 
        inlier_interval: list = [None, None]
    ) -> np.ndarray:
    """
    Remove outliers from a signal. Outliers are defined as values that are smaller than lower_bound (replaced 
    by lower_bound) or larger than upper_bound (replaced by upper_bound).

    lower_bound = inlier_interval[0], upper_bound = inlier_interval[1]

    RETURNS:
    ------------------------------
    cleaned_signal: np.ndarray
        The signal with outliers removed.
    
    ARGUMENTS:
    ------------------------------
    signal: np.ndarray
        The signal to be cleaned.
    inlier_interval: list
        The interval in which inliers are expected. Outliers will be set to the closest value in this interval.
        If interval value is None, no lower and/or upper bound will be applied.
    """

    signal = np.array(signal) # type: ignore

    cleaned_signal = np.copy(signal)

    lower_bound = inlier_interval[0]
    if lower_bound is not None:
        cleaned_signal[cleaned_signal < lower_bound] = lower_bound
    
    upper_bound = inlier_interval[1]
    if upper_bound is not None:
        cleaned_signal[cleaned_signal > upper_bound] = upper_bound

    return cleaned_signal


def signal_normalization(
        signal: list,
        normalization_technique: str = "z-score", # "z_score" or "min-max"
        normalization_mode: str = "global", # "global" or "local"
        **kwargs,
    ) -> np.ndarray: # type: ignore
    """
    Normalize the single- or multidimensional-signal

    RETURNS:
    ------------------------------
    normalized_signal: np.ndarray
        The normalized signal.
    
    ARGUMENTS:
    ------------------------------
    signal: np.ndarray
        The signal to be normalized.
    normalization_technique: str
        The normalization technique to be used.
        if "z-score":   Standardizes the signal to have a mean of 0 and a standard deviation of 1.
        if "min-max":   Scales the signal to a specified range, typically [0, 1] or [-1, 1].
    normalization_mode: str
        The normalization mode.
        if "global":    Scales all elements in the entire multi-dimensional array relative to the global
                        maximum and minimum values across all arrays.
        if "local":     Normalizes each sub-array independently, scaling the elements within relative to its
                        own maximum and minimum values.
    
    KEYWORD ARGUMENTS:
    ------------------------------
    normalization_max: float
        The new maximum value for "min-max" normalization.
        If not specified, defaults to 1.0.
    normalization_min: float
        The new minimum value for "min-max" normalization.
        If not specified, defaults to 0.0.
    """

    # Define default values
    kwargs.setdefault("normalization_max", 1.0)
    kwargs.setdefault("normalization_min", 0.0)

    if kwargs["normalization_max"] <= kwargs["normalization_min"]:
        raise ValueError("The new maximum must be larger than the new minimum, obviously...")

    signal = np.array(signal, dtype=float) # type: ignore

    dimension = signal.ndim # type: ignore

    if dimension == 1 or normalization_mode == "global":
        if normalization_technique == "z-score":
            mean = np.mean(signal)
            std_dev = np.std(signal)

            if std_dev == 0:
                return np.full_like(signal, 0.0, dtype=float)

            return (signal - mean) / std_dev # type: ignore
        
        elif normalization_technique == "min-max":
            old_max = np.max(signal)
            old_min = np.min(signal)

            if old_max == old_min:
                return np.full_like(signal, (kwargs["normalization_max"] - kwargs["normalization_min"]) / 2 + kwargs["normalization_min"], dtype=float)

            return (signal - old_min) / (old_max - old_min) * (kwargs["normalization_max"] - kwargs["normalization_min"]) + kwargs["normalization_min"]

    elif dimension == 2 and normalization_mode == "local":
        if normalization_technique == "z-score":
            for i in range(len(signal)):
                mean = np.mean(signal[i])
                std_dev = np.std(signal[i])

                if std_dev == 0:
                    signal[i] = np.full_like(signal[i], 0.0, dtype=float)
                else:
                    signal[i] = (signal[i] - mean) / std_dev

            return np.array(signal)
        
        elif normalization_technique == "min-max":
            for i in range(len(signal)):
                old_max = np.max(signal[i])
                old_min = np.min(signal[i])

                if old_max == old_min:
                    signal[i] = np.full_like(signal[i], (kwargs["normalization_max"] - kwargs["normalization_min"]) / 2 + kwargs["normalization_min"], dtype=float)
                else:
                    signal[i] = (signal[i] - old_min) / (old_max - old_min) * (kwargs["normalization_max"] - kwargs["normalization_min"]) + kwargs["normalization_min"]

            return np.array(signal)
    
    elif dimension > 2 and normalization_mode == "local":
        for i in range(len(signal)):
            signal[i] = signal_normalization(signal[i], normalization_technique, normalization_mode, **kwargs)

        return np.array(signal)


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


def retrieve_possible_shift_lengths(
        min_shift_seconds: float,
        max_shift_seconds: float,
        all_signal_frequencies: list,
    ) -> list:
    """
    The neural network must always be applied with a signal of the same length. If the signal is longer
    than the desired length, it will be split into multiple signals of the desired length. To create more data, 
    the signal will only be shifted by a certain amount. 
    
    Because every signal has a different sampling frequency, the shift length must be chosen so, that
    the factor of the shift length (in seconds) and every signal frequency is an integer number.

    This function calculates all shift lengths that fulfill this condition and returns them.

    If no shift length can be found within the allowed deviation, the function raises an error.
    Therefore, this function can also be used to check if the set signal frequencies within the 
    SleepDataManager class can be processed.

    RETURNS:
    ------------------------------
    possible_shift_length: int
        The optimal shift length in seconds.
    
    ARGUMENTS:
    ------------------------------
    min_shift_seconds: float
        The minimum shift length in seconds.
    max_shift_seconds: float
        The maximum shift length in seconds.
    all_signal_frequencies: list
        The frequencies of all signals that must be split.
    """

    # collect all shift length seconds that result in an integer number shift length for all signals
    possible_shift_lengths = []
    for shift_length in range(int(min_shift_seconds), int(np.ceil(max_shift_seconds))+1):
        possible = True
        for signal_frequency in all_signal_frequencies:
            if shift_length * signal_frequency != int(shift_length * signal_frequency):
                possible = False
                break
        if possible:
            possible_shift_lengths.append(shift_length)
    
    if len(possible_shift_lengths) == 0:
        raise ValueError("For the current set of signal frequencies, no reasonable shift length can be found. Change the parameters that control the shift length, or use different signal frequencies.")
    
    return possible_shift_lengths


def calculate_optimal_shift_length(
        signal_length_seconds: float, 
        desired_length_seconds: float, 
        wanted_shift_length_seconds: float,
        absolute_shift_deviation_seconds: float,
        all_signal_frequencies: list,
    ) -> int: # type: ignore
    """
    The neural network must always be applied with a signal of the same length. If the signal is longer
    than the desired length, it will be split into multiple signals of the desired length. To create more data, 
    the signal will only be shifted by a certain amount. 
    
    Because every signal has a different sampling frequency, the shift length must be chosen so, that
    the factor of the shift length (in seconds) and every signal frequency is an integer number.

    This function calculates the optimal shift length. It tries to find the shift length that is closest to the
    wanted shift length, but still fulfills the condition above. If no shift length can be found within the
    allowed deviation, the function raises an error.

    RETURNS:
    ------------------------------
    possible_shift_length: int
        The optimal shift length in seconds.
    
    ARGUMENTS:
    ------------------------------
    signal_length_seconds: float
        The length of the signal in seconds.
    desired_length_seconds: float
        The desired length of the signal in seconds.
    wanted_shift_length_seconds: float
        The shift length that is desired in seconds.
    absolute_shift_deviation_seconds: float
        The allowed deviation from the wanted shift length in seconds.
    all_signal_frequencies: list
        The frequencies of all signals that must be split.
    """

    # Calculate min and max shift length
    min_shift = wanted_shift_length_seconds - absolute_shift_deviation_seconds
    if min_shift < 1:
        min_shift = 1
    max_shift = wanted_shift_length_seconds + absolute_shift_deviation_seconds

    # Retrieve possible shift lengths
    possible_shift_lengths = retrieve_possible_shift_lengths(
        min_shift_seconds = min_shift,
        max_shift_seconds = max_shift,
        all_signal_frequencies = all_signal_frequencies
    )
    
    possible_shift_lengths = np.array(possible_shift_lengths)

    # Initialize variables
    number_shifts = 1

    collect_shift_length = []
    collect_wanted_deviation = []
    collect_possible_deviation = []

    # Find shift lengths that result in an integer number of shifts:
    # The idea is that the number of shifts is always increased by 1, until the shift length is 
    # smaller than the minimum shift length.
    while True:
        current_shift_length_seconds = (signal_length_seconds-desired_length_seconds) / number_shifts
        current_shift_deviation_seconds = abs(current_shift_length_seconds - wanted_shift_length_seconds)

        if current_shift_length_seconds < min_shift:
            break

        if current_shift_length_seconds <= max_shift:
            collect_shift_length.append(current_shift_length_seconds)
            collect_wanted_deviation.append(current_shift_deviation_seconds)

            deviation_to_possible_shifts = copy.deepcopy(possible_shift_lengths) - current_shift_length_seconds
            # remove negative values (shift length must be integer, if integer is smaller than float, then not all entries will be collected)
            this_max = max(deviation_to_possible_shifts)
            for i in range(0, len(deviation_to_possible_shifts)):
                if deviation_to_possible_shifts[i] < 0:
                    deviation_to_possible_shifts[i] = this_max
            
            collect_possible_deviation.append(min(deviation_to_possible_shifts))

        number_shifts += 1
    
    if len(collect_shift_length) == 0:
        possible_wanted_deviation = np.abs(possible_shift_lengths - wanted_shift_length_seconds)
        return possible_shift_lengths[np.argmin(possible_wanted_deviation)]
    
    # return shift length with smallest deviation to wanted shift length and possible shift length, 
    # slightly prioritize deviation to possible shift length
    collect_possible_deviation = np.array(collect_possible_deviation) + 0.1
    collect_wanted_deviation = np.array(collect_wanted_deviation) + 0.1
    collect_wanted_deviation *= 2

    criterium = collect_possible_deviation * collect_wanted_deviation
    best_possible_shift_length = collect_shift_length[np.argmin(criterium)]

    for possible_shift_length in possible_shift_lengths:
        if possible_shift_length >= best_possible_shift_length:
            return possible_shift_length
    
    return possible_shift_lengths[-1]


def split_long_signal(
        signal: list, 
        sampling_frequency: int,
        signal_length_seconds: int = 10*3600,
        wanted_shift_length_seconds: int = 3600,
        absolute_shift_deviation_seconds: int = 1800,
        use_shift_length_seconds: int = 0
    ) -> list:
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
    signal_type: str
        The type of signal. Either 'feature' or 'target'.
    
    signal_length_seconds: int
        The duration of the signal that will be passed to the neural network.
    wanted_shift_length_seconds: int
        The shift length that is desired by user.
    absolute_shift_deviation_seconds: int
        The allowed deviation from the wanted shift length.
    all_signal_frequencies: list
        The frequencies of all signals that must be split.
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

    signal = np.array(signal) # type: ignore
        
    # Calculate number of datapoints from signal length in seconds
    number_nn_datapoints = int(signal_length_seconds * sampling_frequency)

    # Check if signal is shorter than those that will be passed to the neural network
    if len(signal) <= number_nn_datapoints:
        raise ValueError("Signal is shorter than the desired signal length. Splitting unnecessary.")
    
    optimal_shift_length = int(use_shift_length_seconds * sampling_frequency)
    
    splitted_signals = list() # type: ignore
    
    # Split long signal into multiple signals by shifting
    start_index = 0
    while True:
        end_index = start_index + number_nn_datapoints
        splitted_signals.append(np.array(signal[start_index:end_index]))
        start_index += optimal_shift_length
        if end_index >= len(signal):
            break

    return splitted_signals


def split_signals_within_dictionary(
        data_dict: dict,
        id_key: str,
        signal_keys: list,
        signal_frequencies: list,
        signal_length_seconds: int,
        wanted_shift_length_seconds: int,
        absolute_shift_deviation_seconds: int
    ):
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
    signal_keys: list
        The keys of all signals that should be split.
    signal_frequencies: list
        The frequencies of the above signals.
    signal_length_seconds: int
        The duration of the signal that will be passed to the neural network.
    wanted_shift_length_seconds: int
        The shift length that is desired by user.
    absolute_shift_deviation_seconds: int
        The allowed deviation from the wanted shift length.
    """

    splitted_signals = list()
    for key_index in range(len(signal_keys)):
        if signal_keys[key_index] in data_dict:
            current_signal_length_seconds = len(data_dict[signal_keys[key_index]]) / signal_frequencies[key_index]
            break

    # Calculate optimal shift length
    use_shift_length_seconds = calculate_optimal_shift_length(
        signal_length_seconds = current_signal_length_seconds,
        desired_length_seconds = signal_length_seconds,
        wanted_shift_length_seconds = wanted_shift_length_seconds,
        absolute_shift_deviation_seconds = absolute_shift_deviation_seconds,
        all_signal_frequencies = signal_frequencies
        )

    # split signals if they are too long
    for signal_key_index in range(0, len(signal_keys)):
        signal_key = signal_keys[signal_key_index]
        if signal_key not in data_dict:
            continue

        this_splitted_signal = split_long_signal(
            signal = copy.deepcopy(data_dict[signal_key]), # type: ignore
            sampling_frequency = copy.deepcopy(signal_frequencies[signal_key_index]),
            signal_length_seconds = signal_length_seconds,
            wanted_shift_length_seconds = wanted_shift_length_seconds,
            absolute_shift_deviation_seconds = absolute_shift_deviation_seconds,
            use_shift_length_seconds = use_shift_length_seconds
            )

        splitted_signals.append(this_splitted_signal)
        del this_splitted_signal

    # create new dictionaries for splitted signals
    new_dictionaries_for_splitted_signals = list()
    
    identifier = data_dict[id_key]
    present_signal_keys = [key for key in data_dict if key in signal_keys]
    additional_keys = {key: data_dict[key] for key in data_dict if key != id_key and key not in signal_keys}
    
    # create and append first dictionary (additionally contains the shift length and other common keys)
    first_dict = {
        id_key: identifier,
        "shift_length_seconds": use_shift_length_seconds,
        "shift": 0
    }
    for key in present_signal_keys:
        first_dict[key] = np.array(splitted_signals[signal_keys.index(key)][0])
    first_dict.update(additional_keys)
    new_dictionaries_for_splitted_signals.append(first_dict)
    del first_dict

    # append remaining dictionaries
    identifier += "*"
    for i in range(1, len(splitted_signals[0])):
        splitted_data_dict = dict()
        splitted_data_dict[id_key] = identifier
        splitted_data_dict["shift"] = i
        for key in present_signal_keys:
            splitted_data_dict[key] = np.array(splitted_signals[signal_keys.index(key)][i])

        new_dictionaries_for_splitted_signals.append(splitted_data_dict)
    
    return new_dictionaries_for_splitted_signals


def fuse_splitted_signals(
        signals: list,
        shift_lengths: list,
        signal_type: str = "feature",
        summarize_overlapping_entries: bool = True
    ):
    """
    Reverse Operation to 'split_long_signal'.

    RETURNS:
    ------------------------------
    fused_signal: list
        The fused signal.
    
    ARGUMENTS:
    ------------------------------
    signals: list
        The splitted signals.
    shift_lengths: list
        Shift length of starting point to corresponding signal in amount of datapoints.
    signal_type: str
        The type of signal. Either 'feature' or 'target'.
        If 'feature':   The signals will be fused by taking the mean of the overlapping entries.
        If 'target':    The signals will be fused by taking the majority of the overlapping entries.
    summarize_overlapping_entries: bool
        If True, the overlapping entries will be summarized by taking the mean or majority of the entries, depending on signal type.
        If False, the overlapping entries will be collected into a list.
    """

    # calculate original signal length
    max_shift_length = max(shift_lengths)
    last_splitted_signal_index = shift_lengths.index(max_shift_length)
    original_signal_length = len(signals[last_splitted_signal_index]) + max_shift_length

    # choose return data type
    data_types = [np.array(i).dtype for i in signals]
    use_data_type = data_types[0]
    for data_type in data_types:
        if data_type == float:
            use_data_type = data_type
            break

    # initialize fused signal
    fused_signal = [[] for _ in range(original_signal_length)]

    # fuse signals (collect duplicate entries into list)
    for signal_index in range(len(signals)):
        this_signal = signals[signal_index]
        this_shift_length = shift_lengths[signal_index]

        for i in range(0, len(this_signal)):
            fused_signal[i + this_shift_length].append(this_signal[i])
    
    # mean collected entries or return them
    if summarize_overlapping_entries:
        if signal_type == "feature":
            if isinstance(fused_signal[0][0], list):
                return np.array([np.array(i).sum(axis=0) / len(i) for i in fused_signal], dtype=use_data_type)
            return np.array([sum(i) / len(i) for i in fused_signal], dtype=use_data_type)
        elif signal_type == "target":
            if isinstance(fused_signal[0][0], list):
                raise ValueError("This should not exist.")
            return np.array([max(set(i), key=i.count) for i in fused_signal], dtype=use_data_type)


def fuse_splitted_signals_within_dictionaries(
        data_dictionaries: list,
        valid_signal_keys: list,
        valid_signal_frequencies: list,
    ):
    """
    Reverse Operation to 'split_signals_within_dictionary'.

    ATTENTION: The order of the signals in 'valid_signal_keys' must be the same as the order of the frequencies
    in 'signal_frequencies'.

    RETURNS:
    ------------------------------
    restored_dictionary: list
        Dictionary with refused signals.
    
    ARGUMENTS:
    ------------------------------
    data_dictionaries: list
        List of dictionaries, containing the splitted signals.
    valid_signal_keys: list
        The keys of the signals that should be refused.
    signal_frequencies: list
        The frequencies of the above signals.
    """

    # initialize restored dictionary and extract original ID
    restored_dictionary = dict()
    for data_dict in data_dictionaries:
        if "*" != data_dict["ID"][-1]:
            restored_dictionary["ID"] = data_dict["ID"]
            shift_length_seconds = data_dict["shift_length_seconds"]
            additional_keys = {key: data_dict[key] for key in data_dict if key not in ["ID", "shift_length_seconds"].extend(valid_signal_keys)}

    # Iterate over different signals, fuse and store them
    for signal_key_index in range(0, len(valid_signal_keys)):
        # check if signal key is present, skip if not
        signal_key = valid_signal_keys[signal_key_index]
        if signal_key not in data_dictionaries[0]:
            continue

        signal_frequency = valid_signal_frequencies[signal_key_index]
        
        shift_length = shift_length_seconds * signal_frequency
        if shift_length != int(shift_length):
            raise ValueError("Shift length must be an integer number of datapoints.")
        
        signals = [data_dict[signal_key] for data_dict in data_dictionaries]
        shift_lengths = [data_dict["shift"] * shift_length for data_dict in data_dictionaries]
        
        signal_type = "feature"
        summarize_overlapping_entries = True
        if signal_key in ["SLP_predicted", "SLP"]:
            signal_type = "target"
        if signal_key == "SLP_predicted":
            summarize_overlapping_entries = False

        fused_signal = fuse_splitted_signals(
            signals = signals,
            shift_lengths = shift_lengths,
            signal_type = signal_type,
            summarize_overlapping_entries = summarize_overlapping_entries
        )

        restored_dictionary[signal_key] = fused_signal
    
    # Add all other entries to restored dictionary
    restored_dictionary.update(additional_keys)
    
    return restored_dictionary


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


def remove_padding_from_windows(
        signal_in_windows: list, 
        target_frequency: int,
        original_signal_length: int,
        window_duration_seconds: int = 120, 
        overlap_seconds: int = 90,
    ) -> np.ndarray:
    """
    Later, we want to estimate the performance of the neural network. We will calculate multiple metrics
    using the predictions reshaped to overlapping windows (nn output), as well as the predictions reshaped to
    the original signal. 

    To gain a more precise estimation, we want to exclude the padding. The function that reshapes the signal
    with overlapping windows to the original structure also removes the padding. However, for the signal with 
    overlapping windows, we need to remove the padding with this function.

    RETURNS:
    ------------------------------
    signal_without_padding: np.ndarray
        The signal with overlapping windows without padding.
    
    ARGUMENTS:
    ------------------------------
    signal_in_windows: list
        The signal with overlapping windows.
    target_frequency: int
        Frequency of signal in the neural network.
    original_signal_length: int
        The number of datapoints in the original signal.
    window_duration_seconds: int
        The window length in seconds.
    overlap_seconds: int
        The overlap between windows in seconds.
    """

    # calculate number of windows that contain signal data
    """
    Formula Explanation:
    ------------------------------
    -   need to find last window that contains signal data -> find number of shifts it takes to reach the end 
        of the signal

    window shift = (window_duration_seconds - overlap_seconds) * target_frequency
    number windows with signal data = signal length / window shift
    """
    number_windows_with_data = original_signal_length / ((window_duration_seconds - overlap_seconds) * target_frequency)
    number_windows_with_data = int(np.ceil(number_windows_with_data))

    # return signal without windows that only contain padding
    return signal_in_windows[:number_windows_with_data] # type: ignore


def reverse_signal_to_windows_reshape(
        signal_in_windows: list, 
        target_frequency: int,
        original_signal_length: int,
        number_windows: int = 1197, 
        window_duration_seconds: int = 120, 
        overlap_seconds: int = 90,
    ) -> np.ndarray:
    """
    Revert the Reshape that was done by 'reshape_signal_to_overlapping_windows'.

    From the overlapping part of each window, the mean value will be taken to create the original signal.

    Mainly used so that Predictions can be transformed back, using it for different purposes is not recommended.

    RETURNS:
    ------------------------------
    recovered_signal: list
        The recovered signal. 
    
    ARGUMENTS:
    ------------------------------
    signal: list
        The signal to be split into windows.
    target_frequency: int
        Frequency of signal in the neural network.
    original_signal_length: int
        The number of datapoints in the original signal.
    number_windows: int
        The number of windows the signal was split into.
    window_duration_seconds: int
        The window length in seconds.
    overlap_seconds: int
        The overlap between windows in seconds.
    """

    assert len(signal_in_windows) == number_windows, "Number of windows does not match signal length."

    datapoints_per_window = int(window_duration_seconds * target_frequency)
    if window_duration_seconds * target_frequency != datapoints_per_window:
        raise ValueError("Datapoints per window must be an integer. Choose 'window_duration_seconds' and 'target_frequency' accordingly.")
    
    overlapping_datapoints = int(overlap_seconds * target_frequency)
    if overlap_seconds * target_frequency != overlapping_datapoints:
        raise ValueError("Overlap must be an integer. Choose 'overlap_seconds' and 'target_frequency' accordingly.")
    
    new_datapoints_per_window = datapoints_per_window - overlapping_datapoints

    count_overlaps = [0 for _ in range(new_datapoints_per_window*(number_windows-1)+datapoints_per_window)]
    sum_up_windows = [0 for _ in range(new_datapoints_per_window*(number_windows-1)+datapoints_per_window)]

    for i in range(0, len(signal_in_windows)):
        assert datapoints_per_window == len(signal_in_windows[i]), "Window length does not match signal length."

        for j in range(0, datapoints_per_window):
            sum_up_windows[j+i*new_datapoints_per_window] += signal_in_windows[i][j]
            count_overlaps[j+i*new_datapoints_per_window] += 1
    
    recovered_signal = [sum_up_windows[i] / count_overlaps[i] for i in range(0, len(sum_up_windows))]

    return np.array(recovered_signal[:original_signal_length])


def slp_label_transformation(
        current_labels: dict,
        desired_labels: dict = {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": 0},
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
    
    # alter dtype of slp labels if necessary (float is not accepted)
    slp_labels = np.array(slp_labels) # type: ignore
    if slp_labels.dtype == float: # type: ignore
        print("\nWARNING: Sleep labels are of type float. Converting to int.")
        slp_labels = slp_labels.astype(int) # type: ignore

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


def summarize_predicted_signal(predicted_signal: list, mode: str):
    """
    As result of predicting the sleep stages for unknown data, we will create the following signals:
    - "SLP_predicted_probability": Holds the probability of each sleep stage for each time point.
    - "SLP_predicted": Lists the sleep stages with highest probability for each time point.

    Afterwards we need to reverse the signal split, causing multiple prediction results for some time points.
    -   In "SLP_predicted_probability", the mean probability is calculated for each sleep stage across 
        the overlapping predictions.
    -   For "SLP_predicted", all overlapping predictions for each time point are gathered into an 
        array. Naturally, the number of data points per time point will vary.
    
    Now, we might want to summarize the signal, so that we have a single prediction for each time point.
    Both signals are 2D arrays, but need to be summarized differently, declared by the parameter "mode".

    RETURNS:
    ------------------------------
    summarized_signal: list
        The summarized signal.
    
    ARGUMENTS:
    ------------------------------
    predicted_signal: list
        The predicted signal.
    mode: str
        The mode to summarize the signal. Can be "probability" or "majority".
        If "probability":   The function assumes that the signal lists for every time point (iteration) 
                            a list of probabilities for each sleep stage. The function will return the 
                            sleep stage with the highest probability.
        If "majority":      The function assumes that the signal lists for every time point (iteration)
                            a list of sleep stages. The function will return the sleep stage that appears
                            the most.
    """

    if mode == "probability":
        return np.argmax(predicted_signal, axis=1)
    
    elif mode == "majority":
        summarized_signal = np.empty(len(predicted_signal), dtype = np.int32)
        for time_point_index in range(len(predicted_signal)):
            # collect unique labels and their counts
            different_labels, label_counts = np.unique(predicted_signal[time_point_index], return_counts=True)

            # remove labels that did not appear the most
            max_count = max(label_counts)
            most_common_labels = different_labels[label_counts == max_count]

            # prioritize labels with higher index
            summarized_signal = np.append(summarized_signal, max(most_common_labels))
        
        return summarized_signal
    
    else:
        raise ValueError("\"mode\" parameter not recognized.")


"""
================================
Handling Pickle Files And Paths
================================
"""


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
            except:
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
        if "/" == file_path[0]:
            path_parts = file_path[1:].split("/")
            path_parts[0] = "/" + path_parts[0]
        else:
            path_parts = file_path.split("/")
        for i in range(1, len(path_parts)):
            path = "/".join(path_parts[:i])

            if not os.path.exists(path):
                os.mkdir(path)


"""
================
Signal Checking
================
"""


def check_signal_length(
        data_point: dict,
        database_configuration: dict,
        signal_keys: list,
        signal_frequency_keys: list,
        tolerance: float = 0.05
    ):
    """
    Checks if all signals in a data point have approximately the same length. Raises error if not.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    data_point: dict
        Dictionary that stores signals that need to be checked
    database_configuration: dict
        Dictionary that holds information on signal frequencies
    signal_keys: list
        List of signal keys (strings) that access the signals in data_point
    signal_frequency_keys: list
        List of signal frequency keys (strings) that access the signals sampling frequency in file_info
    tolerance: float
        Tolerance that decides whether the signal lengths are close enough
    """

    temporary_signal_length_seconds = list()

    for signal_key_index in range(0, len(signal_keys)):
        signal_key = signal_keys[signal_key_index]
        signal_frequency_key = signal_frequency_keys[signal_key_index]
        if signal_key in data_point:
            temporary_signal_length_seconds.append(len(data_point[signal_key]) / database_configuration[signal_frequency_key])

    if len(temporary_signal_length_seconds) > 1:
        temporary_signal_length_seconds = np.array(temporary_signal_length_seconds)
        if not np.allclose(temporary_signal_length_seconds, np.min(temporary_signal_length_seconds), rtol = tolerance):
            raise ValueError("Signal lengths are not uniform. Check signal lengths and frequencies.")
    

"""
===================
Data Manager Class
===================
"""

class SleepDataManager:
    # Define class variables
    signal_keys = ["RRI", "MAD", "SLP", "SLP_predicted", "SLP_predicted_probability"]
    signal_frequency_keys = ["RRI_frequency", "MAD_frequency", "SLP_frequency", "SLP_frequency", "SLP_frequency"] # same order is important
    
    database_configuration = dict()
    database_configuration["RRI_frequency"] = 4
    database_configuration["MAD_frequency"] = 1
    database_configuration["SLP_frequency"] = 1/30
    
    # database_configuration["RRI_inlier_interval"] = [0.3, 2.0] # RRI > 2, RRI < 0.3 are set to 2, 0.3 respectively 
    # database_configuration["MAD_inlier_interval"] = [None, None] # No MAD values are altered
    # database_configuration["sleep_stage_label"] = {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": 0}

    database_configuration["signal_length_seconds"] = None
    database_configuration["wanted_shift_length_seconds"] = None
    database_configuration["absolute_shift_deviation_seconds"] = None

    database_configuration["number_datapoints"] = None

    database_configuration["train_val_test_split_applied"] = False


    def __init__(self, directory_path: str):
        """
        Initialize the SleepDataManager class.

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        directory_path: str
            path to the directory which will store all data in appropriate files
        """

        self.directory_path = directory_path
        self.configuration_path = directory_path + "configuration.pkl"
        self.main_file_path = directory_path + "data.pkl"
        self.train_file_path = directory_path + "training_pid.pkl"
        self.validation_file_path = directory_path + "validation_pid.pkl"
        self.test_file_path = directory_path + "test_pid.pkl"

        # load general information from file
        with open(self.configuration_path, "rb") as f:
            self.database_configuration = pickle.load(f)

    # Getter and setter for file_path
    @property
    def directory_path(self):
        return self._directory_path

    @directory_path.setter
    def directory_path(self, value):

        create_directories_along_path(value)

        self._directory_path = value

        # Check if file exists, if not create it and save default file information
        if not os.path.exists(self.configuration_path):
            # Check if the parameters ensure correct signal processing
            signal_frequencies = [self.database_configuration[key] for key in self.signal_frequency_keys]
            minimum_signal_frequency = min(signal_frequencies)
            if minimum_signal_frequency <= 0:
                raise ValueError("Signal Frequencies must be larger than 0!")
            
            save_to_pickle(data = self.database_configuration, file_name = self.configuration_path)
    

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
        
        valid_datapoint_keys = ["ID", "sleep_stage_label", "RRI", "MAD", "SLP", "RRI_frequency", "MAD_frequency", "SLP_frequency", "SLP_predicted", "SLP_predicted_probability"]
        
        # Check if ID key is misleading
        if new_data["ID"] in valid_datapoint_keys:
            raise ValueError("Value for ID key: \"ID\" must not be the same as a key in the datapoint dictionary!")
        if "*" == new_data["ID"][-1]:
            file_generator = load_from_pickle(self.main_file_path)
            found_id = False
            for existing_data in file_generator:
                if existing_data["ID"] == new_data["ID"]:
                    found_id = True
                    break
            del file_generator

            if not found_id:
                raise ValueError("Value for ID key: \"ID\" must not end with the character: \"*\"!")
        
        # Check if key in new_data is unknown
        for new_data_key in new_data:
            if new_data_key not in valid_datapoint_keys:
                raise ValueError(f"Unknown key in datapoint: {new_data_key}")
        
        # Check if frequency keys are provided if signal keys are provided
        for signal_key_index in range(0, len(self.signal_keys)):
            signal_key = self.signal_keys[signal_key_index]
            signal_frequency_key = self.signal_frequency_keys[signal_key_index]
            if signal_key in new_data and signal_frequency_key not in new_data:
                raise ValueError(f"If you want to add a {signal_key} Signal (key: \"{signal_key}\"), then you must also provide the sampling frequency: \"{signal_frequency_key}\" !")
            
        # Check if declaration of sleep stage labels was provided when SLP signal is provided
        if "SLP" in new_data and "sleep_stage_label" not in new_data:
            raise ValueError("If you provide a SLP signal, you must also provide the sleep stage labels!")
        
        """
        -------------------------------
        Alter Entries In New Datapoint
        -------------------------------
        """

        # make sure ID is a string
        new_data["ID"] = str(new_data["ID"])

        # transform signals to np.ndarray and remove outliers
        for signal_key in self.signal_keys:
            if signal_key in new_data:
                new_data[signal_key] = np.array(new_data[signal_key])

        # make sure sampling frequency matches the one in the file, rescale signal if necessary
        for signal_key_index in range(0, len(self.signal_keys)):
            signal_key = self.signal_keys[signal_key_index]
            signal_frequency_key = self.signal_frequency_keys[signal_key_index]
            if signal_key in new_data and new_data[signal_frequency_key] != self.database_configuration[signal_frequency_key]:
                this_signal_type = "classification" if signal_key == "SLP" else "continuous"
                new_data[signal_key] = scale_signal(
                    signal = new_data[signal_key],
                    signal_frequency = new_data[signal_frequency_key],
                    target_frequency = self.database_configuration[signal_frequency_key],
                    signal_type = this_signal_type
                    )
                del this_signal_type
                new_data[signal_frequency_key] = self.database_configuration[signal_frequency_key]
        
        # check if signal length is uniform in new data point (in case ID does not exist in database)
        check_signal_length(
            data_point = new_data,
            database_configuration = self.database_configuration,
            signal_keys = self.signal_keys,
            signal_frequency_keys = self.signal_frequency_keys,
        )
        
        # Remove frequency keys from new_data (frequency should match the one in the file, saving it is unnecessary)
        for signal_frequency_key in self.signal_frequency_keys:
            if signal_frequency_key in new_data:
                del new_data[signal_frequency_key]
        
        # if datapoints in database were split, also split the new datapoint
        split_signals_needed = False
        if self.database_configuration["signal_length_seconds"] is not None:
            # make sure signal length is not longer than requested
            for signal_key_index in range(0, len(self.signal_keys)):
                signal_key = self.signal_keys[signal_key_index]
                signal_frequency_key = self.signal_frequency_keys[signal_key_index]
                if signal_key in new_data:   
                    if len(new_data[signal_key]) > np.ceil(self.database_configuration["signal_length_seconds"] * self.database_configuration[signal_frequency_key]):
                        split_signals_needed = True
                        break
            
        # split signals in dictionary and create new data dictionaries with ID that marks them as splits
        if split_signals_needed:
            splitted_data_dictionaries = split_signals_within_dictionary(
                data_dict = new_data,
                id_key = "ID",
                signal_keys = self.signal_keys,
                signal_frequencies = [self.database_configuration[key] for key in self.signal_frequency_keys],
                signal_length_seconds = self.database_configuration["signal_length_seconds"],
                wanted_shift_length_seconds = self.database_configuration["wanted_shift_length_seconds"],
                absolute_shift_deviation_seconds = self.database_configuration["absolute_shift_deviation_seconds"]
                ) # returns a list of dictionaries
            
            return splitted_data_dictionaries
        
        return [new_data]
    

    def _save_datapoint(self, new_data, unique_id = False, number_new_datapoints = 1):
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
        number_new_datapoints: int
            The number of splitted parts the datapoint was split into. If 0, the datapoint was not split.
        """
        
        if unique_id:
            self.database_configuration["number_datapoints"] += number_new_datapoints
            save_to_pickle(self.database_configuration, self.configuration_path)
            
            # Append new data point to the file
            append_to_pickle(data = new_data, file_name = self.main_file_path)

        else:
            # Create temporary file to save data in progress
            working_file_path = self.directory_path + "save_in_progress"
            working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")
            working_file = open(working_file_path, "ab")

            # save database configuration
            if self.database_configuration["number_datapoints"] is None:
                self.database_configuration["number_datapoints"] = number_new_datapoints
            else:
                self.database_configuration["number_datapoints"] += number_new_datapoints
            save_to_pickle(self.database_configuration, self.configuration_path)

            # Load data generator from the file
            file_generator = load_from_pickle(self.main_file_path)

            # Check if ID already exists in the data file, then overwrite keys
            not_appended = True
            for data_point in file_generator:
                if data_point["ID"] == new_data["ID"]:
                    print(f"ID \'{new_data['ID']}\' already exists in the data file. Existing keys will be overwritten with new values.")
                    not_appended = False

                    for key in new_data:
                        data_point[key] = new_data[key]
                        
                    # check if signal length is uniform with new signals
                    check_signal_length(
                        data_point = data_point,
                        database_configuration = self.database_configuration,
                        signal_keys = self.signal_keys,
                        signal_frequency_keys = self.signal_frequency_keys,
                    )
                
                # Append data point to the working file
                pickle.dump(data_point, working_file)
            
            # Append new data point if ID was not found
            if not_appended:
                pickle.dump(data_point, working_file)
            
            # close the working file
            working_file.close()
            
            # Remove the old file and rename the working file
            if os.path.exists(self.main_file_path):
                os.remove(self.main_file_path)
            os.rename(working_file_path, self.main_file_path)
    

    def save(self, data_dict, unique_id = False):
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
        # only first dictionary needs to be checked if present in database, as appending id's containing "shift" will not be allowed
        self._save_datapoint(corrected_data_dicts[0], unique_id)

        # append all other dictionaries at once
        with open(self.file_path, "ab") as f:
            for corrected_data_dict in corrected_data_dicts[1:]:
                pickle.dump(corrected_data_dict, f)

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
        valid_keys = ["ID", "RRI", "MAD", "SLP", "SLP_predicted", "SLP_predicted_probability"]
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
            raise ValueError("\'key_id_index\' must be a string, integer, or a key (also a string).")

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
    

    def split_to_uniform_length(
            self,
            signal_length_seconds: int,
            wanted_shift_length_seconds: int,
            absolute_shift_deviation_seconds: int
        ):
        """
        """

        try:
            # reverse signal split if it was applied
            if self.file_info["signal_length_seconds"] is not None:
                if signal_length_seconds == self.file_info["signal_length_seconds"] and wanted_shift_length_seconds == self.file_info["wanted_shift_length_seconds"] and absolute_shift_deviation_seconds == self.file_info["absolute_shift_deviation_seconds"]:
                    print("\nSignals were already split into uniform length using current settings. No need to split again.")
                    return
                print("\nSignals were already split into uniform length. Reversing the split to apply new split.")
                self.reverse_signal_split()
            
            # Check if signals can be split and fused correctly
            retrieve_possible_shift_lengths(
                min_shift_seconds = max(wanted_shift_length_seconds-absolute_shift_deviation_seconds, 1),
                max_shift_seconds = wanted_shift_length_seconds+absolute_shift_deviation_seconds,
                all_signal_frequencies = [self.file_info[key] for key in self.signal_frequency_keys]
            )

            # Create temporary file to save data in progress
            working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
            working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

            # update and save file information to working file
            self.file_info["signal_length_seconds"] = signal_length_seconds
            self.file_info["wanted_shift_length_seconds"] = wanted_shift_length_seconds
            self.file_info["absolute_shift_deviation_seconds"] = absolute_shift_deviation_seconds
            self.file_info["number_datapoints"] = None # reset number of datapoints, as it will be updated after splitting
            save_to_pickle(data = self.file_info, file_name = working_file_path)

            # Initialize progress bar
            print(f"\nSplitting database entries into multiple ones to ensure the contained signals span at most across: {signal_length_seconds} seconds.")
            progress_bar = DynamicProgressBar(total = len(self)) # type: ignore

            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            # skip file information
            next(file_generator)

            # iterate over database entries
            for data_point in file_generator:
                # make sure signal length is not longer than requested
                split_signals_needed = False
                for signal_key_index in range(0, len(self.signal_keys)):
                    signal_key = self.signal_keys[signal_key_index]
                    signal_frequency_key = self.signal_frequency_keys[signal_key_index]
                    if signal_key in data_point:   
                        if len(data_point[signal_key]) > np.ceil(signal_length_seconds * self.file_info[signal_frequency_key]):
                            split_signals_needed = True
                            break
                    
                # split signals in dictionary and create new data dictionaries with ID that marks them as splits
                if split_signals_needed:
            
                    splitted_data_dictionaries = split_signals_within_dictionary(
                        data_dict = data_point,
                        id_key = "ID",
                        signal_keys = self.signal_keys,
                        signal_frequencies = [self.file_info[key] for key in self.signal_frequency_keys],
                        signal_length_seconds = signal_length_seconds,
                        wanted_shift_length_seconds = wanted_shift_length_seconds,
                        absolute_shift_deviation_seconds = absolute_shift_deviation_seconds
                        ) # returns a list of dictionaries
                    
                    # append all other dictionaries at once
                    with open(working_file_path, "ab") as f:
                        for splitted_data_dict in splitted_data_dictionaries:
                            pickle.dump(splitted_data_dict, f)
                else:
                    # Append data point to the working file
                    append_to_pickle(data = data_point, file_name = working_file_path)
                
                # Update progress bar
                progress_bar.update()
            
            # Remove the old file and rename the working file
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            os.rename(working_file_path, self.file_path)
        
        finally:
            # remove working file if it still exists
            if os.path.exists(working_file_path):
                os.remove(working_file_path)
    

    def remove(self, key_id_index):
        """
        Remove data from the file path. The data can be removed by ID, key, or index.

        If removed key is one of the reshaped signals (RRI_windows, MAD_windows, SLP_windows), all reshaped 
        signals are removed (using 'remove_reshaped_signals' function).

        If removed key is one of the predicted signals (SLP_predicted, SLP_predicted_probability), all 
        predicted signals are removed (using 'remove_predicted_signals' function).

        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        key_id_index: str, int
            The ID, key, or index of the data to be removed.
        """

        # check if key_id_index is an id, a key, or an index
        valid_keys = ["ID", "RRI", "MAD", "SLP", "SLP_predicted", "SLP_predicted_probability"]

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # update and save file information to working file
        current_file_info = next(file_generator)
        current_file_info["number_datapoints"] = None
        save_to_pickle(data = current_file_info, file_name = working_file_path)

        if isinstance(key_id_index, str):
            if key_id_index[-1] == "*":
                key_id_index = key_id_index[:-1]  # remove trailing '*' if present

            # remove all keys from the data points
            if key_id_index in valid_keys:
                # Load data generator from the file
                with open(working_file_path, "ab") as f:
                    for data_point in file_generator:
                        if key_id_index in data_point:
                            del data_point[key_id_index]
                        pickle.dump(data_point, f)
            else:
                id_not_found = True
                with open(working_file_path, "ab") as f:
                    for data_point in file_generator:
                        if data_point["ID"] == key_id_index or data_point["ID"] == key_id_index + "*":
                            id_not_found = False
                            continue
                        pickle.dump(data_point, f)
                
                if id_not_found:
                    raise ValueError(f"ID {key_id_index} not found in the data file.")
                
        elif isinstance(key_id_index, int):
            count = 0
            index_out_of_bounds = True

            for data_point in file_generator:
                if count == key_id_index:
                    index_out_of_bounds = False
                    self.remove(data_point["ID"])  # remove data point by ID
                    break
                count += 1
            
            if index_out_of_bounds:
                raise ValueError(f"Index {key_id_index} out of bounds in the data file.")

        else:
            raise ValueError("\'key_id_index\' must be a string, integer, or a key (also a string).")

        del file_generator

        # Remove the old file and rename the working file
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        os.rename(working_file_path, self.file_path)
    

    def reverse_signal_split(self):
        """
        Reverses the signal split that was applied to the data during saving process.

        ATTENTION:  This function should be applied after the predicted Sleep Stage signals were added to 
                    the data.

                    Afterwards the data will not be suitable for our neural network model anymore, as the
                    signals might have overlength.
        
        RETURNS:
        ------------------------------
        None

        ARGUMENTS:
        ------------------------------
        None
        """

        # prevent running this function signals are not split
        if self.file_info["signal_length_seconds"] is None:
            print("\nData was not split yet. No need to reverse the split.")
            return

        # load all data point ids
        all_ids = self.load("ID")

        # put all ids that are part of a splitted signal into a list
        split_ids = []
        for this_id in all_ids: # type: ignore
            if "*" == this_id[-1]:
                if this_id[:-1] not in split_ids:
                    split_ids.append(this_id[:-1])

        # Create temporary files to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # update and save file information to working file
        self.file_info["signal_length_seconds"] = None
        self.file_info["wanted_shift_length_seconds"] = None
        self.file_info["absolute_shift_deviation_seconds"] = None
        self.file_info["number_datapoints"] = None # reset number of datapoints
        save_to_pickle(data = self.file_info, file_name = working_file_path)
        working_file = open(working_file_path, "ab")

        # create a separate file for each splitted parts (massively reduces computation time)
        id_list_paths = list()
        for i in range(len(split_ids)):
            id_list_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/_splitted_signals_" + str(i)
            id_list_path = find_non_existing_path(path_without_file_type = id_list_path, file_type = "pkl")
            id_list_paths.append(id_list_path)

        open_files = [open(path, "ab") for path in id_list_paths]

        # Initialize progress bar
        print("\nDistributing splitted data parts into individual files (Subprocess of Reversing Signal Split):")
        progress_bar = DynamicProgressBar(total = len(self)) # type: ignore

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # skip file information
        next(file_generator)

        # iterate over entries and append them to appropriate files
        for data_point in file_generator:
            this_id = data_point["ID"]
            appended = False
            for id_list_index in range(len(split_ids)):
                if this_id == split_ids[id_list_index] or this_id == split_ids[id_list_index] + "*":
                    pickle.dump(data_point, open_files[id_list_index])
                    appended = True
                    break
            if not appended:
                pickle.dump(data_point, working_file)
            
            # update progress bar
            progress_bar.update()
        
        del file_generator, progress_bar

        # Close all open files
        for open_file in open_files:
            open_file.close()

        # Initialize progress bar
        print("\nMerging data points back into the main file and reversing the Signal Split:")
        progress_bar = DynamicProgressBar(total = len(id_list_paths))

        # iterate over entries in database and reverse signal split
        for id_path in id_list_paths:

            # Load data generator from the file
            file_generator = load_from_pickle(id_path)

            # collect splitted data dictionaries
            splitted_data_dictionaries = list()

            for data_dict in file_generator:
                splitted_data_dictionaries.append(data_dict)
            
            del file_generator

            # fuse splitted data dictionaries
            fused_dictionary = fuse_splitted_signals_within_dictionaries(
                data_dictionaries = splitted_data_dictionaries,
                valid_signal_keys = self.signal_keys,
                valid_signal_frequencies = [self.file_info[key] for key in self.signal_frequency_keys],
            )

            # append fused dictionary to working file
            pickle.dump(fused_dictionary, working_file)

            # remove file containing transferred data points
            os.remove(id_path)

            # update progress bar
            progress_bar.update()
        
        # close working file
        working_file.close()
        
        # Remove the old file and rename the working file
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        os.rename(working_file_path, self.file_path)
    

    def separate_train_test_validation(
            self, 
            train_size = 0.8, 
            validation_size = 0.1,
            test_size = None, 
            random_state = None, 
            shuffle = True,
            join_splitted_parts = True,
            equally_distribute_signal_durations = True,
        ):
        """
        Depending whether "test_size" = None/float: Separate the data in the file into training and validation 
        data / training, validation, and test data. New files will be created in the same directory as the 
        main file. The file information will be saved to each file.

        Data that can not be used to train the network (i.e. missing "RRI" and/or "SLP") will be left in the
        main file.

        Data saved to this file likely has been split. You can choose whether to assign all splitted parts to
        one of the pids (training, validation, test) or distribute them equally or randomly across the pids.

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
        test_size: float or None
            The ratio of the test data.
        random_state: int
            The random state for the train_test_split function.
        shuffle: bool
            If True, the data will be shuffled before splitting.
        join_splitted_parts: bool
            If True, all splitted parts of a datapoint will be joined together and assigned to the same pid.
            If False, the splitted parts will be randomly distributed across the pids.
        stratify: bool
            If True, the data will be stratified by the sleep stage labels. This means that the distribution of 
            sleep stages in the training, validation, and test data will be similar to the distribution in the 
            original data. If False, the data will be split randomly.

            To enable stratification, each datapoint will be represented by only one sleep stage label, which
            is the most frequent one in the datapoint. Of course, this will not be representative if the datapoints
            represent long time periods and therefore contain a large amount of sleep stage labels.
        """

        # check arguments:
        if join_splitted_parts and self.file_info["signal_length_seconds"] is not None:
            signal_length_seconds = self.file_info["signal_length_seconds"]
            wanted_shift_length_seconds = self.file_info["wanted_shift_length_seconds"]
            absolute_shift_deviation_seconds = self.file_info["absolute_shift_deviation_seconds"]
            print("\nAttention: 'join_splitted_parts' is set to True, but the data was already split into uniform length. Depending on the number of datapoints, this could cause long computation times. Reversing signal split.")
            self.reverse_signal_split()
            self.separate_train_test_validation(train_size, validation_size, test_size, random_state, shuffle, join_splitted_parts, equally_distribute_signal_durations, stratify)
            self.split_to_uniform_length(signal_length_seconds, wanted_shift_length_seconds, absolute_shift_deviation_seconds)

        if test_size == 0:
            test_size = None
        
        if validation_size == 0 or validation_size is None and test_size is not None:
            validation_size = test_size
            test_size = None

        if test_size is None:
            if train_size + validation_size != 1: # type: ignore
                raise ValueError("The sum of train_size and validation_size must be 1.")
        else:
            if train_size + validation_size + test_size != 1: # type: ignore
                raise ValueError("The sum of train_size, validation_size, and test_size must be 1.")

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

        # iterate over entries and collect ids
        num_invalid_data_points = 0
        for data_point in file_generator:
            if "SLP" in data_point and "RRI" in data_point:
                if "MAD" in data_point:
                    id_with_rri_and_mad.append(data_point["ID"])
                else:
                    id_with_rri.append(data_point["ID"])
            else:
                num_invalid_data_points += 1
        
        del file_generator
        
        if num_invalid_data_points > 0:
            print(f"Attention: {num_invalid_data_points} datapoints do not contain a SLP and/or RRI signal and will be left in the main file.")
        del num_invalid_data_points

        # check which id's are more numerous
        if len(id_with_rri_and_mad) > len(id_with_rri):
            consider_identifications = id_with_rri_and_mad
            if len(id_with_rri) != 0:
                print(f"Attention: {len(id_with_rri)} datapoints without MAD signal will be left in the main file.")
        else:
            consider_identifications = id_with_rri
            if len(id_with_rri_and_mad) != 0:
                print(f"Attention: {len(id_with_rri_and_mad)} datapoints with MAD signal will be left in the main file.")
        
        del id_with_rri_and_mad, id_with_rri

        """
        To enable that the duration of the signals is equally distributed across the pids, we will assign
        the duration as class label and enable stratification.
        """
        if equally_distribute_signal_durations:
            # collect signal durations (we'll exploit just the rri length as this signal must be present) for each id
            rri_signal_lengths = [0 for _ in range(len(consider_identifications))]

            file_generator = load_from_pickle(self.file_path)
            next(file_generator)
            for data_point in file_generator:
                if data_point["ID"] in consider_identifications:
                    # calculate signal duration in seconds
                    rri_signal_lengths[consider_identifications.index(data_point["ID"])] = len(data_point["RRI"])

            # round signal durations to next 0.5 hours
            binwidth = self.file_info["RRI_frequency"]*1800 # 0.5 hour of RRI datapoints
            rri_signal_lengths = np.array(rri_signal_lengths, dtype=np.float64)
            rri_signal_lengths = np.round(rri_signal_lengths / binwidth)
            rri_signal_lengths = rri_signal_lengths.astype(np.int64)
            
        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # Change file information
        self.file_info["train_val_test_split_applied"] = True

        # save file information to working file
        save_to_pickle(data = self.file_info, file_name = working_file_path)
        working_file = open(working_file_path, "ab")

        if test_size is None:
            """
            split into training and validation data based on the chosen distribution method
            """

            if equally_distribute_signal_durations:
                train_data_ids, validation_data_ids = train_test_split(
                    copy.deepcopy(consider_identifications),
                    train_size = train_size,
                    random_state = random_state,
                    shuffle = shuffle,
                    stratify = rri_signal_lengths
                )
            
            else:                
                train_data_ids, validation_data_ids = train_test_split(
                    copy.deepcopy(consider_identifications),
                    train_size = train_size,
                    random_state = random_state,
                    shuffle = shuffle,
                )

            # ensure files are empty before writing
            for file_path in [self.file_info["train_file_path"], self.file_info["validation_file_path"]]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # open files
            open_files = [open(self.file_info["train_file_path"], "ab"), open(self.file_info["validation_file_path"], "ab")]

            # print progress
            print(f"\nDistributing {round(train_size*100,1)}% / {round(validation_size*100,1)}% of datapoints into training / validation pids, respectively:") # type: ignore
            progress_bar = DynamicProgressBar(total = len(self))
            
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            # skip file information
            next(file_generator)

            # save each data point to corresponding file
            for data_point in file_generator:
                if data_point["ID"] in train_data_ids:
                    pickle.dump(data_point, open_files[0])
                elif data_point["ID"] in validation_data_ids:
                    pickle.dump(data_point, open_files[1])
                else:
                    pickle.dump(data_point, working_file)
                
                # print progress
                progress_bar.update()
        
        else:
            """
            split into training validation and test data
            """

            if equally_distribute_signal_durations:
                train_data_ids, rest_data_ids = train_test_split(
                    copy.deepcopy(consider_identifications),
                    train_size = train_size,
                    random_state = random_state,
                    shuffle = shuffle,
                    stratify = rri_signal_lengths
                )
                validation_data_ids, test_data_ids = train_test_split(
                    rest_data_ids,
                    train_size = validation_size / (1 - train_size), # type: ignore
                    random_state = random_state,
                    shuffle = shuffle,
                    stratify = [rri_signal_lengths[consider_identifications.index(id)] for id in rest_data_ids]
                )

            else:                
                train_data_ids, rest_data_ids = train_test_split(
                    copy.deepcopy(consider_identifications),
                    train_size = train_size,
                    random_state = random_state,
                    shuffle = shuffle,
                )
                validation_data_ids, test_data_ids = train_test_split(
                    rest_data_ids,
                    train_size = validation_size / (1 - train_size), # type: ignore
                    random_state = random_state,
                    shuffle = shuffle,
                )
            
            # ensure files are empty before writing
            for file_path in [self.file_info["train_file_path"], self.file_info["validation_file_path"], self.file_info["test_file_path"]]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            # open files
            open_files = [open(self.file_info["train_file_path"], "ab"), open(self.file_info["validation_file_path"], "ab"), open(self.file_info["test_file_path"], "ab")]

            # print progress
            print(f"\nDistributing {round(train_size*100,1)}% / {round(validation_size*100,1)}% / {round(test_size*100,1)}% of datapoints into training / validation / test pids, respectively:") # type: ignore
            progress_bar = DynamicProgressBar(total = len(self))
            
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            # skip file information
            next(file_generator)

            # save each data point to corresponding file
            for data_point in file_generator:
                if data_point["ID"] in train_data_ids:
                    pickle.dump(data_point, open_files[0])
                elif data_point["ID"] in validation_data_ids:
                    pickle.dump(data_point, open_files[1])
                elif data_point["ID"] in test_data_ids:
                    pickle.dump(data_point, open_files[2])
                else:
                    pickle.dump(data_point, working_file)
                
                # print progress
                progress_bar.update()
        
        # close all open files
        for open_file in open_files:
            open_file.close()
        working_file.close()
        
        # Remove the old file and rename the working file
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            
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

        # Skip if train_val_test_split was not applied
        if not self.file_info["train_val_test_split_applied"]:
            return
        
        # edit file information
        self.file_info["train_val_test_split_applied"] = False

        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")
        working_file = open(working_file_path, "ab")

        # Append data points from main file
        main_file_generator = load_from_pickle(self.file_info["main_file_path"])
        next(main_file_generator)
        for data_point in main_file_generator:
            pickle.dump(data_point, working_file)
        
        # Remove main file
        if os.path.exists(self.file_info["main_file_path"]):
            os.remove(self.file_info["main_file_path"])

        # Append data points from training file
        training_file_generator = load_from_pickle(self.file_info["train_file_path"])
        next(training_file_generator)
        for data_point in training_file_generator:
            pickle.dump(data_point, working_file)
        
        # Remove training file
        if os.path.exists(self.file_info["train_file_path"]):
            os.remove(self.file_info["train_file_path"])
        
        # Append data points from validation file
        validation_file_generator = load_from_pickle(self.file_info["validation_file_path"])
        next(validation_file_generator)
        for data_point in validation_file_generator:
            pickle.dump(data_point, working_file)
        
        # Remove validation file
        if os.path.exists(self.file_info["validation_file_path"]):
            os.remove(self.file_info["validation_file_path"])
        
        # Append data points from test file if it exists
        if os.path.exists(self.file_info["test_file_path"]):
            test_file_generator = load_from_pickle(self.file_info["test_file_path"])
            next(test_file_generator)
            for data_point in test_file_generator:
                pickle.dump(data_point, working_file)
            
            os.remove(self.file_info["test_file_path"])
        
        # close working file
        working_file.close()
        
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

        # prevent runnning this function if data was split into training, validation, and test files
        if self.file_info["train_val_test_split_applied"]:
            raise ValueError("This function can only be called before data was split into training, validation, and test files.")

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
            if key in ["train_val_test_split_applied", "main_file_path", "train_file_path", "validation_file_path", "test_file_path", "signal_split_reversed"]:
                print(f"Attention: Key {key} is a reserved key and cannot be changed.")
                continue
            self.file_info[key] = new_file_info[key]
        
        # Check if the parameters ensure correct signal processing
        all_signal_frequency_keys = copy.deepcopy(self.signal_frequency_keys)
        all_signal_frequency_keys.append("SLP_predicted_frequency")
        all_signal_frequencies = [self.file_info[key] for key in all_signal_frequency_keys]
        minimum_signal_frequency = min(all_signal_frequencies)
        if minimum_signal_frequency <= 0:
            raise ValueError("Signal Frequencies must be larger than 0!")
        
        # Check if signals can be split and fused correctly
        retrieve_possible_shift_lengths(
            min_shift_seconds = max(self.file_info["wanted_shift_length_seconds"]-self.file_info["absolute_shift_deviation_seconds"], 1),
            max_shift_seconds = self.file_info["wanted_shift_length_seconds"]+self.file_info["absolute_shift_deviation_seconds"],
            all_signal_frequencies = all_signal_frequencies
        )

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
        Returns the number of data points in the file. 
        Usage: len(data_manager_instance)

        RETURNS:
        ------------------------------
        count: int
            The number of data points in the file.
        
        ARGUMENTS:
        ------------------------------
        None
        """

        if self.file_info["number_datapoints"] is not None:
            return self.file_info["number_datapoints"]
        
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        # Skip file information
        next(file_generator)

        count = 0
        for _ in file_generator:
            count += 1
        
        del file_generator

        # Update file information
        self.file_info["number_datapoints"] = count

        # Save file information to the file
        working_file_path = os.path.split(copy.deepcopy(self.file_path))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # save file information to working file
        save_to_pickle(data = self.file_info, file_name = working_file_path)

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        with open(working_file_path, "ab") as f:
            for data_point in file_generator:
                # Save data point to the working file
                pickle.dump(data_point, f)
        
        # Remove the old file and rename the working file
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        os.rename(working_file_path, self.file_path)

        return count
    

    def __contains__(self, id):
        """
        Check if the ID is in the file. 
        Usage: id in data_manager_instance:

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
        Iterate over all data points in the file. 
        Usage: for data_point in data_manager_instance:

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
    

    def __getitem__(self, key_id_index):
        """
        Loads data from file path. The data can be loaded by ID, key or index.
        Usage: data_manager_instance[key]

        Identical to data_manager_instance.load(key_id_index)

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

        return self.load(key_id_index)
    

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