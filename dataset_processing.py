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
        signal_type: str = "feature"
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
        The type of signal. Either 'feature' or 'target'.
    """

    signal = np.array(signal) # type: ignore

    if signal_frequency == target_frequency:
        return signal # type: ignore
    
    if signal_type == "feature":
        return interpolate_signal(signal, signal_frequency, target_frequency)
    elif signal_type == "target":
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
        signal_type: str = "feature",
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

    signal = np.array(signal) # type: ignore

    # Scale number of datapoints in signal if sampling frequency is not equal to target frequency
    if sampling_frequency != target_frequency:
        signal = scale_signal(
            signal = signal, # type: ignore
            signal_frequency = sampling_frequency,
            target_frequency = target_frequency,
            signal_type = signal_type
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
    If signal was recorded with a different frequency than the neural network expects, the signal will be
    resampled to the target frequency, using following parameters:
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

    # Scale number of datapoints in signal if sampling frequency is not equal to target frequency
    if sampling_frequency != target_frequency:
        signal = scale_signal(
            signal = signal, # type: ignore
            signal_frequency = sampling_frequency,
            target_frequency = target_frequency,
            signal_type = signal_type
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