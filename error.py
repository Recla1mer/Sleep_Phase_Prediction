from dataset_processing import *
from neural_network_model import *

processed_shhs_path = "Processed_Data/shhs_data.pkl"
shhs_training_data_path = processed_shhs_path[:-4] + "_training_pid.pkl"

default_window_reshape_parameters = {
        "nn_signal_duration_seconds": 10*3600,
        "number_windows": 1197, 
        "window_duration_seconds": 120, 
        "overlap_seconds": 90,
        "priority_order": [0, 1, 2, 3, 5, -1],
        "signal_type": "target",
        "pad_with": 0
}

data_manager = SleepDataManager(shhs_training_data_path)

for dict in data_manager:
    slp = dict['SLP']
    if slp.dtype == float:
        print("hey", dict['ID'])
    reshaped_slp = reshape_signal_to_overlapping_windows(slp, target_frequency=1/30, **default_window_reshape_parameters)
    if reshaped_slp.dtype == float:
        print(dict['ID'])