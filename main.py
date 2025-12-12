"""
Author: Johannes Peter Knoll

This file executes all the code needed to preprocess the data and train the neural network.
It is basically the less commented version of the notebook: "Classification_Demo.ipynb".
"""

# IMPORTS
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
import random

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *


"""
====================================
Setting Global Paths and File Names
====================================
"""

# paths to the data
original_shhs_data_path = "Raw_Data/SHHS_dataset.h5"
# original_gif_data_path = "Raw_Data/GIF_dataset.h5"
original_gif_ssg_data_path = "Raw_Data/gif_sleep_stages.pkl"
original_gif_sae_data_path = "Raw_Data/gif_sleep_apnea_events.pkl"

# file names
project_configuration_file = "Project_Configuration.pkl"

model_state_after_shhs_file = "Model_State_SHHS.pth"
model_state_after_shhs_gif_file = "Model_State.pth"

loss_per_epoch_shhs_file = "Loss_per_Epoch_SHHS.pkl"
loss_per_epoch_gif_file = "Loss_per_Epoch_GIF.pkl"

model_performance_file = "Model_Performance.pkl"


"""
=================================================================
Available (not always required) Project Configuration Parameters
=================================================================
"""

# parameters used to keep sampling frequencies uniform across all datapoints, see 'SleepDataManager' class in dataset_processing.py
sampling_frequency_parameters = {
    "RRI_frequency": 4,
    "MAD_frequency": 1,
    "SLP_frequency": 1/30,
}

# parameters that define how signals with overlength are cropped, see 'crop_oversized_data' function of 'SleepDataManager' class in dataset_processing.py
signal_cropping_parameters = {
    "signal_length_seconds": 36000,
    "shift_length_seconds_interval": (3600, 7200)
}

# parameters needed to ensure signals have uniform shape when passed to the network
padding_parameters = {
    "pad_feature_with": 0,
    "pad_target_with": 0
}

# parameters that alter the values of the signals, see 'map_slp_labels' and 'remove_outliers' functions in dataset_processing.py
value_mapping_parameters = {
    "rri_inlier_interval": (0.3, 2),
    "mad_inlier_interval": (None, None),
    "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
}

# parameters that set train, validation, and test sizes and how data is shuffled, see 'separate_train_test_validation' function of 'SleepDataManager' class in dataset_processing.py
pid_distribution_parameters = {
    "train_size": 0.8,
    "validation_size": 0.2,
    "test_size": None,
    "random_state": None,
    "shuffle": True,
    "join_splitted_parts": True,
    "equally_distribute_signal_durations": True,
    "stratify_by_target": False,
    "consider_targets_for_stratification": [],
}

# transformations applied to the data, see 'CustomSleepDataset' class in neural_network_model.py
def custom_transform(x):
    """
    This function is used to transform the data into a format that can be used by the neural network.
    It is used in the 'CustomSleepDataset' class in neural_network_model.py.
    
    RETURNS:
    ------------------------------
    x: torch.Tensor
        the transformed data
    """
    return torch.from_numpy(x).unsqueeze(0).float() # add a batch dimension and convert to float tensor


dataset_class_transform_parameters = {
    "feature_transform": custom_transform,
    "target_transform": None,
}

# parameters that alter the way the data is reshaped into windows, see 'reshape_signal_to_overlapping_windows' function in dataset_processing.py
window_reshape_parameters = {
    "reshape_to_overlapping_windows": True, # whether to reshape the signals to overlapping windows
    # following parameters are not required if 'reshape_to_overlapping_windows' is False
    "windows_per_signal": 1197, # 'number_windows' in 'reshape_signal_to_overlapping_windows' function
    "window_duration_seconds": 120,
    "overlap_seconds": 90,
    "priority_order": [3, 2, 1, 0],
    # following parameters are already included above
    # "signal_length_seconds": signal_cropping_parameters["signal_length_seconds"], # 'nn_signal_duration_seconds' in 'reshape_signal_to_overlapping_windows' function
    # "pad_feature_with": padding_parameters["pad_feature_with"],
    # "pad_target_with": padding_parameters["pad_target_with"]
}

# parameters that are used to normalize the data, see 'signal_normalization' function in dataset_processing.py
signal_normalization_parameters = {
    "normalize_rri": False, # whether to normalize the RRI signal
    "normalize_mad": False, # whether to normalize the MAD signal
    # following parameters are not required if 'normalize_rri' AND 'normalize_mad' are False
    "normalization_technique": "z-score", # "z-score" or "min-max"
    "normalization_mode": "local", # "local" or "global"
    # following parameters are not required if 'normalize_technique' is not "min-max"
    "normalization_max": 1,
    "normalization_min": 0,
}

# neural network model parameters, see 'DemoWholeNightModel' and 'DemoLocalSleepStageModel' class in neural_network_model.py
neural_network_model_parameters = {
    "neural_network_model": LongSequenceModel,
    # parameters necessary for neural network models based on whole night signals AND short time signals
    "number_target_classes": 4,
    "rri_convolutional_channels": [1, 8, 16, 32, 64],
    "mad_convolutional_channels": [1, 8, 16, 32, 64],
    "max_pooling_layers": 5,
    "fully_connected_features": 128,
    "convolution_dilations": [2, 4, 8, 16, 32],
    # parameters necessary for neural network models only based on whole night signals (do not append if using a model based on short time signals)
    "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * window_reshape_parameters["window_duration_seconds"]),
    "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * window_reshape_parameters["window_duration_seconds"]),
    "windows_per_signal": window_reshape_parameters["windows_per_signal"],
    # parameters necessary for neural network models only based on short time signals (do not append if using a model based on whole night signals)
    "rri_datapoints": int(sampling_frequency_parameters["RRI_frequency"] * signal_cropping_parameters["signal_length_seconds"]),
    "mad_datapoints": int(sampling_frequency_parameters["MAD_frequency"] * signal_cropping_parameters["signal_length_seconds"]),
}

# neural network hyperparameters for SHHS dataset, see 'main_model_training' function in this file
neural_network_hyperparameters_shhs = {
    "batch_size": 8,
    "number_epochs": 40,
    "lr_scheduler_parameters": {
        "number_updates_to_max_lr": 4,
        "start_learning_rate": 1 * 1e-5,
        "max_learning_rate": 1 * 1e-3,
        "end_learning_rate": 1 * 1e-6
    }
}

# neural network hyperparameters for GIF dataset, see 'main_model_training' function in this file
neural_network_hyperparameters_gif = {
    "batch_size": 8,
    "number_epochs": 40,
    "lr_scheduler_parameters": {
        "number_updates_to_max_lr": 4,
        "start_learning_rate": 1 * 1e-5,
        "max_learning_rate": 1 * 1e-3,
        "end_learning_rate": 1 * 1e-6
    }
}

# parameters used to filter the SHHS dataset
filter_shhs_data_parameters = {
    "shhs_min_duration_hours": 7,
    "shhs_filter_ids": []
}

gif_error_code_1 = ["SL007", "SL010", "SL012", "SL014", "SL022", "SL026", "SL039", "SL044", "SL049", "SL064", "SL070", "SL146", "SL150", "SL261", "SL266", "SL296", "SL303", "SL306", "SL342", "SL350", "SL410", "SL411", "SL416"]
gif_error_code_2 = ["SL032", "SL037", "SL079", "SL088", "SL114", "SL186", "SL255", "SL328", "SL336", "SL341", "SL344", "SL424"]
gif_error_code_3 = ["SL001", "SL004", "SL011", "SL025", "SL027", "SL034", "SL055", "SL057", "SL073", "SL075", "SL076", "SL083", "SL085", "SL087", "SL089", "SL096", "SL111", "SL116", "SL126", "SL132", "SL138", "SL141", "SL151", "SL157", "SL159", "SL166", "SL173", "SL174", "SL176", "SL178", "SL179", "SL203", "SL207", "SL208", "SL210", "SL211", "SL214", "SL217", "SL218", "SL221", "SL228", "SL229", "SL236", "SL237", "SL240", "SL245", "SL250", "SL252", "SL269", "SL286", "SL293", "SL294", "SL315", "SL348", "SL382", "SL384", "SL386", "SL389", "SL397", "SL406", "SL408", "SL418", "SL422", "SL428"]
gif_error_code_4 = ["SL061", "SL066", "SL091", "SL105", "SL202", "SL204", "SL205", "SL216", "SL305", "SL333", "SL349", "SL430", "SL439", "SL440"]
gif_error_code_5 = ["SL016", "SL040", "SL145", "SL199", "SL246", "SL268", "SL290", "SL316", "SL332", "SL365", "SL392", "SL426", "SL433", "SL438"]

# parameters used to filter the GIF dataset
filter_gif_data_parameters = {
    "gif_min_duration_hours": 7,
    "gif_filter_ids": gif_error_code_4 + gif_error_code_5
}

del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, neural_network_hyperparameters_shhs, neural_network_hyperparameters_gif, filter_shhs_data_parameters, filter_gif_data_parameters, gif_error_code_1, gif_error_code_2, gif_error_code_3, gif_error_code_4, gif_error_code_5

def check_project_configuration(parameters: dict):
    """
    This function checks if a correct computation is guaranteed by the chosen signal processing parameters.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    parameters: dict
        the parameters to check
    """

    # Check if unknown keys are present in the parameters
    known_keys = [
        "RRI_frequency", "MAD_frequency", "SLP_frequency",
        "signal_length_seconds", "shift_length_seconds_interval",
        "pad_feature_with", "pad_target_with",
        "rri_inlier_interval", "mad_inlier_interval", "target_classes", "disregard_classes_when_possible",
        "train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification",
        "feature_transform", "target_transform",
        "reshape_to_overlapping_windows", "windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order",
        "normalize_rri", "normalize_mad", "normalization_technique", "normalization_mode", "normalization_max", "normalization_min",
        "neural_network_model", "number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "rri_datapoints", "mad_datapoints",
        "shhs_min_duration_hours", "shhs_filter_ids", "gif_min_duration_hours", "gif_filter_ids"
    ]

    unknown_keys = [key for key in parameters if key not in known_keys]
    if len(unknown_keys) > 0:
        raise ValueError(f"Unknown parameters: {', '.join(unknown_keys)}. Please check your project configuration file.")

    # Check if all necessary parameters are provided
    required_keys = [
        "RRI_frequency", "MAD_frequency", "SLP_frequency",
        "rri_inlier_interval", "mad_inlier_interval", "target_classes",
        "signal_length_seconds", "shift_length_seconds_interval",
        "pad_feature_with", "pad_target_with",
        "train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification",
        "feature_transform", "target_transform",
        "reshape_to_overlapping_windows", 
        "normalize_rri", "normalize_mad"
    ]

    missing_keys = [key for key in required_keys if key not in parameters]
    if len(missing_keys) > 0:
        raise ValueError(f"Missing parameters: {', '.join(missing_keys)}.")
    
    # check if parameters for pid distribution are correct
    equals_one = parameters["train_size"]
    if parameters["validation_size"] is not None:
        equals_one += parameters["validation_size"]
    if parameters["test_size"] is not None:
        equals_one += parameters["test_size"]
    if equals_one != 1:
        raise ValueError("The sum of train_size, validation_size (and test_size) must be 1.")
    
    if not parameters["join_splitted_parts"] and parameters["equally_distribute_signal_durations"]:
        raise ValueError("If 'equally_distribute_signal_durations' is True, 'join_splitted_parts' must also be True. It does not make sense to equally distribute signal durations if you do not intend to keep all splitted parts of a datapoint together in the same pid.")

    # Check if all parameters for the neural network model are correctly provided
    neural_network_model = parameters["neural_network_model"]
    nnm_params = {key: parameters[key] for key in parameters if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters
    neural_network_model = neural_network_model(**nnm_params)
    
    # Check if all necessary parameters for reshaping the signal to overlapping windows are provided
    required_keys = ["signal_length_seconds", "windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order", "pad_feature_with", "pad_target_with"]
    
    if parameters["reshape_to_overlapping_windows"]:
        missing_keys = [key for key in required_keys if key not in parameters]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing parameters for reshaping to overlapping windows: {', '.join(missing_keys)}.")
    else:
        required_keys.remove("signal_length_seconds")
        required_keys.remove("pad_feature_with")
        required_keys.remove("pad_target_with")

        unnecessary_keys = [key for key in required_keys if key in parameters]
        if len(unnecessary_keys) > 0:
            raise ValueError(f"Unnecessary parameters as window reshaping is disabled: {', '.join(unnecessary_keys)}.")
    
    # Check if all necessary parameters for normalizing the signal are provided
    required_keys = ["normalization_mode"]
    if parameters["normalize_rri"] or parameters["normalize_mad"]:
        if "normalization_technique" not in parameters:
            raise ValueError("Parameter 'normalization_technique' is missing. Check 'signal_normalization_parameters' in 'main.py'.")
        if parameters["normalization_technique"] == "min-max":
            required_keys += ["normalization_max", "normalization_min"]

        missing_keys = [key for key in required_keys if key not in parameters]
        if len(missing_keys) > 0:
            raise ValueError(f"Missing parameters for signal normalization: {', '.join(missing_keys)}.")
    else:
        required_keys = ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]
        unnecessary_keys = [key for key in required_keys if key in parameters]
        if len(unnecessary_keys) > 0:
            raise ValueError(f"Unnecessary parameters as signal normalization is disabled: {', '.join(unnecessary_keys)}.")
    
    frequencies = [parameters["RRI_frequency"], parameters["MAD_frequency"], parameters["SLP_frequency"]]
    for freq in frequencies:
        # Check if number of datapoints is an integer
        number_nn_datapoints = parameters["signal_length_seconds"] * freq
        if int(number_nn_datapoints) != number_nn_datapoints:
            raise ValueError("Number of datapoints must be an integer. Choose 'signal_length_seconds' and 'target_frequency' accordingly.")
        number_nn_datapoints = int(number_nn_datapoints)

        if parameters["reshape_to_overlapping_windows"]:
            # Check parameters needed for reshaping the signal to overlapping windows
            datapoints_per_window = parameters["window_duration_seconds"] * freq
            window_overlap = parameters["overlap_seconds"] * freq

            if int(datapoints_per_window) != datapoints_per_window:
                raise ValueError("Datapoints per window must be an integer. Choose 'window_duration_seconds' and 'target_frequency' accordingly.")
            datapoints_per_window = int(datapoints_per_window)

            if int(window_overlap) != window_overlap:
                raise ValueError("Window overlap must be an integer. Choose 'overlap_seconds' and 'target_frequency' accordingly.")
            window_overlap = int(window_overlap)

            check_overlap = calculate_overlap(
                signal_length = number_nn_datapoints, 
                number_windows = parameters["windows_per_signal"], 
                datapoints_per_window = datapoints_per_window
                )
    
            if window_overlap != check_overlap:
                raise ValueError("Overlap does not match the number of windows and datapoints per window. Check parameters.")
    
    # check if number sleep stages matches the sleep stage label
    sleep_stages = []
    for key in parameters["target_classes"]:
        value = parameters["target_classes"][key]
        if value not in sleep_stages:
            sleep_stages.append(value)
    
    if len(sleep_stages) != parameters["number_target_classes"]:
        raise ValueError("Number of sleep stages does not match the target classes. Adjust parameters accordingly.")
    
    # check parameters that should be equal
    if parameters["reshape_to_overlapping_windows"]:
        if parameters["datapoints_per_rri_window"] != int(parameters["RRI_frequency"] * parameters["window_duration_seconds"]):
            raise ValueError("'datapoints_per_rri_window' must be equal to 'RRI_frequency' * 'window_duration_seconds'. Adjust parameters accordingly.")
        
        if parameters["datapoints_per_mad_window"] != int(parameters["MAD_frequency"] * parameters["window_duration_seconds"]):
            raise ValueError("'datapoints_per_mad_window' must be equal to 'MAD_frequency' * 'window_duration_seconds'. Adjust parameters accordingly.")


"""
============================================
Applying SleepDataManager Class To Our Data
============================================
"""

import h5py

def Process_SHHS_SSG_Dataset(
        path_to_shhs_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str,
    ):
    """
    This function processes our SHHS dataset for Sleep Stage Annotations (SSA). It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the SHHS dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    parameters this function accesses from "path_to_project_configuration". Afterwards we will use the 
    class to split the data into training, validation, and test pids (individual files).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_shhs_dataset: str
        the path to the SHHS dataset
    path_to_save_processed_data: str
        the path to save the processed SHHS dataset

    ### Parameters for change_file_information function in dataset_processing.py ###

    path_to_project_configuration: str
        the path to all signal processing parameters 
        (includes more parameters, but only the sleep_data_manager_parameters are needed here)
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    shhs_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    shhs_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "shift_length_seconds_interval"]} # signal_cropping_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["shhs_min_duration_hours"] * 3600
    filter_ids = project_configuration["shhs_filter_ids"]

    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')
    
    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    shhs_target_classes = {"wake": [0], "LS": [1,2], "DS": [3], "REM": [5], "artifact": ["other"]}

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    # check if patient ids are unique:
    shhs_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the SHHS dataset:")
    progress_bar = DynamicProgressBar(total = len(patients))

    # saving all data from SHHS dataset to the pickle file
    for patient_id in patients:
        # filter data
        if patient_id in filter_ids:
            continue
        if len(shhs_dataset["rri"][patient_id][:]) / shhs_dataset["rri"].attrs["freq"] < minimum_length_seconds: # type: ignore
            continue

        new_datapoint = {
            "ID": patient_id,
            "RRI": shhs_dataset["rri"][patient_id][:], # type: ignore
            "SLP": shhs_dataset["slp"][patient_id][:], # type: ignore
            "RRI_frequency": shhs_dataset["rri"].attrs["freq"], # type: ignore
            "SLP_frequency": shhs_dataset["slp"].attrs["freq"], # type: ignore
            "target_classes": copy.deepcopy(shhs_target_classes)
        }

        shhs_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()
    
    # if all splitted parts resulting from cropping an original datapoint are supposed to end up in the same pid,
    # we'll apply the signal cropping after pid distribution, otherwise before.

    if distribution_params["join_splitted_parts"]:
        # Train-, Validation- and Test-Pid Distribution
        shhs_data_manager.separate_train_test_validation(**distribution_params)

        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        shhs_data_manager.crop_oversized_data(**signal_crop_params)
    else:
        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        shhs_data_manager.crop_oversized_data(**signal_crop_params)
        
        # Train-, Validation- and Test-Pid Distribution
        shhs_data_manager.separate_train_test_validation(**distribution_params)


def Process_GIF_Dataset_h5(
        path_to_gif_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str
    ):
    """
    This function processes our GIF dataset. It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the GIF dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    parameters this function accesses from "path_to_project_configuration". Afterwards we will use the 
    class to split the data into training, validation, and test pids (individual files).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_gif_dataset: str
        the path to the GIF dataset

    Others: See 'Process_SHHS_Sleep_Dataset' function
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    gif_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    gif_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "shift_length_seconds_interval"]} # signal_cropping_parameters

    # access the GIF dataset
    gif_dataset = h5py.File(path_to_gif_dataset, 'r')

    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"wake": [0], "LS": [1, 2], "DS": [3], "REM": [5], "artifact": ["other"]}

    # accessing patient ids:
    patients = list(gif_dataset['stage'].keys()) # type: ignore

    # check if patient ids are unique:
    gif_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the GIF dataset:")
    progress_bar = DynamicProgressBar(total = len(patients))

    # saving all data from GIF dataset to the pickle file
    for patient_id in patients:
        new_datapoint = {
            "ID": patient_id,
            "RRI": gif_dataset["rri"][patient_id][:], # type: ignore
            "MAD": gif_dataset["mad"][patient_id][:], # type: ignore
            "SLP": np.array(gif_dataset["stage"][patient_id][:]).astype(int), # type: ignore
            "RRI_frequency": gif_dataset["rri"].attrs["freq"], # type: ignore
            "MAD_frequency": gif_dataset["mad"].attrs["freq"], # type: ignore
            "SLP_frequency": 1/30, # type: ignore
            "target_classes": copy.deepcopy(gif_target_classes)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # if all splitted parts resulting from cropping an original datapoint are supposed to end up in the same pid,
    # we'll apply the signal cropping after pid distribution, otherwise before.

    if distribution_params["join_splitted_parts"]:
        # Train-, Validation- and Test-Pid Distribution
        gif_data_manager.separate_train_test_validation(**distribution_params)

        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        gif_data_manager.crop_oversized_data(**signal_crop_params)
    else:
        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        gif_data_manager.crop_oversized_data(**signal_crop_params)

        # Train-, Validation- and Test-Pid Distribution
        gif_data_manager.separate_train_test_validation(**distribution_params)


def Process_GIF_SSG_Dataset(
        path_to_gif_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str
    ):
    """
    This function processes our GIF dataset for Sleep Stage Annotations (SSA). It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the GIF dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    parameters this function accesses from "path_to_project_configuration". Afterwards we will use the 
    class to split the data into training, validation, and test pids (individual files).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_gif_dataset: str
        the path to the GIF dataset

    Others: See 'Process_SHHS_Sleep_Dataset' function
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    gif_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    gif_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "shift_length_seconds_interval"]} # signal_cropping_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["gif_min_duration_hours"] * 3600
    filter_ids = project_configuration["gif_filter_ids"]

    # define the target classes (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"wake": [0], "LS": [1, 2], "DS": [3], "REM": [5], "artifact": ["other"]}

    gif_data_generator = load_from_pickle(path_to_gif_dataset)
    gif_length = 0
    for _ in gif_data_generator:
        gif_length += 1
    del gif_data_generator

    gif_data_generator = load_from_pickle(path_to_gif_dataset)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the GIF dataset:")
    progress_bar = DynamicProgressBar(total = gif_length)

    # saving all data from GIF dataset to the pickle file
    for generator_entry in gif_data_generator:
        # filter data
        if generator_entry["ID"] in filter_ids:
            continue
        if len(generator_entry["RRI"]) / generator_entry["RRI_frequency"] < minimum_length_seconds: # type: ignore
            continue

        new_datapoint = {
            "ID": generator_entry["ID"],
            "RRI": generator_entry["RRI"],
            "MAD": generator_entry["MAD"],
            "SLP": np.array(generator_entry["SLP"]).astype(int),
            "RRI_frequency": generator_entry["RRI_frequency"],
            "MAD_frequency": generator_entry["MAD_frequency"],
            "SLP_frequency": generator_entry["SLP_frequency"],
            "target_classes": copy.deepcopy(gif_target_classes)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # if all splitted parts resulting from cropping an original datapoint are supposed to end up in the same pid,
    # we'll apply the signal cropping after pid distribution, otherwise before.

    if distribution_params["join_splitted_parts"]:
        # Train-, Validation- and Test-Pid Distribution
        gif_data_manager.separate_train_test_validation(**distribution_params)

        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        gif_data_manager.crop_oversized_data(**signal_crop_params)
    else:
        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        gif_data_manager.crop_oversized_data(**signal_crop_params)

        # Train-, Validation- and Test-Pid Distribution
        gif_data_manager.separate_train_test_validation(**distribution_params)


def Process_GIF_SAE_Dataset(
        path_to_gif_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str
    ):
    """
    This function processes our GIF dataset for Sleep Apnea Events (SAE). It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the GIF dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    parameters this function accesses from "path_to_project_configuration". Afterwards we will use the 
    class to split the data into training, validation, and test pids (individual files).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_gif_dataset: str
        the path to the GIF dataset

    Others: See 'Process_SHHS_Sleep_Dataset' function
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    gif_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    gif_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "shift_length_seconds_interval"]} # signal_cropping_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["gif_min_duration_hours"] * 3600
    filter_ids = project_configuration["gif_filter_ids"]

    # define the target classes (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"Normal": [0], "Apnea": [1], "Obstructive Apnea": [2], "Central Apnea": [3], "Mixed Apnea": [4], "Hypopnea": [5], "Obstructive Hypopnea": [6], "Central Hypopnea": [7]}

    gif_data_generator = load_from_pickle(path_to_gif_dataset)
    gif_length = 0
    for _ in gif_data_generator:
        gif_length += 1
    del gif_data_generator

    gif_data_generator = load_from_pickle(path_to_gif_dataset)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the GIF dataset:")
    progress_bar = DynamicProgressBar(total = gif_length)

    # saving all data from GIF dataset to the pickle file
    for generator_entry in gif_data_generator:
        # filter data
        if generator_entry["ID"] in filter_ids:
            continue
        if len(generator_entry["RRI"]) / generator_entry["RRI_frequency"] < minimum_length_seconds: # type: ignore
            continue

        new_datapoint = {
            "ID": generator_entry["ID"],
            "RRI": generator_entry["RRI"],
            "MAD": generator_entry["MAD"],
            "SLP": np.array(generator_entry["SAE"]).astype(int),
            "RRI_frequency": generator_entry["RRI_frequency"],
            "MAD_frequency": generator_entry["MAD_frequency"],
            "SLP_frequency": generator_entry["SAE_frequency"],
            "target_classes": copy.deepcopy(gif_target_classes)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # if all splitted parts resulting from cropping an original datapoint are supposed to end up in the same pid,
    # we'll apply the signal cropping after pid distribution, otherwise before.

    if distribution_params["join_splitted_parts"]:
        # Train-, Validation- and Test-Pid Distribution
        gif_data_manager.separate_train_test_validation(**distribution_params)

        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        gif_data_manager.crop_oversized_data(**signal_crop_params)
    else:
        # Cropping datapoints with overlength (resulting in multiple splitted parts)
        gif_data_manager.crop_oversized_data(**signal_crop_params)

        # Train-, Validation- and Test-Pid Distribution
        gif_data_manager.separate_train_test_validation(**distribution_params)


def Process_NAKO_Dataset(
        path_to_nako_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str
    ):
    """
    If you processed the NAKO dataset using my other project: 'EDF_Processing', then the results of every .edf
    file should be saved as dictionaries to a pickle file, in the following format:

    {
        "ID":     
                Variation of the (.edf) file name the results were calculated for, 
                (number appended if multiple valid ecgregions)
        
        "time_period":
                List of the start and end time points (in seconds) of the time period in seconds
        
        "RRI":
                List of RR-intervals calculated from the r-peak locations within this time period.
        
        "RRI_frequency":
                Sampling frequency of the RR-intervals.
        
        "MAD":
                List of Mean Amplitude Deviation values calculated from the wrist acceleration data within 
                this time period.
        
        "MAD_frequency":
                Sampling frequency of the MAD values. Corresponds to 1 / parameters["mad_time_period_seconds"].
    }

    This function processes our NAKO dataset. It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the NAKO dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    parameters this function accesses from "path_to_project_configuration". Afterwards we will use the 
    class to split the data into training, validation, and test pids (individual files).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_nako_dataset: str
        the path to the NAKO dataset

    Others: See 'Process_SHHS_Sleep_Dataset' function
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    nako_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    nako_data_manager.change_uniform_frequencies(freq_params)

    ########################### REWORK THIS PART IF YOU WANT TO USE THE NAKO DATASET ###########################

    # access the NAKO dataset
    nako_dataset_generator = load_from_pickle(path_to_nako_dataset)

    # count total data points in dataset
    collect_ids = []
    total_data_points = 0
    for generator_entry in nako_dataset_generator:
        collect_ids.append(generator_entry["ID"])
        total_data_points += 1
    
    # check if all ids are unique:
    nako_data_manager.check_if_ids_are_unique(collect_ids)
    del collect_ids
    
    # reaccess the NAKO dataset
    del nako_dataset_generator
    nako_dataset_generator = load_from_pickle(path_to_nako_dataset)

    # showing progress bar
    print("\nPreproccessing datapoints from NAKO dataset (ensuring uniformity):")
    progress_bar = DynamicProgressBar(total = total_data_points)

    # saving all data from NAKO dataset to the pickle file
    for generator_entry in nako_dataset_generator:
        new_datapoint = {
            "ID": generator_entry["ID"],
            "RRI": generator_entry["RRI"],
            "MAD": generator_entry["MAD"],
            "RRI_frequency": generator_entry["RRI_frequency"],
            "MAD_frequency": generator_entry["MAD_frequency"],
        }

        nako_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()


"""
===========================================
Training And Applying Neural Network Model
===========================================
"""


def main_model_training_stage(
        neural_network_hyperparameters: dict,
        path_to_training_data_directory: str,
        path_to_project_configuration: str,
        path_to_model_state,
        path_to_updated_model_state: str,
        paths_to_validation_data_directories: list,
        path_to_loss_per_epoch: str,
    ):
    """
    Full implementation of project, with ability to easily change most important parameters to test different
    neural network architecture configurations. Some Parameters are hardcoded by design.

    The Data is accessed using the CustomSleepDataset class from neural_network_model.py. Data preprocessing
    adjustments performed through this class can be made using the parameters this function accesses from
    "path_to_project_configuration".

    Afterwards the neural network model is trained and tested. The accuracy and loss are saved in a pickle file
    for every epoch. The final model state dictionary is saved in a .pth file.

    The performance values are saved in a dictionary with the following format:
    {
        "train_accuracy": train_accuracy for each epoch (list),
        "train_avg_loss": train_avg_loss for each epoch (list),
        "{validation_file_name_without_extension}_accuracy": accuracy for each epoch (list) (multiple entries like this for each file in paths_to_processed_validation_data),
        "{validation_file_name_without_extension}_avg_loss": average loss for each epoch (list) (multiple entries like this for each file in paths_to_processed_validation_data),
    }

    
    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_hyperparameters: dict
        the hyperparameters for the neural network model training
        (batch_size, number_epochs, lr_scheduler_parameters)
    path_to_processed_training_data: str
        the path to the processed dataset containing the training data
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_updated_model_state: str
        the path to save the model state dictionary
    paths_to_processed_validation_data: list (of str)
        list of paths to the processed datasets containing the validation data (might be multiple)
    path_to_loss_per_epoch: str
        the path to save the accuracy values
    """

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters
    number_classes = project_configuration["number_target_classes"]

    # initialize parameters adjusting the data preprocessing
    CustomDatasetKeywords = dict()

    # retrieve dictionary needed to map sleep stage labels
    data_manager = BigDataManager(directory_path = path_to_training_data_directory)
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    CustomDatasetKeywords["slp_label_mapping"] = slp_label_mapping
    del data_manager

    # add data cleaning parameters
    CustomDatasetKeywords["rri_inlier_interval"] = project_configuration["rri_inlier_interval"]
    CustomDatasetKeywords["mad_inlier_interval"] = project_configuration["mad_inlier_interval"]

    # add desired signal length
    CustomDatasetKeywords["signal_length_seconds"] = project_configuration["signal_length_seconds"]

    # add padding parameters
    CustomDatasetKeywords["pad_feature_with"] = project_configuration["pad_feature_with"]
    CustomDatasetKeywords["pad_target_with"] = project_configuration["pad_target_with"]

    # add window_reshape_parameters
    CustomDatasetKeywords["reshape_to_overlapping_windows"] = project_configuration["reshape_to_overlapping_windows"]
    if project_configuration["reshape_to_overlapping_windows"]:
        CustomDatasetKeywords.update({key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]}) # window_reshape_parameters

    # add signal_normalization_parameters
    CustomDatasetKeywords["normalize_rri"] = project_configuration["normalize_rri"]
    CustomDatasetKeywords["normalize_mad"] = project_configuration["normalize_mad"]
    if project_configuration["normalize_rri"] or project_configuration["normalize_mad"]:
        CustomDatasetKeywords.update({key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]}) # signal_normalization_parameters

    # add transform parameters
    CustomDatasetKeywords.update({key: project_configuration[key] for key in ["feature_transform", "target_transform"]}) # dataset_class_transform_parameters
    CustomDatasetKeywords["modeling_task"] = "sleep_staging"

    training_dataset = AccessTransformDataset(path_to_data_directory = path_to_training_data_directory, pid = "train", **CustomDatasetKeywords)
    validation_datasets = [AccessTransformDataset(path_to_data_directory = path, pid = "validation", **CustomDatasetKeywords) for path in paths_to_validation_data_directories]

    del CustomDatasetKeywords, project_configuration
    
    """
    ----------------
    Hyperparameters
    ----------------
    """

    batch_size = neural_network_hyperparameters["batch_size"]
    number_epochs = neural_network_hyperparameters["number_epochs"]

    learning_rate_scheduler = CosineScheduler(
        number_updates_total = number_epochs,
        **neural_network_hyperparameters["lr_scheduler_parameters"]
    )

    """
    ---------------------------------------------
    Preparing Data For Training With Dataloaders
    ---------------------------------------------
    """

    train_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle=True)
    validation_dataloaders = [DataLoader(validation_dataset, batch_size = batch_size, shuffle=True) for validation_dataset in validation_datasets]
    # test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True)

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to learn on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    if path_to_model_state is not None:
        neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    """
    ----------------------------
    Loss Function And Optimizer
    ----------------------------
    """

    loss_function = nn.CrossEntropyLoss()
    optimizer_function = optim.Adam # type: ignore

    """
    ------------------------
    Training Neural Network
    ------------------------
    """
    # clearing sequence to remove progress bars of previous epoch
    clearing_sequence = "\033[2K"
    for _ in range(7+6*len(paths_to_validation_data_directories)):
        clearing_sequence += "\033[F" # Move cursor up
        clearing_sequence += "\033[2K" # Clear line

    # variables to store accuracy progress
    train_avg_loss = list()
    train_confusion_matrices = list()

    test_avg_loss = [[] for _ in range(len(paths_to_validation_data_directories))]
    test_confusion_matrices = [[] for _ in range(len(paths_to_validation_data_directories))]

    for t in range(1, number_epochs+1):
        # clearing previous epoch progress bars
        if t > 1:
            print(clearing_sequence, end='')

        print("")
        print("-"*10)
        print(f"Epoch {t}:")
        print("-"*10)

        train_loss, train_confusion_matrix = train_loop(
            dataloader = train_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            optimizer_fn = optimizer_function,
            lr_scheduler = learning_rate_scheduler,
            current_epoch = t,
            batch_size = batch_size,
            number_classes = number_classes
        )
        train_avg_loss.append(train_loss)
        train_confusion_matrices.append(train_confusion_matrix)

        for i, validation_dataloader in enumerate(validation_dataloaders):
            test_loss, test_confusion_matrix = test_loop(
                dataloader = validation_dataloader,
                model = neural_network_model,
                device = device,
                loss_fn = loss_function,
                batch_size = batch_size,
                number_classes = number_classes
            )

            test_avg_loss[i].append(test_loss)
            test_confusion_matrices[i].append(test_confusion_matrix)
        
        # acc = test_confusion_matrix.diagonal().sum() / test_confusion_matrix.sum()
        # print(acc)

    """
    ----------------------------------
    Saving Neural Network Model State
    ----------------------------------
    """

    create_directories_along_path(path_to_updated_model_state)
    
    torch.save(neural_network_model.state_dict(), path_to_updated_model_state)

    
    """
    --------------------------
    Saving Performance Values
    --------------------------
    """

    create_directories_along_path(path_to_loss_per_epoch)

    performance_values = {
        "train_avg_loss": train_avg_loss,
        "train_confusion_matrix": train_confusion_matrices,
    }
    short_names = copy.deepcopy(paths_to_validation_data_directories)
    for i in range(len(short_names)):
        if "SHHS" in short_names[i]:
            short_names[i] = "SHHS"
        elif "GIF" in short_names[i]:
            short_names[i] = "GIF"

    for i, name in enumerate(short_names):
        performance_values[f"{name}_avg_loss"] = test_avg_loss[i]
        performance_values[f"{name}_confusion_matrix"] = test_confusion_matrices[i]

    save_to_pickle(performance_values, path_to_loss_per_epoch)


def main_model_training_apnea(
        neural_network_hyperparameters: dict,
        path_to_training_data_directory: str,
        path_to_project_configuration: str,
        path_to_model_state,
        path_to_updated_model_state: str,
        paths_to_validation_data_directories: list,
        path_to_loss_per_epoch: str,
    ):
    """
    Full implementation of project, with ability to easily change most important parameters to test different
    neural network architecture configurations. Some Parameters are hardcoded by design.

    The Data is accessed using the CustomSleepDataset class from neural_network_model.py. Data preprocessing
    adjustments performed through this class can be made using the parameters this function accesses from
    "path_to_project_configuration".

    Afterwards the neural network model is trained and tested. The accuracy and loss are saved in a pickle file
    for every epoch. The final model state dictionary is saved in a .pth file.

    The performance values are saved in a dictionary with the following format:
    {
        "train_accuracy": train_accuracy for each epoch (list),
        "train_avg_loss": train_avg_loss for each epoch (list),
        "{validation_file_name_without_extension}_accuracy": accuracy for each epoch (list) (multiple entries like this for each file in paths_to_processed_validation_data),
        "{validation_file_name_without_extension}_avg_loss": average loss for each epoch (list) (multiple entries like this for each file in paths_to_processed_validation_data),
    }

    
    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_hyperparameters: dict
        the hyperparameters for the neural network model training
        (batch_size, number_epochs, lr_scheduler_parameters)
    path_to_processed_training_data: str
        the path to the processed dataset containing the training data
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_updated_model_state: str
        the path to save the model state dictionary
    paths_to_processed_validation_data: list (of str)
        list of paths to the processed datasets containing the validation data (might be multiple)
    path_to_loss_per_epoch: str
        the path to save the accuracy values
    """

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters
    number_classes = project_configuration["number_target_classes"]

    # initialize parameters adjusting the data preprocessing
    CustomDatasetKeywords = dict()

    # retrieve dictionary needed to map sleep stage labels
    data_manager = BigDataManager(directory_path = path_to_training_data_directory)
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    CustomDatasetKeywords["slp_label_mapping"] = slp_label_mapping
    del data_manager

    # add data cleaning parameters
    CustomDatasetKeywords["rri_inlier_interval"] = project_configuration["rri_inlier_interval"]
    CustomDatasetKeywords["mad_inlier_interval"] = project_configuration["mad_inlier_interval"]

    # add desired signal length
    CustomDatasetKeywords["signal_length_seconds"] = project_configuration["signal_length_seconds"]

    # add padding parameters
    CustomDatasetKeywords["pad_feature_with"] = project_configuration["pad_feature_with"]
    CustomDatasetKeywords["pad_target_with"] = project_configuration["pad_target_with"]

    # add window_reshape_parameters
    CustomDatasetKeywords["reshape_to_overlapping_windows"] = project_configuration["reshape_to_overlapping_windows"]
    if project_configuration["reshape_to_overlapping_windows"]:
        CustomDatasetKeywords.update({key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]}) # window_reshape_parameters

    # add signal_normalization_parameters
    CustomDatasetKeywords["normalize_rri"] = project_configuration["normalize_rri"]
    CustomDatasetKeywords["normalize_mad"] = project_configuration["normalize_mad"]
    if project_configuration["normalize_rri"] or project_configuration["normalize_mad"]:
        CustomDatasetKeywords.update({key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]}) # signal_normalization_parameters

    # add transform parameters
    CustomDatasetKeywords.update({key: project_configuration[key] for key in ["feature_transform", "target_transform"]}) # dataset_class_transform_parameters
    CustomDatasetKeywords["modeling_task"] = "apnea_detection"

    training_dataset = AccessTransformDataset(path_to_data_directory = path_to_training_data_directory, pid = "train", **CustomDatasetKeywords)
    validation_datasets = [AccessTransformDataset(path_to_data_directory = path, pid = "validation", **CustomDatasetKeywords) for path in paths_to_validation_data_directories]

    del CustomDatasetKeywords, project_configuration
    
    """
    ----------------
    Hyperparameters
    ----------------
    """

    batch_size = neural_network_hyperparameters["batch_size"]
    number_epochs = neural_network_hyperparameters["number_epochs"]

    learning_rate_scheduler = CosineScheduler(
        number_updates_total = number_epochs,
        **neural_network_hyperparameters["lr_scheduler_parameters"]
    )

    """
    ---------------------------------------------
    Preparing Data For Training With Dataloaders
    ---------------------------------------------
    """

    train_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle=True)
    validation_dataloaders = [DataLoader(validation_dataset, batch_size = batch_size, shuffle=True) for validation_dataset in validation_datasets]
    # test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True)

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to learn on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    if path_to_model_state is not None:
        neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    """
    ----------------------------
    Loss Function And Optimizer
    ----------------------------
    """

    loss_function = nn.CrossEntropyLoss()
    optimizer_function = optim.Adam # type: ignore

    """
    ------------------------
    Training Neural Network
    ------------------------
    """
    # clearing sequence to remove progress bars of previous epoch
    clearing_sequence = "\033[2K"
    for _ in range(7+6*len(paths_to_validation_data_directories)):
        clearing_sequence += "\033[F" # Move cursor up
        clearing_sequence += "\033[2K" # Clear line

    # variables to store accuracy progress
    train_avg_loss = list()
    train_confusion_matrices = list()

    test_avg_loss = [[] for _ in range(len(paths_to_validation_data_directories))]
    test_confusion_matrices = [[] for _ in range(len(paths_to_validation_data_directories))]

    for t in range(1, number_epochs+1):
        # clearing previous epoch progress bars
        if t > 1:
            print(clearing_sequence, end='')

        print("")
        print("-"*10)
        print(f"Epoch {t}:")
        print("-"*10)

        train_loss, train_confusion_matrix = train_loop(
            dataloader = train_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            optimizer_fn = optimizer_function,
            lr_scheduler = learning_rate_scheduler,
            current_epoch = t,
            batch_size = batch_size,
            number_classes = number_classes
        )
        train_avg_loss.append(train_loss)
        train_confusion_matrices.append(train_confusion_matrix)

        for i, validation_dataloader in enumerate(validation_dataloaders):
            test_loss, test_confusion_matrix = test_loop(
                dataloader = validation_dataloader,
                model = neural_network_model,
                device = device,
                loss_fn = loss_function,
                batch_size = batch_size,
                number_classes = number_classes
            )

            test_avg_loss[i].append(test_loss)
            test_confusion_matrices[i].append(test_confusion_matrix)
        
        # acc = test_confusion_matrix.diagonal().sum() / test_confusion_matrix.sum()
        # print(acc)

    """
    ----------------------------------
    Saving Neural Network Model State
    ----------------------------------
    """

    create_directories_along_path(path_to_updated_model_state)
    
    torch.save(neural_network_model.state_dict(), path_to_updated_model_state)

    
    """
    --------------------------
    Saving Performance Values
    --------------------------
    """

    create_directories_along_path(path_to_loss_per_epoch)

    performance_values = {
        "train_avg_loss": train_avg_loss,
        "train_confusion_matrix": train_confusion_matrices,
    }
    short_names = copy.deepcopy(paths_to_validation_data_directories)
    for i in range(len(short_names)):
        if "SHHS" in short_names[i]:
            short_names[i] = "SHHS"
        elif "GIF" in short_names[i]:
            short_names[i] = "GIF"

    for i, name in enumerate(short_names):
        performance_values[f"{name}_avg_loss"] = test_avg_loss[i]
        performance_values[f"{name}_confusion_matrix"] = test_confusion_matrices[i]

    save_to_pickle(performance_values, path_to_loss_per_epoch)


"""
======================================
Applying Trained Neural Network Model
======================================
"""


def main_model_predicting_stage_validation_set(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_save_results: str
        If actual results exist, predicted and actual results will be saved to this path
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------

    During Training, reshaping the signals to overlapping windows is done using the CustomSleepDataset class,
    which uses the SleepDataManager class from dataset_processing.py to access the data.
    Using this here would be possible as well, but not as clear because we want to add the predicted sleep
    stages directly to the database. Making the code more complex than necessary. Therefore, we will use the
    SleepDataManager class directly to access the data and reshape the signals to overlapping windows.
    """

    # accessing database
    data_manager = BigDataManager(directory_path = path_to_data_directory, pid = pid)

    # retrieve rri, mad, and slp frequencies
    rri_frequency = data_manager.database_configuration["RRI_frequency"]
    mad_frequency = data_manager.database_configuration["MAD_frequency"]
    slp_frequency = data_manager.database_configuration["SLP_frequency"]

    # determine if data contains sleep phases
    if not "SLP" in data_manager.load(0): # type: ignore
        raise ValueError("The apnea validation dataset must contain actual sleep phases for comparison.")

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]
    pad_target_with = project_configuration["pad_target_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]
    target_transform = project_configuration["target_transform"]

    del project_configuration

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    # prepare file that stores results
    if os.path.exists(path_to_save_results):
        os.remove(path_to_save_results)
    else:
        create_directories_along_path(path_to_save_results)

    results_file = open(path_to_save_results, "ab")

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = len(data_manager))


    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:

            try:

                """
                Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                """

                rri = final_data_preprocessing(
                    signal = copy.deepcopy(data_dict["RRI"]), # type: ignore
                    signal_id = "RRI",
                    inlier_interval = rri_inlier_interval,
                    target_frequency = rri_frequency,
                    signal_length_seconds = signal_length_seconds,
                    pad_with = pad_feature_with,
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    normalize = normalize_rri,
                    **common_signal_normalization_params,
                    datatype_mappings = [(np.float64, np.float32)],
                    transform = feature_transform
                )

                rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                rri = rri.to(device) # type: ignore

                # Ensure RRI is of the correct data type
                if not isinstance(rri, torch.FloatTensor):
                    rri = rri.float()

                # MAD preparation analogously to RRI
                if "MAD" in data_dict:
                    mad = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["MAD"]), # type: ignore
                        signal_id = "MAD",
                        inlier_interval = mad_inlier_interval,
                        target_frequency = mad_frequency,
                        signal_length_seconds = signal_length_seconds,
                        pad_with = pad_feature_with,
                        reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                        **common_window_reshape_params,
                        normalize = normalize_mad,
                        **common_signal_normalization_params,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = feature_transform
                    )

                    mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                    mad = mad.to(device) # type: ignore

                    if not isinstance(mad, torch.FloatTensor):
                        mad = mad.float()
                else:
                    mad = None

                actual_original_structure = map_slp_labels(
                    slp_labels = copy.deepcopy(data_dict["SLP"]), # type: ignore
                    slp_label_mapping = slp_label_mapping
                )
                original_signal_length = len(copy.deepcopy(actual_original_structure))

                # access processed target, reshape to overlapping windows and apply transformations
                slp = final_data_preprocessing(
                    signal = copy.deepcopy(data_dict["SLP"]),
                    signal_id = "SLP_stage_predict",
                    slp_label_mapping = slp_label_mapping,
                    target_frequency = slp_frequency,
                    signal_length_seconds = signal_length_seconds,
                    pad_with = pad_target_with,
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    normalize = False, # SLP is not normalized
                    datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                    transform = target_transform
                )
                
                """
                Applying Neural Network Model
                """

                # predictions in windows
                if reshape_to_overlapping_windows:
                    predictions_in_windows = neural_network_model(rri, mad)

                    """
                    Preparing Predicted Sleep Phases
                    """

                    predictions_in_windows = predictions_in_windows.cpu().numpy()

                    # reshape windows to original signal structure
                    # Lot of stuff happening below, so i explain the process:
                    # predictions_in_windows is a 2D array with shape (windows_per_signal, number_target_classes)
                    predictions_probability = np.empty((original_signal_length, 0))
                    for i in range(predictions_in_windows.shape[1]):
                        # get a list of probabilities for this target class
                        this_slp_stage_pred_probability = copy.deepcopy(predictions_in_windows[:, i])
                        # transform every probability to a list with the size of the SLP windows, with every entry 
                        # being the probability
                        pred_prob_expanded_to_windows = []
                        for pred_prob in this_slp_stage_pred_probability:
                            pred_prob_expanded_to_windows.append([pred_prob for _ in range(int(common_window_reshape_params["window_duration_seconds"]*slp_frequency))])
                        # if we now pass this list to the reverse window reshape function, using the frequency of the
                        # SLP signal, we get the probability for this sleep stage in the same sampling frequency as 
                        # the SLP signal
                        temp_original_structure = reverse_signal_to_windows_reshape(
                            signal_in_windows = pred_prob_expanded_to_windows,
                            target_frequency = slp_frequency,
                            original_signal_length = original_signal_length,
                            number_windows = common_window_reshape_params["windows_per_signal"],
                            window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                            overlap_seconds = common_window_reshape_params["overlap_seconds"],
                        )
                        temp_original_structure = np.array([[temp_val] for temp_val in temp_original_structure])
                        predictions_probability = np.append(predictions_probability, temp_original_structure, axis=1)
                    
                    # convert probabilities to sleep stages
                    predictions_original_structure = np.argmax(copy.deepcopy(predictions_probability), axis=1)

                    """
                    Saving Predicted (and Actual) Sleep Phases
                    """
                    
                    # remove padding from signals with overlapping windows
                    predictions_in_windows = remove_padding_from_windows(
                        signal_in_windows = predictions_in_windows,
                        target_frequency = slp_frequency,
                        original_signal_length = original_signal_length,
                        window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                        overlap_seconds = common_window_reshape_params["overlap_seconds"],
                    )

                    slp = remove_padding_from_windows(
                        signal_in_windows = slp, # type: ignore
                        target_frequency = slp_frequency,
                        original_signal_length = original_signal_length,
                        window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                        overlap_seconds = common_window_reshape_params["overlap_seconds"],
                    )

                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": predictions_probability,
                        "Predicted": predictions_original_structure,
                        "Actual": actual_original_structure,
                        "Predicted_in_windows": predictions_in_windows.argmax(1).flatten(),
                        "Actual_in_windows": slp
                    }
                
                # predictions not in windows
                else:
                    predictions_probability = neural_network_model(rri, mad)

                    predictions_probability = predictions_probability.cpu().numpy()
                    predicted = predictions_probability.argmax(1)

                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": np.array([predictions_probability[0]]), # remove padding
                        "Predicted": np.array([predicted[0]]), # remove padding
                        "Actual": np.array([slp[0]]),
                    }
                
                pickle.dump(results, results_file)

            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore
                    
            # update progress
            progress_bar.update()

    # Remove the old file and rename the working file
    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def main_model_predicting_stage(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        inference = False,
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_save_results: str
        If actual results exist, predicted and actual results will be saved to this path
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------

    During Training, reshaping the signals to overlapping windows is done using the CustomSleepDataset class,
    which uses the SleepDataManager class from dataset_processing.py to access the data.
    Using this here would be possible as well, but not as clear because we want to add the predicted sleep
    stages directly to the database. Making the code more complex than necessary. Therefore, we will use the
    SleepDataManager class directly to access the data and reshape the signals to overlapping windows.
    """

    # accessing database
    data_manager = BigDataManager(directory_path = path_to_data_directory, pid = pid)
    pid_file_path = data_manager.pid_paths[data_manager.current_pid]

    # retrieve rri, mad, and slp frequencies
    rri_frequency = data_manager.database_configuration["RRI_frequency"]
    mad_frequency = data_manager.database_configuration["MAD_frequency"]
    slp_frequency = data_manager.database_configuration["SLP_frequency"]

    # determine if data contains sleep phases
    actual_results_available = False
    if "SLP" in data_manager.load(0): # type: ignore
        actual_results_available = True

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]
    pad_target_with = project_configuration["pad_target_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]
    target_transform = project_configuration["target_transform"]

    del project_configuration

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    # prepare path that stores results, if necessary
    if actual_results_available:
        if os.path.exists(path_to_save_results):
            os.remove(path_to_save_results)
        else:
            create_directories_along_path(path_to_save_results)

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    if inference:
        stride_seconds = int(signal_length_seconds / 4)
    else:
        stride_seconds = int(signal_length_seconds)
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]
    
    slp_duration_seconds = int(1 / slp_frequency)

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = len(data_manager))

    results_file = open(path_to_save_results, "ab")

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:
            
            try:
                total_duration = int(len(data_dict["RRI"])/rri_frequency)
                
                strided_prediction_probabilities = [[] for i in range(total_duration)]
                strided_predicted_classes = [[] for _ in range(total_duration)]

                if actual_results_available:
                    actual_original_structure = map_slp_labels(
                        slp_labels = copy.deepcopy(data_dict["SLP"]), # type: ignore
                        slp_label_mapping = slp_label_mapping
                    )
                    original_signal_length = len(copy.deepcopy(actual_original_structure))
                else:
                    original_signal_length = int(np.ceil(signal_length_seconds * slp_frequency))

                start_time = -stride_seconds
                upper_bound = 0
                # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                while upper_bound < total_duration:
                    start_time += stride_seconds
                    upper_bound = start_time + signal_length_seconds
                    if upper_bound > total_duration:
                        upper_bound = total_duration
                        start_time = upper_bound - signal_length_seconds
                    if start_time < 0: # happens when signal length is longer than total duration
                        start_time = 0
                        upper_bound = signal_length_seconds

                    """
                    Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                    """

                    rri = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["RRI"][int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
                        signal_id = "RRI",
                        inlier_interval = rri_inlier_interval,
                        target_frequency = rri_frequency,
                        signal_length_seconds = signal_length_seconds,
                        pad_with = pad_feature_with,
                        reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                        **common_window_reshape_params,
                        normalize = normalize_rri,
                        **common_signal_normalization_params,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = feature_transform
                    )

                    rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                    rri = rri.to(device) # type: ignore

                    # Ensure RRI is of the correct data type
                    if not isinstance(rri, torch.FloatTensor):
                        rri = rri.float()

                    # MAD preparation analogously to RRI
                    if "MAD" in data_dict:
                        mad = final_data_preprocessing(
                            signal = copy.deepcopy(data_dict["MAD"][int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
                            signal_id = "MAD",
                            inlier_interval = mad_inlier_interval,
                            target_frequency = mad_frequency,
                            signal_length_seconds = signal_length_seconds,
                            pad_with = pad_feature_with,
                            reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                            **common_window_reshape_params,
                            normalize = normalize_mad,
                            **common_signal_normalization_params,
                            datatype_mappings = [(np.float64, np.float32)],
                            transform = feature_transform
                        )

                        mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                        mad = mad.to(device) # type: ignore

                        if not isinstance(mad, torch.FloatTensor):
                            mad = mad.float()
                    else:
                        mad = None
                
                    """
                    Applying Neural Network Model
                    """

                    # predictions in windows
                    if reshape_to_overlapping_windows:
                        predictions_probability_in_windows = neural_network_model(rri, mad)

                        """
                        Preparing Predicted Sleep Phases
                        """

                        predictions_probability_in_windows = predictions_probability_in_windows.cpu().numpy()
                        predictions_in_windows = predictions_probability_in_windows.argmax(1)

                        for i in range(len(predictions_in_windows)):
                            for j in range(window_duration_seconds):
                                this_index = start_time + int(i*(window_duration_seconds - overlap_seconds)) + j # type: ignore
                                if this_index >= total_duration:
                                    break
                                strided_prediction_probabilities[this_index].append(predictions_probability_in_windows[i])
                                strided_predicted_classes[this_index].append(predictions_in_windows[i])
                    
                    # predictions not in windows
                    else:
                        predictions_probability = neural_network_model(rri, mad)

                        predictions_probability = predictions_probability.cpu().numpy()
                        predicted = predictions_probability.argmax(1)

                        for i in range(start_time, upper_bound):
                            strided_prediction_probabilities[i].append(predictions_probability[0])
                            strided_predicted_classes[i].append(predicted[0])
                
                # combine strided predictions
                resolution_seconds = 30
                combined_predicted_probabilities = []
                combined_predicted_classes = []
                for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                    collected_probabilities = list()
                    collected_classes = list()
                    max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))

                    for j in range(i, max_bound):
                        collected_probabilities.extend(strided_prediction_probabilities[j])
                        collected_classes.extend(strided_predicted_classes[j])
                    
                    combined_predicted_probabilities.append(collected_probabilities)
                    combined_predicted_classes.append(collected_classes)

                mean_combined_prediction_probabilities = []
                for i in range(len(combined_predicted_probabilities)):
                    mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)

                predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                
                predictions_from_combined_classes = list()
                for row in combined_predicted_classes:
                    values, counts = np.unique(row, return_counts=True)
                    predictions_from_combined_classes.append(values[np.argmax(counts)])
                predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                if actual_results_available:
                    if slp_duration_seconds == resolution_seconds:
                        slp = actual_original_structure

                        if len(slp) != len(predictions_from_combined_probabilities):
                            raise ValueError("Length of actual sleep stages and predicted sleep stages do not match after rescaling.")
                    else:
                        slp = scale_classification_signal(
                            signal = actual_original_structure, # type: ignore
                            signal_frequency = slp_frequency,
                            target_frequency = 1/resolution_seconds
                        )

                        if len(slp) != len(predictions_from_combined_probabilities):
                            crop_to = min(len(slp), len(predictions_from_combined_probabilities))
                            slp = slp[:crop_to]
                            mean_combined_prediction_probabilities = mean_combined_prediction_probabilities[:crop_to]
                            predictions_from_combined_probabilities = predictions_from_combined_probabilities[:crop_to]
                            predictions_from_combined_classes = predictions_from_combined_classes[:crop_to]


                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": mean_combined_prediction_probabilities,
                        "Predicted": predictions_from_combined_probabilities,
                        "Predicted_2": predictions_from_combined_classes,
                        "Actual": slp,
                    }
                else:
                    # save results to existing dictionary
                    results = copy.deepcopy(data_dict)
                    results["SLP_prediction_probability"] = mean_combined_prediction_probabilities
                    results["SLP_from_prob"] = predictions_from_combined_probabilities
                    results["SLP_prediction_classes"] = combined_predicted_classes
                    results["SLP_from_class"] = predictions_from_combined_classes
                    results["SLP"] = predictions_from_combined_probabilities
                    
                
                pickle.dump(results, results_file)
            
            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore

                if not actual_results_available:
                    results = copy.deepcopy(data_dict)
                    pickle.dump(results, results_file)

                continue

            finally:        
                # update progress
                progress_bar.update()

    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def main_model_predicting_stage_inference(
        path_to_model_state: str,
        path_to_data_directory: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        inference = False,
        results_key = "SLP",
        data_length = None
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_save_results: str
        If actual results exist, predicted and actual results will be saved to this path
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------
    """

    if data_length == None:
        data_generator = load_from_pickle(path_to_data_directory)
        dataset_length = 0
        for _ in data_generator:
            dataset_length += 1
        del data_generator
    else:
        dataset_length = data_length

    data_generator = load_from_pickle(path_to_data_directory)

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    rri_frequency = project_configuration["RRI_frequency"]
    mad_frequency = project_configuration["MAD_frequency"]

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    target_classes = project_configuration["target_classes"]
    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]

    del project_configuration

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    working_file_path = "save_in_progress"
    for char_pos in range(len(path_to_save_results)-1, -1, -1):
        if path_to_save_results[char_pos] == "/":
            working_file_path = path_to_save_results[:char_pos+1] + "save_in_progress"
            break

    working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")
    working_file = open(working_file_path, "ab")

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    if inference:
        stride_seconds = int(signal_length_seconds / 4)
    else:
        stride_seconds = int(signal_length_seconds)
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = dataset_length)

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_generator:
            
            try:
                total_duration = min(int(len(data_dict["RRI"])/rri_frequency), int(len(data_dict["MAD"])/mad_frequency))
                
                strided_prediction_probabilities = [[] for i in range(total_duration)]
                strided_predicted_classes = [[] for _ in range(total_duration)]

                start_time = -stride_seconds
                upper_bound = 0
                # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                while upper_bound < total_duration:
                    start_time += stride_seconds
                    upper_bound = start_time + signal_length_seconds
                    if upper_bound > total_duration:
                        upper_bound = total_duration
                        start_time = upper_bound - signal_length_seconds
                    if start_time < 0: # happens when signal length is longer than total duration
                        start_time = 0
                        upper_bound = signal_length_seconds

                    """
                    Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                    """

                    rri = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["RRI"][int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
                        signal_id = "RRI",
                        inlier_interval = rri_inlier_interval,
                        target_frequency = rri_frequency,
                        signal_length_seconds = signal_length_seconds,
                        pad_with = pad_feature_with,
                        reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                        **common_window_reshape_params,
                        normalize = normalize_rri,
                        **common_signal_normalization_params,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = feature_transform
                    )

                    rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                    rri = rri.to(device) # type: ignore

                    # Ensure RRI is of the correct data type
                    if not isinstance(rri, torch.FloatTensor):
                        rri = rri.float()

                    # MAD preparation analogously to RRI
                    if "MAD" in data_dict:
                        mad = final_data_preprocessing(
                            signal = copy.deepcopy(data_dict["MAD"][int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
                            signal_id = "MAD",
                            inlier_interval = mad_inlier_interval,
                            target_frequency = mad_frequency,
                            signal_length_seconds = signal_length_seconds,
                            pad_with = pad_feature_with,
                            reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                            **common_window_reshape_params,
                            normalize = normalize_mad,
                            **common_signal_normalization_params,
                            datatype_mappings = [(np.float64, np.float32)],
                            transform = feature_transform
                        )

                        mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                        mad = mad.to(device) # type: ignore

                        if not isinstance(mad, torch.FloatTensor):
                            mad = mad.float()
                    else:
                        mad = None
                
                    """
                    Applying Neural Network Model
                    """

                    # predictions in windows
                    if reshape_to_overlapping_windows:
                        predictions_probability_in_windows = neural_network_model(rri, mad)

                        """
                        Preparing Predicted Sleep Phases
                        """

                        predictions_probability_in_windows = predictions_probability_in_windows.cpu().numpy()
                        predictions_in_windows = predictions_probability_in_windows.argmax(1)

                        for i in range(len(predictions_in_windows)):
                            for j in range(window_duration_seconds):
                                this_index = start_time + int(i*(window_duration_seconds - overlap_seconds)) + j # type: ignore
                                if this_index >= total_duration:
                                    break
                                strided_prediction_probabilities[this_index].append(predictions_probability_in_windows[i])
                                strided_predicted_classes[this_index].append(predictions_in_windows[i])
                    
                    # predictions not in windows
                    else:
                        predictions_probability = neural_network_model(rri, mad)

                        predictions_probability = predictions_probability.cpu().numpy()
                        predicted = predictions_probability.argmax(1)

                        for i in range(start_time, upper_bound):
                            strided_prediction_probabilities[i].append(predictions_probability[0])
                            strided_predicted_classes[i].append(predicted[0])
                
                # combine strided predictions
                resolution_seconds = 30
                combined_predicted_probabilities = []
                combined_predicted_classes = []
                for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                    collected_probabilities = list()
                    collected_classes = list()
                    max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))

                    for j in range(i, max_bound):
                        collected_probabilities.extend(strided_prediction_probabilities[j])
                        collected_classes.extend(strided_predicted_classes[j])
                    
                    combined_predicted_probabilities.append(collected_probabilities)
                    combined_predicted_classes.append(collected_classes)

                mean_combined_prediction_probabilities = []
                for i in range(len(combined_predicted_probabilities)):
                    mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)

                predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                
                predictions_from_combined_classes = list()
                for row in combined_predicted_classes:
                    values, counts = np.unique(row, return_counts=True)
                    predictions_from_combined_classes.append(values[np.argmax(counts)])
                predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                # save results to existing dictionary
                results = copy.deepcopy(data_dict)
                results[results_key + "_target_classes"] = target_classes
                results[results_key + "_frequency"] = 1 / resolution_seconds
                results[results_key + "_prediction_probability"] = mean_combined_prediction_probabilities
                results[results_key + "_from_probability"] = predictions_from_combined_probabilities
                results[results_key + "_prediction_classes"] = combined_predicted_classes
                results[results_key + "_from_majority"] = predictions_from_combined_classes
                results[results_key] = predictions_from_combined_probabilities
                
                pickle.dump(results, working_file)
            
            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore
                results = copy.deepcopy(data_dict)
                pickle.dump(results, working_file)

                continue

            finally:        
                # update progress
                progress_bar.update()

    working_file.close()

    # rename working file
    if os.path.exists(path_to_save_results):
        os.remove(path_to_save_results)
    create_directories_along_path(path_to_save_results)
    os.rename(working_file_path, path_to_save_results)
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def main_model_predicting_apnea_validation_set(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_save_results: str
        If actual results exist, predicted and actual results will be saved to this path
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------

    During Training, reshaping the signals to overlapping windows is done using the CustomSleepDataset class,
    which uses the SleepDataManager class from dataset_processing.py to access the data.
    Using this here would be possible as well, but not as clear because we want to add the predicted sleep
    stages directly to the database. Making the code more complex than necessary. Therefore, we will use the
    SleepDataManager class directly to access the data and reshape the signals to overlapping windows.
    """

    # accessing database
    data_manager = BigDataManager(directory_path = path_to_data_directory, pid = pid)

    # retrieve rri, mad, and slp frequencies
    rri_frequency = data_manager.database_configuration["RRI_frequency"]
    mad_frequency = data_manager.database_configuration["MAD_frequency"]
    slp_frequency = data_manager.database_configuration["SLP_frequency"]

    # determine if data contains sleep phases
    if not "SLP" in data_manager.load(0): # type: ignore
        raise ValueError("The apnea validation dataset must contain actual sleep phases for comparison.")

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]
    pad_target_with = project_configuration["pad_target_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]
    target_transform = project_configuration["target_transform"]

    del project_configuration

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    # prepare file that stores results
    if os.path.exists(path_to_save_results):
        os.remove(path_to_save_results)
    else:
        create_directories_along_path(path_to_save_results)

    results_file = open(path_to_save_results, "ab")

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Apnea Events:")
    progress_bar = DynamicProgressBar(total = len(data_manager))


    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:

            try:

                """
                Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                """

                rri = final_data_preprocessing(
                    signal = copy.deepcopy(data_dict["RRI"]), # type: ignore
                    signal_id = "RRI",
                    inlier_interval = rri_inlier_interval,
                    target_frequency = rri_frequency,
                    signal_length_seconds = signal_length_seconds,
                    pad_with = pad_feature_with,
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    normalize = normalize_rri,
                    **common_signal_normalization_params,
                    datatype_mappings = [(np.float64, np.float32)],
                    transform = feature_transform
                )

                rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                rri = rri.to(device) # type: ignore

                # Ensure RRI is of the correct data type
                if not isinstance(rri, torch.FloatTensor):
                    rri = rri.float()

                # MAD preparation analogously to RRI
                if "MAD" in data_dict:
                    mad = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["MAD"]), # type: ignore
                        signal_id = "MAD",
                        inlier_interval = mad_inlier_interval,
                        target_frequency = mad_frequency,
                        signal_length_seconds = signal_length_seconds,
                        pad_with = pad_feature_with,
                        reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                        **common_window_reshape_params,
                        normalize = normalize_mad,
                        **common_signal_normalization_params,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = feature_transform
                    )

                    mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                    mad = mad.to(device) # type: ignore

                    if not isinstance(mad, torch.FloatTensor):
                        mad = mad.float()
                else:
                    mad = None

                actual_original_structure = map_slp_labels(
                    slp_labels = copy.deepcopy(data_dict["SLP"]), # type: ignore
                    slp_label_mapping = slp_label_mapping
                )
                original_signal_length = len(copy.deepcopy(actual_original_structure))

                # access processed target, reshape to overlapping windows and apply transformations
                slp = final_data_preprocessing(
                    signal = copy.deepcopy(data_dict["SLP"]),
                    signal_id = "SLP_apnea_predict",
                    slp_label_mapping = slp_label_mapping,
                    target_frequency = slp_frequency,
                    signal_length_seconds = signal_length_seconds,
                    pad_with = pad_target_with,
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    normalize = False, # SLP is not normalized
                    datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                    transform = target_transform
                )
                
                """
                Applying Neural Network Model
                """

                # predictions in windows
                if reshape_to_overlapping_windows:
                    predictions_in_windows = neural_network_model(rri, mad)

                    """
                    Preparing Predicted Sleep Phases
                    """

                    predictions_in_windows = predictions_in_windows.cpu().numpy()

                    # reshape windows to original signal structure
                    # Lot of stuff happening below, so i explain the process:
                    # predictions_in_windows is a 2D array with shape (windows_per_signal, number_target_classes)
                    predictions_probability = np.empty((original_signal_length, 0))
                    for i in range(predictions_in_windows.shape[1]):
                        # get a list of probabilities for this target class
                        this_slp_stage_pred_probability = copy.deepcopy(predictions_in_windows[:, i])
                        # transform every probability to a list with the size of the SLP windows, with every entry 
                        # being the probability
                        pred_prob_expanded_to_windows = []
                        for pred_prob in this_slp_stage_pred_probability:
                            pred_prob_expanded_to_windows.append([pred_prob for _ in range(int(common_window_reshape_params["window_duration_seconds"]*slp_frequency))])
                        # if we now pass this list to the reverse window reshape function, using the frequency of the
                        # SLP signal, we get the probability for this sleep stage in the same sampling frequency as 
                        # the SLP signal
                        temp_original_structure = reverse_signal_to_windows_reshape(
                            signal_in_windows = pred_prob_expanded_to_windows,
                            target_frequency = slp_frequency,
                            original_signal_length = original_signal_length,
                            number_windows = common_window_reshape_params["windows_per_signal"],
                            window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                            overlap_seconds = common_window_reshape_params["overlap_seconds"],
                        )
                        temp_original_structure = np.array([[temp_val] for temp_val in temp_original_structure])
                        predictions_probability = np.append(predictions_probability, temp_original_structure, axis=1)
                    
                    # convert probabilities to sleep stages
                    predictions_original_structure = np.argmax(copy.deepcopy(predictions_probability), axis=1)

                    """
                    Saving Predicted (and Actual) Sleep Phases
                    """
                    
                    # remove padding from signals with overlapping windows
                    predictions_in_windows = remove_padding_from_windows(
                        signal_in_windows = predictions_in_windows,
                        target_frequency = slp_frequency,
                        original_signal_length = original_signal_length,
                        window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                        overlap_seconds = common_window_reshape_params["overlap_seconds"],
                    )

                    slp = remove_padding_from_windows(
                        signal_in_windows = slp, # type: ignore
                        target_frequency = slp_frequency,
                        original_signal_length = original_signal_length,
                        window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                        overlap_seconds = common_window_reshape_params["overlap_seconds"],
                    )

                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": predictions_probability,
                        "Predicted": predictions_original_structure,
                        "Actual": actual_original_structure,
                        "Predicted_in_windows": predictions_in_windows.argmax(1).flatten(),
                        "Actual_in_windows": slp
                    }
                
                # predictions not in windows
                else:
                    predictions_probability = neural_network_model(rri, mad)

                    predictions_probability = predictions_probability.cpu().numpy()
                    predicted = predictions_probability.argmax(1)

                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": np.array([predictions_probability[0]]), # remove padding
                        "Predicted": np.array([predicted[0]]), # remove padding
                        "Actual": np.array([slp[0]]),
                    }
                
                pickle.dump(results, results_file)

            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore
                    
            # update progress
            progress_bar.update()

    # Remove the old file and rename the working file
    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def better_int(value):
    if int(value) == value:
        return int(value)
    else:
        raise ValueError("Value cannot be converted to int without loss of information.")


def main_model_predicting_apnea(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        inference = False,
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_save_results: str
        If actual results exist, predicted and actual results will be saved to this path
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------

    During Training, reshaping the signals to overlapping windows is done using the CustomSleepDataset class,
    which uses the SleepDataManager class from dataset_processing.py to access the data.
    Using this here would be possible as well, but not as clear because we want to add the predicted sleep
    stages directly to the database. Making the code more complex than necessary. Therefore, we will use the
    SleepDataManager class directly to access the data and reshape the signals to overlapping windows.
    """

    # accessing database
    data_manager = BigDataManager(directory_path = path_to_data_directory, pid = pid)
    pid_file_path = data_manager.pid_paths[data_manager.current_pid]

    # retrieve rri, mad, and slp frequencies
    rri_frequency = data_manager.database_configuration["RRI_frequency"]
    mad_frequency = data_manager.database_configuration["MAD_frequency"]
    slp_frequency = data_manager.database_configuration["SLP_frequency"]

    # determine if data contains sleep phases
    actual_results_available = False
    if "SLP" in data_manager.load(0): # type: ignore
        actual_results_available = True

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]
    number_target_classes = project_configuration["number_target_classes"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]
    pad_target_with = project_configuration["pad_target_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]
    target_transform = project_configuration["target_transform"]

    del project_configuration

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    # prepare path that stores results, if necessary
    if actual_results_available:
        if os.path.exists(path_to_save_results):
            os.remove(path_to_save_results)
        else:
            create_directories_along_path(path_to_save_results)

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    if inference:
        stride_seconds = int(signal_length_seconds / 4)
    else:
        stride_seconds = int(signal_length_seconds)
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Apnea Events:")
    progress_bar = DynamicProgressBar(total = len(data_manager))

    results_file = open(path_to_save_results, "ab")

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:
            
            try:
                total_duration = int(len(data_dict["RRI"])/rri_frequency)
                
                strided_prediction_probabilities = [[] for i in range(total_duration)]
                strided_predicted_classes = [[] for _ in range(total_duration)]

                # if actual_results_available:
                #     actual_original_structure = map_slp_labels(
                #         slp_labels = copy.deepcopy(data_dict["SLP"]), # type: ignore
                #         slp_label_mapping = slp_label_mapping
                #     )
                #     original_signal_length = len(copy.deepcopy(actual_original_structure))
                # else:
                #     original_signal_length = int(np.ceil(signal_length_seconds * slp_frequency))

                start_time = -stride_seconds
                upper_bound = 0
                # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                while upper_bound < total_duration:
                    start_time += stride_seconds
                    upper_bound = start_time + signal_length_seconds
                    if upper_bound > total_duration:
                        upper_bound = total_duration
                        start_time = upper_bound - signal_length_seconds
                    if start_time < 0: # happens when signal length is longer than total duration
                        start_time = 0
                        upper_bound = signal_length_seconds

                    """
                    Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                    """

                    rri = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["RRI"][int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
                        signal_id = "RRI",
                        inlier_interval = rri_inlier_interval,
                        target_frequency = rri_frequency,
                        signal_length_seconds = signal_length_seconds,
                        pad_with = pad_feature_with,
                        reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                        **common_window_reshape_params,
                        normalize = normalize_rri,
                        **common_signal_normalization_params,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = feature_transform
                    )

                    rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                    rri = rri.to(device) # type: ignore

                    # Ensure RRI is of the correct data type
                    if not isinstance(rri, torch.FloatTensor):
                        rri = rri.float()

                    # MAD preparation analogously to RRI
                    if "MAD" in data_dict:
                        mad = final_data_preprocessing(
                            signal = copy.deepcopy(data_dict["MAD"][int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
                            signal_id = "MAD",
                            inlier_interval = mad_inlier_interval,
                            target_frequency = mad_frequency,
                            signal_length_seconds = signal_length_seconds,
                            pad_with = pad_feature_with,
                            reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                            **common_window_reshape_params,
                            normalize = normalize_mad,
                            **common_signal_normalization_params,
                            datatype_mappings = [(np.float64, np.float32)],
                            transform = feature_transform
                        )

                        mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                        mad = mad.to(device) # type: ignore

                        if not isinstance(mad, torch.FloatTensor):
                            mad = mad.float()
                    else:
                        mad = None
                
                    """
                    Applying Neural Network Model
                    """

                    # predictions in windows
                    if reshape_to_overlapping_windows:
                        predictions_probability_in_windows = neural_network_model(rri, mad)

                        """
                        Preparing Predicted Sleep Phases
                        """

                        predictions_probability_in_windows = predictions_probability_in_windows.cpu().numpy()
                        predictions_in_windows = predictions_probability_in_windows.argmax(1)

                        for i in range(len(predictions_in_windows)):
                            for j in range(window_duration_seconds):
                                this_index = start_time + int(i*(window_duration_seconds - overlap_seconds)) + j # type: ignore
                                if this_index >= total_duration:
                                    break
                                strided_prediction_probabilities[this_index].append(predictions_probability_in_windows[i])
                                strided_predicted_classes[this_index].append(predictions_in_windows[i])
                    
                    # predictions not in windows
                    else:
                        predictions_probability = neural_network_model(rri, mad)

                        predictions_probability = predictions_probability.cpu().numpy()
                        predicted = predictions_probability.argmax(1)

                        # for i in range(start_time, upper_bound):
                        for i in range(int(start_time+0.1*signal_length_seconds), int(upper_bound-0.2*signal_length_seconds)):
                            strided_prediction_probabilities[i].append(predictions_probability[0])
                            strided_predicted_classes[i].append(predicted[0])
                
                # fill empty entries
                normal_breathing_probability = [0 for _ in range(number_target_classes)]
                normal_breathing_probability[0] = 1
                for i in range(0, len(strided_prediction_probabilities)):
                    if len(strided_prediction_probabilities[i]) != 0:
                        break
                    if len(strided_prediction_probabilities[i]) == 0:
                        strided_prediction_probabilities[i].append(normal_breathing_probability)
                        strided_predicted_classes[i].append(0)
                for i in range(len(strided_prediction_probabilities)-1, -1, -1):
                    if len(strided_prediction_probabilities[i]) != 0:
                        break
                    if len(strided_prediction_probabilities[i]) == 0:
                        strided_prediction_probabilities[i].append(normal_breathing_probability)
                        strided_predicted_classes[i].append(0)
                
                # combine strided predictions
                resolution_seconds = signal_length_seconds

                combined_predicted_probabilities = []
                combined_predicted_classes = []

                current_apnea_event_position = -1
                start_appending_at = -1
                for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                    collected_probabilities = list()
                    collected_classes = list()
                    max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))
                    current_apnea_event = 0

                    for j in range(i, max_bound):
                        collected_probabilities.extend(strided_prediction_probabilities[j])

                        if j > start_appending_at: # prevent that apnea event is assigned to multiple time intervals (only let apneas pass that dont result from overlapping segments)
                            collected_classes.extend(strided_predicted_classes[j])

                            if current_apnea_event == 0:
                                for k in range(len(strided_predicted_classes[j])):
                                    if strided_predicted_classes[j][k] > 0:
                                        current_apnea_event = strided_predicted_classes[j][k]
                                        current_apnea_event_position = j + int(0.8*signal_length_seconds)
                                        break
                        else:
                            collected_classes.append(0)
                    
                    start_appending_at = current_apnea_event_position
                    
                    combined_predicted_probabilities.append(collected_probabilities)
                    combined_predicted_classes.append(collected_classes)

                mean_combined_prediction_probabilities = []
                for i in range(len(combined_predicted_probabilities)):
                    mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)

                predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                
                predictions_from_combined_classes = list()
                for row in combined_predicted_classes:
                    values, counts = np.unique(row, return_counts=True)
                    if len(values) > 1:
                        # remove 0 predictions if other classes are present
                        if 0 in values:
                            index_of_0 = np.where(values == 0)[0][0]
                            values = np.delete(values, index_of_0)
                            counts = np.delete(counts, index_of_0)

                    predictions_from_combined_classes.append(values[np.argmax(counts)])
                predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                if actual_results_available:
                    slp = []
                    original_signal_length = len(data_dict["SLP"])
                    stepsize_seconds = better_int(resolution_seconds * slp_frequency)
                    for lower_border in range(0, original_signal_length, stepsize_seconds):
                        upper_border = lower_border + stepsize_seconds
                        if upper_border > original_signal_length:
                            upper_border = original_signal_length

                        this_slp = final_data_preprocessing(
                            signal = data_dict["SLP"][lower_border:upper_border],
                            signal_id = "SLP_apnea_predict",
                            slp_label_mapping = slp_label_mapping,
                            target_frequency = slp_frequency,
                            signal_length_seconds = signal_length_seconds,
                            pad_with = pad_target_with,
                            reshape_to_overlapping_windows = False,
                            **common_window_reshape_params,
                            normalize = False, # SLP is not normalized
                            datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                            transform = target_transform
                        )

                        slp.append(this_slp[0])
                    
                    slp = np.array(slp)

                    for i in range(1, len(slp)-1):
                        if slp[i] != 0: 
                            # if predictions_from_combined_probabilities[i-1] != 0 and slp[i-1] == 0:
                            #     predictions_from_combined_probabilities[i-1] = 0
                            #     if predictions_from_combined_probabilities[i] == 0:
                            #         predictions_from_combined_probabilities[i] = slp[i]
                            # if predictions_from_combined_probabilities[i+1] != 0 and slp[i+1] == 0:
                            #     predictions_from_combined_probabilities[i+1] = 0
                            #     if predictions_from_combined_probabilities[i] == 0:
                            #         predictions_from_combined_probabilities[i] = slp[i]
                            
                            # if predictions_from_combined_classes[i-1] != 0 and slp[i-1] == 0:
                            #     predictions_from_combined_classes[i-1] = 0
                            #     if predictions_from_combined_classes[i] == 0:
                            #         predictions_from_combined_classes[i] = slp[i]
                            # if predictions_from_combined_classes[i+1] != 0 and slp[i+1] == 0:
                            #     predictions_from_combined_classes[i+1] = 0
                            #     if predictions_from_combined_classes[i] == 0:
                            #         predictions_from_combined_classes[i] = slp[i]

                            if predictions_from_combined_probabilities[i] == 0:
                                if predictions_from_combined_probabilities[i-1] != 0 and slp[i-1] == 0:
                                    predictions_from_combined_probabilities[i] = predictions_from_combined_probabilities[i-1]
                                    predictions_from_combined_probabilities[i-1] = 0
                                if predictions_from_combined_probabilities[i+1] != 0 and slp[i+1] == 0:
                                    predictions_from_combined_probabilities[i] = predictions_from_combined_probabilities[i+1]
                                    predictions_from_combined_probabilities[i+1] = 0

                            if predictions_from_combined_classes[i] == 0:
                                if predictions_from_combined_classes[i-1] != 0 and slp[i-1] == 0:
                                    predictions_from_combined_classes[i] = predictions_from_combined_classes[i-1]
                                    predictions_from_combined_classes[i-1] = 0
                                if predictions_from_combined_classes[i+1] != 0 and slp[i+1] == 0:
                                    predictions_from_combined_classes[i] = predictions_from_combined_classes[i+1]
                                    predictions_from_combined_classes[i+1] = 0
                    
                    # next_blocked = False
                    # for i in range(1, len(slp)):
                    #     if next_blocked:
                    #         next_blocked = False
                    #         continue
                    #     if slp[i-1] == 0 and slp[i] == 0:
                    #         if predictions_from_combined_probabilities[i-1] != 0 and predictions_from_combined_probabilities[i] != 0:
                    #             predictions_from_combined_probabilities[i] = 0
                    #             next_blocked = True
                    #         if predictions_from_combined_classes[i-1] != 0 and predictions_from_combined_classes[i] != 0:
                    #             predictions_from_combined_classes[i] = 0
                    #             next_blocked = True

                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": mean_combined_prediction_probabilities,
                        "Predicted": predictions_from_combined_probabilities,
                        "Predicted_2": predictions_from_combined_classes,
                        "Actual": slp,
                    }

                else:
                    # next_blocked = False
                    # for i in range(1, len(predictions_from_combined_classes)):
                    #     if next_blocked:
                    #         next_blocked = False
                    #         continue
                    #     if predictions_from_combined_probabilities[i-1] != 0 and predictions_from_combined_probabilities[i] != 0:
                    #         predictions_from_combined_probabilities[i] = 0
                    #         next_blocked = True
                    #     if predictions_from_combined_classes[i-1] != 0 and predictions_from_combined_classes[i] != 0:
                    #         predictions_from_combined_classes[i] = 0
                    #         next_blocked = True

                    # save results to existing dictionary
                    results = copy.deepcopy(data_dict)
                    results["SLP_prediction_probability"] = mean_combined_prediction_probabilities
                    results["SLP_from_prob"] = predictions_from_combined_probabilities
                    results["SLP_prediction_classes"] = combined_predicted_classes
                    results["SLP_from_class"] = predictions_from_combined_classes
                    results["SLP"] = predictions_from_combined_probabilities
                    
                
                pickle.dump(results, results_file)
            
            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore

                if not actual_results_available:
                    results = copy.deepcopy(data_dict)
                    pickle.dump(results, results_file)

                continue

            finally:        
                # update progress
                progress_bar.update()

    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def main_model_predicting_apnea_inference(
        path_to_model_state: str,
        path_to_data_directory: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        inference = False,
        results_key = "SAE",
        data_length = None
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_save_results: str
        If actual results exist, predicted and actual results will be saved to this path
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------
    """

    if data_length == None:
        data_generator = load_from_pickle(path_to_data_directory)
        dataset_length = 0
        for _ in data_generator:
            dataset_length += 1
        del data_generator
    else:
        dataset_length = data_length

    data_generator = load_from_pickle(path_to_data_directory)

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    rri_frequency = project_configuration["RRI_frequency"]
    mad_frequency = project_configuration["MAD_frequency"]

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]
    number_target_classes = project_configuration["number_target_classes"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    target_classes = project_configuration["target_classes"]
    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]

    del project_configuration

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    ----------------------------------
    Initializing Neural Network Model
    ----------------------------------
    """

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    working_file_path = "save_in_progress"
    for char_pos in range(len(path_to_save_results)-1, -1, -1):
        if path_to_save_results[char_pos] == "/":
            working_file_path = path_to_save_results[:char_pos+1] + "save_in_progress"
            break

    working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")
    working_file = open(working_file_path, "ab")

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    if inference:
        stride_seconds = int(signal_length_seconds / 4)
    else:
        stride_seconds = int(signal_length_seconds)
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = dataset_length)

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_generator:
            
            try:
                total_duration = min(int(len(data_dict["RRI"])/rri_frequency), int(len(data_dict["MAD"])/mad_frequency))
                
                strided_prediction_probabilities = [[] for i in range(total_duration)]
                strided_predicted_classes = [[] for _ in range(total_duration)]

                start_time = -stride_seconds
                upper_bound = 0
                # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                while upper_bound < total_duration:
                    start_time += stride_seconds
                    upper_bound = start_time + signal_length_seconds
                    if upper_bound > total_duration:
                        upper_bound = total_duration
                        start_time = upper_bound - signal_length_seconds
                    if start_time < 0: # happens when signal length is longer than total duration
                        start_time = 0
                        upper_bound = signal_length_seconds

                    """
                    Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                    """

                    rri = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["RRI"][int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
                        signal_id = "RRI",
                        inlier_interval = rri_inlier_interval,
                        target_frequency = rri_frequency,
                        signal_length_seconds = signal_length_seconds,
                        pad_with = pad_feature_with,
                        reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                        **common_window_reshape_params,
                        normalize = normalize_rri,
                        **common_signal_normalization_params,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = feature_transform
                    )

                    rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                    rri = rri.to(device) # type: ignore

                    # Ensure RRI is of the correct data type
                    if not isinstance(rri, torch.FloatTensor):
                        rri = rri.float()

                    # MAD preparation analogously to RRI
                    if "MAD" in data_dict:
                        mad = final_data_preprocessing(
                            signal = copy.deepcopy(data_dict["MAD"][int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
                            signal_id = "MAD",
                            inlier_interval = mad_inlier_interval,
                            target_frequency = mad_frequency,
                            signal_length_seconds = signal_length_seconds,
                            pad_with = pad_feature_with,
                            reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                            **common_window_reshape_params,
                            normalize = normalize_mad,
                            **common_signal_normalization_params,
                            datatype_mappings = [(np.float64, np.float32)],
                            transform = feature_transform
                        )

                        mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                        mad = mad.to(device) # type: ignore

                        if not isinstance(mad, torch.FloatTensor):
                            mad = mad.float()
                    else:
                        mad = None
                
                    """
                    Applying Neural Network Model
                    """

                    # predictions in windows
                    if reshape_to_overlapping_windows:
                        predictions_probability_in_windows = neural_network_model(rri, mad)

                        """
                        Preparing Predicted Sleep Phases
                        """

                        predictions_probability_in_windows = predictions_probability_in_windows.cpu().numpy()
                        predictions_in_windows = predictions_probability_in_windows.argmax(1)

                        for i in range(len(predictions_in_windows)):
                            for j in range(window_duration_seconds):
                                this_index = start_time + int(i*(window_duration_seconds - overlap_seconds)) + j # type: ignore
                                if this_index >= total_duration:
                                    break
                                strided_prediction_probabilities[this_index].append(predictions_probability_in_windows[i])
                                strided_predicted_classes[this_index].append(predictions_in_windows[i])
                    
                    # predictions not in windows
                    else:
                        predictions_probability = neural_network_model(rri, mad)

                        predictions_probability = predictions_probability.cpu().numpy()
                        predicted = predictions_probability.argmax(1)

                        # for i in range(start_time, upper_bound):
                        for i in range(int(start_time+0.1*signal_length_seconds), int(upper_bound-0.2*signal_length_seconds)):
                            strided_prediction_probabilities[i].append(predictions_probability[0])
                            strided_predicted_classes[i].append(predicted[0])
                
                # fill empty entries
                normal_breathing_probability = [0 for _ in range(number_target_classes)]
                normal_breathing_probability[0] = 1
                for i in range(0, len(strided_prediction_probabilities)):
                    if len(strided_prediction_probabilities[i]) != 0:
                        break
                    if len(strided_prediction_probabilities[i]) == 0:
                        strided_prediction_probabilities[i].append(normal_breathing_probability)
                        strided_predicted_classes[i].append(0)
                for i in range(len(strided_prediction_probabilities)-1, -1, -1):
                    if len(strided_prediction_probabilities[i]) != 0:
                        break
                    if len(strided_prediction_probabilities[i]) == 0:
                        strided_prediction_probabilities[i].append(normal_breathing_probability)
                        strided_predicted_classes[i].append(0)
                
                # combine strided predictions
                resolution_seconds = signal_length_seconds

                combined_predicted_probabilities = []
                combined_predicted_classes = []

                current_apnea_event_position = -1
                start_appending_at = -1
                for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                    collected_probabilities = list()
                    collected_classes = list()
                    max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))
                    current_apnea_event = 0

                    for j in range(i, max_bound):
                        collected_probabilities.extend(strided_prediction_probabilities[j])

                        if j > start_appending_at: # prevent that apnea event is assigned to multiple time intervals (only let apneas pass that dont result from overlapping segments)
                            collected_classes.extend(strided_predicted_classes[j])

                            if current_apnea_event == 0:
                                for k in range(len(strided_predicted_classes[j])):
                                    if strided_predicted_classes[j][k] > 0:
                                        current_apnea_event = strided_predicted_classes[j][k]
                                        current_apnea_event_position = j + int(signal_length_seconds) + 1
                                        break
                        else:
                            collected_classes.append(0)
                    
                    start_appending_at = current_apnea_event_position
                    
                    combined_predicted_probabilities.append(collected_probabilities)
                    combined_predicted_classes.append(collected_classes)

                mean_combined_prediction_probabilities = []
                for i in range(len(combined_predicted_probabilities)):
                    mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)

                predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                
                predictions_from_combined_classes = list()
                for row in combined_predicted_classes:
                    values, counts = np.unique(row, return_counts=True)
                    if len(values) > 1:
                        # remove 0 predictions if other classes are present
                        if 0 in values:
                            index_of_0 = np.where(values == 0)[0][0]
                            values = np.delete(values, index_of_0)
                            counts = np.delete(counts, index_of_0)

                    predictions_from_combined_classes.append(values[np.argmax(counts)])
                predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                # save results to existing dictionary
                results = copy.deepcopy(data_dict)
                results[results_key + "_target_classes"] = target_classes
                results[results_key + "_frequency"] = 1 / resolution_seconds
                results[results_key + "_prediction_probability"] = mean_combined_prediction_probabilities
                results[results_key + "_from_prob"] = predictions_from_combined_probabilities
                results[results_key + "_prediction_classes"] = combined_predicted_classes
                results[results_key + "_from_class"] = predictions_from_combined_classes
                results[results_key] = predictions_from_combined_probabilities
                    
                
                pickle.dump(results, working_file)
            
            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore

                results = copy.deepcopy(data_dict)
                pickle.dump(results, working_file)

                continue

            finally:        
                # update progress
                progress_bar.update()

    working_file.close()

    # rename working file
    if os.path.exists(path_to_save_results):
        os.remove(path_to_save_results)
    create_directories_along_path(path_to_save_results)
    os.rename(working_file_path, path_to_save_results)
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


"""
===========================
Evaluate Model Performance
===========================
"""


def print_model_performance(
        paths_to_pkl_files: list,
        path_to_project_configuration: str,
        prediction_result_key: str,
        actual_result_key: str,
        additional_score_function_args: dict = {"zero_division": np.nan},
        number_of_decimals = 2
    ):
    """
    This function calculates various performance parameters from the given pickle files (need to contain
    actual and predicted values).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_pkl_file: list
        the paths to the pickle files containing the data
    path_to_project_configuration: str
        the path to all signal processing parameters
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "test_predicted_results")
    actual_result_key: str
        the key that accesses the actual results in the data (for example: "test_actual_results")
    additional_score_function_args: dict
        additional arguments for some of the score functions (precision_score, recall_score, f1_score), e.g.:
            - average: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
                average parameter
            - zero_division: {"warn", 0.0, 1.0, np.nan}
                zero division parameter
            - Attention: if average not specified, the performance values are printed for average = 'macro', 'weighted' and None
    number_of_decimals: int
        the number of decimals to round the results to
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access dictionary that maps sleep stages (display labels) to integers
    sleep_stage_to_label = project_configuration["target_classes"]

    # Create a list of the integer labels, sorted
    integer_labels = np.array([value for value in sleep_stage_to_label.values()])
    integer_labels = np.unique(integer_labels)
    integer_labels.sort()

    # Create a list of the display labels
    display_labels = []
    for integer_label in integer_labels:
        for key, value in sleep_stage_to_label.items():
            if value == integer_label:
                display_labels.append(key)
                break
    
    # append labels to additional_score_function_args
    additional_score_function_args["labels"] = integer_labels

    # variables to store results
    all_predicted_results = np.empty(0)
    all_actual_results = np.empty(0)

    for file_path in paths_to_pkl_files:
        # Load the data
        data_generator = load_from_pickle(file_path)

        for data in data_generator:
            
            # Get the predicted and actual results
            predicted_results = data[prediction_result_key]
            actual_results = data[actual_result_key]

            # Flatten the arrays
            # print(predicted_results.shape, actual_results.shape)
            predicted_results = predicted_results.flatten()
            actual_results = actual_results.flatten()

            # Add the results to the arrays
            all_predicted_results = np.append(all_predicted_results, predicted_results)
            all_actual_results = np.append(all_actual_results, actual_results)

    # Define description of performance parameters
    accuracy_description = "Accuracy"
    kappa_description = "Cohen's Kappa"
    precision_description = "Precision"
    recall_description = "Recall"
    f1_description = "f1"

    # Calculate and print accuracy and cohen's kappa score
    accuracy = accuracy_score(all_actual_results, all_predicted_results)
    kappa = cohen_kappa_score(all_actual_results, all_predicted_results)

    print()
    print(f"{accuracy_description:^{15}}| {round(accuracy, number_of_decimals)}")
    print(f"{kappa_description:^{15}}| {round(kappa, number_of_decimals)}")
    print("\n")

    # Print the results
    if "average" in additional_score_function_args:
        # Calculate performance values
        precision = precision_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        recall = recall_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        f1 = f1_score(all_actual_results, all_predicted_results, **additional_score_function_args)

        if additional_score_function_args["average"] is not None:
            print(precision_description, round(precision, number_of_decimals)) # type: ignore
            print(recall_description, round(recall, number_of_decimals)) # type: ignore
            print(f1_description, round(f1, number_of_decimals)) # type: ignore

        return
    
    if "average" not in additional_score_function_args:
        precision = np.empty(0)
        recall = np.empty(0)
        f1 = np.empty(0)

        # Calculate Macro performance values
        additional_score_function_args["average"] = "macro"
        additional_score_function_args["zero_division"] = np.nan

        precision = np.append(precision, precision_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        recall = np.append(recall, recall_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        f1 = np.append(f1, f1_score(all_actual_results, all_predicted_results, **additional_score_function_args))

        # Calculate Micro performance values
        additional_score_function_args["average"] = "micro"

        precision = np.append(precision, precision_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        recall = np.append(recall, recall_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        f1 = np.append(f1, f1_score(all_actual_results, all_predicted_results, **additional_score_function_args))

        # Calculate Weighted performance values
        additional_score_function_args["average"] = "weighted"

        precision = np.append(precision, precision_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        recall = np.append(recall, recall_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        f1 = np.append(f1, f1_score(all_actual_results, all_predicted_results, **additional_score_function_args))

        # Round the results
        precision = np.round(precision, number_of_decimals)
        recall = np.round(recall, number_of_decimals)
        f1 = np.round(f1, number_of_decimals)

        # Calculate column width
        longest_precision_value = max([len(str(value)) for value in precision])
        longest_recall_value = max([len(str(value)) for value in recall])
        longest_f1_value = max([len(str(value)) for value in f1])

        column_header = ["Macro", "Micro", "Weighted"]
        column_width = max([len(label) for label in column_header])
        column_width = max([column_width, longest_precision_value, longest_recall_value, longest_f1_value])
        column_width += 2

        first_column_width = max([len(precision_description), len(recall_description), len(f1_description)]) + 1

        # Print the header
        print(" "*first_column_width, end = "")
        for label in column_header:
            print(f"|{label:^{column_width}}", end = "")
        print()
        print("-"*(first_column_width + len(column_header)*(column_width + 1)))

        # Print the results
        print(f"{precision_description:<{first_column_width}}", end = "")
        for value in precision:
            print(f"|{value:^{column_width}}", end = "")
        print()
        print(f"{recall_description:<{first_column_width}}", end = "")
        for value in recall:
            print(f"|{value:^{column_width}}", end = "")
        print()
        print(f"{f1_description:<{first_column_width}}", end = "")
        for value in f1:
            print(f"|{value:^{column_width}}", end = "")
        print("\n\n")

        # cleaning up
        del precision, recall, f1

        # Calculate the performance values for "average"=None
        additional_score_function_args["average"] = None

        precision = precision_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        recall = recall_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        f1 = f1_score(all_actual_results, all_predicted_results, **additional_score_function_args)

    # Round the results
    precision = np.round(precision, number_of_decimals)
    recall = np.round(recall, number_of_decimals)
    f1 = np.round(f1, number_of_decimals)

    # Calculate column width
    longest_precision_value = max([len(str(value)) for value in precision])
    longest_recall_value = max([len(str(value)) for value in recall])
    longest_f1_value = max([len(str(value)) for value in f1])

    column_width = max([len(label) for label in display_labels])
    column_width = max([column_width, longest_precision_value, longest_recall_value, longest_f1_value])
    column_width += 2

    first_column_width = max([len(precision_description), len(recall_description), len(f1_description)]) + 1
    
    # Print the header
    print(" "*first_column_width, end = "")
    for label in display_labels:
        print(f"|{label:^{column_width}}", end = "")
    print()
    print("-"*(first_column_width + len(display_labels)*(column_width + 1)))

    # Print the results
    print(f"{precision_description:<{first_column_width}}", end = "")
    for value in precision:
        print(f"|{value:^{column_width}}", end = "")
    print()
    print(f"{recall_description:<{first_column_width}}", end = "")
    for value in recall:
        print(f"|{value:^{column_width}}", end = "")
    print()
    print(f"{f1_description:<{first_column_width}}", end = "")
    for value in f1:
        print(f"|{value:^{column_width}}", end = "")
    print()


"""
=======================
Run Main Functionality
=======================

This file/project provides three main functionalities using the functions implemented above:
    1. Processing datasets and training the neural network model
    2. Evaluating the neural network model's performance
    3. Applying the trained neural network model to new data

Each functionality requires specific functions to be called in the correct order. The necessary sequence of 
operations is executed within the following functions.
"""


def run_ssg_model_training(
        path_to_model_directory: str,
        path_to_shhs_directory: str,
        path_to_gif_directory: str,
        neural_network_hyperparameters_shhs: dict,
        neural_network_hyperparameters_gif: dict,
    ):
    """
    Corresponds to the 1st main functionality: Processing datasets and training the neural network model.

    Before executing each step, the system checks whether data or model states already exist in the specified 
    paths. To prevent accidental loss of results and due to the potentially long computation time for each 
    step, the user will be prompted to confirm whether they want to overwrite the existing data or model 
    states.

    Most of the parameters are hardcoded in the functions called below. This ensures that the results of
    training different models or training with different configurations are stored analogously. 

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_model_directory: str
        the path to the directory where all results are stored
    path_to_processed_shhs: str
        the path to the file where the processed SHHS data is stored
    path_to_processed_gif: str
        the path to the file where the processed GIF data is stored
    neural_network_hyperparameters_shhs: dict
        the hyperparameters for the neural network model trained on SHHS data
    neural_network_hyperparameters_gif: dict
        the hyperparameters for the neural network model trained on GIF data
    """

    """
    ------------------------
    Preprocessing SHHS Data
    ------------------------
    """

    # check if processed data already exists
    user_response = "y"
    if os.path.exists(path_to_shhs_directory):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are attempting to process and save SHHS data to an existing directory. " +
                "The existing data may have been used to train and validate a model. " +
                "Overwriting it may prevent accurate assessment of the model's performance, " + 
                "as the validation pid will change. Do you want to overwrite? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            clean_and_remove_directory(path_to_shhs_directory)

    # process SHHS data
    if user_response == "y":
        Process_SHHS_SSG_Dataset(
            path_to_shhs_dataset = original_shhs_data_path,
            path_to_save_processed_data = path_to_shhs_directory,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            )
    
    """
    -----------------------
    Preprocessing GIF Data
    -----------------------
    """

    # check if processed data already exists
    user_response = "y"
    if os.path.exists(path_to_gif_directory):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are attempting to process and save GIF data to an existing directory. " +
                "The existing data may have been used to train and validate a model. " +
                "Overwriting it may prevent accurate assessment of the model's performance, " + 
                "as the validation pid will change. Do you want to overwrite? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            clean_and_remove_directory(path_to_gif_directory)

    # process GIF data
    if user_response == "y":
        Process_GIF_SSG_Dataset(
            path_to_gif_dataset = original_gif_ssg_data_path,
            path_to_save_processed_data = path_to_gif_directory,
            path_to_project_configuration = path_to_model_directory + project_configuration_file
            )
    
    """
    ------------------------------
    Training Network on SHHS Data
    ------------------------------
    """

    # check if model state already exists
    user_response = "y"
    if os.path.exists(path_to_model_directory + model_state_after_shhs_file):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are attempting to train the neural network on SHHS data and save its " +
                "final model state to an existing path. The existing model may have been used for further " +
                "analysis. Overwriting it will replace the model state and all subsequent results derived " +
                "from it. Do you want to overwrite? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_directory_files(directory_path = path_to_model_directory, keep_files = [project_configuration_file])

    # train neural network on SHHS data
    if user_response == "y":
        main_model_training_stage(
            neural_network_hyperparameters = neural_network_hyperparameters_shhs,
            path_to_training_data_directory = path_to_shhs_directory,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = None,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_file,
            paths_to_validation_data_directories = [path_to_shhs_directory, path_to_gif_directory],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_shhs_file,
            )

    """
    -----------------------------
    Training Network on GIF Data
    -----------------------------
    """

    # check if model state already exists
    user_response = "y"
    if os.path.exists(path_to_model_directory + model_state_after_shhs_gif_file):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are attempting to train the neural network on GIF data and save its " +
                "final model state to an existing path. The existing model may have been used for further " +
                "analysis. Overwriting it will replace the model state and all subsequent results derived " +
                "from it. Do you want to overwrite? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_directory_files(directory_path = path_to_model_directory, keep_files = [project_configuration_file, model_state_after_shhs_file])

    # train neural network on GIF data
    if user_response == "y":
        main_model_training_stage(
            neural_network_hyperparameters = neural_network_hyperparameters_gif,
            path_to_training_data_directory = path_to_gif_directory,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = path_to_model_directory + model_state_after_shhs_file,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
            paths_to_validation_data_directories = [path_to_shhs_directory, path_to_gif_directory],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_gif_file,
            )


def run_model_performance_evaluation_SSG(
        path_to_model_directory: str,
        path_to_splitted_shhs_directory: str,
        path_to_complete_shhs_directory: str,
        path_to_splitted_gif_directory: str,
        path_to_complete_gif_directory: str,
    ):
    """
    Corresponds to the 2nd main functionality: Evaluating the neural network model's performance.

    In order to do that, this function will first predict the sleep stages for the SHHS and GIF validation
    datasets using the trained model and save the results to a pkl-file. Then, the performance of the model
    will be calculated from these results and printed to the console.

    Before executing the functions to predict the results, the system checks whether those already exist in 
    the specified paths. Due to the potentially long computation time, the user will be prompted to confirm 
    whether they want to overwrite the existing data.

    Most of the parameters are hardcoded in the functions called below. This ensures that the results of
    the predictions for different models are stored analogously. 

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_model_directory: str
        the path to the directory where all results are stored
    path_to_splitted_shhs_directory: str
        the path to the directory where the SHHS validation data is stored after signal splitting
    path_to_complete_shhs_directory: str
        the path to the directory where the SHHS validation data is stored before signal splitting
    path_to_splitted_gif_directory: str
        the path to the directory where the GIF validation data is stored after signal splitting
    path_to_complete_gif_directory: str
        the path to the directory where the GIF validation data is stored before signal splitting
    """

    # check wether model performance can be evaluated with inference (predict and combine results for
    # overlapping segments of data of variable lengths)
    # possible if neural network input length is substantially shorther than the available data
    apply_inference = True
    with open(path_to_model_directory + project_configuration_file, "rb") as f:
        project_configuration = pickle.load(f)
        signal_length_seconds = project_configuration["signal_length_seconds"]
    # if signal_length_seconds > 7 * 3600:
    #     apply_inference = False
    del project_configuration, signal_length_seconds

    """
    ----------
    SHHS Data
    ----------
    """

    # path to save the predictions
    shhs_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_SHHS_Splitted_Validation_Pid.pkl"
    shhs_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_SHHS_Complete_Validation_Pid.pkl"

    """
    -------------------------------------
    Predicting Sleep Phases of SHHS Data
    -------------------------------------
    """

    # remove predictions if they already exist
    delete_files([shhs_splitted_validation_pid_results_path, shhs_complete_validation_pid_results_path])

    # make predictions for the splitted data
    main_model_predicting_stage_validation_set(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_data_directory = path_to_splitted_shhs_directory,
        pid = "validation",
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        path_to_save_results = shhs_splitted_validation_pid_results_path,
    )

    # make predictions for the complete data
    main_model_predicting_stage(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_data_directory = path_to_complete_shhs_directory,
        pid = "validation",
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        path_to_save_results = shhs_complete_validation_pid_results_path,
        inference = apply_inference,
    )

    """
    -------------------------------
    Print Performance on SHHS Data
    -------------------------------
    """

    # calculate and print performance results for splitted data
    print_headline("Performance on Splitted SHHS Data:", symbol_sequence="=")
    print("   - Reflecting performance on single data segments of required signal length")

    print_model_performance(
        paths_to_pkl_files = [shhs_splitted_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    # calculate and print performance results for complete data
    print_headline("Performance on Complete SHHS Data:", symbol_sequence="=")
    print("   - Reflecting performance on data segments of variable lengths, as they occur in practice")
    if apply_inference:
        print("   - Predictions were made using inference, combining results from overlapping data segments")
    else:
        print("   - Predictions were made without inference, as the required input length is too long")
    
    print_headline("Performance calculated from combined Probability:", symbol_sequence="-")

    print_model_performance(
        paths_to_pkl_files = [shhs_complete_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    print_headline("Performance calculated from majority vote:", symbol_sequence="-")

    print_model_performance(
        paths_to_pkl_files = [shhs_complete_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted_2",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    """
    ---------
    GIF Data
    ---------
    """

    # path to save the predictions
    gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
    gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

    """
    ------------------------------------
    Predicting Sleep Phases of GIF Data
    ------------------------------------
    """

    # remove predictions if they already exist
    delete_files([gif_splitted_validation_pid_results_path, gif_complete_validation_pid_results_path])
    # make predictions for the splitted data
    main_model_predicting_stage_validation_set(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_data_directory = path_to_splitted_gif_directory,
        pid = "validation",
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        path_to_save_results = gif_splitted_validation_pid_results_path,
    )

    # make predictions for the complete data
    main_model_predicting_stage(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_data_directory = path_to_complete_gif_directory,
        pid = "validation",
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        path_to_save_results = gif_complete_validation_pid_results_path,
        inference = apply_inference,
    )

    """
    ------------------------------
    Print Performance on GIF Data
    ------------------------------
    """

    # calculate and print performance results for splitted data
    print_headline("Performance on Splitted GIF Data:", symbol_sequence="=")
    print("   - Reflecting performance on single data segments of required signal length")

    print_model_performance(
        paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    # calculate and print performance results for complete data
    print_headline("Performance on Complete GIF Data:", symbol_sequence="=")
    print("   - Reflecting performance on data segments of variable lengths, as they occur in practice")
    if apply_inference:
        print("   - Predictions were made using inference, combining results from overlapping data segments")
    else:
        print("   - Predictions were made without inference, as the required input length is too long")
    
    print_headline("Performance calculated from combined Probability:", symbol_sequence="-")

    print_model_performance(
        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    print_headline("Performance calculated from majority vote:", symbol_sequence="-")

    print_model_performance(
        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted_2",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )


def run_model_performance_evaluation_SAE(
        path_to_model_directory: str,
        path_to_splitted_gif_directory: str,
        path_to_complete_gif_directory: str,
    ):
    """
    Corresponds to the 2nd main functionality: Evaluating the neural network model's performance.

    In order to do that, this function will first predict the sleep stages for the SHHS and GIF validation
    datasets using the trained model and save the results to a pkl-file. Then, the performance of the model
    will be calculated from these results and printed to the console.

    Before executing the functions to predict the results, the system checks whether those already exist in 
    the specified paths. Due to the potentially long computation time, the user will be prompted to confirm 
    whether they want to overwrite the existing data.

    Most of the parameters are hardcoded in the functions called below. This ensures that the results of
    the predictions for different models are stored analogously. 

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_model_directory: str
        the path to the directory where all results are stored
    path_to_splitted_gif_directory: str
        the path to the directory where the GIF validation data is stored after signal splitting
    path_to_complete_gif_directory: str
        the path to the directory where the GIF validation data is stored before signal splitting
    """

    # check wether model performance can be evaluated with inference (predict and combine results for
    # overlapping segments of data of variable lengths)
    # possible if neural network input length is substantially shorther than the available data
    apply_inference = True
    with open(path_to_model_directory + project_configuration_file, "rb") as f:
        project_configuration = pickle.load(f)
        signal_length_seconds = project_configuration["signal_length_seconds"]
    if signal_length_seconds > 7 * 3600:
        apply_inference = False
    del project_configuration, signal_length_seconds

    """
    ---------
    GIF Data
    ---------
    """

    # path to save the predictions
    gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
    gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

    """
    ------------------------------------
    Predicting Sleep Phases of GIF Data
    ------------------------------------
    """

    # remove predictions if they already exist
    delete_files([gif_splitted_validation_pid_results_path, gif_complete_validation_pid_results_path])
    # make predictions for the splitted data
    main_model_predicting_apnea_validation_set(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_data_directory = path_to_splitted_gif_directory,
        pid = "validation",
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        path_to_save_results = gif_splitted_validation_pid_results_path,
    )

    # make predictions for the complete data
    main_model_predicting_apnea(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_data_directory = path_to_complete_gif_directory,
        pid = "validation",
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        path_to_save_results = gif_complete_validation_pid_results_path,
        inference = apply_inference,
    )

    """
    ------------------------------
    Print Performance on GIF Data
    ------------------------------
    """

    # calculate and print performance results for splitted data
    print_headline("Performance on Splitted GIF Data:", symbol_sequence="=")
    print("   - Reflecting performance on single data segments of required signal length")

    print_model_performance(
        paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    # calculate and print performance results for complete data
    print_headline("Performance on Complete GIF Data:", symbol_sequence="=")
    print("   - Reflecting performance on data segments of variable lengths, as they occur in practice")
    if apply_inference:
        print("   - Predictions were made using inference, combining results from overlapping data segments")
    else:
        print("   - Predictions were made without inference, as the required input length is too long")
    
    print_headline("Performance calculated from combined Probability:", symbol_sequence="-")

    print_model_performance(
        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    print_headline("Performance calculated from majority vote:", symbol_sequence="-")

    print_model_performance(
        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
        prediction_result_key = "Predicted_2",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )


def run_model_predicting(
        path_to_model_directory: str,
        path_to_unknown_dataset: str,
        path_to_processed_unknown_dataset: str,
    ):
    """
    Corresponds to the 3rd main functionality: Applying the trained neural network model to new data.

    Make sure that all files, you want to predict the sleep stages for, have a different name.

    In order to do that, this function will first process the unknown dataset and save the results to a 
    pkl-file. Then, the trained model will be used to predict the sleep stages for the unknown dataset
    and save the results to the same pkl-file. The processing caused the signals to be split into overlapping
    parts that fit the neural networks requirements. Therefore, all signals will be reconstructed to their
    original form (causing to have multiple predictions for the overlapping parts).

    Most of the parameters are hardcoded in the functions called below. This ensures that the results of
    the predictions for different models are stored analogously. 

    ATTENTION:  This function was designed to predict sleep stages for the NAKO dataset. However, this does
                not mean that it is limited to this dataset.
    
                For more information on how the unknown dataset must be formatted to be processable, please 
                refer to the Process_NAKO_Dataset function in dataset_processing.py. If your dataset is 
                formatted differently, you may need to create a new function, analogous to the one mentioned
                above, to process your dataset.

                For more information on how the predictions are saved, please refer to the 
                main_model_predicting function in this file.


    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_model_directory: str
        the path to the directory where all results are stored
    path_to_unknown_dataset: str
        the path to the file where the unknown dataset is stored
    path_to_processed_unknown_dataset: str
        the path to the file where the processed unknown dataset and the predictions are stored
    """

    """
    ---------------------------
    Preprocessing Unknown Data
    ---------------------------
    """

    # check if processed data already exists
    user_response = "y"
    if os.path.exists(path_to_processed_unknown_dataset):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are about to process and save NAKO data to an existing path. " +
                "The existing file may contain results from a previous prediction using a different file " +
                "path but the same file name. Only overwrite if you intend to reprocess older results. " +
                "(This must be done even if you only intend to repredict the sleep stages.)" + 
                "Do you want to proceed? (y/n)",
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_files([path_to_processed_unknown_dataset])
        else:
            return

    # process unknown dataset
    Process_NAKO_Dataset(
        path_to_nako_dataset = path_to_unknown_dataset,
        path_to_save_processed_data = path_to_processed_unknown_dataset,
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
    )

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    # main_model_predicting(
    #     path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
    #     path_to_processed_data = path_to_processed_unknown_dataset,
    #     path_to_project_configuration = path_to_model_directory + project_configuration_file,
    # )

    """
    ------------------------
    Reverse Data Alteration
    ------------------------
    """
    
    # data_manager = SleepDataManager(file_path = path_to_processed_unknown_dataset)

    # reverse signal split
    # data_manager.reverse_signal_split()

    # crop padded signals
    # data_manager.crop_predicted_signals()


if False:
    
    """
    ===============
    Set File Paths
    ===============
    """

    # Create directory to store configurations and results
    model_directory_path = "Neural_Network/"
    create_directories_along_path(model_directory_path)
    
    shhs_directory_path = model_directory_path + "SHHS_Data/"
    gif_directory_path = model_directory_path + "GIF_Data/"

    """
    ==========================
    Set Project Configuration
    ==========================
    """

    sampling_frequency_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1/30,
    }

    signal_cropping_parameters = {
        "signal_length_seconds": 36000,
        "shift_length_seconds_interval": (3600, 7200)
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None), # (0.3, 2)
        "mad_inlier_interval": (None, None),
        "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
    }

    pid_distribution_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True,
        "stratify_by_target": False,
        "consider_targets_for_stratification": [],
    }

    dataset_class_transform_parameters = {
        "feature_transform": custom_transform,
        "target_transform": None,
    }

    window_reshape_parameters = {
        "reshape_to_overlapping_windows": True,
        #
        "windows_per_signal": 1197,
        "window_duration_seconds": 120,
        "overlap_seconds": 90,
        "priority_order": [3, 2, 1, 0],
    }

    signal_normalization_parameters = {
        "normalize_rri": True,
        "normalize_mad": True,
        #
        "normalization_technique": "z-score",
        "normalization_mode": "local",
    }

    neural_network_model_parameters = {
        "neural_network_model": LongSequenceModel,
        "number_target_classes": 4,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "fully_connected_features": 128,
        "convolution_dilations": [2, 4, 8, 16, 32],
        #
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "windows_per_signal": window_reshape_parameters["windows_per_signal"],
        #
        # "rri_datapoints": int(sampling_frequency_parameters["RRI_frequency"] * sampling_frequency_parameters["signal_length_seconds"]),
        # "mad_datapoints": int(sampling_frequency_parameters["MAD_frequency"] * sampling_frequency_parameters["signal_length_seconds"]),
    }

    neural_network_hyperparameters_shhs = {
        "batch_size": 8,
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    neural_network_hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    filter_shhs_data_parameters = {
        "shhs_min_duration_hours": 7,
        "shhs_filter_ids": []
    }

    gif_error_code_1 = ["SL007", "SL010", "SL012", "SL014", "SL022", "SL026", "SL039", "SL044", "SL049", "SL064", "SL070", "SL146", "SL150", "SL261", "SL266", "SL296", "SL303", "SL306", "SL342", "SL350", "SL410", "SL411", "SL416"]
    gif_error_code_2 = ["SL032", "SL037", "SL079", "SL088", "SL114", "SL186", "SL255", "SL328", "SL336", "SL341", "SL344", "SL424"]
    gif_error_code_3 = ["SL001", "SL004", "SL011", "SL025", "SL027", "SL034", "SL055", "SL057", "SL073", "SL075", "SL076", "SL083", "SL085", "SL087", "SL089", "SL096", "SL111", "SL116", "SL126", "SL132", "SL138", "SL141", "SL151", "SL157", "SL159", "SL166", "SL173", "SL174", "SL176", "SL178", "SL179", "SL203", "SL207", "SL208", "SL210", "SL211", "SL214", "SL217", "SL218", "SL221", "SL228", "SL229", "SL236", "SL237", "SL240", "SL245", "SL250", "SL252", "SL269", "SL286", "SL293", "SL294", "SL315", "SL348", "SL382", "SL384", "SL386", "SL389", "SL397", "SL406", "SL408", "SL418", "SL422", "SL428"]
    gif_error_code_4 = ["SL061", "SL066", "SL091", "SL105", "SL202", "SL204", "SL205", "SL216", "SL305", "SL333", "SL349", "SL430", "SL439", "SL440"]
    gif_error_code_5 = ["SL016", "SL040", "SL145", "SL199", "SL246", "SL268", "SL290", "SL316", "SL332", "SL365", "SL392", "SL426", "SL433", "SL438"]

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 7,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5
    }

    project_configuration = dict()
    project_configuration.update(sampling_frequency_parameters)
    project_configuration.update(signal_cropping_parameters)
    project_configuration.update(padding_parameters)
    project_configuration.update(value_mapping_parameters)
    project_configuration.update(pid_distribution_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(signal_normalization_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)
    project_configuration.update(filter_shhs_data_parameters)
    project_configuration.update(filter_gif_data_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_shhs_data_parameters, filter_gif_data_parameters, gif_error_code_1, gif_error_code_2, gif_error_code_3, gif_error_code_4, gif_error_code_5

    check_project_configuration(project_configuration)

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    del project_configuration

    """
    ==============================
    Training Neural Network Model
    ==============================
    """

    run_ssg_model_training(
        path_to_model_directory = model_directory_path,
        path_to_shhs_directory = shhs_directory_path,
        path_to_gif_directory = gif_directory_path,
        neural_network_hyperparameters_shhs = neural_network_hyperparameters_shhs,
        neural_network_hyperparameters_gif = neural_network_hyperparameters_gif,
    )

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    run_model_performance_evaluation_SSG(
        path_to_model_directory = model_directory_path,
        path_to_splitted_shhs_directory = shhs_directory_path,
        path_to_complete_shhs_directory = shhs_directory_path,
        path_to_splitted_gif_directory = gif_directory_path,
        path_to_complete_gif_directory = gif_directory_path,
    )

    """
    ===========================================
    Predict Sleep Phases for Non-Training Data
    ===========================================
    """

    unknown_dataset_paths = ["/Volumes/NaKo-UniHalle/RRI_and_MAD/NAKO-33a.pkl"]

    # predict sleep stages for unknown data
    for unknown_dataset_path in unknown_dataset_paths:
        processed_unknown_dataset_path = "Processed_NAKO/" + os.path.split(unknown_dataset_path)[1]

        run_model_predicting(
            path_to_model_directory = model_directory_path,
            path_to_unknown_dataset = unknown_dataset_path,
            path_to_processed_unknown_dataset = processed_unknown_dataset_path,
        )
    
    """
    =========================
    Access Predicted Results
    =========================
    """

    results_data_manager = BigDataManager(directory_path = processed_unknown_dataset_path)
    
    # accessing random datapoint
    random_datapoint = results_data_manager.load(random.randint(0, len(results_data_manager) - 1))
    
    # access predicted results of random datapoint (2D arrays, provide more information on sleep stage likelihood)
    slp_predicted_probability = random_datapoint["SLP_predicted_probability"] # type: ignore
    slp_predicted = random_datapoint["SLP_predicted"] # type: ignore

    # summarize predicted results (1D arrays, provide final sleep stage prediction)
    slp_predicted_probability_summarized = summarize_predicted_signal(predicted_signal = slp_predicted_probability, mode = "probability")
    slp_predicted_summarized = summarize_predicted_signal(predicted_signal = slp_predicted, mode = "majority")


# IDEAS: max conv channels for mad

# compare different predictions for same time point depending on input signal (after splitting because of length)
# > 2s, < 1/3s rauswerfen

# remove class function to turn signal into wi dows in sleepdatamanager class

# why predicted freuqency = 1/30?
# why does accuracy not match? slp stage is reversed to 1/30, meaning one value becomes 4
# make function that performs all data transformations and use this witin predicting and training