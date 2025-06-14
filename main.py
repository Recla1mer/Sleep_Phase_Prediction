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
original_gif_data_path = "Raw_Data/GIF_dataset.h5"

# file names
project_configuration_file = "Project_Configuration.pkl"

model_state_after_shhs_file = "Model_State_SHHS.pth"
model_state_after_shhs_gif_file = "Model_State.pth"

loss_per_epoch_shhs_file = "Loss_per_Epoch_SHHS.pkl"
loss_per_epoch_gif_file = "Loss_per_Epoch_GIF.pkl"

model_performance_file = "Model_Performance.pkl"


"""
==============================
Default Project Configuration
==============================
"""

# parameters that are used to keep data uniform, see SleepDataManager class in dataset_processing.py
sleep_data_manager_parameters = {
    "RRI_frequency": 4,
    "MAD_frequency": 1,
    "SLP_frequency": 1/30,
    "RRI_inlier_interval": [0.3, 2],
    "MAD_inlier_interval": [None, None],
    "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": 0},
    "signal_length_seconds": 36000,
    "wanted_shift_length_seconds": 5400,
    "absolute_shift_deviation_seconds": 1800,
}
sleep_data_manager_parameters["SLP_predicted_frequency"] = sleep_data_manager_parameters["SLP_frequency"]

# parameters that set train, validation, and test sizes and how data is shuffled, see separate_train_test_validation
split_data_parameters = {
    "train_size": 0.8,
    "validation_size": 0.2,
    "test_size": None,
    "random_state": None,
    "shuffle": True,
    "join_splitted_parts": False,
    "stratify": False
}

# transformations applied to the data, see CustomSleepDataset class in neural_network_model.py
dataset_class_transform_parameters = {
    "feature_transform": ToTensor(),
    "target_transform": None,
}

# parameters that alter the way the data is reshaped into windows, see reshape_signal_to_overlapping_windows 
# function in dataset_processing.py
window_reshape_parameters = {
    "reshape_to_overlapping_windows": True, # whether to reshape the signals to overlapping windows
    # following parameters are not required if 'reshape_to_overlapping_windows' is False
    "signal_length_seconds": sleep_data_manager_parameters["signal_length_seconds"], # 'nn_signal_duration_seconds' in 'reshape_signal_to_overlapping_windows' function
    "windows_per_signal": 1197, # 'number_windows' in 'reshape_signal_to_overlapping_windows' function
    "window_duration_seconds": 120,
    "overlap_seconds": 90,
    "priority_order": [3, 2, 1, 0],
    "pad_feature_with": 0,
    "pad_target_with": 0
}

# parameters that are used to normalize the data, see unity_based_normalization function in dataset_processing.py
signal_normalization_parameters = {
    "normalize_rri": False, # whether to normalize the RRI signal
    "normalize_mad": False, # whether to normalize the MAD signal
    # following parameters are not required if 'normalize_rri' and 'normalize_mad' are False
    "normalization_max": 1,
    "normalization_min": 0,
    "normalization_mode": "global"
}

# neural network model parameters, see DemoWholeNightModel and DemoLocalSleepStageModel class in neural_network_model.py
neural_network_model_parameters = {
    "neural_network_model": SleepStageModel,
    # parameters necessary for neural network models based on whole night signals AND short time signals
    "number_sleep_stages": 4,
    "rri_convolutional_channels": [1, 8, 16, 32, 64],
    "mad_convolutional_channels": [1, 8, 16, 32, 64],
    "max_pooling_layers": 5,
    "number_window_learning_features": 128,
    "window_learning_dilations": [2, 4, 8, 16, 32],
    # parameters necessary for neural network models only based on whole night signals (do not append if using a model based on short time signals)
    "datapoints_per_rri_window": int(sleep_data_manager_parameters["RRI_frequency"] * window_reshape_parameters["window_duration_seconds"]),
    "datapoints_per_mad_window": int(sleep_data_manager_parameters["MAD_frequency"] * window_reshape_parameters["window_duration_seconds"]),
    "windows_per_signal": window_reshape_parameters["windows_per_signal"],
    # parameters necessary for neural network models only based on short time signals (do not append if using a model based on whole night signals)
    "rri_datapoints": int(sleep_data_manager_parameters["RRI_frequency"] * sleep_data_manager_parameters["signal_length_seconds"]),
    "mad_datapoints": int(sleep_data_manager_parameters["MAD_frequency"] * sleep_data_manager_parameters["signal_length_seconds"]),
}

# neural network hyperparameters, see main_model_training function in this file
neural_network_hyperparameters_shhs = {
    "batch_size": 8,
    "number_epochs": 40,
    "lr_scheduler_parameters": {
        "number_updates_to_max_lr": 10,
        "start_learning_rate": 2.5 * 1e-5,
        "max_learning_rate": 1 * 1e-4,
        "end_learning_rate": 5 * 1e-5
    }
}

neural_network_hyperparameters_gif = {
    "batch_size": 8,
    "number_epochs": 100,
    "lr_scheduler_parameters": {
        "number_updates_to_max_lr": 25,
        "start_learning_rate": 2.5 * 1e-5,
        "max_learning_rate": 1 * 1e-4,
        "end_learning_rate": 1 * 1e-5
    }
}


def check_project_configuration(parameters: dict):
    """
    This function checks if a correct computation is guaranteed by the chosen signal processing parameters.

    Running this function does not mean that the parameters ensure an error-free computation. But further
    along the way, especially when calculating number of datapoints from signal duration and sampling 
    frequency, you can be sure that the computation will not lead to unexpected results.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    parameters: dict
        the parameters to check
    """

    # Check if all necessary parameters for the SleepDataManager class are provided
    required_keys = [
        "RRI_frequency", "MAD_frequency", "SLP_frequency", "SLP_predicted_frequency", "RRI_inlier_interval", 
        "MAD_inlier_interval", "sleep_stage_label", "signal_length_seconds", 
        "wanted_shift_length_seconds", "absolute_shift_deviation_seconds"
    ]

    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Parameter '{key}' is missing. Check 'sleep_data_manager_parameters' in 'main.py'.")
    
    # Check if all necessary parameters for the train, validation, and test split are provided and valid
    required_keys = ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "stratify"]
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Parameter '{key}' is missing. Check 'split_data_parameters' in 'main.py'.")
    
    equals_one = parameters["train_size"]
    if parameters["validation_size"] is not None:
        equals_one += parameters["validation_size"]
    if parameters["test_size"] is not None:
        equals_one += parameters["test_size"]
    if equals_one != 1:
        raise ValueError("The sum of train_size, validation_size (and test_size) must be 1.")
    
    if parameters["join_splitted_parts"] and parameters["stratify"]:
        raise ValueError("Cannot join splitted parts and stratify at the same time (see 'separate_train_test_validation' function of 'SleepDataManager' class in 'dataset_processing.py'). Set 'join_splitted_parts' to False or 'stratify' to False.")

    if parameters["stratify"] and parameters["signal_length_seconds"] > 900:
        raise ValueError("Stratification only makes sense for short signals ('signal_length_seconds' <= 900). Set 'stratify' to False or 'signal_length_seconds' to a smaller value.")

    # Check if all parameters for the neural network model are provided
    try:
        neural_network_model = parameters["neural_network_model"]
        nnm_params = {key: parameters[key] for key in parameters if key in neural_network_model_parameters and key != "neural_network_model"}
        neural_network_model = neural_network_model(**nnm_params)
    except:
        raise ValueError("Neural network model parameters are not provided correctly. Check 'neural_network_model_parameters' in 'main.py' and compare parameters to corresponding model class in 'neural_network_model.py'.")
    
    # Check if all necessary parameters for reshaping the signal to overlapping windows are provided
    if parameters["reshape_to_overlapping_windows"]:
        required_keys = [
            "reshape_to_overlapping_windows", "signal_length_seconds", "windows_per_signal",
            "window_duration_seconds", "overlap_seconds", "priority_order", 
            "pad_feature_with", "pad_target_with"
        ]
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Parameter '{key}' is missing. Check 'window_reshape_parameters' in 'main.py'.")
    
    # Check if all necessary parameters for normalizing the signal are provided
    required_keys = ["normalization_max", "normalization_min", "normalization_mode"]
    if parameters["normalize_rri"] or parameters["normalize_mad"]:
        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Parameter '{key}' is missing. Check 'signal_normalization_parameters' in 'main.py'.")
    
    # Check if all necessary parameters for the dataset class transformations are provided
    required_keys = ["feature_transform", "target_transform"]
    for key in required_keys:
        if key not in parameters:
            raise ValueError(f"Parameter '{key}' is missing. Check 'dataset_class_transform_parameters' in 'main.py'.")
    
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
    
    # Neural Network Model Parameters
        if len(parameters["mad_convolutional_channels"]) % 2 != 1 or len(parameters["mad_convolutional_channels"]) < 3:
            raise ValueError("Number of convolutional channels in MAD branch must be odd and more than 2.")
        if parameters["datapoints_per_rri_window"] % 2**(len(parameters["rri_convolutional_channels"])-1) != 0:
            raise ValueError("Number of RRI datapoints per window must be dividable by 2^(number of RRI convolutional layers - 1) without rest.")
        if parameters["datapoints_per_mad_window"] % 2 ** ((len(parameters["rri_convolutional_channels"])-1)/2) != 0:
            raise ValueError("Number of MAD datapoints per window must be dividable by 2^((number of MAD convolutional layers - 1)/2) without rest.")
        if parameters["rri_convolutional_channels"][-1] != parameters["mad_convolutional_channels"][-1]:
            raise ValueError("Number of channels in last convolutional layer of RRI and MAD branch must be equal.")
        if 2**(len(parameters["rri_convolutional_channels"]) - 1) / 2**((len(parameters["mad_convolutional_channels"]) - 1) / 2) != parameters["datapoints_per_rri_window"] / parameters["datapoints_per_mad_window"]:
            raise ValueError("Number of remaining values after Signal Learning must be equal for RRI and MAD branch. Adjust number of convolutional channels accordingly.")
        
    # check if number sleep stages matches the sleep stage label
    sleep_stages = []
    for key in parameters["sleep_stage_label"]:
        value = parameters["sleep_stage_label"][key]
        if value not in sleep_stages:
            sleep_stages.append(value)
    
    if len(sleep_stages) != parameters["number_sleep_stages"]:
        raise ValueError("Number of sleep stages does not match the sleep stage label. Adjust parameters accordingly.")
    
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

def Process_SHHS_Dataset(
        path_to_shhs_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str
    ):
    """
    This function processes our SHHS dataset. It is designed to be a more specific. So, if you are not using
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
    shhs_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access parameters used for splitting the data
    split_data_params = {key: project_configuration[key] for key in split_data_parameters}

    # access data manager parameters
    sdm_params = {key: project_configuration[key] for key in sleep_data_manager_parameters}
    shhs_data_manager.change_file_information(sdm_params)

    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')
    
    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    shhs_sleep_stage_label = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifect": ["other"]}

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    # check if patient ids are unique:
    shhs_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nPreproccessing datapoints from SHHS dataset (ensuring uniformity):")
    progress_bar = DynamicProgressBar(total = len(patients))

    # saving all data from SHHS dataset to the pickle file
    for patient_id in patients:
        new_datapoint = {
            "ID": patient_id,
            "RRI": shhs_dataset["rri"][patient_id][:], # type: ignore
            "SLP": shhs_dataset["slp"][patient_id][:], # type: ignore
            "RRI_frequency": shhs_dataset["rri"].attrs["freq"], # type: ignore
            "SLP_frequency": shhs_dataset["slp"].attrs["freq"], # type: ignore
            "sleep_stage_label": copy.deepcopy(shhs_sleep_stage_label)
        }

        shhs_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # Train-, Validation- and Test-Split
    shhs_data_manager.separate_train_test_validation(**split_data_params)


def Process_GIF_Dataset(
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
    
    Others: See 'Process_SHHS_Dataset' function
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    gif_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access parameters used for splitting the data
    split_data_params = {key: project_configuration[key] for key in split_data_parameters}

    # access data manager parameters
    sdm_params = {key: project_configuration[key] for key in sleep_data_manager_parameters}
    gif_data_manager.change_file_information(sdm_params)

    # access the GIF dataset
    gif_dataset = h5py.File(path_to_gif_dataset, 'r')

    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    gif_sleep_stage_label = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifect": ["other"]}

    # accessing patient ids:
    patients = list(gif_dataset['stage'].keys()) # type: ignore

    # check if patient ids are unique:
    gif_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nPreproccessing datapoints from GIF dataset (ensuring uniformity):")
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
            "sleep_stage_label": copy.deepcopy(gif_sleep_stage_label)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # Train-, Validation- and Test-Split
    gif_data_manager.separate_train_test_validation(**split_data_params)


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
    
    Others: See 'Process_SHHS_Dataset' function
    """

    # abort if destination path exists to avoid accidental overwriting
    if os.path.exists(path_to_save_processed_data):
        return

    # initializing the database
    nako_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access data manager parameters
    sdm_params = {key: project_configuration[key] for key in sleep_data_manager_parameters}
    nako_data_manager.change_file_information(sdm_params)

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


def main_model_training(
        neural_network_hyperparameters: dict = neural_network_hyperparameters_shhs,
        path_to_processed_data: str = "Processed_Data/shhs_data.pkl",
        path_to_project_configuration: str = "Neural_Network/Project_Configuration.pkl",
        path_to_model_state = None,
        path_to_updated_model_state: str = "Neural_Network/Model_State.pth",
        path_to_loss_per_epoch: str = "Neural_Network/Loss_per_Epoch_SHHS.pkl",
    ):
    """
    Full implementation of project, with ability to easily change most important parameters to test different
    neural network architecture configurations. Some Parameters are hardcoded by design.

    The Data is accessed using the CustomSleepDataset class from neural_network_model.py. Before returning 
    the data, this class reshapes the data into windows. Adjustments can be made using the 
    parameters this function accesses from "path_to_project_configuration".

    Afterwards the neural network model is trained and tested. The accuracy and loss are saved in a pickle file
    for every epoch. The final model state dictionary is saved in a .pth file.

    The accuracy values are saved in a dictionary with the following format:
    {
        "train_accuracy": train_accuracy for each epoch (list),
        "train_avg_loss": train_avg_loss for each epoch (list),
        "test_accuracy": test_accuracy for each epoch (list),
        "test_avg_loss": test_avg_loss for each epoch (list),
    }

    
    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_hyperparameters: dict
        the hyperparameters for the neural network model training
        (batch_size, number_epochs, lr_scheduler_parameters)
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_updated_model_state: str
        the path to save the model state dictionary
    path_to_loss_per_epoch: str
        the path to save the accuracy values
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------
    """

    # paths to access the training, validation, and test datasets
    training_data_path = path_to_processed_data[:-4] + "_training_pid.pkl"
    validation_data_path = path_to_processed_data[:-4] + "_validation_pid.pkl"
    test_data_path = path_to_processed_data[:-4] + "_test_pid.pkl"

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

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in neural_network_model_parameters}
    del nnm_params["neural_network_model"]
    
    # access parameters adjusting the final data preprocessing
    CustomDatasetKeywords = dict()

    # add window_reshape_parameters
    CustomDatasetKeywords["reshape_to_overlapping_windows"] = project_configuration["reshape_to_overlapping_windows"]
    if project_configuration["reshape_to_overlapping_windows"]:
        CustomDatasetKeywords.update({key: project_configuration[key] for key in window_reshape_parameters})
    del CustomDatasetKeywords["signal_length_seconds"]

    # add signal_normalization_parameters
    CustomDatasetKeywords["normalize_rri"] = project_configuration["normalize_rri"]
    CustomDatasetKeywords["normalize_mad"] = project_configuration["normalize_mad"]
    if project_configuration["normalize_rri"] or project_configuration["normalize_mad"]:
        CustomDatasetKeywords.update({key: project_configuration[key] for key in signal_normalization_parameters})

    # add transform parameters
    CustomDatasetKeywords.update({key: project_configuration[key] for key in dataset_class_transform_parameters})
    
    training_data = CustomSleepDataset(path_to_data = training_data_path, **CustomDatasetKeywords)
    validation_data = CustomSleepDataset(path_to_data = validation_data_path, **CustomDatasetKeywords)
    # test_data = CustomSleepDataset(path_to_data = test_data_path, **CustomDatasetKeywords)
    
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

    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size = batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True)
    
    del training_data_path, validation_data_path, test_data_path

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to learn on mps device, option to use it is removed
    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
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

    # variables to store accuracy progress
    train_accuracy = []
    train_avg_loss = []

    test_accuracy = []
    test_avg_loss = []

    for t in range(number_epochs):
        print(f"\nEpoch {t+1}:")
        print("-"*130)

        train_results = train_loop(
            dataloader = train_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            optimizer_fn = optimizer_function,
            lr_scheduler = learning_rate_scheduler,
            current_epoch = t,
            batch_size = batch_size
        )
        train_avg_loss.append(train_results[0])
        train_accuracy.append(train_results[1])

        test_results = test_loop(
            dataloader = validation_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            batch_size = batch_size
        )

        test_avg_loss.append(test_results[0])
        test_accuracy.append(test_results[1])

    
    """
    -----------------------
    Saving Accuracy Values
    -----------------------
    """

    create_directories_along_path(path_to_loss_per_epoch)

    accuracy_values = {
        "train_accuracy": train_accuracy,
        "train_avg_loss": train_avg_loss,
        "test_accuracy": test_accuracy,
        "test_avg_loss": test_avg_loss
    }

    save_to_pickle(accuracy_values, path_to_loss_per_epoch)

    """
    ----------------------------------
    Saving Neural Network Model State
    ----------------------------------
    """

    create_directories_along_path(path_to_updated_model_state)
    
    torch.save(neural_network_model.state_dict(), path_to_updated_model_state)


"""
======================================
Applying Trained Neural Network Model
======================================
"""


def main_model_predicting(
        path_to_model_state: str = "Neural_Network/Model_State.pth",
        path_to_processed_data: str = "Processed_Data/shhs_data.pkl",
        path_to_project_configuration: str = "Neural_Network/Project_Configuration.pkl",
        path_to_save_results: str = "Neural_Network/Model_Performance.pkl",
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
            - shape: (number datapoints, number_sleep_stages) 
            - probabilities for each sleep stage,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted sleep stage with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual sleep stages,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted sleep stages with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual sleep stages, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted sleep stages and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted sleep stages:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_sleep_stages) 
            - probabilities for each sleep stage,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted sleep stage with highest probability,
    }

    Note:   The algorithm already crops the sleep stages to the correct length of the original signal. This is
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
    data_manager = SleepDataManager(file_path = path_to_processed_data)

    # retrieve rri, mad, and slp frequencies
    rri_frequency = data_manager.file_info["RRI_frequency"]
    mad_frequency = data_manager.file_info["MAD_frequency"]
    slp_frequency = data_manager.file_info["SLP_frequency"]

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

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in neural_network_model_parameters}
    del nnm_params["neural_network_model"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order", "signal_length_seconds"]:
            common_window_reshape_params[key] = project_configuration[key]
        
        pad_feature_with = project_configuration["pad_feature_with"]
        pad_target_with = project_configuration["pad_target_with"]
    else:
        pad_feature_with = None
        pad_target_with = None

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        for key in ["normalization_max", "normalization_min", "normalization_mode"]:
            common_signal_normalization_params[key] = project_configuration[key]

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
        # else "mps"
        # if torch.backends.mps.is_available()
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
    else:
        # Create temporary file to save data in progress
        working_file_path = os.path.split(copy.deepcopy(path_to_processed_data))[0] + "/save_in_progress"
        working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

        # save file information to working file
        save_to_pickle(data = data_manager.file_info, file_name = working_file_path)

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
            
            # retrieve signal length in seconds
            signal_length_seconds = len(data_dict["RRI"]) / rri_frequency

            """
            Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
            """

            rri = final_data_preprocessing(
                signal = copy.deepcopy(data_dict["RRI"]), # type: ignore
                reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                **common_window_reshape_params,
                target_frequency = rri_frequency,
                pad_with = pad_feature_with,
                signal_type = "feature",
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
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    target_frequency = mad_frequency,
                    pad_with = pad_feature_with,
                    signal_type = "feature",
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

            if actual_results_available:
                actual_original_structure = data_dict["SLP"]
                original_signal_length = len(copy.deepcopy(actual_original_structure))

                # access processed target, reshape to overlapping windows and apply transformations
                slp = final_data_preprocessing(
                    signal = copy.deepcopy(actual_original_structure),
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    target_frequency = slp_frequency,
                    pad_with = pad_target_with,
                    signal_type = "target",
                    normalize = False, # SLP is not normalized
                    datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                    transform = target_transform
                )

            else:
                original_signal_length = int(np.ceil(signal_length_seconds * slp_frequency))
            
            """
            Applying Neural Network Model
            """

            try:
                predictions_in_windows = neural_network_model(rri, mad)
            except:
                unpredictable_signals.append(data_dict["ID"])
                continue

            """
            Preparing Predicted Sleep Phases
            """

            predictions_in_windows = predictions_in_windows.cpu().numpy()

            # reshape windows to original signal structure
            # Lot of stuff happening below, so i explain the process:
            # predictions_in_windows is a 2D array with shape (windows_per_signal, number_sleep_stages)
            predictions_probability = np.empty((original_signal_length, 0))
            for i in range(predictions_in_windows.shape[1]):
                # get a list of probabilities for this sleep stage
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
            
            if actual_results_available:
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

                # save results to file
                append_to_pickle(results, path_to_save_results)
            
            else:
                # save results to exisitng dictionary
                data_dict["SLP_predicted_probability"] = predictions_probability
                data_dict["SLP_predicted"] = predictions_original_structure

                # save updated dictionary to working file
                append_to_pickle(data_dict, working_file_path)
            
            # update progress
            progress_bar.update()

    
    # Remove the old file and rename the working file
    if not actual_results_available:
        if os.path.isfile(path_to_processed_data):
            os.remove(path_to_processed_data)
            
        os.rename(working_file_path, path_to_processed_data)
    
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
    
        # collect results if requested
            if collect_results:
                this_predicted_results_reshaped = pred.argmax(1).view(int(slp.shape[0]/windows_per_signal), windows_per_signal).cpu().numpy()
                this_actual_results_reshaped = slp.view(int(slp.shape[0]/windows_per_signal), windows_per_signal).cpu().numpy()
                
                predicted_results = np.append(predicted_results, this_predicted_results_reshaped, axis=0)
                actual_results = np.append(actual_results, this_actual_results_reshaped, axis=0)
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access dictionary that maps sleep stages (display labels) to integers
    sleep_stage_to_label = project_configuration["sleep_stage_label"]

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


def run_model_training(
        path_to_model_directory: str,
        path_to_processed_shhs: str,
        path_to_processed_gif: str,
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
    """

    """
    ------------------------
    Preprocessing SHHS Data
    ------------------------
    """

    # check if processed data already exists
    user_response = "y"
    if os.path.exists(path_to_processed_shhs):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are attempting to process and save SHHS data to an existing path. " +
                "The existing data may have been used to train and validate a model. " +
                "Overwriting it may prevent accurate assessment of the model's performance, " + 
                "as the validation pid will change. Do you want to overwrite? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_files([path_to_processed_shhs, path_to_processed_shhs[:-4] + "_training_pid.pkl", path_to_processed_shhs[:-4] + "_validation_pid.pkl", path_to_processed_shhs[:-4] + "_test_pid.pkl"])

    # process SHHS data
    if user_response == "y":
        Process_SHHS_Dataset(
            path_to_shhs_dataset = original_shhs_data_path,
            path_to_save_processed_data = path_to_processed_shhs,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
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
        main_model_training(
            neural_network_hyperparameters = neural_network_hyperparameters_shhs,
            path_to_processed_data = path_to_processed_shhs,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = None,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_file,
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_shhs_file,
            )
    
    """
    -----------------------
    Preprocessing GIF Data
    -----------------------
    """

    # check if processed data already exists
    user_response = "y"
    if os.path.exists(path_to_processed_gif):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are attempting to process and save GIF data to an existing path. " +
                "The existing data may have been used to train and validate a model. " +
                "Overwriting it may prevent accurate assessment of the model's performance, " + 
                "as the validation pid will change. Do you want to overwrite? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_files([path_to_processed_gif, path_to_processed_gif[:-4] + "_training_pid.pkl", path_to_processed_gif[:-4] + "_validation_pid.pkl", path_to_processed_gif[:-4] + "_test_pid.pkl"])

    # process GIF data
    if user_response == "y":
        Process_GIF_Dataset(
            path_to_gif_dataset = original_gif_data_path,
            path_to_save_processed_data = path_to_processed_gif,
            path_to_project_configuration = path_to_model_directory + project_configuration_file
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
        main_model_training(
            neural_network_hyperparameters = neural_network_hyperparameters_gif,
            path_to_processed_data = path_to_processed_gif,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = path_to_model_directory + model_state_after_shhs_file,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_gif_file,
            )


def run_model_performance_evaluation(
        path_to_model_directory: str,
        path_to_processed_shhs: str,
        path_to_processed_gif: str,
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
    path_to_processed_shhs: str
        the path to the file where the processed SHHS data is stored
    path_to_processed_gif: str
        the path to the file where the processed GIF data is stored
    """

    """
    ----------
    SHHS Data
    ----------
    """

    # path to save the predictions
    path_to_save_shhs_results = path_to_model_directory + model_performance_file[:-4] + "_SHHS.pkl"
    shhs_validation_pid_results_path = path_to_save_shhs_results[:-4] + "_Validation_Pid.pkl"
    # shhs_training_pid_results_path = path_to_save_shhs_results[:-4] + "_Training_Pid.pkl"
    # shhs_test_pid_results_path = path_to_save_shhs_results[:-4] + "_Test_Pid.pkl"

    # paths to access the training, validation, and test pids
    shhs_validation_data_path = path_to_processed_shhs[:-4] + "_validation_pid.pkl"
    # shhs_training_data_path = path_to_processed_shhs[:-4] + "_training_pid.pkl"
    # shhs_test_data_path = path_to_processed_shhs[:-4] + "_test_pid.pkl"

    """
    -------------------------------------
    Predicting Sleep Phases of SHHS Data
    -------------------------------------
    """

    # check if predictions already exist
    user_response = "y"
    if os.path.exists(shhs_validation_pid_results_path):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are about to predict sleep stages for the SHHS validation pid and " +
                "save the results to an existing file. If you skip the prediction, performance values will " +
                "be calculated from the existing results, saving significant computation time. Do you still " +
                "want to overwrite the existing file and re-run the prediction? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_files([shhs_validation_pid_results_path])

    # make predictions for the relevant files
    if user_response == "y":
        main_model_predicting(
            path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
            path_to_processed_data = shhs_validation_data_path,
            path_to_project_configuration = model_directory_path + project_configuration_file,
            path_to_save_results = shhs_validation_pid_results_path,
        )

    # calculate and print performance results
    print_headline("Performance on SHHS Data:", symbol_sequence="=")

    print_model_performance(
        paths_to_pkl_files = [shhs_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted_in_windows", # or: "Predicted"
        actual_result_key = "Actual_in_windows", # or: "Actual"
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    """
    ---------
    GIF Data
    ---------
    """

    # path to save the predictions
    path_to_save_gif_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl"
    gif_validation_pid_results_path = path_to_save_gif_results[:-4] + "_Validation_Pid.pkl"
    # gif_training_pid_results_path = path_to_save_gif_results[:-4] + "_Training_Pid.pkl"
    # gif_test_pid_results_path = path_to_save_gif_results[:-4] + "_Test_Pid.pkl"

    # paths to access the training, validation, and test pids
    gif_validation_data_path = path_to_processed_gif[:-4] + "_validation_pid.pkl"
    # gif_training_data_path = path_to_processed_gif[:-4] + "_training_pid.pkl"
    # gif_test_data_path = path_to_processed_gif[:-4] + "_test_pid.pkl"

    # check if predictions already exist
    user_response = "y"
    if os.path.exists(gif_validation_pid_results_path):
        # ask the user if they want to overwrite
        user_response = retrieve_user_response(
            message = "ATTENTION: You are about to predict sleep stages for the GIF validation pid and " +
                "save the results to an existing file. If you skip the prediction, performance values will " +
                "be calculated from the existing results, saving significant computation time. Do you still " +
                "want to overwrite the existing file and re-run the prediction? (y/n)", 
            allowed_responses = ["y", "n"]
        )

        if user_response == "y":
            delete_files([gif_validation_pid_results_path])

    # make predictions for the relevant files
    if user_response == "y":
        main_model_predicting(
            path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
            path_to_processed_data = gif_validation_data_path,
            path_to_project_configuration = model_directory_path + project_configuration_file,
            path_to_save_results = gif_validation_pid_results_path,
        )

    # calculate and print performance results
    print_headline("Performance on GIF Data:", symbol_sequence="=")

    print_model_performance(
        paths_to_pkl_files = [gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted_in_windows", # or: "Predicted"
        actual_result_key = "Actual_in_windows", # or: "Actual"
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

    main_model_predicting(
        path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
        path_to_processed_data = path_to_processed_unknown_dataset,
        path_to_project_configuration = path_to_model_directory + project_configuration_file,
    )

    """
    ------------------------
    Reverse Data Alteration
    ------------------------
    """
    
    data_manager = SleepDataManager(file_path = path_to_processed_unknown_dataset)

    # reverse signal split
    data_manager.reverse_signal_split()

    # crop padded signals
    # data_manager.crop_predicted_signals()


if __name__ == "__main__":
    
    """
    ===============
    Set File Paths
    ===============
    """

    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    processed_gif_path = "Processed_Data/gif_data.pkl"

    # Create directory to store configurations and results
    model_directory_path = "Neural_Network_test/"
    create_directories_along_path(model_directory_path)


    """
    ==========================
    Set Project Configuration
    ==========================
    """

    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(signal_normalization_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

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

    run_model_training(
        path_to_model_directory = model_directory_path,
        path_to_processed_shhs = processed_shhs_path,
        path_to_processed_gif = processed_gif_path,
    )

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    run_model_performance_evaluation(
        path_to_model_directory = model_directory_path,
        path_to_processed_shhs = processed_shhs_path,
        path_to_processed_gif = processed_gif_path,
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

    results_data_manager = SleepDataManager(file_path = processed_unknown_dataset_path)
    
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