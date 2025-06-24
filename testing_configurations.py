"""
Author: Johannes Peter Knoll

This file is unnecessary for the user. I used this to test different data processing configurations and 
neural network models.
"""

from main import *
from plot_helper import *

import os

def print_project_configuration():
    """
    """
    all_directories = os.listdir()
    for directory in all_directories:
        if "SSM" == directory[:3] or "Yao" == directory[:3]:
            with open(directory + "/" + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            print(directory)
            print(this_project_configuration)
            print("\n")
            print("-"*50)
            print("\n")


def fix_project_configuration():
    """
    """
    all_directories = os.listdir()
    for directory in all_directories:
        if "SSM" == directory[:3]:
            with open(directory + "/" + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            this_project_configuration["neural_network_model"] = SleepStageModel
        elif "Yao" == directory[:3]:
            with open(directory + "/" + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            this_project_configuration["neural_network_model"] = YaoModel
        else:
            continue

        os.remove(directory + "/" + project_configuration_file)
        save_to_pickle(this_project_configuration, directory + "/" + project_configuration_file)


def fix_project_configuration_2():
    """
    """
    all_directories = os.listdir()
    for directory in all_directories:
        if "SSM" == directory[:3] or "Yao" == directory[:3]:
            with open(directory + "/" + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            this_project_configuration["SLP_predicted_frequency"] = 1/30
            if "SLP_expected_predicted_frequency" in this_project_configuration:
                del this_project_configuration["SLP_expected_predicted_frequency"]
        else:
            continue

        os.remove(directory + "/" + project_configuration_file)
        save_to_pickle(this_project_configuration, directory + "/" + project_configuration_file)


def fix_project_configuration_3(directory = ""):
    """
    """
    if directory == "":
        directory = os.getcwd() + "/"
    all_files = os.listdir(directory)
    for file in all_files:
        if os.path.isdir(file):
            fix_project_configuration_3(directory + file + "/")
        if file == project_configuration_file:
            print("Fixing project configuration in", directory + project_configuration_file)
            with open(directory + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            if "number_windows" in this_project_configuration:
                del this_project_configuration["number_windows"]
            if "nn_signal_duration_seconds" in this_project_configuration:
                del this_project_configuration["nn_signal_duration_seconds"]
            if not this_project_configuration["normalize_rri"] and not this_project_configuration["normalize_mad"]:
                for key in ["normalization_max", "normalization_min", "normalization_mode"]:
                    if key in this_project_configuration:
                        del this_project_configuration[key]
            this_project_configuration["reshape_to_overlapping_windows"] = True
            this_project_configuration["max_pooling_layers"] = 5
            if "transform" in this_project_configuration:
                this_project_configuration["feature_transform"] = this_project_configuration["transform"]
                del this_project_configuration["transform"]
        else:
            continue

        os.remove(directory + "/" + project_configuration_file)
        save_to_pickle(this_project_configuration, directory + "/" + project_configuration_file)


def fix_file_info(path):
    """
    """
    # Create temporary file to save data in progress
    working_file_path = os.path.split(copy.deepcopy(path))[0] + "/save_in_progress"
    working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

    gen = load_from_pickle(path)
    file_info = next(gen)
    if "SLP_expected_predicted_frequency" in file_info:
        del file_info["SLP_expected_predicted_frequency"]
    file_info["SLP_predicted_frequency"] = 1/30

    save_to_pickle(file_info, working_file_path)

    # Save the rest of the data
    for data in gen:
        append_to_pickle(data, working_file_path)

    os.remove(path)
    os.rename(working_file_path, path)


"""
===============
Model Training
===============
"""


def main_pipeline(project_configuration, model_directory_path):
    """
    Main function to run the entire pipeline
    """

    """
    ===============
    Set File Paths
    ===============
    """

    # Create directory to store configurations and results
    create_directories_along_path(model_directory_path + "Processed_Data/")

    processed_shhs_path = model_directory_path + "Processed_Data/shhs_data.pkl"
    processed_gif_path = model_directory_path + "Processed_Data/gif_data.pkl"


    """
    ==========================
    Set Project Configuration
    ==========================
    """

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


def advanced_pipeline(
        name_addition: str = "",
        project_configuration: dict = dict()
    ):
    
    print_headline("SleepStageModel (BatchAfterActivation)", "-")

    project_configuration["neural_network_model"] = SleepStageModelNew

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "SSM_BAA" + name_addition + "/",
    )

    print_headline("YaoModel", "-")

    project_configuration["neural_network_model"] = YaoModel

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "Yao" + name_addition + "/",
    )

    print_headline("YaoModel (BatchAfterActivation)", "-")

    project_configuration["neural_network_model"] = YaoModelNew

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "Yao_BAA" + name_addition + "/",
    )


if True:

    """
    ======================================================
    Default Project Configuration for Whole Night Signals
    ======================================================
    """

    # parameters that are used to keep data uniform, see SleepDataManager class in dataset_processing.py
    sleep_data_manager_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1/30,
        "RRI_inlier_interval": [None, None],
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
        "random_state": 0,
        "shuffle": True,
        "join_splitted_parts": True,
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

    # parameters that are used to normalize the data, see signal_normalization function in dataset_processing.py
    signal_normalization_parameters = {
        "normalize_rri": False, # whether to normalize the RRI signal
        "normalize_mad": False, # whether to normalize the MAD signal
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

    default_project_configuration = dict()
    default_project_configuration.update(sleep_data_manager_parameters)
    default_project_configuration.update(window_reshape_parameters)
    default_project_configuration.update(signal_normalization_parameters)
    default_project_configuration.update(split_data_parameters)
    default_project_configuration.update(dataset_class_transform_parameters)
    default_project_configuration.update(neural_network_model_parameters)

    """
    ====
    RAW
    ====
    """

    print_headline("RAW Data Processing", "=")

    """
    ---------------------------------
    Window Overlap, Artifact as Wake
    ---------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    
    print_headline("Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_ArtifactAsWake_RAW",
        project_configuration = project_configuration,
    )

    """
    ------------------------------------
    No Window Overlap, Artifact as Wake
    ------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["overlap_seconds"] = 0
    project_configuration["number_windows"] = 300
    project_configuration["windows_per_signal"] = 300

    print_headline("No Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_NoOverlap_ArtifactAsWake_RAW",
        project_configuration = project_configuration,
    )

    """
    -----------------------------------------
    Window Overlap, Artifact as Unique Stage
    -----------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["sleep_stage_label"] = {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}
    project_configuration["priority_order"] = [4, 3, 2, 1, 0]
    project_configuration["number_sleep_stages"] = 5

    print_headline("Window Overlap, Artifact as Unique Stage", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_FullClass_RAW",
        project_configuration = project_configuration,
    )

    """
    ================
    Remove Outliers
    ================
    """

    default_project_configuration["RRI_inlier_interval"] = [0.3, 2]

    """
    ---------------------------------
    Window Overlap, Artifact as Wake
    ---------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    
    print_headline("Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_ArtifactAsWake_Cleaned",
        project_configuration = project_configuration,
    )

    """
    ------------------------------------
    No Window Overlap, Artifact as Wake
    ------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["overlap_seconds"] = 0
    project_configuration["number_windows"] = 300
    project_configuration["windows_per_signal"] = 300

    print_headline("No Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_NoOverlap_ArtifactAsWake_Cleaned",
        project_configuration = project_configuration,
    )

    """
    -----------------------------------------
    Window Overlap, Artifact as Unique Stage
    -----------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["sleep_stage_label"] = {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}
    project_configuration["priority_order"] = [4, 3, 2, 1, 0]
    project_configuration["number_sleep_stages"] = 5

    print_headline("Window Overlap, Artifact as Unique Stage", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_FullClass_Cleaned",
        project_configuration = project_configuration,
    )

    """
    =====================================
    Remove Outliers & Normalize Globally
    =====================================
    """

    default_project_configuration["RRI_inlier_interval"] = [0.3, 2]
    default_project_configuration["normalize_rri"] = True
    default_project_configuration["normalize_mad"] = True
    default_project_configuration["normalization_technique"] = "z-score" # "z-score" or "min-max"
    default_project_configuration["normalization_mode"] = "global" # "local" or "global"
    # default_project_configuration["normalization_max"] = 1
    # default_project_configuration["normalization_min"] = 0

    """
    ---------------------------------
    Window Overlap, Artifact as Wake
    ---------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    
    print_headline("Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_ArtifactAsWake_GlobalNorm",
        project_configuration = project_configuration,
    )

    """
    ------------------------------------
    No Window Overlap, Artifact as Wake
    ------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["overlap_seconds"] = 0
    project_configuration["number_windows"] = 300
    project_configuration["windows_per_signal"] = 300

    print_headline("No Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_NoOverlap_ArtifactAsWake_GlobalNorm",
        project_configuration = project_configuration,
    )

    """
    -----------------------------------------
    Window Overlap, Artifact as Unique Stage
    -----------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["sleep_stage_label"] = {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}
    project_configuration["priority_order"] = [4, 3, 2, 1, 0]
    project_configuration["number_sleep_stages"] = 5

    print_headline("Window Overlap, Artifact as Unique Stage", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_FullClass_GlobalNorm",
        project_configuration = project_configuration,
    )

    """
    ====================================
    Remove Outliers & Normalize Locally
    ====================================
    """

    default_project_configuration["RRI_inlier_interval"] = [0.3, 2]
    default_project_configuration["normalize_rri"] = True
    default_project_configuration["normalize_mad"] = True
    default_project_configuration["normalization_technique"] = "z-score" # "z-score" or "min-max"
    default_project_configuration["normalization_mode"] = "local" # "local" or "global"
    # default_project_configuration["normalization_max"] = 1
    # default_project_configuration["normalization_min"] = 0

    """
    ---------------------------------
    Window Overlap, Artifact as Wake
    ---------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    
    print_headline("Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_ArtifactAsWake_LocalNorm",
        project_configuration = project_configuration,
    )

    """
    ------------------------------------
    No Window Overlap, Artifact as Wake
    ------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["overlap_seconds"] = 0
    project_configuration["number_windows"] = 300
    project_configuration["windows_per_signal"] = 300

    print_headline("No Window Overlap, Artifact as Wake", "= ")

    advanced_pipeline(
        name_addition = "_NoOverlap_ArtifactAsWake_LocalNorm",
        project_configuration = project_configuration,
    )

    """
    -----------------------------------------
    Window Overlap, Artifact as Unique Stage
    -----------------------------------------
    """

    project_configuration = copy.deepcopy(default_project_configuration)
    project_configuration["sleep_stage_label"] = {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}
    project_configuration["priority_order"] = [4, 3, 2, 1, 0]
    project_configuration["number_sleep_stages"] = 5

    print_headline("Window Overlap, Artifact as Unique Stage", "= ")

    advanced_pipeline(
        name_addition = "_Overlap_FullClass_LocalNorm",
        project_configuration = project_configuration,
    )


if False:
    # fix_project_configuration_3()
    # print_project_configuration()
    # raise SystemExit("Testing configurations...")

    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(signal_normalization_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

    project_configuration["neural_network_model"] = YaoModel
    del project_configuration["rri_datapoints"]
    del project_configuration["mad_datapoints"]
    
    project_configuration["RRI_inlier_interval"] = [0.3, 2]
    
    project_configuration["normalize_rri"] = True
    project_configuration["normalize_mad"] = True
    project_configuration["normalization_max"] = 1
    project_configuration["normalization_min"] = 0
    project_configuration["normalization_mode"] = "local"

    project_configuration["random_state"] = 0

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "Test_Yao_Local_Original/",
    )

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
        "MAD_inlier_interval": [None, None],
        "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": 0},
    }
    sleep_data_manager_parameters["SLP_predicted_frequency"] = sleep_data_manager_parameters["SLP_frequency"]

    # parameters that set train, validation, and test sizes and how data is shuffled, see separate_train_test_validation
    split_data_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": 0,
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
        "reshape_to_overlapping_windows": False, # whether to reshape the signals to overlapping windows
    }

    neural_network_model_parameters = {
        # parameters necessary for neural network models based on whole night signals AND short time signals
        "number_sleep_stages": 4,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "number_window_learning_features": 128,
        "window_learning_dilations": [2, 4, 8, 16, 32],
    }

    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

    neural_network_hyperparameters_shhs["batch_size"] = 150 # 5h for 120s data
    neural_network_hyperparameters_gif["batch_size"] = 30 # 1h for 120s data

    # pre preprocess
    project_configuration["RRI_inlier_interval"] = [0.3, 2]
    project_configuration["signal_length_seconds"] = 120
    project_configuration["wanted_shift_length_seconds"] = 120
    project_configuration["absolute_shift_deviation_seconds"] = 10
    project_configuration["join_splitted_parts"] = True

    # final preprocess
    project_configuration["normalize_rri"] = True
    project_configuration["normalize_mad"] = True
    project_configuration["normalization_max"] = 1
    project_configuration["normalization_min"] = 0
    project_configuration["normalization_mode"] = "local"

    # nn
    project_configuration["neural_network_model"] = LocalIntervalModel
    project_configuration["rri_datapoints"] = int(sleep_data_manager_parameters["RRI_frequency"] * project_configuration["signal_length_seconds"])
    project_configuration["mad_datapoints"] = int(sleep_data_manager_parameters["MAD_frequency"] * project_configuration["signal_length_seconds"])

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "Short_Time_Norm_120/",
    )

    neural_network_hyperparameters_shhs["batch_size"] = 600 # 5h for 30s data
    neural_network_hyperparameters_gif["batch_size"] = 120 # 1h for 30s data

    # pre preprocess
    project_configuration["RRI_inlier_interval"] = [0.3, 2]
    project_configuration["signal_length_seconds"] = 30
    project_configuration["wanted_shift_length_seconds"] = 30
    project_configuration["absolute_shift_deviation_seconds"] = 1
    project_configuration["join_splitted_parts"] = True

    # final preprocess
    project_configuration["normalize_rri"] = True
    project_configuration["normalize_mad"] = True
    project_configuration["normalization_max"] = 1
    project_configuration["normalization_min"] = 0
    project_configuration["normalization_mode"] = "local"

    # nn
    project_configuration["neural_network_model"] = LocalIntervalModel
    project_configuration["rri_datapoints"] = int(sleep_data_manager_parameters["RRI_frequency"] * project_configuration["signal_length_seconds"])
    project_configuration["mad_datapoints"] = int(sleep_data_manager_parameters["MAD_frequency"] * project_configuration["signal_length_seconds"])

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "Short_Time_Norm_30/",
    )


if False:
    name_addition = "_NEW"
    project_configuration_change = {
        "RRI_inlier_interval": [None, None],
        "normalize_rri": False,
        "normalize_mad": False,
    }
    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = "_rm_outliers_NEW"
    project_configuration_change = {
        "RRI_inlier_interval": [0.3, 2],
        "normalize_rri": False,
        "normalize_mad": False,
        "normalization_mode": "global",
    }

    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = "_norm_global_NEW"
    project_configuration_change = {
        "RRI_inlier_interval": [0.3, 2],
        "normalize_rri": True,
        "normalize_mad": False,
        "normalization_mode": "global",
    }

    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = "_norm_local_NEW"
    project_configuration_change = {
        "RRI_inlier_interval": [0.3, 2],
        "normalize_rri": True,
        "normalize_mad": False,
        "normalization_mode": "local",
    }

    # train_multiple_configurations(name_addition, project_configuration_change)
    # predict_multiple_configurations(name_addition)

    name_addition = ""
    accuracy_multiple_configurations(name_addition)

    name_addition = "_rm_outliers"
    accuracy_multiple_configurations(name_addition)

    name_addition = "_norm_global"
    accuracy_multiple_configurations(name_addition)

    name_addition = "_norm_local"
    accuracy_multiple_configurations(name_addition)


if False:
    # fix_project_configuration_2()
    # print_project_configuration()
    
    # fix_file_info("Processed_NAKO/NAKO-33a.pkl")

    # data_manager = SleepDataManager(file_path="Processed_NAKO/NAKO-33a.pkl")
    # print(data_manager.file_info)

    data_manager = SleepDataManager(file_path="Processed_NAKO/NAKO-33a.pkl")
    data_manager.reverse_signal_split()