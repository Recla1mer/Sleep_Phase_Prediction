"""
Author: Johannes Peter Knoll

This file is unnecessary for the user. I used this to test different data processing configurations and 
neural network models.
"""

from main import *
from plot_helper import *
import shutil

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


def main_pipeline(
        project_configuration, 
        model_directory_path,
        neural_network_hyperparameters_shhs: dict,
        neural_network_hyperparameters_gif: dict,
        default_shhs_database: str,
        default_gif_database: str,
        ):
    """
    Main function to run the entire pipeline
    """

    """
    ===============
    Set File Paths
    ===============
    """

    # Create directory to store configurations and results
    create_directories_along_path(model_directory_path)

    shhs_directory_path = model_directory_path + "SHHS_Data/"
    gif_directory_path = model_directory_path + "GIF_Data/"


    """
    ==========================
    Set Project Configuration
    ==========================
    """

    check_project_configuration(project_configuration)

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    """
    =======================
    Copy Default Databases
    =======================
    """

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "wanted_shift_length_seconds", "absolute_shift_deviation_seconds"]} # signal_cropping_parameters
    del project_configuration

    # copy default SHHS and GIF databases
    if not os.path.exists(shhs_directory_path):
        shutil.copytree(default_shhs_database, shhs_directory_path)

    if not os.path.exists(gif_directory_path):
        shutil.copytree(default_gif_database, gif_directory_path)
    
    # process SHHS dataset
    shhs_data_manager = SleepDataManager(directory_path = shhs_directory_path)
    shhs_data_manager.crop_oversized_data(**signal_crop_params)

    # process GIF dataset
    gif_data_manager = SleepDataManager(directory_path = gif_directory_path)
    gif_data_manager.crop_oversized_data(**signal_crop_params)

    del signal_crop_params, shhs_data_manager, gif_data_manager

    """
    ==============================
    Training Neural Network Model
    ==============================
    """

    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_shhs,
        path_to_training_data_directory = shhs_directory_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        paths_to_validation_data_directories = [shhs_directory_path, gif_directory_path],
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
    )

    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_gif,
        path_to_training_data_directory = gif_directory_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        paths_to_validation_data_directories = [shhs_directory_path, gif_directory_path],
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
    )

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    run_model_performance_evaluation(
        path_to_model_directory = model_directory_path,
        path_to_shhs_directory = shhs_directory_path,
        path_to_gif_directory = gif_directory_path,
    )


def Reduced_Process_SHHS_Dataset(
        path_to_shhs_dataset: str,
        path_to_save_processed_data: str,
        path_to_project_configuration: str,
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
    shhs_data_manager = SleepDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    shhs_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations"]} # pid_distribution_parameters

    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')
    
    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    shhs_sleep_stage_label = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifact": ["other"]}

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    # check if patient ids are unique:
    shhs_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the SHHS dataset:")
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

    shhs_data_manager.separate_train_test_validation(**distribution_params)


def Reduced_Process_GIF_Dataset(
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
    gif_data_manager = SleepDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    gif_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations"]} # pid_distribution_parameters

    # access the GIF dataset
    gif_dataset = h5py.File(path_to_gif_dataset, 'r')

    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    gif_sleep_stage_label = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifact": ["other"]}

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
            "sleep_stage_label": copy.deepcopy(gif_sleep_stage_label)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # Train-, Validation- and Test-Pid Distribution
    gif_data_manager.separate_train_test_validation(**distribution_params)


default_shhs_path = "Default_SHHS_Data/"
default_gif_path = "Default_GIF_Data/"


if True:
    """
    =======================
    Build Default Database
    =======================
    """
    limited_project_configuration_file = "default_project_configuration.pkl"
    project_configuration = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1/30,
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True,
    }
    with open(limited_project_configuration_file, "wb") as file:
        pickle.dump(project_configuration, file)

    Reduced_Process_SHHS_Dataset(
        path_to_shhs_dataset = original_shhs_data_path,
        path_to_save_processed_data = default_shhs_path,
        path_to_project_configuration = limited_project_configuration_file,
        )
    
    Reduced_Process_GIF_Dataset(
        path_to_gif_dataset = original_gif_data_path,
        path_to_save_processed_data = default_gif_path,
        path_to_project_configuration = limited_project_configuration_file
        )
    
    os.remove(limited_project_configuration_file)


if True:

    """
    ======================================================
    Default Project Configuration for Whole Night Signals
    ======================================================
    """

    sampling_frequency_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1/30,
    }

    signal_cropping_parameters = {
        "signal_length_seconds": 36000,
        "wanted_shift_length_seconds": 5400,
        "absolute_shift_deviation_seconds": 1800,
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None), # (0.3, 2)
        "mad_inlier_interval": (None, None),
        "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
    }

    pid_distribution_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True
    }

    dataset_class_transform_parameters = {
        "feature_transform": ToTensor(),
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
        "normalize_rri": False,
        "normalize_mad": False,
    }

    neural_network_model_parameters = {
        "neural_network_model": SleepStageModel,
        "number_sleep_stages": 4,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "number_window_learning_features": 128,
        "window_learning_dilations": [2, 4, 8, 16, 32],
        #
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "windows_per_signal": window_reshape_parameters["windows_per_signal"],
    }

    neural_network_hyperparameters_shhs = {
        "batch_size": 8, # 80h for 10h data | 7K (6712) / 8 => 839 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 10,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 5 * 1e-5
        }
    }

    neural_network_hyperparameters_gif = {
        "batch_size": 2, # 80h for 10h data | 584 / 2 => 292 steps per epoch
        "number_epochs": 100,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 25,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 1 * 1e-5
        }
    }

    default_project_configuration = dict()
    default_project_configuration.update(sampling_frequency_parameters)
    default_project_configuration.update(signal_cropping_parameters)
    default_project_configuration.update(padding_parameters)
    default_project_configuration.update(value_mapping_parameters)
    default_project_configuration.update(pid_distribution_parameters)
    default_project_configuration.update(window_reshape_parameters)
    default_project_configuration.update(signal_normalization_parameters)
    default_project_configuration.update(dataset_class_transform_parameters)
    default_project_configuration.update(neural_network_model_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters

    overlap_artifact_as_wake = {
        "number_sleep_stages": 4,
        "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
        "window_duration_seconds": 120,
        "windows_per_signal": 1197,
        "overlap_seconds": 90,
        "priority_order": [3, 2, 1, 0],
    }

    no_overlap_artifact_as_wake = {
        "number_sleep_stages": 4,
        "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
        "window_duration_seconds": 120,
        "windows_per_signal": 300,
        "overlap_seconds": 0,
        "priority_order": [3, 2, 1, 0],
    }
    
    overlap_full_class = {
        "number_sleep_stages": 5, # 5 sleep stages: wake, LS, DS, REM, artifact
        "sleep_stage_label": {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifact": 0},
        "window_duration_seconds": 120,
        "windows_per_signal": 1197,
        "overlap_seconds": 90,
        "priority_order": [4, 3, 2, 1, 0], # REM, DS, LS, wake, artifact
    }

    raw = {
        "rri_inlier_interval": (None, None),
        "mad_inlier_interval": (None, None),
        "normalize_rri": False,
        "normalize_mad": False,
    }

    cleaned = {
        "rri_inlier_interval": (0.3, 2),
        "mad_inlier_interval": (None, None),
        "normalize_rri": False,
        "normalize_mad": False,
    }

    global_norm = {
        "rri_inlier_interval": (0.3, 2),
        "mad_inlier_interval": (None, None),
        "normalize_rri": True,
        "normalize_mad": True,
        "normalization_technique": "z-score", # "z-score" or "min-max"
        "normalization_mode": "global", # "local" or "global"
    }

    local_norm = {
        "rri_inlier_interval": (0.3, 2),
        "mad_inlier_interval": (None, None),
        "normalize_rri": True,
        "normalize_mad": True,
        "normalization_technique": "z-score",
        "normalization_mode": "local",
    }

    window_and_class_adjustments = [overlap_artifact_as_wake, no_overlap_artifact_as_wake, overlap_full_class]
    window_and_class_names = ["Overlap_ArtifactAsWake", "NoOverlap_ArtifactAsWake", "Overlap_FullClass"]

    cleaning_adjustments = [raw, cleaned, global_norm, local_norm]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]

    network_models = [SleepStageModelNew, YaoModel, YaoModelNew]
    network_model_names = ["LTM_BAA", "Yao", "Yao_BAA"]

    for clean_index in range(len(cleaning_adjustments)):
        for window_index in range(len(window_and_class_adjustments)):
            for model_index in range(len(network_models)):

                project_configuration = copy.deepcopy(default_project_configuration)
                project_configuration.update(cleaning_adjustments[clean_index])
                project_configuration.update(window_and_class_adjustments[window_index])
                project_configuration["neural_network_model"] = network_models[model_index]

                identifier = network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index]

                print_headline("Running " + identifier, "=")

                main_pipeline(
                    project_configuration = project_configuration,
                    model_directory_path = identifier + "/",
                    neural_network_hyperparameters_shhs = neural_network_hyperparameters_shhs,
                    neural_network_hyperparameters_gif = neural_network_hyperparameters_gif,
                    default_shhs_database = default_shhs_path,
                    default_gif_database = default_gif_path,
                )


if True:

    """
    ==========================================================
    Default Project Configuration for Local Short-Time Models
    ==========================================================
    """

    sampling_frequency_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1/30,
    }

    signal_cropping_parameters = {
        "signal_length_seconds": 30,
        "wanted_shift_length_seconds": 30,
        "absolute_shift_deviation_seconds": 0,
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None),
        "mad_inlier_interval": (None, None),
        "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
    }

    pid_distribution_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True
    }

    dataset_class_transform_parameters = {
        "feature_transform": ToTensor(),
        "target_transform": None,
    }

    window_reshape_parameters = {
        "reshape_to_overlapping_windows": False,
    }

    signal_normalization_parameters = {
        "normalize_rri": False,
        "normalize_mad": False,
    }

    neural_network_model_parameters = {
        "neural_network_model": LocalIntervalModel,
        "number_sleep_stages": 4,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "number_window_learning_features": 128,
        "window_learning_dilations": [2, 4, 8, 16, 32],
        "rri_datapoints": int(sampling_frequency_parameters["RRI_frequency"] * sampling_frequency_parameters["signal_length_seconds"]),
        "mad_datapoints": int(sampling_frequency_parameters["MAD_frequency"] * sampling_frequency_parameters["signal_length_seconds"]),
    }

    default_project_configuration = dict()
    default_project_configuration.update(sampling_frequency_parameters)
    default_project_configuration.update(signal_cropping_parameters)
    default_project_configuration.update(padding_parameters)
    default_project_configuration.update(value_mapping_parameters)
    default_project_configuration.update(pid_distribution_parameters)
    default_project_configuration.update(window_reshape_parameters)
    default_project_configuration.update(signal_normalization_parameters)
    default_project_configuration.update(dataset_class_transform_parameters)
    default_project_configuration.update(neural_network_model_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters

    artifact_as_wake = {
        "number_sleep_stages": 4,
        "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
    }
    
    full_class = {
        "number_sleep_stages": 5, # 5 sleep stages: wake, LS, DS, REM, artifact
        "sleep_stage_label": {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifact": 0},
    }

    raw = {
        "rri_inlier_interval": (None, None),
        "mad_inlier_interval": (None, None),
        "normalize_rri": False,
        "normalize_mad": False,
    }

    cleaned = {
        "rri_inlier_interval": (0.3, 2),
        "mad_inlier_interval": (None, None),
        "normalize_rri": False,
        "normalize_mad": False,
    }

    norm = {
        "rri_inlier_interval": (0.3, 2),
        "mad_inlier_interval": (None, None),
        "normalize_rri": True,
        "normalize_mad": True,
        "normalization_technique": "z-score", # "z-score" or "min-max"
        "normalization_mode": "local", # "local" or "global" makes no difference for 1D data
    }

    thirty_second_network = {
        "signal_length_seconds": 30,
        "wanted_shift_length_seconds": 30,
        "absolute_shift_deviation_seconds": 0,
        "rri_datapoints": int(project_configuration["RRI_frequency"] * 30),
        "mad_datapoints": int(project_configuration["MAD_frequency"] * 30),
    }

    thirty_second_hyperparameters_shhs = {
        "batch_size": 1024, # 8,5h for 30s data | 6M (5931923) / 1024 => 5792 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 10,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 5 * 1e-5
        }
    }

    thirty_second_hyperparameters_gif = {
        "batch_size": 128, # 1h for 30s data | 350K (348524) / 128 => 2720 steps per epoch
        "number_epochs": 100,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 25,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 1 * 1e-5
        }
    }

    sixty_second_network = {
        "signal_length_seconds": 60,
        "wanted_shift_length_seconds": 60,
        "absolute_shift_deviation_seconds": 0,
        "rri_datapoints": int(project_configuration["RRI_frequency"] * 60),
        "mad_datapoints": int(project_configuration["MAD_frequency"] * 60),
    }

    sixty_second_hyperparameters_shhs = {
        "batch_size": 512, # 8,5h for 60s data | 3M (2966296) / 512 => 5792 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 10,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 5 * 1e-5
        }
    }

    sixty_second_hyperparameters_gif = {
        "batch_size": 64, # 1h for 60s data | 175K (174374) / 64 => 2720 steps per epoch
        "number_epochs": 100,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 25,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 1 * 1e-5
        }
    }

    hundred_twenty_second_network = {
        "signal_length_seconds": 120,
        "wanted_shift_length_seconds": 120,
        "absolute_shift_deviation_seconds": 0,
        "rri_datapoints": int(project_configuration["RRI_frequency"] * 120),
        "mad_datapoints": int(project_configuration["MAD_frequency"] * 120),
    }

    hundred_twenty_second_hyperparameters_shhs = {
        "batch_size": 256, # 8,5h for 120s data | 1.5M (1484839) / 256 => 5792 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 10,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 5 * 1e-5
        }
    }

    hundred_twenty_second_hyperparameters_gif = {
        "batch_size": 32, # 8,5h for 120s data | 90K (87221) / 32 => 2720 steps per epoch
        "number_epochs": 100,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 25,
            "start_learning_rate": 2.5 * 1e-5,
            "max_learning_rate": 1 * 1e-4,
            "end_learning_rate": 1 * 1e-5
        }
    }

    class_adjustments = [artifact_as_wake, full_class]
    class_names = ["ArtifactAsWake", "FullClass"]

    cleaning_adjustments = [raw, cleaned, norm]
    cleaning_names = ["RAW", "Cleaned", "Norm"]

    network_adjustments = [thirty_second_network, sixty_second_network, hundred_twenty_second_network]
    shhs_hyperparameter_adjustments = [thirty_second_hyperparameters_shhs, sixty_second_hyperparameters_shhs, hundred_twenty_second_hyperparameters_shhs]
    gif_hyperparameter_adjustments = [thirty_second_hyperparameters_gif, sixty_second_hyperparameters_gif, hundred_twenty_second_hyperparameters_gif]
    network_names = ["Local_30s", "Local_60s", "Local_120s"]

    for clean_index in range(len(cleaning_adjustments)):
        project_configuration = copy.deepcopy(default_project_configuration)
        project_configuration.update(cleaning_adjustments[clean_index])

        for class_index in range(len(class_adjustments)):
            project_configuration.update(class_adjustments[class_index])
            
            for network_index in range(len(network_adjustments)):
                project_configuration.update(network_adjustments[network_index])

                identifier = network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index]

                print_headline("Running " + identifier, "=")

                main_pipeline(
                    project_configuration = project_configuration,
                    model_directory_path = identifier + "/",
                    neural_network_hyperparameters_shhs = shhs_hyperparameter_adjustments[network_index],
                    neural_network_hyperparameters_gif = gif_hyperparameter_adjustments[network_index],
                    default_shhs_database = default_shhs_path,
                    default_gif_database = default_gif_path,
                )


if False:
    # fix_project_configuration_3()
    # print_project_configuration()
    # raise SystemExit("Testing configurations...")

    # fix_project_configuration_2()
    # print_project_configuration()
    
    # fix_file_info("Processed_NAKO/NAKO-33a.pkl")

    # data_manager = SleepDataManager(file_path="Processed_NAKO/NAKO-33a.pkl")
    # print(data_manager.file_info)

    data_manager = SleepDataManager(file_path="Processed_NAKO/NAKO-33a.pkl")
    data_manager.reverse_signal_split()

    # -----------------------------------------------------------

    shhs_data_manager = SleepDataManager(directory_path = default_shhs_path)
    shhs_training_data_manager = SleepDataManager(directory_path = default_shhs_path, pid = "train")
    shhs_validation_data_manager = SleepDataManager(directory_path = default_shhs_path, pid = "validation")

    gif_data_manager = SleepDataManager(directory_path = default_gif_path)
    gif_training_data_manager = SleepDataManager(directory_path = default_gif_path, pid = "train")
    gif_validation_data_manager = SleepDataManager(directory_path = default_gif_path, pid = "validation")

    print(len(shhs_data_manager), len(shhs_training_data_manager), len(shhs_validation_data_manager))
    print(len(gif_data_manager), len(gif_training_data_manager), len(gif_validation_data_manager))

    signal_cropping_parameters = {
        "signal_length_seconds": 30,
        "wanted_shift_length_seconds": 30,
        "absolute_shift_deviation_seconds": 0,
    }

    # gif_data_manager.crop_oversized_data(**signal_cropping_parameters)
    # shhs_data_manager.crop_oversized_data(**signal_cropping_parameters)

    print(len(shhs_data_manager), len(shhs_training_data_manager), len(shhs_validation_data_manager))
    print(len(gif_data_manager), len(gif_training_data_manager), len(gif_validation_data_manager))