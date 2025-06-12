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


def ask_to_override_files(file_paths: list):
    """
    Asks the user if they want to override files.

    ARGUMENTS:
    ------------------------------
    file_paths: list
        list of file paths to ask for
    
    RETURNS:
    ------------------------------
    str
        "y" if the user wants to override the files, "n" otherwise
    """

    for file_path in file_paths:
        file_exists = False
        if os.path.exists(file_path):
            file_exists = True
            break
    
    if file_exists:
        while True:
            print("At least one of the following files already exists:")
            print(file_paths)
            answer = input("\nDo you want to override all of them? (y/n): ")
            if answer == "y":
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                return "y"
            elif answer == "n":
                return "n"
            else:
                print("Please enter 'y' or 'n'.")


def predictions_for_model_accuracy_evaluation(
        path_to_model_state: str = "Neural_Network/Model_State.pth",
        path_to_processed_data: str = "Processed_Data/shhs_data.pkl",
        path_to_project_configuration: str = "Neural_Network/Project_Configuration.pkl",
        path_to_save_results: str = "Neural_Network/Model_Accuracy.pkl",
    ):
    """
    Applies the trained neural network model to the processed data (training and validation datasets), for
    the purpose of evaluating the model accuracy.

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

    # paths to access the training, validation, and test datasets
    training_data_path = path_to_processed_data[:-4] + "_training_pid.pkl"
    validation_data_path = path_to_processed_data[:-4] + "_validation_pid.pkl"
    test_data_path = path_to_processed_data[:-4] + "_test_pid.pkl"

    training_pid_results_path = path_to_save_results[:-4] + "_Training_Pid.pkl"
    validation_pid_results_path = path_to_save_results[:-4] + "_Validation_Pid.pkl"

    user_answer = ask_to_override_files([training_pid_results_path, validation_pid_results_path])
    if user_answer == "n":
        return

    # make predictions for the relevant files
    main_model_predicting(
        path_to_model_state = path_to_model_state,
        path_to_processed_data = training_data_path,
        path_to_project_configuration = path_to_project_configuration,
        path_to_save_results = training_pid_results_path,
    )

    main_model_predicting(
        path_to_model_state = path_to_model_state,
        path_to_processed_data = validation_data_path,
        path_to_project_configuration = path_to_project_configuration,
        path_to_save_results = validation_pid_results_path,
    )


"""
===============
Model Training
===============
"""

def train_multiple_configurations(
        name_addition: str = "",
        global_project_configuration_change: dict = dict(),
    ):

    """
    ---------------------------------------------------------------------
    Testing Original Idea: Overlapping Windows and artifect = wake stage
    ---------------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data" + name_addition + "/shhs_data_original.pkl"
    processed_gif_path = "Processed_Data" + name_addition + "/gif_data_original.pkl"

    # Set Signal Processing Parameters
    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(signal_normalization_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)
    project_configuration.update(global_project_configuration_change)

    check_project_configuration(project_configuration)

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Original" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    # project_configuration["neural_network_model"] = SleepStageModel
    project_configuration["neural_network_model"] = SleepStageModelNew

    # if os.path.isfile(model_directory_path + project_configuration_file):
    #     os.remove(model_directory_path + project_configuration_file)
    # save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Preprocess SHHS Data
    # Process_SHHS_Dataset(
    #     path_to_shhs_dataset = original_shhs_data_path,
    #     path_to_save_processed_data = processed_shhs_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     )

    # Training Network on SHHS Data
    # main_model_training(
    #     neural_network_hyperparameters = neural_network_hyperparameters_shhs,
    #     path_to_processed_data = processed_shhs_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_model_state = None,
    #     path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
    #     path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
    #     )

    # Preprocess GIF Data
    # Process_GIF_Dataset(
    #     path_to_gif_dataset = original_gif_data_path,
    #     path_to_save_processed_data = processed_gif_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file
    #     )

    # Training Network on GIF Data
    # main_model_training(
    #     neural_network_hyperparameters = neural_network_hyperparameters_gif,
    #     path_to_processed_data = processed_gif_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_model_state = model_directory_path + model_state_after_shhs_file,
    #     path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #     path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
    #     )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Original" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    # project_configuration["neural_network_model"] = YaoModel
    project_configuration["neural_network_model"] = YaoModelNew

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_shhs,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Training Network on GIF Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_gif,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )

    """
    ---------------------------------------------------------------
    Testing with non-overlapping windows and artifect = wake stage
    ---------------------------------------------------------------
    """

    # Set File Paths
    # processed_shhs_path = "Processed_Data" + name_addition + "/shhs_data_no_overlap.pkl"
    # processed_gif_path = "Processed_Data" + name_addition + "/gif_data_no_overlap.pkl"

    # Set Signal Processing Parameters
    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(signal_normalization_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)
    project_configuration.update(global_project_configuration_change)

    project_configuration["overlap_seconds"] = 0
    project_configuration["number_windows"] = 300
    project_configuration["windows_per_signal"] = 300

    check_project_configuration(project_configuration)

    # Using SleepStageModel

    model_directory_path = "SSM_no_overlap" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    # project_configuration["neural_network_model"] = SleepStageModel
    project_configuration["neural_network_model"] = SleepStageModelNew

    # if os.path.isfile(model_directory_path + project_configuration_file):
    #     os.remove(model_directory_path + project_configuration_file)
    # save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    # main_model_training(
    #     neural_network_hyperparameters = neural_network_hyperparameters_shhs,
    #     path_to_processed_data = processed_shhs_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_model_state = None,
    #     path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
    #     path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
    #     )

    # Training Network on GIF Data
    # main_model_training(
    #     neural_network_hyperparameters = neural_network_hyperparameters_gif,
    #     path_to_processed_data = processed_gif_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_model_state = model_directory_path + model_state_after_shhs_file,
    #     path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #     path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
    #     )

    # Using YaoModel

    model_directory_path = "Yao_no_overlap" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    # project_configuration["neural_network_model"] = YaoModel
    project_configuration["neural_network_model"] = YaoModelNew

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_shhs,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Training Network on GIF Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_gif,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )

    """
    -------------------------------------------------------------------
    Testing with Overlapping windows but artifect being a unique stage
    -------------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data" + name_addition + "/shhs_data_artifect.pkl"
    processed_gif_path = "Processed_Data" + name_addition + "/gif_data_artifect.pkl"

    # Set Signal Processing Parameters
    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(signal_normalization_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)
    project_configuration.update(global_project_configuration_change)

    project_configuration["sleep_stage_label"] = {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}
    project_configuration["priority_order"] = [4, 3, 2, 1, 0]
    project_configuration["number_sleep_stages"] = 5

    check_project_configuration(project_configuration)

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Artifect" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    # project_configuration["neural_network_model"] = SleepStageModel
    project_configuration["neural_network_model"] = SleepStageModelNew

    # if os.path.isfile(model_directory_path + project_configuration_file):
    #     os.remove(model_directory_path + project_configuration_file)
    # save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Preprocess SHHS Data
    # Process_SHHS_Dataset(
    #     path_to_shhs_dataset = original_shhs_data_path,
    #     path_to_save_processed_data = processed_shhs_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     )

    # Training Network on SHHS Data
    # main_model_training(
    #     neural_network_hyperparameters = neural_network_hyperparameters_shhs,
    #     path_to_processed_data = processed_shhs_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_model_state = None,
    #     path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
    #     path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
    #     )

    # Preprocess GIF Data
    # Process_GIF_Dataset(
    #     path_to_gif_dataset = original_gif_data_path,
    #     path_to_save_processed_data = processed_gif_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file
    #     )

    # Training Network on GIF Data
    # main_model_training(
    #     neural_network_hyperparameters = neural_network_hyperparameters_gif,
    #     path_to_processed_data = processed_gif_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_model_state = model_directory_path + model_state_after_shhs_file,
    #     path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #     path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
    #     )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Artifect" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    # project_configuration["neural_network_model"] = YaoModel
    project_configuration["neural_network_model"] = YaoModelNew

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_shhs,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Training Network on GIF Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_gif,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )


def predict_multiple_configurations(
        name_addition: str = "",
    ):
    """
    ==============
    Model Testing
    ==============
    """

    """
    ---------------------------------------------------------------------
    Testing Original Idea: Overlapping Windows and artifect = wake stage
    ---------------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data" + name_addition + "/shhs_data_original.pkl"
    processed_gif_path = "Processed_Data" + name_addition + "/gif_data_original.pkl"

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Original" + name_addition + "/"

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl",
    )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Original" + name_addition + "/"

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl",
    )

    """
    ---------------------------------------------------------------
    Testing with non-overlapping windows and artifect = wake stage
    ---------------------------------------------------------------
    """

    # Set File Paths
    # processed_shhs_path = "Processed_Data" + name_addition + "/shhs_data_no_overlap.pkl"
    # processed_gif_path = "Processed_Data" + name_addition + "/gif_data_no_overlap.pkl"

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_no_overlap" + name_addition + "/"

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl",
    )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_no_overlap" + name_addition + "/"

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl",
    )

    """
    -------------------------------------------------------------------
    Testing with Overlapping windows but artifect being a unique stage
    -------------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data" + name_addition + "/shhs_data_artifect.pkl"
    processed_gif_path = "Processed_Data" + name_addition + "/gif_data_artifect.pkl"

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Artifect" + name_addition + "/"

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl",
    )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Artifect" + name_addition + "/"

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl",
    )


def extensive_accuracy_printing(model_directory_path: str):
    """
    Prining accuracy results for a specific model directory
    """

    """
    SHHS Data
    """
    
    path_to_save_shhs_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl"
    shhs_training_pid_results_path = path_to_save_shhs_results[:-4] + "_Training_Pid.pkl"
    shhs_validation_pid_results_path = path_to_save_shhs_results[:-4] + "_Validation_Pid.pkl"

    # message = "Accuracy of SHHS Training and Validation Data in windows:"
    # print()
    # print("-"*len(message))
    # print(message)
    # print("-"*len(message))

    # print_model_performance(
    #     paths_to_pkl_files = [shhs_training_pid_results_path, shhs_validation_pid_results_path],
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     prediction_result_key = "Predicted_in_windows",
    #     actual_result_key = "Actual_in_windows",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3
    # )

    # message = "Accuracy of SHHS Training and Validation Data reshaped to original signal structure:"
    # print()
    # print("-"*len(message))
    # print(message)
    # print("-"*len(message))

    # print_model_performance(
    #     paths_to_pkl_files = [shhs_training_pid_results_path, shhs_validation_pid_results_path],
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3
    # )

    # message = "Accuracy of SHHS Validation Data in windows:"
    # print()
    # print("-"*len(message))
    # print(message)
    # print("-"*len(message))

    # print_model_performance(
    #     paths_to_pkl_files = [shhs_validation_pid_results_path],
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     prediction_result_key = "Predicted_in_windows",
    #     actual_result_key = "Actual_in_windows",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3
    # )

    message = "Accuracy of SHHS Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [shhs_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )

    """
    GIF Data
    """

    path_to_save_gif_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl"
    gif_training_pid_results_path = path_to_save_gif_results[:-4] + "_Training_Pid.pkl"
    gif_validation_pid_results_path = path_to_save_gif_results[:-4] + "_Validation_Pid.pkl"
    
    # message = "Accuracy of GIF Training and Validation Data in windows:"
    # print()
    # print("-"*len(message))
    # print(message)
    # print("-"*len(message))

    # print_model_performance(
    #     paths_to_pkl_files = [gif_training_pid_results_path, gif_validation_pid_results_path],
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     prediction_result_key = "Predicted_in_windows",
    #     actual_result_key = "Actual_in_windows",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3
    # )

    # message = "Accuracy of GIF Training and Validation Data reshaped to original signal structure:"
    # print()
    # print("-"*len(message))
    # print(message)
    # print("-"*len(message))

    # print_model_performance(
    #     paths_to_pkl_files = [gif_training_pid_results_path, gif_validation_pid_results_path],
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3
    # )

    # message = "Accuracy of GIF Validation Data in windows:"
    # print()
    # print("-"*len(message))
    # print(message)
    # print("-"*len(message))

    # print_model_performance(
    #     paths_to_pkl_files = [gif_validation_pid_results_path],
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     prediction_result_key = "Predicted_in_windows",
    #     actual_result_key = "Actual_in_windows",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3
    # )

    message = "Accuracy of GIF Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"zero_division": np.nan},
        number_of_decimals = 3
    )


def accuracy_multiple_configurations(
        name_addition: str = "",
    ):
    """
    Printing accuracy results for different configurations
    """

    """
    ---------------------------------------------------------------------
    Testing Original Idea: Overlapping Windows and artifect = wake stage
    ---------------------------------------------------------------------
    """

    message = "Original Idea: Overlapping Windows and artifect = wake stage:"
    print("="*len(message))
    print("="*len(message))
    print(message)
    print("="*len(message))
    print("="*len(message))

    """
    Using SleepStageModel
    """

    message = "SleepStageModel (directory: 'SSM_Original'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "SSM_Original" + name_addition + "/"

    extensive_accuracy_printing(model_directory_path)

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_Original'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_Original" + name_addition + "/"

    extensive_accuracy_printing(model_directory_path)

    """
    -------------------------------------------------------------------
    Testing with Overlapping windows but artifect being a unique stage
    -------------------------------------------------------------------
    """

    message = "Overlapping windows but artifect being a unique stage:"
    print("\n")
    print("="*len(message))
    print("="*len(message))
    print(message)
    print("="*len(message))
    print("="*len(message))

    """
    Using SleepStageModel
    """

    message = "SleepStageModel (directory: 'SSM_Artifect'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "SSM_Artifect" + name_addition + "/"

    extensive_accuracy_printing(model_directory_path)

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_Artifect'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_Artifect" + name_addition + "/"

    extensive_accuracy_printing(model_directory_path)

    """
    ---------------------------------------------------------------
    Testing with non-overlapping windows and artifect = wake stage
    ---------------------------------------------------------------
    """

    message = "Non-Overlapping windows and artifect = wake stage:"
    print("\n")
    print("="*len(message))
    print("="*len(message))
    print(message)
    print("="*len(message))
    print("="*len(message))

    """
    Using SleepStageModel
    """

    message = "SleepStageModel (directory: 'SSM_no_overlap'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "SSM_no_overlap" + name_addition + "/"

    extensive_accuracy_printing(model_directory_path)

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_no_overlap'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_no_overlap" + name_addition + "/"

    extensive_accuracy_printing(model_directory_path)


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


if True:
    # fix_project_configuration_3()
    # print_project_configuration()
    # raise SystemExit("Testing configurations...")

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
        "reshape_to_overlapping_windows": False, # whether to reshape the signals to overlapping windows
    }

    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

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
    project_configuration["neural_network_model"] = SleepStageModelNew
    project_configuration["number_sleep_stages"] = 4
    project_configuration["rri_datapoints"] = int(sleep_data_manager_parameters["RRI_frequency"] * project_configuration["signal_length_seconds"])
    project_configuration["mad_datapoints"] = int(sleep_data_manager_parameters["MAD_frequency"] * project_configuration["signal_length_seconds"])
    project_configuration["number_window_learning_features"] = 128
    project_configuration["rri_convolutional_channels"] = [1, 8, 16, 32, 64]
    project_configuration["mad_convolutional_channels"] = [1, 8, 16, 32, 64]
    project_configuration["window_learning_dilations"] = [2, 4, 8, 16, 32]

    main_pipeline(
        project_configuration = project_configuration,
        model_directory_path = "SSM_Original",
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