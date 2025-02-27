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


def fix_file_info(path):
    """
    """
    data_manager = SleepDataManager(file_path=path)
    data_manager.change_file_information({"SLP_predicted_frequency": 1/120})
    del data_manager


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
        neural_network_model = SleepStageModel,
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

    project_configuration["neural_network_model"] = SleepStageModel

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Preprocess SHHS Data
    Process_SHHS_Dataset(
        path_to_shhs_dataset = original_shhs_data_path,
        path_to_save_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        )

    # Training Network on SHHS Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_shhs,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Preprocess GIF Data
    Process_GIF_Dataset(
        path_to_gif_dataset = original_gif_data_path,
        path_to_save_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file
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
    Using YaoModel
    """

    model_directory_path = "Yao_Original" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    project_configuration["neural_network_model"] = YaoModel

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

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_no_overlap" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    project_configuration["neural_network_model"] = SleepStageModel

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
    Using YaoModel
    """

    model_directory_path = "Yao_no_overlap" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    project_configuration["neural_network_model"] = YaoModel

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

    project_configuration["neural_network_model"] = SleepStageModel

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Preprocess SHHS Data
    Process_SHHS_Dataset(
        path_to_shhs_dataset = original_shhs_data_path,
        path_to_save_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        )

    # Training Network on SHHS Data
    main_model_training(
        neural_network_hyperparameters = neural_network_hyperparameters_shhs,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Preprocess GIF Data
    Process_GIF_Dataset(
        path_to_gif_dataset = original_gif_data_path,
        path_to_save_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file
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
    Using YaoModel
    """

    model_directory_path = "Yao_Artifect" + name_addition + "/"
    create_directories_along_path(model_directory_path)

    project_configuration["neural_network_model"] = YaoModel

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
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
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
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
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
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
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
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
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
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
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
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_performance_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
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

    message = "Accuracy of SHHS Training and Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [shhs_training_pid_results_path, shhs_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

    message = "Accuracy of SHHS Training and Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [shhs_training_pid_results_path, shhs_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

    message = "Accuracy of SHHS Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [shhs_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

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
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

    """
    GIF Data
    """

    path_to_save_gif_results = model_directory_path + model_performance_file[:-4] + "_GIF.pkl"
    gif_training_pid_results_path = path_to_save_gif_results[:-4] + "_Training_Pid.pkl"
    gif_validation_pid_results_path = path_to_save_gif_results[:-4] + "_Validation_Pid.pkl"
    
    message = "Accuracy of GIF Training and Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [gif_training_pid_results_path, gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

    message = "Accuracy of GIF Training and Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [gif_training_pid_results_path, gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

    message = "Accuracy of GIF Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_performance(
        paths_to_pkl_files = [gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

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
        additional_score_function_args = {"average": None, "zero_division": np.nan},
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


if False:
    name_addition = ""
    project_configuration_change = {
        "RRI_inlier_interval": [None, None],
        "normalize_rri": False,
        "normalize_mad": False,
    }
    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = "_rm_outliers"
    project_configuration_change = {
        "RRI_inlier_interval": [0.3, 2],
        "normalize_rri": False,
        "normalize_mad": False,
        "normalization_mode": "global",
    }

    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = "_norm_global"
    project_configuration_change = {
        "RRI_inlier_interval": [0.3, 2],
        "normalize_rri": True,
        "normalize_mad": False,
        "normalization_mode": "global",
    }

    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = "_norm_local"
    project_configuration_change = {
        "RRI_inlier_interval": [0.3, 2],
        "normalize_rri": True,
        "normalize_mad": False,
        "normalization_mode": "local",
    }

    train_multiple_configurations(name_addition, project_configuration_change)
    predict_multiple_configurations(name_addition)

    name_addition = ""
    accuracy_multiple_configurations(name_addition)

    name_addition = "_rm_outliers"
    accuracy_multiple_configurations(name_addition)

    name_addition = "_norm_global"
    accuracy_multiple_configurations(name_addition)

    name_addition = "_norm_local"
    accuracy_multiple_configurations(name_addition)


if __name__ == "__main__":
    # fix_file_info("Processed_NAKO/NAKO-33a.pkl")
    # fix_project_configuration_2()
    print_project_configuration()

    # data_manager = SleepDataManager(file_path="Processed_NAKO/NAKO-33b.pkl")
    # print(data_manager.file_info)