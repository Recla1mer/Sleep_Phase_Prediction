"""
Author: Johannes Peter Knoll

This file is unnecessary for the user. I used this to test different data processing configurations and 
neural network models.
"""

from main import *
from plot_helper import *

"""
===============
Model Training
===============
"""

def train_multiple_configurations():

    """
    ---------------------------------------------------------------------
    Testing Original Idea: Overlapping Windows and artifect = wake stage
    ---------------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data/shhs_data_original.pkl"
    processed_gif_path = "Processed_Data/gif_data_original.pkl"

    # Set Signal Processing Parameters
    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

    check_project_configuration(project_configuration)

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Original/"
    create_directories_along_path(model_directory_path)

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
        neural_network_model = SleepStageModel,
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
        neural_network_model = SleepStageModel,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Original/"
    create_directories_along_path(model_directory_path)

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    main_model_training(
        neural_network_model = YaoModel, # type: ignore
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Training Network on GIF Data
    main_model_training(
        neural_network_model = YaoModel, # type: ignore
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
    processed_shhs_path = "Processed_Data/shhs_data_artifect.pkl"
    processed_gif_path = "Processed_Data/gif_data_artifect.pkl"

    # Set Signal Processing Parameters
    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

    project_configuration["sleep_stage_label"] = {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}
    project_configuration["priority_order"] = [4, 3, 2, 1, 0]
    project_configuration["number_sleep_stages"] = 5

    check_project_configuration(project_configuration)

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Artifect/"
    create_directories_along_path(model_directory_path)

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
        neural_network_model = SleepStageModel,
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
        neural_network_model = SleepStageModel,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Artifect/"
    create_directories_along_path(model_directory_path)

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    main_model_training(
        neural_network_model = YaoModel, # type: ignore
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Training Network on GIF Data
    main_model_training(
        neural_network_model = YaoModel, # type: ignore
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
    processed_shhs_path = "Processed_Data/shhs_data_no_overlap.pkl"
    processed_gif_path = "Processed_Data/gif_data_no_overlap.pkl"

    # Set Signal Processing Parameters
    project_configuration = dict()
    project_configuration.update(sleep_data_manager_parameters)
    project_configuration.update(window_reshape_parameters)
    project_configuration.update(split_data_parameters)
    project_configuration.update(dataset_class_transform_parameters)
    project_configuration.update(neural_network_model_parameters)

    project_configuration["overlap_seconds"] = 0
    project_configuration["number_windows"] = 300
    project_configuration["windows_per_signal"] = 300

    check_project_configuration(project_configuration)

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_no_overlap/"
    create_directories_along_path(model_directory_path)

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
        neural_network_model = SleepStageModel,
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
        neural_network_model = SleepStageModel,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_no_overlap/"
    create_directories_along_path(model_directory_path)

    if os.path.isfile(model_directory_path + project_configuration_file):
        os.remove(model_directory_path + project_configuration_file)
    save_to_pickle(project_configuration, model_directory_path + project_configuration_file)

    # Training Network on SHHS Data
    main_model_training(
        neural_network_model = YaoModel, # type: ignore
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = None,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_shhs_file,
        )

    # Training Network on GIF Data
    main_model_training(
        neural_network_model = YaoModel, # type: ignore
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_model_state = model_directory_path + model_state_after_shhs_file,
        path_to_updated_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_loss_per_epoch = model_directory_path + loss_per_epoch_gif_file,
        )


def predict_multiple_configurations():
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
    processed_shhs_path = "Processed_Data/shhs_data_original.pkl"
    processed_gif_path = "Processed_Data/gif_data_original.pkl"

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Original/"

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Original/"

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    )

    """
    -------------------------------------------------------------------
    Testing with Overlapping windows but artifect being a unique stage
    -------------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data/shhs_data_artifect.pkl"
    processed_gif_path = "Processed_Data/gif_data_artifect.pkl"

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_Artifect/"

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_Artifect/"

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    )

    """
    ---------------------------------------------------------------
    Testing with non-overlapping windows and artifect = wake stage
    ---------------------------------------------------------------
    """

    # Set File Paths
    processed_shhs_path = "Processed_Data/shhs_data_no_overlap.pkl"
    processed_gif_path = "Processed_Data/gif_data_no_overlap.pkl"

    """
    Using SleepStageModel
    """

    model_directory_path = "SSM_no_overlap/"

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = SleepStageModel,
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    )

    """
    Using YaoModel
    """

    model_directory_path = "Yao_no_overlap/"

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_shhs_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    )

    predictions_for_model_accuracy_evaluation(
        neural_network_model = YaoModel, # type: ignore
        path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
        path_to_processed_data = processed_gif_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    )


def extensive_accuracy_printing(model_directory_path: str):
    """
    Prinintg accuracy results for a specific model directory
    """

    """
    SHHS Data
    """
    
    path_to_save_shhs_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl"
    shhs_training_pid_results_path = path_to_save_shhs_results[:-4] + "_Training_Pid.pkl"
    shhs_validation_pid_results_path = path_to_save_shhs_results[:-4] + "_Validation_Pid.pkl"

    message = "Accuracy of SHHS Training and Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
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

    print_model_accuracy(
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

    print_model_accuracy(
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

    print_model_accuracy(
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

    path_to_save_gif_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl"
    gif_training_pid_results_path = path_to_save_gif_results[:-4] + "_Training_Pid.pkl"
    gif_validation_pid_results_path = path_to_save_gif_results[:-4] + "_Validation_Pid.pkl"
    
    message = "Accuracy of GIF Training and Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
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

    print_model_accuracy(
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

    print_model_accuracy(
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

    print_model_accuracy(
        paths_to_pkl_files = [gif_validation_pid_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )


def accuracy_multiple_configurations():
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

    model_directory_path = "SSM_Original/"

    extensive_accuracy_printing(model_directory_path)

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_Original'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_Original/"

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

    model_directory_path = "SSM_Artifect/"

    extensive_accuracy_printing(model_directory_path)

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_Artifect'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_Artifect/"

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

    model_directory_path = "SSM_no_overlap/"

    extensive_accuracy_printing(model_directory_path)

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_no_overlap'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_no_overlap/"

    extensive_accuracy_printing(model_directory_path)


if __name__ == "__main__":
    # train_multiple_configurations()
    # predict_multiple_configurations()
    # accuracy_multiple_configurations()

    """
    =============================
    Plot Accuracy/Loss per Epoch
    =============================
    """

    # plot_accuracy_per_epoch(
    #     paths_to_pkl_files = ["Yao_no_overlap/Loss_per_Epoch_SHHS.pkl"],
    #     result_keys = ["train_accuracy", "train_avg_loss", "test_accuracy", "test_avg_loss"],
    #     label = ["train_accuracy", "train_avg_loss", "test_accuracy", "test_avg_loss"],
    #     title = "Training YaoModel on SHHS Data",
    #     xlabel = "Epoch",
    #     ylabel = "Accuracy / Loss",
    # )

    # plot_accuracy_per_epoch(
    #     paths_to_pkl_files = ["Yao_no_Overlap/Loss_per_Epoch_GIF.pkl", "Yao_no_overlap/Loss_per_Epoch_SHHS.pkl"],
    #     result_keys = ["test_accuracy"],
    #     label = ["GIF", "SHHS"],
    #     title = "History of YaoModel",
    #     xlabel = "Epoch",
    #     ylabel = "Test Accuracy",
    # )

    """
    ================================================
    Plot Distribution for different Score Functions
    ================================================
    """

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Yao_no_overlap/Model_Accuracy_GIF_Training_Pid.pkl", "Yao_no_overlap/Model_Accuracy_GIF_Validation_Pid.pkl"],
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     score_function = metrics.accuracy_score, # metrics.cohen_kappa_score
    #     combine_file_predictions = False,
    #     title = "Distribution of Accuracy",
    #     xlabel = "Accuracy",
    #     label = ["Training Data", "Validation Data"],
    #     binrange = (0, 1),
    #     binwidth = 0.05,
    #     xlim = (0.6, 1.01),
    # )

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Yao_no_overlap/Model_Accuracy_GIF_Training_Pid.pkl", "Yao_no_overlap/Model_Accuracy_GIF_Validation_Pid.pkl"],
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     score_function = metrics.accuracy_score, # metrics.cohen_kappa_score
    #     combine_file_predictions = True,
    #     title = "Combined Training and Validation Accuracy Distribution",
    #     xlabel = "Accuracy",
    #     binrange = (0, 1),
    #     binwidth = 0.05,
    #     xlim = (0.6, 1.01),
    # )

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Yao_no_overlap/Model_Accuracy_GIF_Validation_Pid.pkl"],
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     score_function = metrics.precision_score, # metrics.f1_score
    #     additional_score_function_args = {"average": None, "labels": [0,1,2,3], "zero_division": np.nan}, # or: None, 'micro', 'macro', 'weighted' ('binary', 'samples')
    #     title = "Distribution of Precision for GIF Validation Data",
    #     xlabel = "Precision",
    #     label = ["Wake", "LS", "DS", "REM"],
    #     binrange = (0, 1),
    #     binwidth = 0.05,
    #     xlim = (0.0, 1.01),
    # )

    # plot_distribution_of_score(
    #     paths_to_pkl_files = ["Yao_no_overlap/Model_Accuracy_GIF_Validation_Pid.pkl"],
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     score_function =  metrics.recall_score, # metrics.f1_score
    #     additional_score_function_args = {"average": 'weighted', "labels": [0,1,2,3], "zero_division": np.nan}, # or: None, 'micro', 'macro', 'weighted' ('binary', 'samples')
    #     title = "Distribution of Recall for GIF Validation Data",
    #     xlabel = "Weighted Recall",
    #     binrange = (0, 1),
    #     binwidth = 0.05,
    #     xlim = (0.5, 1.01),
    # )

    """
    ======================
    Plot Confusion Matrix
    ======================
    """

    # plot_confusion_matrix(
    #     path_to_pkl_file = "Yao_no_overlap/Model_Accuracy_GIF_Validation_Pid.pkl",
    #     path_to_project_configuration = "Yao_no_overlap/Project_Configuration.pkl",
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     title = "Confusion Matrix of Neural Network",
    #     xlabel = "predicted stage",
    #     ylabel = "actual stage",
    #     normalize = None, # 'true', 'pred', 'all'
    #     values_format = None, # 'd', 'f', '.1%'
    # )

    """
    ==========================
    Plot Actual vs. Predicted
    ==========================
    """

    with open("Yao_no_overlap/Project_Configuration.pkl", "rb") as f:
        project_configuration = pickle.load(f)
    
    reciprocal_slp_frequency = round(1 / project_configuration['SLP_frequency'])

    # plot_actual_predicted(
    #     path_to_pkl_file = "Yao_no_overlap/Model_Accuracy_GIF_Validation_Pid.pkl",
    #     path_to_project_configuration = "Yao_no_overlap/Project_Configuration.pkl",
    #     prediction_result_key = "Predicted",
    #     actual_result_key = "Actual",
    #     data_position = None,
    #     title = "Sleep Stages",
    #     xlabel = r"Time $\left(\text{in } %i \text{s}\right)$" % reciprocal_slp_frequency,
    #     ylabel = "Sleep Stage",
    # )