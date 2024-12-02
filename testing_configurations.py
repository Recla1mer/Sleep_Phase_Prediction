"""
Author: Johannes Peter Knoll

This file is unnecessary for the user. I used this to test different data processing configurations and 
neural network models.
"""

from main import *

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


def extensive_accuracy_printing(model_directory_path: str, display_labels = ["Wake", "LS", "DS", "REM"]):
    """
    Prinintg accuracy results for a specific model directory
    """

    """
    SHHS Data
    """

    message = "Accuracy of SHHS Training and Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_SHHS_Training_Pid.pkl", model_directory_path + "Model_Accuracy_SHHS_Validation_Pid.pkl"],
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    message = "Accuracy of SHHS Training and Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_SHHS_Training_Pid.pkl", model_directory_path + "Model_Accuracy_SHHS_Validation_Pid.pkl"],
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    message = "Accuracy of SHHS Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_SHHS_Validation_Pid.pkl"],
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    message = "Accuracy of SHHS Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_SHHS_Validation_Pid.pkl"],
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    """
    GIF Data
    """
    
    message = "Accuracy of GIF Training and Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_GIF_Training_Pid.pkl", model_directory_path + "Model_Accuracy_GIF_Validation_Pid.pkl"],
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    message = "Accuracy of GIF Training and Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_GIF_Training_Pid.pkl", model_directory_path + "Model_Accuracy_GIF_Validation_Pid.pkl"],
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    message = "Accuracy of GIF Validation Data in windows:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_GIF_Validation_Pid.pkl"],
        prediction_result_key = "Predicted_in_windows",
        actual_result_key = "Actual_in_windows",
        display_labels = display_labels,
        average = None,
        number_of_decimals = 3
    )

    message = "Accuracy of GIF Validation Data reshaped to original signal structure:"
    print()
    print("-"*len(message))
    print(message)
    print("-"*len(message))

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_GIF_Validation_Pid.pkl"],
        prediction_result_key = "Predicted",
        actual_result_key = "Actual",
        display_labels = display_labels,
        average = None,
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

    extensive_accuracy_printing(model_directory_path, display_labels = ["Artifect", "Wake", "LS", "DS", "REM"])

    """
    Using YaoModel
    """

    message = "YaoModel (directory: 'Yao_Artifect'):"
    print()
    print("="*len(message))
    print(message)
    print("="*len(message))

    model_directory_path = "Yao_Artifect/"

    extensive_accuracy_printing(model_directory_path, display_labels = ["Artifect", "Wake", "LS", "DS", "REM"])

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
    accuracy_multiple_configurations()