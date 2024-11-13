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


def test_multiple_configurations():

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

    # predictions_for_model_accuracy_evaluation(
    #     neural_network_model = SleepStageModel,
    #     path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #     path_to_processed_data = processed_shhs_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_SHHS.pkl",
    # )

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_SHHS_Training_Pid.pkl", model_directory_path + "Model_Accuracy_SHHS_Validation_Pid.pkl"],
        prediction_result_key = "Predicted_in_windows",
        actual_result_keys = "Actual_in_windows",
        display_labels = ["Wake", "LS", "DS", "REM"],
        average = None,
        number_of_decimals = 3
    )

    print_model_accuracy(
        paths_to_pkl_files = [model_directory_path + "Model_Accuracy_SHHS_Training_Pid.pkl", model_directory_path + "Model_Accuracy_SHHS_Validation_Pid.pkl"],
        prediction_result_key = "Predicted",
        actual_result_keys = "Actual",
        display_labels = ["Wake", "LS", "DS", "REM"],
        average = None,
        number_of_decimals = 3
    )

    # predictions_for_model_accuracy_evaluation(
    #     neural_network_model = SleepStageModel,
    #     path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #     path_to_processed_data = processed_gif_path,
    #     path_to_project_configuration = model_directory_path + project_configuration_file,
    #     path_to_save_results = model_directory_path + model_accuracy_file[:-4] + "_GIF.pkl",
    # )

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


if __name__ == "__main__":
    # train_multiple_configurations()
    test_multiple_configurations()