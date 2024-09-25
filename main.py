"""
Author: Johannes Peter Knoll

This file executes all the code needed to preprocess the data and train the neural network.
It is basically the less commented version of the notebook: "Classification_Demo.ipynb".
"""

# IMPORTS
from sklearn.metrics import cohen_kappa_score, f1_score

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *

"""
============================================
Applying SleepDataManager Class To Our Data
============================================
"""

import h5py

def Process_SHHS_Dataset(
        path_to_shhs_dataset: str,
        path_to_save_processed_data: str,
        change_data_parameters: dict = {},
        train_size = 0.8, 
        validation_size = 0.1, 
        test_size = 0.1, 
        random_state = None, 
        shuffle = True,
    ):
    """
    This function processes our SHHS dataset. It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the SHHS dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    change_data_parameters argument. Afterwards we will use the class to split the data into training,
    validation, and test pids (individual files).

    If already processed, the function will only shuffle the data in the pids again.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_shhs_dataset: str
        the path to the SHHS dataset
    path_to_save_processed_data: str
        the path to save the processed SHHS dataset
    change_data_parameters: dict
        the parameters that are used to keep data uniform 
        (see SleepDataManager class in dataset_processing.py)
    train_size: float
        the size of the training dataset
    validation_size: float
        the size of the validation dataset
    test_size: float
        the size of the test dataset
    random_state: int
        the random state to use for the train-test-validation split
    shuffle: bool
        whether to shuffle the data before splitting
    """

    # following path will be created at the end of this function, if it exists, skip the processing part
    shhs_training_data_path = path_to_save_processed_data[:-4] + "_training_pid.pkl"

    # initializing the database
    shhs_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

    # update data parameters, if necessary
    if len(change_data_parameters) > 0:
        shhs_data_manager.change_file_information(change_data_parameters)

    if not os.path.exists(shhs_training_data_path):

        # access the SHHS dataset
        shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')
        
        # define the sleep stage labels (attention: a different dataset will most likely have different labels)
        shhs_sleep_stage_label = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifect": ["other"]}

        # accessing patient ids:
        patients = list(shhs_dataset['slp'].keys()) # type: ignore

        # check if patient ids are unique:
        shhs_data_manager.check_if_ids_are_unique(patients)

        # showing progress bar
        start_time = time.time()
        total_data_points = len(patients)
        print("\nPreproccessing Datapoints from SHHS Dataset:")
        progress_bar(0, total_data_points, 1, start_time, None, None)

        # saving all data from SHHS dataset to the shhs_data.pkl file
        for patient_index in range(total_data_points):
            patient_id = patients[patient_index]
            new_datapoint = {
                "ID": patient_id,
                "RRI": shhs_dataset["rri"][patient_id][:], # type: ignore
                "SLP": shhs_dataset["slp"][patient_id][:], # type: ignore
                "RRI_frequency": shhs_dataset["rri"].attrs["freq"], # type: ignore
                "SLP_frequency": shhs_dataset["slp"].attrs["freq"], # type: ignore
                "sleep_stage_label": copy.deepcopy(shhs_sleep_stage_label)
            }

            shhs_data_manager.save(new_datapoint, unique_id=True)
            progress_bar(patient_index+1, total_data_points, 1, start_time, None, None)
    
    else:
        print("\nATTENTION: SHHS dataset seems to be processed already. Skipping processing. Only the datapoints in the training-, validation, and test pid will be randomly distributed again.")

    # transform signal to overlapping windows
    shhs_data_manager.apply_signal_reshape()
    
    # Train-, Validation- and Test-Split
    shhs_data_manager.separate_train_test_validation(
        train_size = train_size, 
        validation_size = validation_size, 
        test_size = test_size, 
        random_state = random_state, 
        shuffle = shuffle
    )


def Process_GIF_Dataset(
        path_to_gif_dataset: str,
        path_to_save_processed_data: str,
        change_data_parameters: dict = {},
        train_size = 0.8, 
        validation_size = 0.1, 
        test_size = 0.1, 
        random_state = None, 
        shuffle = True,
    ):
    """
    This function processes our GIF dataset. It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the SleepDataManager class from dataset_processing.py 
    to process a dataset.

    The datapoints from the GIF dataset are resaved to a pickle file using the SleepDataManager class.
    The class is designed to save the data in a uniform way. How exactly can be altered using the
    change_data_parameters argument. Afterwards we will use the class to split the data into training,
    validation, and test pids (individual files).

    If already processed, the function will only shuffle the data in the pids again.

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_shhs_dataset: str
        the path to the SHHS dataset
    path_to_save_processed_data: str
        the path to save the processed SHHS dataset
    change_data_parameters: dict
        the parameters that are used to keep data uniform 
        (see SleepDataManager class in dataset_processing.py)
    train_size: float
        the size of the training dataset
    validation_size: float
        the size of the validation dataset
    test_size: float
        the size of the test dataset
    random_state: int
        the random state to use for the train-test-validation split
    shuffle: bool
        whether to shuffle the data before splitting
    """

    # following path will be created at the end of this function, if it exists, skip the processing part
    gif_training_data_path = path_to_save_processed_data[:-4] + "_training_pid.pkl"

    # initializing the database
    gif_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

    # update data parameters, if necessary
    if len(change_data_parameters) > 0:
        gif_data_manager.change_file_information(change_data_parameters)

    if not os.path.exists(gif_training_data_path):

        # access the GIF dataset
        gif_dataset = h5py.File(path_to_gif_dataset, 'r')

        # define the sleep stage labels (attention: a different dataset will most likely have different labels)
        gif_sleep_stage_label = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifect": ["other"]}

        # accessing patient ids:
        patients = list(gif_dataset['stage'].keys()) # type: ignore

        # check if patient ids are unique:
        gif_data_manager.check_if_ids_are_unique(patients)

        # showing progress bar
        start_time = time.time()
        total_data_points = len(patients)
        print("\nPreproccessing Datapoints from GIF Dataset:")
        progress_bar(0, total_data_points, 1, start_time, None, None)

        # saving all data from GIF dataset to the gif_data.pkl file
        for patient_index in range(total_data_points):
            patient_id = patients[patient_index]
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

            progress_bar(patient_index+1, total_data_points, 1, start_time, None, None)

    else:
        print("\nATTENTION: GIF dataset seems to be processed already. Skipping processing. Only the datapoints in the training-, validation, and test pid will be randomly distributed again.")
    
    # Train-, Validation- and Test-Split
    gif_data_manager.separate_train_test_validation(
        train_size = train_size, 
        validation_size = validation_size, 
        test_size = test_size, 
        random_state = random_state, 
        shuffle = shuffle
    )


"""
==========================================
Training And Testing Neural Network Model
==========================================
"""


def main_model_training(
        neural_network_model = SleepStageModel(),
        load_model_state_path = None,
        processed_path = "Processed_Data/shhs_data.pkl",
        save_accuracy_values_path: str = "Model_Accuracy/Neural_Network.pkl",
        save_model_state_path: str = "Model_State/Neural_Network.pth",
        window_reshape_parameters: dict = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
    ):
    """
    Full implementation of project, with ability to easily change most important parameters to test different
    neural network architecture configurations. Some Parameters are hardcoded by design.

    The Data is accessed using the CustomSleepDataset class from neural_network_model.py. Before returning 
    the data, this class reshapes the data into windows. Adjustments can be made using the 
    window_reshape_parameters argument.

    Afterwards the neural network model is trained and tested. The accuracy results are saved in a pickle file
    and the model state dictionary is saved in a .pth file.

    The accuracy values are saved in a dictionary with the following format:
    {
        "train_accuracy": train_accuracy for each epoch (list),
        "train_avg_loss": train_avg_loss for each epoch (list),
        "train_predicted_results": all predicted results for last epoch (list),
        "train_actual_results": all actual results for last epoch (list),
        "test_accuracy": test_accuracy for each epoch (list),
        "test_avg_loss": test_avg_loss for each epoch (list),
        "test_predicted_results": all predicted results for last epoch (list),
        "test_actual_results": all actual results for last epoch (list)
    }

    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    load_model_state_path: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    processed_path: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    save_accuracy_values_path: str
        the path to save the accuracy values
    save_model_state_path: str
        the path to save the model state dictionary
    window_reshape_parameters: dict
        the parameters used when reshaping the signal to windows 
        (see reshape_signal_to_overlapping_windows function in dataset_processing.py)
    pad_feature_with : int
        Value to pad feature (RRI and MAD) with if signal too short, by default 0
    pad_target_with : int
        Value to pad target (SLP) with if signal too short, by default 0
    """

    
    """
    ------------------
    Accessing Dataset
    ------------------
    """

    CustomSleepDataset_keywords = {
        "transform": ToTensor(), 
        "window_reshape_parameters": window_reshape_parameters, 
        "pad_feature_with": pad_feature_with, 
        "pad_target_with": pad_target_with
    }

    training_data_path = processed_path[:-4] + "_training_pid.pkl"
    validation_data_path = processed_path[:-4] + "_validation_pid.pkl"
    test_data_path = processed_path[:-4] + "_test_pid.pkl"

    training_data = CustomSleepDataset(path_to_data = training_data_path, **CustomSleepDataset_keywords)
    validation_data = CustomSleepDataset(path_to_data = validation_data_path, **CustomSleepDataset_keywords)
    test_data = CustomSleepDataset(path_to_data = test_data_path, **CustomSleepDataset_keywords)
    
    """
    ----------------
    Hyperparameters
    ----------------
    """

    batch_size = 8
    number_epochs = 2

    learning_rate_scheduler = CosineScheduler(
        number_updates_total = number_epochs,
        number_updates_to_max_lr = 10,
        start_learning_rate = 2.5 * 1e-5,
        max_learning_rate = 1 * 1e-4,
        end_learning_rate = 5 * 1e-5
    )

    """
    ---------------------------------------------
    Preparing Data For Training With Dataloaders
    ---------------------------------------------
    """

    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size = batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True)
    
    del CustomSleepDataset_keywords, training_data_path, validation_data_path, test_data_path

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
   
    if load_model_state_path is not None:
        neural_network_model.load_state_dict(torch.load(load_model_state_path, map_location=device, weights_only=True))
    
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

        # get all the results from the training and testing loop in the last epoch
        collect_results = False
        if t == number_epochs - 1:
            collect_results = True

        train_results = train_loop(
            dataloader = train_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            optimizer_fn = optimizer_function,
            lr_scheduler = learning_rate_scheduler,
            current_epoch = t,
            batch_size = batch_size,
            collect_results = collect_results
        )
        train_avg_loss.append(train_results[0])
        train_accuracy.append(train_results[1])

        test_results = test_loop(
            dataloader = validation_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            batch_size = batch_size,
            collect_results = collect_results
        )

        test_avg_loss.append(test_results[0])
        test_accuracy.append(test_results[1])

    
    """
    -----------------------
    Saving Accuracy Values
    -----------------------
    """

    create_directories_along_path(save_accuracy_values_path)

    accuracy_values = {
        "train_accuracy": train_accuracy,
        "train_avg_loss": train_avg_loss,
        "train_predicted_results": train_results[2],
        "train_actual_results": train_results[3],
        "test_accuracy": test_accuracy,
        "test_avg_loss": test_avg_loss,
        "test_predicted_results": test_results[2],
        "test_actual_results": test_results[3]
    }

    save_to_pickle(accuracy_values, save_accuracy_values_path)

    """
    ----------------------------------
    Saving Neural Network Model State
    ----------------------------------
    """

    create_directories_along_path(save_model_state_path)
    
    torch.save(neural_network_model.state_dict(), save_model_state_path)


def predicting_sleep_stage_using_trained_model(
        neural_network_model = SleepStageModel(),
        path_to_model_state: str = "Model_State/Neural_Network.pth",
        path_to_processed_data: str = "Processed_Data/shhs_data_validation_pid.pkl",
        path_to_save_results: str = "Results/Neural_Network.pkl",
    ):
    """
    """


if __name__ == "__main__":
    train_size = 0.05
    validation_size = 0.05
    test_size = 0.9

    # Preprocess SHHS Data
    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    Process_SHHS_Dataset(
        path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", 
        path_to_save_processed_data = processed_shhs_path,
        change_data_parameters = {},
        train_size = train_size, 
        validation_size = validation_size, 
        test_size = test_size, 
        )

    # Train and test model on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/NN_SHHS.pkl",
        save_model_state_path = "Model_State/NN_SHHS.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )

    # Preprocess GIF Data
    processed_gif_path = "Processed_Data/gif_data.pkl"
    Process_GIF_Dataset(
        path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", 
        path_to_save_processed_data = processed_gif_path,
        change_data_parameters = {},
        train_size = train_size, 
        validation_size = validation_size, 
        test_size = test_size
        )

    # Train and test model on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(), # type: ignore
        load_model_state_path = "Model_State/NN_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/NN_SHHS_GIF.pkl",
        save_model_state_path = "Model_State/NN_SHHS_GIF.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    raise SystemExit

    """
    ---------------------------------------------------------------------
    Testing Original Idea: Overlapping Windows and artifect = wake stage
    ---------------------------------------------------------------------
    """

    # Preprocess SHHS Data
    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    Process_SHHS_Dataset(path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", path_to_save_processed_data = processed_shhs_path, change_data_parameters = {})

    # Train and test different models on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(),
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/SSM_Original.pkl",
        save_model_state_path = "Model_State/SSM_Original.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    main_model_training(
        neural_network_model = YaoModel(), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/Yao_Original.pkl",
        save_model_state_path = "Model_State/Yao_Original.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    # Preprocess GIF Data
    processed_gif_path = "Processed_Data/gif_data.pkl"
    Process_GIF_Dataset(path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", path_to_save_processed_data = processed_gif_path, change_data_parameters = {})

    # Train and test different models on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(),
        load_model_state_path = "Model_State/SSM_Original.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/SSM_Original_GIF.pkl",
        save_model_state_path = "Model_State/SSM_Original_GIF.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    main_model_training(
        neural_network_model = YaoModel(), # type: ignore
        load_model_state_path = "Model_State/Yao_Original.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/Yao_Original_GIF.pkl",
        save_model_state_path = "Model_State/Yao_Original_GIF.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    """
    -------------------------------------------------------------------
    Testing with Overlapping windows but artifect being a unique stage
    -------------------------------------------------------------------
    """

    # Preprocess SHHS Data
    processed_shhs_path = "Processed_Data/shhs_data_artifect.pkl"
    change_data_parameters = {"sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": -1}}
    
    Process_SHHS_Dataset(path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", path_to_save_processed_data = processed_shhs_path, change_data_parameters = change_data_parameters)
    
    default_window_reshape_parameters["priority_order"] = [3, 2, 1, 0, -1]

    # Train and test different models on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(number_sleep_stages = 5),
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/SSM_Artifect.pkl",
        save_model_state_path = "Model_State/SSM_Artifect.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = -1
        )
    
    main_model_training(
        neural_network_model = YaoModel(number_sleep_stages = 5), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/Yao_Artifect.pkl",
        save_model_state_path = "Model_State/Yao_Artifect.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = -1
        )
    
    # Preprocess GIF Data
    processed_gif_path = "Processed_Data/gif_data_artifect.pkl"
    Process_GIF_Dataset(path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", path_to_save_processed_data = processed_gif_path, change_data_parameters = change_data_parameters)

    # Train and test different models on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(number_sleep_stages = 5),
        load_model_state_path = "Model_State/SSM_Artifect.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/SSM_Artifect_GIF.pkl",
        save_model_state_path = "Model_State/SSM_Artifect_GIF.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = -1
        )
    
    main_model_training(
        neural_network_model = YaoModel(number_sleep_stages = 5), # type: ignore
        load_model_state_path = "Model_State/Yao_Artifect.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/Yao_Artifect_GIF.pkl",
        save_model_state_path = "Model_State/Yao_Artifect_GIF.pth",
        window_reshape_parameters = default_window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = -1
        )
    
    default_window_reshape_parameters["priority_order"] = [3, 2, 1, 0]

    """
    ---------------------------------------------------------------
    Testing with non-overlapping windows and artifect = wake stage
    ---------------------------------------------------------------
    """

    # Preprocess SHHS Data
    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    Process_SHHS_Dataset(path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", path_to_save_processed_data = processed_shhs_path, change_data_parameters = {})
    
    window_reshape_parameters = {
        "nn_signal_duration_seconds": 10*3600,
        "number_windows": 300, 
        "window_duration_seconds": 120, 
        "overlap_seconds": 0,
        "priority_order": [3, 2, 1, 0]
    }

    # Train and test different models on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(),
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/SSM_no_overlap.pkl",
        save_model_state_path = "Model_State/SSM_no_overlap.pth",
        window_reshape_parameters = window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    main_model_training(
        neural_network_model = YaoModel(), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = "Model_Accuracy/Yao_no_overlap.pkl",
        save_model_state_path = "Model_State/Yao_no_overlap.pth",
        window_reshape_parameters = window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    # Preprocess GIF Data
    processed_gif_path = "Processed_Data/gif_data.pkl"
    Process_GIF_Dataset(path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", path_to_save_processed_data = processed_gif_path, change_data_parameters = {})

    # Train and test different models on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(),
        load_model_state_path = "Model_State/SSM_no_overlap.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/SSM_no_overlap_GIF.pkl",
        save_model_state_path = "Model_State/SSM_no_overlap_GIF.pth",
        window_reshape_parameters = window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )
    
    main_model_training(
        neural_network_model = YaoModel(), # type: ignore
        load_model_state_path = "Model_State/Yao_no_overlap.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = "Model_Accuracy/Yao_no_overlap_GIF.pkl",
        save_model_state_path = "Model_State/Yao_no_overlap_GIF.pth",
        window_reshape_parameters = window_reshape_parameters,
        pad_feature_with = 0,
        pad_target_with = 0
        )