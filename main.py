"""
Author: Johannes Peter Knoll

This file executes all the code needed to preprocess the data and train the neural network.
It is basically the less commented version of the notebook: "Classification_Demo.ipynb".
"""

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *

"""
--------------------------------------------
Applying SleepDataManager class to our data
--------------------------------------------
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
    --------------------------------
    None

    ARGUMENTS:
    --------------------------------
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

        # saving all data from SHHS dataset to the shhs_data.pkl file
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
    
    else:
        print("\nATTENTION: SHHS dataset seems to be processed already. Skipping processing. Only the datapoints in the training-, validation, and test pid will be randomly distributed again.")
    
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
    --------------------------------
    None

    ARGUMENTS:
    --------------------------------
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

        # saving all data from GIF dataset to the gif_data.pkl file
        for patient_id in patients:
            new_datapoint = {
                "ID": patient_id,
                "RRI": gif_dataset["rri"][patient_id][:], # type: ignore
                "MAD": gif_dataset["mad"][patient_id][:], # type: ignore
                "SLP": gif_dataset["stage"][patient_id][:], # type: ignore
                "RRI_frequency": gif_dataset["rri"].attrs["freq"], # type: ignore
                "MAD_frequency": gif_dataset["mad"].attrs["freq"], # type: ignore
                "SLP_frequency": 1/30, # type: ignore
                "sleep_stage_label": copy.deepcopy(gif_sleep_stage_label)
            }

            gif_data_manager.save(new_datapoint, unique_id=True)

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
------------------------------------------
Training and Testing Neural Network Model
------------------------------------------
"""


def main(
        neural_network_model = SleepStageModel(),
        save_file_name: str = "Neural_Network",
        save_accuracy_directory: str = "Accuracy",
        save_model_directory: str = "Model_State",
        processed_shhs_path = "Processed_Data/shhs_data.pkl",
        processed_gif_path = "Processed_Data/gif_data.pkl",
        window_reshape_parameters: dict = default_window_reshape_parameters,
        change_data_parameters: dict = {}
    ):
    """
    Full implementation of project, with ability to easily change most important parameters to test different
    dataset preprocessing and neural network architecture configurations.

    First, the available datasets are preprocessed (functions: Process_SHHS_Dataset and Process_GIF_Dataset).
    Using the SleepDataManager class from dataset_processing.py, the data is saved in a uniform way. How exactly
    can be altered using the change_data_parameters argument.

    Afterwards it is split into training, validation, and test datasets and accessed using the 
    CustomSleepDataset class from neural_network_model.py. Before returning the data, this class reshapes the
    data into windows. Adjustments can be made using the window_reshape_parameters argument.

    Afterwards the neural network model is trained and tested. The accuracy results are saved in a pickle file
    and the model state dictionary is saved in a .pth file.

    The accuracy values are saved in a dictionary with the following format:
    {
        "train_accuracy": train_accuracy for each epoch (list),
        "train_avg_loss": train_avg_loss for each epoch (list),
        "test_accuracy": test_accuracy for each epoch (list),
        "test_avg_loss": test_avg_loss for each epoch (list),
        "classification_values": classification_values found in the test dataset (list),
        "true_positives": true_positives for each classifcation value (list) calculated after last epoch,
        "false_positives": false_positives for each classifcation value (list) calculated after last epoch,
        "true_negatives": true_negatives for each classifcation value (list) calculated after last epoch,
        "false_negatives": false_negatives for each classifcation value (list) calculated after last epoch
    }

    RETURNS:
    ================================================================================
    None


    ARGUMENTS:
    ================================================================================
    
    Arguments for Neural Network Section:
    -------------------------------------

    neural_network_model
        the neural network model to use
    save_file_name: str
        the name of the file to save the accuracy values and the model
    save_accuracy_directory: str
        the directory to save the accuracy values
    save_model_directory: str
        the directory to save the model state dictionary

    Arguments for Data Preprocessing Section:
    -----------------------------------------

    processed_shhs_path: str
        the path to save the processed SHHS dataset
    processed_gif_path: str
        the path to save the processed GIF dataset
    window_reshape_parameters: dict
        the parameters used when reshaping the signal to windows 
        (see reshape_signal_to_overlapping_windows function in dataset_processing.py)
    change_data_parameters: dict
        the parameters that are used to keep data uniform 
        (see SleepDataManager class in dataset_processing.py)
    
    """

    """
    ====================
    Preprocess Data
    ====================
    """

    Process_SHHS_Dataset(path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", path_to_save_processed_data = processed_shhs_path, change_data_parameters = change_data_parameters)
    # Process_GIF_Dataset(path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", path_to_save_processed_data = processed_gif_path, change_data_parameters = change_data_parameters)

    """
    -------------------------
    ACCESSING SHHS DATASETS
    -------------------------
    """

    shhs_training_data_path = processed_shhs_path[:-4] + "_training_pid.pkl"
    shhs_validation_data_path = processed_shhs_path[:-4] + "_validation_pid.pkl"
    shhs_test_data_path = processed_shhs_path[:-4] + "_test_pid.pkl"

    apply_transformation = ToTensor()

    if os.path.exists(shhs_training_data_path):
        shhs_training_data = CustomSleepDataset(path_to_data = shhs_training_data_path, transform = apply_transformation, window_reshape_parameters = window_reshape_parameters)
        shhs_validation_data = CustomSleepDataset(path_to_data = shhs_validation_data_path, transform = apply_transformation, window_reshape_parameters = window_reshape_parameters)
        shhs_test_data = CustomSleepDataset(path_to_data = shhs_test_data_path, transform = apply_transformation, window_reshape_parameters = window_reshape_parameters)

    """
    -------------------------
    ACCESSING GIF DATASETS
    -------------------------
    """
    gif_training_data_path = processed_gif_path[:-4] + "_training_pid.pkl"
    gif_validation_data_path = processed_gif_path[:-4] + "_validation_pid.pkl"
    gif_test_data_path = processed_gif_path[:-4] + "_test_pid.pkl"

    if os.path.exists(gif_training_data_path):
        gif_training_data = CustomSleepDataset(path_to_data = gif_training_data_path, transform = apply_transformation, window_reshape_parameters = window_reshape_parameters)
        gif_validation_data = CustomSleepDataset(path_to_data = gif_validation_data_path, transform = apply_transformation, window_reshape_parameters = window_reshape_parameters)
        gif_test_data = CustomSleepDataset(path_to_data = gif_test_data_path, transform = apply_transformation, window_reshape_parameters = window_reshape_parameters)
    
    """
    ====================
    Neural Network
    ====================
    """
    
    """
    -------------------------
    HYPERPARAMETERS
    -------------------------
    """
    batch_size = 8
    number_epochs = 40

    learning_rate_scheduler = CosineScheduler(
        number_updates_total = number_epochs,
        number_updates_to_max_lr = 10,
        start_learning_rate = 2.5 * 1e-5,
        max_learning_rate = 1 * 1e-4,
        end_learning_rate = 5 * 1e-5
    )

    """
    --------------------------------------------
    PREPARING DATA FOR TRAINING WITH DATALOADERS
    --------------------------------------------
    """
    if os.path.exists(shhs_training_data_path):
        shhs_train_dataloader = DataLoader(shhs_training_data, batch_size = batch_size, shuffle=True)
        shhs_validation_dataloader = DataLoader(shhs_validation_data, batch_size = batch_size, shuffle=True)
        shhs_test_dataloader = DataLoader(shhs_test_data, batch_size = batch_size, shuffle=True)
    
    if os.path.exists(gif_training_data_path):
        gif_train_dataloader = DataLoader(gif_training_data, batch_size = batch_size, shuffle=True)
        gif_validation_dataloader = DataLoader(gif_validation_data, batch_size = batch_size, shuffle=True)
        gif_test_dataloader = DataLoader(gif_test_data, batch_size = batch_size, shuffle=True)

    """
    -------------------------
    SETTING DEVICE
    -------------------------
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    -------------------------------
    INITIALIZE NEURAL NETWORK MODEL
    -------------------------------
    """
   
    neural_network_model.to(device)

    """
    -------------------------------
    LOSS AND OPTIMIZER FUNCTIONS
    -------------------------------
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer_function = optim.Adam # type: ignore

    """
    -------------------------------
    TRAINING NEURAL NETWORK
    -------------------------------
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
            dataloader = shhs_train_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            optimizer_fn = optimizer_function,
            lr_scheduler = learning_rate_scheduler,
            current_epoch = t,
            batch_size = batch_size,
        )
        train_avg_loss.append(train_results[0])
        train_accuracy.append(train_results[1])

        test_results = test_loop(
            dataloader = shhs_validation_dataloader,
            model = neural_network_model,
            device = device,
            loss_fn = loss_function,
            batch_size = batch_size
        )

        test_avg_loss.append(test_results[0])
        test_accuracy.append(test_results[1])

        classification_values = test_results[2]
        true_positives = test_results[3]
        false_positives = test_results[4]
        true_negatives = test_results[5]
        false_negatives = test_results[6]
    
    """
    -------------------------------
    SAVING ACCURACY VALUES
    -------------------------------
    """

    if not os.path.exists(save_accuracy_directory):
        os.mkdir(save_accuracy_directory)

    accuracy_values_save_path = save_accuracy_directory + "/" + save_file_name + ".pkl"

    accuracy_values = {
        "train_accuracy": train_accuracy,
        "train_avg_loss": train_avg_loss,
        "test_accuracy": test_accuracy,
        "test_avg_loss": test_avg_loss,
        "classification_values": classification_values,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives
    }

    save_to_pickle(accuracy_values, accuracy_values_save_path)

    """
    -------------------------------
    SAVING NEURAL NETWORK MODEL 
    -------------------------------
    """

    if not os.path.exists(save_model_directory):
        os.mkdir(save_model_directory)
    
    model_save_path = save_model_directory + "/" + save_file_name + ".pth"
    
    torch.save(neural_network_model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()