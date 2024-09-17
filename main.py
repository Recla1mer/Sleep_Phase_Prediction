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
        train_size = 0.8, 
        validation_size = 0.1, 
        test_size = 0.1, 
        random_state = None, 
        shuffle = True
    ):
    """
    This function processes our SHHS dataset. It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the above code to process a dataset.

    If already processed, the function will only shuffle the data in the pids again.
    """

    # following path will be created at the end of this function, if it exists, skip the processing part
    shhs_training_data_path = path_to_save_processed_data[:-4] + "_training_pid.pkl"

    # initializing the database
    shhs_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

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
        train_size = 0.8, 
        validation_size = 0.1, 
        test_size = 0.1, 
        random_state = None, 
        shuffle = True
    ):
    """
    This function processes our GIF dataset. It is designed to be a more specific. So, if you are not using
    the same data as we are, you need to write a similar function for your dataset. Nonetheless, this
    quickly demonstrates how to quickly use the above code to process a dataset.

    If already processed, the function will only shuffle the data in the pids again.
    """

    # following path will be created at the end of this function, if it exists, skip the processing part
    gif_training_data_path = path_to_save_processed_data[:-4] + "_training_pid.pkl"

    # initializing the database
    gif_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

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


if __name__ == "__main__":

    """
    ====================
    Preprocess Data
    ====================
    """
    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    processed_gif_path = "Processed_Data/gif_data.pkl"

    # Process_SHHS_Dataset(path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", path_to_save_processed_data = processed_shhs_path)
    # Process_GIF_Dataset(path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", path_to_save_processed_data = processed_gif_path)


    """
    ====================
    Neural Network
    ====================
    """

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
        shhs_training_data = CustomSleepDataset(path_to_data = shhs_training_data_path, transform = apply_transformation)
        shhs_validation_data = CustomSleepDataset(path_to_data = shhs_validation_data_path, transform = apply_transformation)
        shhs_test_data = CustomSleepDataset(path_to_data = shhs_test_data_path, transform = apply_transformation)

    """
    -------------------------
    ACCESSING GIF DATASETS
    -------------------------
    """
    gif_training_data_path = processed_gif_path[:-4] + "_training_pid.pkl"
    gif_validation_data_path = processed_gif_path[:-4] + "_validation_pid.pkl"
    gif_test_data_path = processed_gif_path[:-4] + "_test_pid.pkl"

    if os.path.exists(gif_training_data_path):
        gif_training_data = CustomSleepDataset(path_to_data = gif_training_data_path, transform = apply_transformation)
        gif_validation_data = CustomSleepDataset(path_to_data = gif_validation_data_path, transform = apply_transformation)
        gif_test_data = CustomSleepDataset(path_to_data = gif_test_data_path, transform = apply_transformation)
    
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
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    """
    -------------------------------
    INITIALIZE NEURAL NETWORK MODEL
    -------------------------------
    """
    # nn_model = SleepStageModel()
    nn_model = YaoModel()
    nn_model.to(device)

    """
    -------------------------------
    LOSS AND OPTIMIZER FUNCTIONS
    -------------------------------
    """
    loss_function = nn.CrossEntropyLoss()
    optimizer_function = optim.Adam

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
            model = nn_model,
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
            model = nn_model,
            device = device,
            loss_fn = loss_function,
        )
        test_avg_loss.append(test_results[0])
        test_accuracy.append(test_results[1])