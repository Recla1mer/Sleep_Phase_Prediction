"""
Author: Johannes Peter Knoll

This file executes all the code needed to preprocess the data and train the neural network.
It is basically the less commented version of the notebook: "Classification_Demo.ipynb".
"""

# IMPORTS
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *

# HARD CODED SIGNAL PROCESSING PARAMETERS

# parameters that are used to keep data uniform, see SleepDataManager class in dataset_processing.py 
sleep_data_manager_parameters = {
    "RRI_frequency": 4,
    "MAD_frequency": 1,
    "SLP_frequency": 1/30,
    "sleep_stage_label": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifect": 0},
    "signal_length_seconds": 36000,
    "wanted_shift_length_seconds": 5400,
    "absolute_shift_deviation_seconds": 1800
}

window_reshape_parameters = {
    "nn_signal_duration_seconds": sleep_data_manager_parameters["signal_length_seconds"],
    "number_windows": 1197,
    "window_duration_seconds": 120,
    "overlap_seconds": 90,
    "priority_order": [3, 2, 1, 0],
    "pad_feature_with": 0,
    "pad_target_with": 0
}

dataset_class_transform_parameters = {
    "transform": ToTensor(), 
    "target_transform": None,
}


"""
============================================
Applying SleepDataManager Class To Our Data
============================================
"""

import h5py

def Process_SHHS_Dataset(
        path_to_shhs_dataset: str,
        path_to_save_processed_data: str,
        path_to_signal_processing_parameters: str,
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

    ### Parameters for change_file_information function in dataset_processing.py ###

    path_to_signal_processing_parameters: str
        the path to all signal processing parameters 
        (includes more parameters, but only the sleep_data_manager_parameters are needed here)

    ### Parameters for separate_train_test_validation function in dataset_processing.py ###

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

    # load signal processing parameters
    with open(path_to_signal_processing_parameters, "rb") as f:
        signal_processing_parameters = pickle.load(f)
    
    # access data manager parameters and update them
    change_data_parameters = {key: signal_processing_parameters[key] for key in sleep_data_manager_parameters}
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
        print("\nPreproccessing datapoints from SHHS dataset (ensuring uniformity):")
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
        path_to_signal_processing_parameters: str,
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
    
    Others: See 'Process_SHHS_Dataset' function
    """

    # following path will be created at the end of this function, if it exists, skip the processing part
    gif_training_data_path = path_to_save_processed_data[:-4] + "_training_pid.pkl"

    # initializing the database
    gif_data_manager = SleepDataManager(file_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_signal_processing_parameters, "rb") as f:
        signal_processing_parameters = pickle.load(f)
    
    # access data manager parameters and update them
    change_data_parameters = {key: signal_processing_parameters[key] for key in sleep_data_manager_parameters}
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
        print("\nPreproccessing datapoints from GIF dataset (ensuring uniformity):")
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
        path_to_model_state = None,
        path_to_processed_data = "Processed_Data/shhs_data.pkl",
        save_accuracy_values_path: str = "Model_Accuracy/Neural_Network.pkl",
        save_model_state_path: str = "Model_State/Neural_Network.pth",
        path_to_signal_processing_parameters: str = "Signal_Processing_Parameters/signal_processing_parameters.pkl",
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
    
    ### Parameters that set the paths to the data and the results ###
        
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    save_accuracy_values_path: str
        the path to save the accuracy values
    save_model_state_path: str
        the path to save the model state dictionary
    
    ### Parameters for CustomSleepDataset class in neural_network_model.py ###

    path_to_signal_processing_parameters: str
        the path to all signal processing parameters 
        (includes more parameters, but only the window_reshape_parameters are needed here)
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------
    """

    # paths to access the training, validation, and test datasets
    training_data_path = path_to_processed_data[:-4] + "_training_pid.pkl"
    validation_data_path = path_to_processed_data[:-4] + "_validation_pid.pkl"
    test_data_path = path_to_processed_data[:-4] + "_test_pid.pkl"

    # load signal processing parameters
    with open(path_to_signal_processing_parameters, "rb") as f:
        signal_processing_parameters = pickle.load(f)
    
    # access window_reshape_parameters
    CustomDatasetKeywords = {key: signal_processing_parameters[key] for key in window_reshape_parameters}

    # add transform parameters
    for key in dataset_class_transform_parameters:
        CustomDatasetKeywords[key] = signal_processing_parameters[key]

    training_data = CustomSleepDataset(path_to_data = training_data_path, **CustomDatasetKeywords)
    validation_data = CustomSleepDataset(path_to_data = validation_data_path, **CustomDatasetKeywords)
    # test_data = CustomSleepDataset(path_to_data = test_data_path, **CustomDatasetKeywords)
    
    del CustomDatasetKeywords
    
    """
    ----------------
    Hyperparameters
    ----------------
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
    ---------------------------------------------
    Preparing Data For Training With Dataloaders
    ---------------------------------------------
    """

    train_dataloader = DataLoader(training_data, batch_size = batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size = batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle=True)
    
    del training_data_path, validation_data_path, test_data_path

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
   
    if path_to_model_state is not None:
        neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
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


"""
============================
Retrieving Accuracy Results
============================
"""


def print_model_accuracy(
        paths_to_pkl_files: list,
        prediction_result_key: str,
        actual_result_keys: str,
        display_labels: list = ["Wake", "LS", "DS", "REM"],
        average = None,
        number_of_decimals = 2
    ):
    """
    This function calculates various accuracy parameters from the given pickle files (need to contain
    actual and predicted values).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_pkl_file: list
        the paths to the pickle files containing the data
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "test_predicted_results")
    actual_result_keys: str
        the key that accesses the actual results in the data (for example: "test_actual_results")
    display_labels: list
        the labels for the sleep stages
    average: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
        average parameter of the sklearn functions: precision_score, recall_score, f1_score
    number_of_decimals: int
        the number of decimals to round the results to
    """

    # variables to store results
    all_predicted_results = np.empty(0)
    all_actual_results = np.empty(0)

    for file_path in paths_to_pkl_files:
        # Load the data
        data_generator = load_from_pickle(file_path)
        data = next(data_generator)

        # Get the predicted and actual results
        predicted_results = data[prediction_result_key]
        actual_results = data[actual_result_keys]

        # Flatten the arrays
        predicted_results = predicted_results.flatten()
        actual_results = actual_results.flatten()

        # Add the results to the arrays
        all_predicted_results = np.append(all_predicted_results, predicted_results)
        all_actual_results = np.append(all_actual_results, actual_results)
    
    # Calculate the accuracy values
    accuracy = accuracy_score(all_actual_results, all_predicted_results)
    kappa = cohen_kappa_score(all_actual_results, all_predicted_results)

    precision = precision_score(all_actual_results, all_predicted_results, average = average)
    recall = recall_score(all_actual_results, all_predicted_results, average = average)
    f1 = f1_score(all_actual_results, all_predicted_results, average = average)

    # Define description of accuracy parameters
    accuracy_description = "Accuracy"
    kappa_description = "Cohen's Kappa"
    precision_description = "Precision"
    recall_description = "Recall"
    f1_description = "f1"

    # Print the results
    if average is not None:
        print()
        print(accuracy_description, round(accuracy, number_of_decimals))
        print(kappa_description, round(kappa, number_of_decimals))
        print(precision_description, round(precision, number_of_decimals)) # type: ignore
        print(recall_description, round(recall, number_of_decimals)) # type: ignore
        print(f1_description, round(f1, number_of_decimals)) # type: ignore
    else:
        # check if the display_labels have the right length
        unique_labels_actual = np.unique(all_actual_results)
        unique_labels_predicted = np.unique(all_predicted_results)
        unique_labels = np.unique(np.concatenate((unique_labels_actual, unique_labels_predicted)))
        if len(display_labels) != len(precision): # type: ignore
            display_labels = np.sort(unique_labels) # type: ignore
            display_labels = np.array(display_labels, dtype = str) # type: ignore
            print("\nGiven display labels do not match the number of labels in the data. Display labels are set to the unique labels in the data.")
        else:
            print("\nThe display labels correspond to the following labels in the data:")
            for i in range(len(display_labels)):
                print(f"{display_labels[i]}: {unique_labels[i]}")

        # Round the results
        precision = np.round(precision, number_of_decimals)
        recall = np.round(recall, number_of_decimals)
        f1 = np.round(f1, number_of_decimals)

        # Calculate column width
        longest_precision_value = max([len(str(value)) for value in precision])
        longest_recall_value = max([len(str(value)) for value in recall])
        longest_f1_value = max([len(str(value)) for value in f1])

        column_width = max([len(label) for label in display_labels])
        column_width = max([column_width, longest_precision_value, longest_recall_value, longest_f1_value])
        column_width += 2

        first_column_width = max([len(precision_description), len(recall_description), len(f1_description)]) + 1

        # Print the results
        print()
        print(accuracy_description, round(accuracy, number_of_decimals))
        print(kappa_description, round(kappa, number_of_decimals))
        print()
        
        # Print the header
        print(" "*first_column_width, end = "")
        for label in display_labels:
            print(f"|{label:^{column_width}}", end = "")
        print()
        print("-"*(first_column_width + len(display_labels)*(column_width + 1)))

        # Print the results
        print(f"{precision_description:<{first_column_width}}", end = "")
        for value in precision:
            print(f"|{value:^{column_width}}", end = "")
        print()
        print(f"{recall_description:<{first_column_width}}", end = "")
        for value in recall:
            print(f"|{value:^{column_width}}", end = "")
        print()
        print(f"{f1_description:<{first_column_width}}", end = "")
        for value in f1:
            print(f"|{value:^{column_width}}", end = "")
        print()


        
# print_model_accuracy(
#     paths_to_pkl_files = ["Model_Accuracy/NN_SHHS.pkl"],
#     prediction_result_key = "test_predicted_results",
#     actual_result_keys = "test_actual_results",
#     display_labels = ["Wake", "LS", "DS", "REM"],
#     average = None,
#     number_of_decimals = 3
# )

        


"""
======================================
Applying Trained Neural Network Model
======================================
"""


def predict(dataloader, model, device):
    """
    Iterate over a given dataset to predict sleep stages

    RETURNS:
    ------------------------------
    test_loss : float
        Average loss value of the test dataset
    correct : float
        Ratio of correctly predicted values of the test dataset
    predicted_results : list
        Predicted sleep stages
    actual_results : list
        Actual sleep stages


    ARGUMENTS:
    ------------------------------
    dataloader : DataLoader
        DataLoader object containing the dataset
    model : nn.Module
        Neural Network model to use
    device : str
        Device to run the model on
    """

    # get number of windows the signals are reshaped to
    windows_per_signal = model.windows_per_signal

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    # variables to track progress
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_number_predictions = 0
    start_time = time.time()
    print("\nPredicting Sleep Stages:")
    progress_bar(0, size, 1, start_time, None, None)

    # variables to save results
    predicted_results = np.empty((0, windows_per_signal))
    actual_results = np.empty((0, windows_per_signal))

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        # Iterate over the dataset
        for batch, (rri, mad, slp) in enumerate(dataloader):
            # check if MAD signal was not provided
            if mad[0] == "None":
                mad = None
            else:
                mad = mad.to(device)

            # Send data to device
            rri = rri.to(device)

            # Compute prediction
            pred = model(rri, mad)
            slp = slp.long()

            # collect results if requested
            if collect_results:
                this_predicted_results_reshaped = pred.argmax(1).view(int(slp.shape[0]/windows_per_signal), windows_per_signal).cpu().numpy()
                this_actual_results_reshaped = slp.view(int(slp.shape[0]/windows_per_signal), windows_per_signal).cpu().numpy()
                
                predicted_results = np.append(predicted_results, this_predicted_results_reshaped, axis=0)
                actual_results = np.append(actual_results, this_actual_results_reshaped, axis=0)

            # print progress bar
            datapoints_done = (batch+1) * batch_size
            if datapoints_done > size:
                datapoints_done = size
            progress_bar(datapoints_done, size, batch_size, start_time, None, None)

    test_loss /= num_batches
    correct /= total_number_predictions
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct, predicted_results, actual_results):


def main_model_predicting(
        neural_network_model = SleepStageModel(),
        path_to_model_state: str = "Model_State/Neural_Network.pth",
        path_to_processed_data = "Processed_Data/shhs_data.pkl",
        save_accuracy_values_path: str = "Model_Accuracy/Neural_Network.pkl",
        save_model_state_path: str = "Model_State/Neural_Network.pth",
        path_to_signal_processing_parameters: str = "Signal_Processing_Parameters/signal_processing_parameters.pkl",
    ):
    """
    

    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_model
        the neural network model to use
    
    ### Parameters that set the paths to the data and the results ###
        
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_processed_data: str
        the path to the processed dataset 
        (must be designed so that adding: '_training_pid.pkl', '_validation_pid.pkl', '_test_pid.pkl' 
        [after removing '.pkl'] accesses the training, validation, and test datasets)
    save_accuracy_values_path: str
        the path to save the accuracy values
    save_model_state_path: str
        the path to save the model state dictionary
    
    ### Parameters for CustomSleepDataset class in neural_network_model.py ###

    path_to_signal_processing_parameters: str
        the path to all signal processing parameters 
        (includes more parameters, but only the window_reshape_parameters are needed here)
    """
    
    """
    ------------------
    Accessing Dataset
    ------------------

    During Training, reshaping the signals to overlapping windows is done using the CustomSleepDataset class,
    which uses the SleepDataManager class from dataset_processing.py to access the data.
    Using this here would be possible as well, but not as clear because we want to add the predicted sleep
    stages directly to the database. Making the code more complex than necessary. Therefore, we will use the
    SleepDataManager class directly to access the data and reshape the signals to overlapping windows.
    """

    # accessing database
    data_manager = SleepDataManager(file_path = path_to_processed_data)

    """
    ---------------------------------------
    Accessing Signal Processing Parameters
    ---------------------------------------
    """

    # load signal processing parameters
    with open(path_to_signal_processing_parameters, "rb") as f:
        signal_processing_parameters = pickle.load(f)
    
    # access window_reshape_parameters

    """
    ---------------------------------------------
    Preparing Data For Training With Dataloaders
    ---------------------------------------------
    """

    dataloader = DataLoader(training_data, batch_size = 1, shuffle=False)

    """
    ---------------
    Setting Device
    ---------------
    """

    # Neural network model is unable to function properly on mps device, option to use it is removed
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
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    # variables to store results
    actual_results = np.empty(0)
    predicted_results = np.empty(0)

    test_accuracy = []
    test_avg_loss = []

    
    """
    -----------------------
    Saving Results
    -----------------------
    """

    create_directories_along_path(path_to_save_results)

    accuracy_values = {
        "actual_results": actual_results,
        "predicted_results": predicted_results
    }

    save_to_pickle(accuracy_values, path_to_save_results)


if __name__ == "__main__":

    """
    ----------
    Settings:
    ----------
    """

    # set window reshape parameters
    window_reshape_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0,
        "number_windows": 1197,
        "window_duration_seconds": 120,
        "overlap_seconds": 90,
        "priority_order": [3, 2, 1, 0]
    }

    # set train, validation, and test sizes
    split_data_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True
    }

    original_shhs_data_path = "Raw_Data/SHHS_dataset.h5"
    original_gif_data_path = "Raw_Data/GIF_dataset.h5"

    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    processed_gif_path = "Processed_Data/gif_data.pkl"

    save_accuracy_values_path = "Model_Accuracy/SSM"
    save_model_state_path = "Model_State/SSM"

    """
    ------------------------
    Preprocessing SHHS Data
    ------------------------
    """

    # Process_SHHS_Dataset(
    #     computation_mode = computation_mode,
    #     path_to_shhs_dataset = original_shhs_data_path, 
    #     path_to_save_processed_data = processed_shhs_path,
    #     change_data_parameters = {},
    #     ** split_data_parameters
    #     )
    
    """
    ------------------------------
    Training Network on SHHS Data
    ------------------------------
    """

    # main_model_training(
    #     computation_mode = computation_mode,
    #     neural_network_model = SleepStageModel(), # type: ignore
    #     load_model_state_path = None,
    #     processed_path = processed_shhs_path,
    #     save_accuracy_values_path = save_accuracy_values_path + "_SHHS.pkl",
    #     save_model_state_path = save_model_state_path + "_SHHS.pth",
    #     ** window_reshape_parameters,
    #     )
    
    """
    -----------------------
    Preprocessing GIF Data
    -----------------------
    """

    # Process_GIF_Dataset(
    #     computation_mode = computation_mode,
    #     path_to_gif_dataset = original_gif_data_path, 
    #     path_to_save_processed_data = processed_gif_path,
    #     change_data_parameters = {},
    #     ** split_data_parameters
    #     )

    """
    -----------------------------
    Training Network on GIF Data
    -----------------------------
    """

    # main_model_training(
    #     computation_mode = computation_mode,
    #     neural_network_model = SleepStageModel(), # type: ignore
    #     load_model_state_path = save_model_state_path + "_SHHS.pth",
    #     processed_path = processed_gif_path,
    #     save_accuracy_values_path = save_accuracy_values_path + "_SHHS_GIF.pkl",
    #     save_model_state_path = save_model_state_path + "_SHHS_GIF.pth",
    #     ** window_reshape_parameters,
    #     )

    """
    ---------------------------------------------------------------------
    Testing Original Idea: Overlapping Windows and artifect = wake stage
    ---------------------------------------------------------------------
    """

    # Parameters
    save_accuracy_values_path_ssm = "Model_Accuracy/SSM"
    save_model_state_path_ssm = "Model_State/SSM"

    save_accuracy_values_path_yao = "Model_Accuracy/Yao"
    save_model_state_path_yao = "Model_State/Yao"

    name_addition = "_Original"

    # Preprocess SHHS Data
    Process_SHHS_Dataset(
        path_to_shhs_dataset = original_shhs_data_path, 
        path_to_save_processed_data = processed_shhs_path,
        change_data_parameters = {},
        ** split_data_parameters
        )

    # Train and test different models on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = save_accuracy_values_path_ssm + name_addition + "_SHHS.pkl",
        save_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS.pth",
        ** window_reshape_parameters,
        )
    
    main_model_training(
        neural_network_model = YaoModel(), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = save_accuracy_values_path_yao + name_addition + "_SHHS.pkl",
        save_model_state_path = save_model_state_path_yao + name_addition + "_SHHS.pth",
        ** window_reshape_parameters,
        )
    
    # Preprocess GIF Data
    Process_GIF_Dataset(
        path_to_gif_dataset = original_gif_data_path, 
        path_to_save_processed_data = processed_gif_path,
        change_data_parameters = {},
        ** split_data_parameters
        )

    # Train and test different models on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(),
        load_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = save_accuracy_values_path_ssm + name_addition + "_SHHS_GIF.pkl",
        save_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS_GIF.pth",
        ** window_reshape_parameters,
        )
    
    main_model_training(
        neural_network_model = YaoModel(), # type: ignore
        load_model_state_path = save_model_state_path_yao + name_addition + "_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = save_accuracy_values_path_yao + name_addition + "_SHHS_GIF.pkl",
        save_model_state_path = save_model_state_path_yao + name_addition + "_SHHS_GIF.pth",
        ** window_reshape_parameters,
        )

    """
    -------------------------------------------------------------------
    Testing with Overlapping windows but artifect being a unique stage
    -------------------------------------------------------------------
    """

    # Parameters
    processed_shhs_path = "Processed_Data/shhs_data_artifect.pkl"
    processed_gif_path = "Processed_Data/gif_data_artifect.pkl"

    name_addition = "_Artifect"

    change_data_parameters = {"sleep_stage_label": {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifect": 0}}

    window_reshape_parameters["priority_order"] = [4, 3, 2, 1, 0]
    
    # Preprocess SHHS Data
    Process_SHHS_Dataset(
        path_to_shhs_dataset = original_shhs_data_path, 
        path_to_save_processed_data = processed_shhs_path,
        change_data_parameters = change_data_parameters,
        ** split_data_parameters
        )

    # Train and test different models on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(number_sleep_stages = 5), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = save_accuracy_values_path_ssm + name_addition + "_SHHS.pkl",
        save_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS.pth",
        ** window_reshape_parameters,
        )
    
    main_model_training(
        neural_network_model = YaoModel(number_sleep_stages = 5), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = save_accuracy_values_path_yao + name_addition + "_SHHS.pkl",
        save_model_state_path = save_model_state_path_yao + name_addition + "_SHHS.pth",
        ** window_reshape_parameters,
        )
    
    # Preprocess GIF Data
    Process_GIF_Dataset(
        path_to_gif_dataset = original_gif_data_path, 
        path_to_save_processed_data = processed_gif_path,
        change_data_parameters = change_data_parameters,
        ** split_data_parameters
        )

    # Train and test different models on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(number_sleep_stages = 5),
        load_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = save_accuracy_values_path_ssm + name_addition + "_SHHS_GIF.pkl",
        save_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS_GIF.pth",
        ** window_reshape_parameters,
        )
    
    main_model_training(
        neural_network_model = YaoModel(number_sleep_stages = 5), # type: ignore
        load_model_state_path = save_model_state_path_yao + name_addition + "_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = save_accuracy_values_path_yao + name_addition + "_SHHS_GIF.pkl",
        save_model_state_path = save_model_state_path_yao + name_addition + "_SHHS_GIF.pth",
        ** window_reshape_parameters,
        )
    
    window_reshape_parameters["priority_order"] = [3, 2, 1, 0]

    """
    ---------------------------------------------------------------
    Testing with non-overlapping windows and artifect = wake stage
    ---------------------------------------------------------------
    """

    # Parameters
    processed_shhs_path = "Processed_Data/shhs_data_no_overlap.pkl"
    processed_gif_path = "Processed_Data/gif_data_no_overlap.pkl"

    name_addition = "_no_overlap"

    window_reshape_parameters["overlap_seconds"] = 0
    window_reshape_parameters["number_windows"] = 300

    # Preprocess SHHS Data
    Process_SHHS_Dataset(
        path_to_shhs_dataset = original_shhs_data_path, 
        path_to_save_processed_data = processed_shhs_path,
        change_data_parameters = {},
        ** split_data_parameters
        )

    # Train and test different models on SHHS Data
    main_model_training(
        neural_network_model = SleepStageModel(windows_per_signal = window_reshape_parameters["number_windows"]), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = save_accuracy_values_path_ssm + name_addition + "_SHHS.pkl",
        save_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS.pth",
        ** window_reshape_parameters,
        )
    
    main_model_training(
        neural_network_model = YaoModel(windows_per_signal = window_reshape_parameters["number_windows"]), # type: ignore
        load_model_state_path = None,
        processed_path = processed_shhs_path,
        save_accuracy_values_path = save_accuracy_values_path_yao + name_addition + "_SHHS.pkl",
        save_model_state_path = save_model_state_path_yao + name_addition + "_SHHS.pth",
        ** window_reshape_parameters,
        )
    
    # Preprocess GIF Data
    Process_GIF_Dataset(
        path_to_gif_dataset = original_gif_data_path, 
        path_to_save_processed_data = processed_gif_path,
        change_data_parameters = {},
        ** split_data_parameters
        )

    # Train and test different models on GIF Data
    main_model_training(
        neural_network_model = SleepStageModel(windows_per_signal = window_reshape_parameters["number_windows"]),
        load_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = save_accuracy_values_path_ssm + name_addition + "_SHHS_GIF.pkl",
        save_model_state_path = save_model_state_path_ssm + name_addition + "_SHHS_GIF.pth",
        ** window_reshape_parameters,
        )
    
    main_model_training(
        neural_network_model = YaoModel(windows_per_signal = window_reshape_parameters["number_windows"]), # type: ignore
        load_model_state_path = save_model_state_path_yao + name_addition + "_SHHS.pth",
        processed_path = processed_gif_path,
        save_accuracy_values_path = save_accuracy_values_path_yao + name_addition + "_SHHS_GIF.pkl",
        save_model_state_path = save_model_state_path_yao + name_addition + "_SHHS_GIF.pth",
        ** window_reshape_parameters,
        )

# IDEAS: max conv channels for mad