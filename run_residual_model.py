"""
Author: Johannes Peter Knoll

This file is unnecessary for the user. I used this to test different data processing configurations and 
neural network models.
"""

# IMPORTS
from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
import random
import shutil
import os

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *
from plot_helper import *
from main import check_project_configuration, model_state_after_shhs_file, model_state_after_shhs_gif_file, loss_per_epoch_shhs_file, loss_per_epoch_gif_file, project_configuration_file

default_complete_shhs_SSG_path = "Default_SHHS_SSG_Data_All/"
default_complete_gif_SSG_path = "Default_GIF_SSG_Data_All/"

default_reduced_shhs_SSG_path = "Default_SHHS_SSG_Data_Reduced/"
default_reduced_gif_SSG_path = "Default_GIF_SSG_Data_Reduced/"

default_complete_gif_SAE_path = "Default_GIF_SAE_Data_All/"
default_reduced_gif_SAE_path = "Default_GIF_SAE_Data_Reduced/"

gif_error_code_1 = ["SL007", "SL010", "SL012", "SL014", "SL022", "SL026", "SL039", "SL044", "SL049", "SL064", "SL070", "SL146", "SL150", "SL261", "SL266", "SL296", "SL303", "SL306", "SL342", "SL350", "SL410", "SL411", "SL416"]
gif_error_code_2 = ["SL032", "SL037", "SL079", "SL088", "SL114", "SL186", "SL255", "SL328", "SL336", "SL341", "SL344", "SL424"]
gif_error_code_3 = ["SL001", "SL004", "SL011", "SL025", "SL027", "SL034", "SL055", "SL057", "SL073", "SL075", "SL076", "SL083", "SL085", "SL087", "SL089", "SL096", "SL111", "SL116", "SL126", "SL132", "SL138", "SL141", "SL151", "SL157", "SL159", "SL166", "SL173", "SL174", "SL176", "SL178", "SL179", "SL203", "SL207", "SL208", "SL210", "SL211", "SL214", "SL217", "SL218", "SL221", "SL228", "SL229", "SL236", "SL237", "SL240", "SL245", "SL250", "SL252", "SL269", "SL286", "SL293", "SL294", "SL315", "SL348", "SL382", "SL384", "SL386", "SL389", "SL397", "SL406", "SL408", "SL418", "SL422", "SL428"]
gif_error_code_4 = ["SL061", "SL066", "SL091", "SL105", "SL202", "SL204", "SL205", "SL216", "SL305", "SL333", "SL349", "SL430", "SL439", "SL440"]
gif_error_code_5 = ["SL016", "SL040", "SL145", "SL199", "SL246", "SL268", "SL290", "SL316", "SL332", "SL365", "SL392", "SL426", "SL433", "SL438"]

"""
===============
Model Training
===============
"""


# TRAINING LOOP
def train_loop(dataloader, model, device, loss_fn, optimizer_fn, lr_scheduler, current_epoch, batch_size, number_classes):
    """
    Iterate over the training dataset and try to converge to optimal parameters.

    Source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    RETURNS:
    ------------------------------
    train_loss : float
        Average loss value of the training dataset
    correct : float
        Ratio of correctly predicted values of the training dataset
    predicted_results : list
        Predicted sleep stages
    actual_results : list
        Actual sleep stages

    ARGUMENTS:
    ------------------------------
    dataloader : DataLoader
        DataLoader object containing the training dataset
    model : nn.Module
        Neural Network model to train
    device : str
        Device to train the model on
    loss_fn : nn.Module
        Loss function to be minimized
    optimizer_fn : torch.optim
        Optimizer to update the model parameters
    lr_scheduler :
        Scheduler for the learning rate
    current_epoch : int
        Current epoch number
    batch_size : int
        Number of samples in each batch
    """

    # set optimizer
    optimizer = optimizer_fn(model.parameters(), lr=lr_scheduler(current_epoch))

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()

    # variables to save accuracy progress
    train_loss = 0
    train_confusion_matrix = np.zeros((number_classes, number_classes))

    # variables to track progress
    num_batches = len(dataloader)
    print("\nTraining Neural Network Model:")
    progress_bar = DynamicProgressBar(total = len(dataloader.dataset), batch_size = batch_size)

    # Iterate over the training dataset
    for batch, (rri, mad, slp) in enumerate(dataloader):

        # check if MAD signal was not provided
        if mad[0] == "None":
            mad = None
        else:
            mad = mad.to(device)
        
        # Send data to device
        rri, slp = rri.to(device), slp.to(device)

        # reshape slp to fit the model output
        slp = slp.view(-1) # Combine batch and windows dimensions

        # Compute prediction and loss
        pred = model(rri, mad)
        slp = slp.long()
        loss = loss_fn(pred, slp)

        # Backpropagation
        loss.backward()
        optimizer.step() # updates the model parameters based on the gradients computed during the backward pass
        optimizer.zero_grad()

        # update confusion matrix
        pred = pred.argmax(1).cpu().numpy()
        slp = slp.cpu().numpy()
        for i in range(len(slp)):
            train_confusion_matrix[slp[i], pred[i]] += 1

        train_loss += loss.item()

        # print progress bar
        accuracy = train_confusion_matrix.diagonal().sum() / train_confusion_matrix.sum()
        progress_bar.update(
            additional_info = f'Loss: {format_float(loss.item(), 3)} | Acc: {(100*accuracy):>0.1f}%',
            )
    
    train_loss /= num_batches

    return train_loss, train_confusion_matrix


# TESTING LOOP
def test_loop(dataloader, model, device, loss_fn, batch_size, number_classes):
    """
    Iterate over the test dataset to check if model performance is improving

    Source: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

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
        DataLoader object containing the test dataset
    model : nn.Module
        Neural Network model to test
    device : str
        Device to test the model on
    loss_fn : nn.Module
        Loss function to be minimized
    batch_size : int
        Number of samples in each batch
    collect_results : bool
        If True, predicted and actual results are collected
    """

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()

    # variables to track progress
    num_batches = len(dataloader)
    print("\nCalculating Prediction Accuracy on Test Data:")
    progress_bar = DynamicProgressBar(total = len(dataloader.dataset), batch_size = batch_size)

    # variables to save performance progress
    test_loss = 0
    test_confusion_matrix = np.zeros((number_classes, number_classes))

    # test_target_true = np.array([], dtype=int)
    # test_target_pred = np.array([], dtype=int)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        # Iterate over the test dataset
        for batch, (rri, mad, slp) in enumerate(dataloader):
            # check if MAD signal was not provided
            if mad[0] == "None":
                mad = None
            else:
                mad = mad.to(device)

            # Send data to device
            rri, slp = rri.to(device), slp.to(device)

            # reshape slp to fit the model output
            slp = slp.view(-1) # Combine batch and windows dimensions

            # Compute prediction and loss
            pred = model(rri, mad)
            slp = slp.long()
            test_loss += loss_fn(pred, slp).item()

            # update confusion matrix
            pred = pred.argmax(1).cpu().numpy()
            slp = slp.cpu().numpy()

            # test_target_true = np.concatenate((test_target_true, slp))
            # test_target_pred = np.concatenate((test_target_pred, pred))

            for i in range(len(slp)):
                test_confusion_matrix[slp[i], pred[i]] += 1

            # print progress bar
            progress_bar.update()

    test_loss /= num_batches
    accuracy = test_confusion_matrix.diagonal().sum() / test_confusion_matrix.sum()

    print(f"\nTest Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")

    # return test_loss, test_confusion_matrix, test_target_true, test_target_pred
    return test_loss, test_confusion_matrix


def main_model_training(
        neural_network_hyperparameters: dict,
        path_to_training_data_directory: str,
        path_to_project_configuration: str,
        path_to_model_state,
        path_to_updated_model_state: str,
        paths_to_validation_data_directories: list,
        path_to_loss_per_epoch: str,
    ):
    """
    Full implementation of project, with ability to easily change most important parameters to test different
    neural network architecture configurations. Some Parameters are hardcoded by design.

    The Data is accessed using the CustomSleepDataset class from neural_network_model.py. Data preprocessing
    adjustments performed through this class can be made using the parameters this function accesses from
    "path_to_project_configuration".

    Afterwards the neural network model is trained and tested. The accuracy and loss are saved in a pickle file
    for every epoch. The final model state dictionary is saved in a .pth file.

    The performance values are saved in a dictionary with the following format:
    {
        "train_accuracy": train_accuracy for each epoch (list),
        "train_avg_loss": train_avg_loss for each epoch (list),
        "{validation_file_name_without_extension}_accuracy": accuracy for each epoch (list) (multiple entries like this for each file in paths_to_processed_validation_data),
        "{validation_file_name_without_extension}_avg_loss": average loss for each epoch (list) (multiple entries like this for each file in paths_to_processed_validation_data),
    }

    
    RETURNS:
    ------------------------------
    None

    
    ARGUMENTS:
    ------------------------------
    neural_network_hyperparameters: dict
        the hyperparameters for the neural network model training
        (batch_size, number_epochs, lr_scheduler_parameters)
    path_to_processed_training_data: str
        the path to the processed dataset containing the training data
    path_to_project_configuration: str
        the path to all signal processing parameters 
        (not all are needed here)
    path_to_model_state: str
        the path to load the model state dictionary
        if None, the model will be trained from scratch
    path_to_updated_model_state: str
        the path to save the model state dictionary
    paths_to_processed_validation_data: list (of str)
        list of paths to the processed datasets containing the validation data (might be multiple)
    path_to_loss_per_epoch: str
        the path to save the accuracy values
    """

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters
    number_classes = project_configuration["number_target_classes"]

    # retrieve dictionary needed to map sleep stage labels
    training_data_manager = BigDataManager(directory_path = path_to_training_data_directory, pid="Train")
    current_target_classes = training_data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    # add window_reshape_parameters
    common_window_reshape_parameters = dict()
    if project_configuration["reshape_to_overlapping_windows"]:
        for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]:
            common_window_reshape_parameters[key] = project_configuration[key]

    # add signal_normalization_parameters
    common_signal_normalization_parameters = dict()
    if project_configuration["normalize_rri"] or project_configuration["normalize_mad"]:
        common_signal_normalization_parameters = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters
    
    """
    ----------------
    Hyperparameters
    ----------------
    """

    number_epochs = neural_network_hyperparameters["number_epochs"]

    learning_rate_scheduler = CosineScheduler(
        number_updates_total = number_epochs,
        **neural_network_hyperparameters["lr_scheduler_parameters"]
    )

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

    neural_network_model = neural_network_model(**nnm_params)
   
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
    # clearing sequence to remove progress bars of previous epoch
    clearing_sequence = "\033[2K"
    for _ in range(7+6*len(paths_to_validation_data_directories)):
        clearing_sequence += "\033[F" # Move cursor up
        clearing_sequence += "\033[2K" # Clear line

    # variables to store accuracy progress
    train_avg_loss = list()
    train_confusion_matrices = list()

    test_avg_loss = [[] for _ in range(len(paths_to_validation_data_directories))]
    test_confusion_matrices = [[] for _ in range(len(paths_to_validation_data_directories))]

    rri_steps = project_configuration["signal_length_seconds"] * project_configuration["rri_frequency"]
    mad_steps = project_configuration["signal_length_seconds"] * project_configuration["mad_frequency"]
    slp_steps = project_configuration["signal_length_seconds"] * project_configuration["slp_frequency"]

    for t in range(1, number_epochs+1):
        # clearing previous epoch progress bars
        if t > 1:
            print(clearing_sequence, end='')

        print("")
        print("-"*10)
        print(f"Epoch {t}:")
        print("-"*10)

        # set optimizer
        optimizer = optimizer_function(neural_network_model.parameters(), lr=learning_rate_scheduler(t)) # type: ignore

        # Set the model to training mode - important for batch normalization and dropout layers
        neural_network_model.train()

        # variables to save accuracy progress
        train_loss = 0
        train_confusion_matrix = np.zeros((number_classes, number_classes))

        # variables to track progress
        print("\nTraining Neural Network Model:")
        progress_bar = DynamicProgressBar(total = len(training_data_manager), batch_size = 1)

        for data_sample in training_data_manager:

            total_length = len(data_sample["RRI"]) # type: ignore

            start = 0
            end = 1
            while True:
                if end * rri_steps > total_length:
                    break

                # extract feature (RRI) from dictionary and perform final preprocessing:
                rri = final_data_preprocessing(
                    signal = data_sample["RRI"][start * rri_steps: end * rri_steps], # type: ignore
                    signal_id = "RRI",
                    inlier_interval = project_configuration["rri_inlier_interval"],
                    target_frequency = project_configuration["rri_frequency"],
                    signal_length_seconds = project_configuration["signal_length_seconds"],
                    pad_with = project_configuration["pad_feature_with"],
                    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"],
                    **common_window_reshape_parameters,
                    normalize = project_configuration["normalize_rri"],
                    **common_signal_normalization_parameters,
                    datatype_mappings = [(np.float64, np.float32)],
                    transform = project_configuration["feature_transform"]
                )

                if "MAD" in data_sample: # type: ignore
                    # extract feature (MAD) from dictionary and perform final preprocessing:
                    mad = final_data_preprocessing(
                        signal = data_sample["MAD"][start * mad_steps: end * mad_steps], # type: ignore
                        signal_id = "MAD",
                        inlier_interval = project_configuration["mad_inlier_interval"],
                        target_frequency = project_configuration["mad_frequency"],
                        signal_length_seconds = project_configuration["signal_length_seconds"],
                        pad_with = project_configuration["pad_feature_with"],
                        reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"],
                        **common_window_reshape_parameters,
                        normalize = project_configuration["normalize_mad"],
                        **common_signal_normalization_parameters,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = project_configuration["feature_transform"],
                    )
                    mad = mad.to(device)
                
                else:
                    mad = None

                # extract labels from dictionary and perform final preprocessing:
                slp = final_data_preprocessing(
                    signal = data_sample["SLP"][start * slp_steps: end * slp_steps], # type: ignore
                    signal_id = "SLP",
                    slp_label_mapping = slp_label_mapping,
                    target_frequency = project_configuration["slp_frequency"],
                    signal_length_seconds = project_configuration["signal_length_seconds"],
                    pad_with = project_configuration["pad_target_with"],
                    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"],
                    **common_window_reshape_parameters,
                    normalize = False,  # SLP labels should not be normalized
                    datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                    transform = project_configuration["target_transform"]
                )
                
                # Send data to device
                rri, slp = rri.to(device), slp.to(device)

                # reshape slp to fit the model output
                slp = slp.view(-1) # Combine batch and windows dimensions

                # Compute prediction and loss
                pred = neural_network_model(rri, mad)
                slp = slp.long()
                loss = loss_function(pred, slp)

                # Backpropagation
                loss.backward()
                optimizer.step() # updates the model parameters based on the gradients computed during the backward pass
                optimizer.zero_grad()

                # update confusion matrix
                pred = pred.argmax(1).cpu().numpy()
                slp = slp.cpu().numpy()
                for i in range(len(slp)):
                    train_confusion_matrix[slp[i], pred[i]] += 1

                train_loss += loss.item()

                # print progress bar
                accuracy = train_confusion_matrix.diagonal().sum() / train_confusion_matrix.sum()
                progress_bar.update(
                    additional_info = f'Loss: {format_float(loss.item(), 3)} | Acc: {(100*accuracy):>0.1f}%',
                    )
                
                start += 1
                end += 1
            
            train_loss /= len(training_data_manager)
            neural_network_model.reset_residual_features() # reset residual features for next data sample

        train_avg_loss.append(train_loss)
        train_confusion_matrices.append(train_confusion_matrix)

        # Set the model to evaluation mode - important for batch normalization and dropout layers
        neural_network_model.eval()

        for path in paths_to_validation_data_directories:
            validation_data_manager = BigDataManager(directory_path = path, pid="Validation")

            # variables to track progress
            print("\nCalculating Prediction Accuracy on Test Data:")
            progress_bar = DynamicProgressBar(total = len(validation_data_manager), batch_size = 1)

            # variables to save performance progress
            test_loss = 0
            test_confusion_matrix = np.zeros((number_classes, number_classes))

            for data_sample in validation_data_manager:

                total_length = len(data_sample["RRI"]) # type: ignore

                start = 0
                end = 1
                while True:
                    if end * rri_steps > total_length:
                        break

                    # extract feature (RRI) from dictionary and perform final preprocessing:
                    rri = final_data_preprocessing(
                        signal = data_sample["RRI"][start * rri_steps: end * rri_steps], # type: ignore
                        signal_id = "RRI",
                        inlier_interval = project_configuration["rri_inlier_interval"],
                        target_frequency = project_configuration["rri_frequency"],
                        signal_length_seconds = project_configuration["signal_length_seconds"],
                        pad_with = project_configuration["pad_feature_with"],
                        reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"],
                        **common_window_reshape_parameters,
                        normalize = project_configuration["normalize_rri"],
                        **common_signal_normalization_parameters,
                        datatype_mappings = [(np.float64, np.float32)],
                        transform = project_configuration["feature_transform"]
                    )

                    if "MAD" in data_sample: # type: ignore
                        # extract feature (MAD) from dictionary and perform final preprocessing:
                        mad = final_data_preprocessing(
                            signal = data_sample["MAD"][start * mad_steps: end * mad_steps], # type: ignore
                            signal_id = "MAD",
                            inlier_interval = project_configuration["mad_inlier_interval"],
                            target_frequency = project_configuration["mad_frequency"],
                            signal_length_seconds = project_configuration["signal_length_seconds"],
                            pad_with = project_configuration["pad_feature_with"],
                            reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"],
                            **common_window_reshape_parameters,
                            normalize = project_configuration["normalize_mad"],
                            **common_signal_normalization_parameters,
                            datatype_mappings = [(np.float64, np.float32)],
                            transform = project_configuration["feature_transform"],
                        )
                        mad = mad.to(device)
                    
                    else:
                        mad = None

                    # extract labels from dictionary and perform final preprocessing:
                    slp = final_data_preprocessing(
                        signal = data_sample["SLP"][start * slp_steps: end * slp_steps], # type: ignore
                        signal_id = "SLP",
                        slp_label_mapping = slp_label_mapping,
                        target_frequency = project_configuration["slp_frequency"],
                        signal_length_seconds = project_configuration["signal_length_seconds"],
                        pad_with = project_configuration["pad_target_with"],
                        reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"],
                        **common_window_reshape_parameters,
                        normalize = False,  # SLP labels should not be normalized
                        datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                        transform = project_configuration["target_transform"]
                    )
                    
                    # Send data to device
                    rri, slp = rri.to(device), slp.to(device)

                    # reshape slp to fit the model output
                    slp = slp.view(-1) # Combine batch and windows dimensions

                    # Compute prediction and loss
                    pred = neural_network_model(rri, mad)
                    slp = slp.long()
                    test_loss += loss_function(pred, slp).item()

                    # update confusion matrix
                    pred = pred.argmax(1).cpu().numpy()
                    slp = slp.cpu().numpy()

                    for i in range(len(slp)):
                        test_confusion_matrix[slp[i], pred[i]] += 1

                # print progress bar
                progress_bar.update()

                neural_network_model.reset_residual_features() # reset residual features for next data sample

                test_loss /= len(validation_data_manager)
                accuracy = test_confusion_matrix.diagonal().sum() / test_confusion_matrix.sum()

                print(f"\nTest Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
                    
                start += 1
                end += 1

            test_avg_loss[i].append(test_loss)
            test_confusion_matrices[i].append(test_confusion_matrix)

    """
    ----------------------------------
    Saving Neural Network Model State
    ----------------------------------
    """

    create_directories_along_path(path_to_updated_model_state)
    
    torch.save(neural_network_model.state_dict(), path_to_updated_model_state)

    
    """
    --------------------------
    Saving Performance Values
    --------------------------
    """

    create_directories_along_path(path_to_loss_per_epoch)

    performance_values = {
        "train_avg_loss": train_avg_loss,
        "train_confusion_matrix": train_confusion_matrices,
    }
    short_names = copy.deepcopy(paths_to_validation_data_directories)
    for i in range(len(short_names)):
        if "SHHS" in short_names[i]:
            short_names[i] = "SHHS"
        elif "GIF" in short_names[i]:
            short_names[i] = "GIF"

    for i, name in enumerate(short_names):
        performance_values[f"{name}_avg_loss"] = test_avg_loss[i]
        performance_values[f"{name}_confusion_matrix"] = test_confusion_matrices[i]

    save_to_pickle(performance_values, path_to_loss_per_epoch)


def main_model_predicting(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
    ):
    """
    Applies the trained neural network model to the processed data. The processed data is accessed using the
    SleepDataManager class from dataset_processing.py. The predictions are retransformed to the original
    signal structure (they were reshaped to overlapping windows during training).
    
    If the database was previously split into training, validation, and test datasets, the algorithm assumes
    that the data also contains the actual sleep stages and you want to do statistics using them and the 
    predictions. Therefore, the results are saved to a pkl-file as individual dictionaries for every patient.
    These dictionaries have the following format:
    {
        "Predicted_Probabilities": 
            - shape: (number datapoints, number_target_classes) 
            - probabilities for each target class,
        "Predicted": 
            - shape: (number datapoints) 
            - predicted target class with highest probability,
        "Actual": 
            - shape: (number datapoints) 
            - actual target class,
        "Predicted_in_windows": 
            - shape: (number datapoints, windows_per_signal) 
            - predicted target classes with highest probability, signal still as overlapping windows (output of neural network), 
        "Actual_in_windows":
            - shape: (number datapoints, windows_per_signal) 
            - actual target classes, signal still as overlapping windows (used by the neural network),
    }

    If the database was not split, the algorithm assumes you want to collect the predicted target classes and 
    saves them directly to the database for easy access. Each appropriate datapoint is updated with the
    predicted target classes:
    {
        "SLP_predicted_probability":
            - shape: (windows_per_signal, number_target_classes) 
            - probabilities for each target class,
        "SLP_predicted":
            - shape: (windows_per_signal) 
            - predicted target class with highest probability,
    }

    Note:   The algorithm already crops the target classes to the correct length of the original signal. This is
            important as the original signal might has been padded to fit the requirements of the neural network.


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
    data_manager = BigDataManager(directory_path = path_to_data_directory, pid = pid)
    pid_file_path = data_manager.pid_paths[data_manager.current_pid]

    # retrieve rri, mad, and slp frequencies
    rri_frequency = data_manager.database_configuration["RRI_frequency"]
    mad_frequency = data_manager.database_configuration["MAD_frequency"]
    slp_frequency = data_manager.database_configuration["SLP_frequency"]

    # determine if data contains sleep phases
    actual_results_available = False
    if "SLP" in data_manager.load(0): # type: ignore
        actual_results_available = True

    """
    --------------------------------
    Accessing Project Configuration
    --------------------------------
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access neural network initialization parameters
    neural_network_model = project_configuration["neural_network_model"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )

    rri_inlier_interval = project_configuration["rri_inlier_interval"]
    mad_inlier_interval = project_configuration["mad_inlier_interval"]

    # parameters needed for ensuring uniform signal shape
    signal_length_seconds = project_configuration["signal_length_seconds"]
    pad_feature_with = project_configuration["pad_feature_with"]
    pad_target_with = project_configuration["pad_target_with"]

    # access common window_reshape_parameters
    reshape_to_overlapping_windows = project_configuration["reshape_to_overlapping_windows"]
    common_window_reshape_params = dict()

    if reshape_to_overlapping_windows:
        common_window_reshape_params = {key: project_configuration[key] for key in ["windows_per_signal", "window_duration_seconds", "overlap_seconds", "priority_order"]} # window_reshape_parameters

    # access common signal_normalization_parameters
    normalize_rri = project_configuration["normalize_rri"]
    normalize_mad = project_configuration["normalize_mad"]
    common_signal_normalization_params = dict()

    if normalize_mad or normalize_rri:
        common_signal_normalization_params = {key: project_configuration[key] for key in project_configuration if key in ["normalization_technique", "normalization_mode", "normalization_max", "normalization_min"]} # signal_normalization_parameters

    # access feature and target transformations
    feature_transform = project_configuration["feature_transform"]
    target_transform = project_configuration["target_transform"]

    del project_configuration

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

    neural_network_model = neural_network_model(**nnm_params)
   
    neural_network_model.load_state_dict(torch.load(path_to_model_state, map_location=device, weights_only=True))
    
    neural_network_model.to(device)

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    neural_network_model.eval()

    """
    -----------------------------
    Preparations for Saving Data
    -----------------------------
    """

    # prepare path that stores results, if necessary
    if actual_results_available:
        if os.path.exists(path_to_save_results):
            os.remove(path_to_save_results)
        else:
            create_directories_along_path(path_to_save_results)

    # Create temporary file to save data in progress
    working_file_path = path_to_data_directory + "save_in_progress"
    working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = len(data_manager))


    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:

            """
            Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
            """

            rri = final_data_preprocessing(
                signal = copy.deepcopy(data_dict["RRI"]), # type: ignore
                signal_id = "RRI",
                inlier_interval = rri_inlier_interval,
                target_frequency = rri_frequency,
                signal_length_seconds = signal_length_seconds,
                pad_with = pad_feature_with,
                reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                **common_window_reshape_params,
                normalize = normalize_rri,
                **common_signal_normalization_params,
                datatype_mappings = [(np.float64, np.float32)],
                transform = feature_transform
            )

            rri = rri.unsqueeze(0) # type: ignore # add batch dimension (= 1)
            rri = rri.to(device) # type: ignore

            # Ensure RRI is of the correct data type
            if not isinstance(rri, torch.FloatTensor):
                rri = rri.float()

            # MAD preparation analogously to RRI
            if "MAD" in data_dict:
                mad = final_data_preprocessing(
                    signal = copy.deepcopy(data_dict["MAD"]), # type: ignore
                    signal_id = "MAD",
                    inlier_interval = mad_inlier_interval,
                    target_frequency = mad_frequency,
                    signal_length_seconds = signal_length_seconds,
                    pad_with = pad_feature_with,
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    normalize = normalize_mad,
                    **common_signal_normalization_params,
                    datatype_mappings = [(np.float64, np.float32)],
                    transform = feature_transform
                )

                mad = mad.unsqueeze(0) # type: ignore # add batch dimension (= 1)
                mad = mad.to(device) # type: ignore

                if not isinstance(mad, torch.FloatTensor):
                    mad = mad.float()
            else:
                mad = None

            if actual_results_available:
                actual_original_structure = map_slp_labels(
                    slp_labels = data_dict["SLP"], # type: ignore
                    slp_label_mapping = slp_label_mapping
                )
                original_signal_length = len(copy.deepcopy(actual_original_structure))

                # access processed target, reshape to overlapping windows and apply transformations
                slp = final_data_preprocessing(
                    signal = copy.deepcopy(actual_original_structure),
                    signal_id = "SLP",
                    slp_label_mapping = slp_label_mapping,
                    target_frequency = slp_frequency,
                    signal_length_seconds = signal_length_seconds,
                    pad_with = pad_target_with,
                    reshape_to_overlapping_windows = reshape_to_overlapping_windows,
                    **common_window_reshape_params,
                    normalize = False, # SLP is not normalized
                    datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                    transform = target_transform
                )

            else:
                original_signal_length = int(np.ceil(signal_length_seconds * slp_frequency))
            
            """
            Applying Neural Network Model
            """

            working_file = open(working_file_path, "ab")

            error_occured = True
            try:
                # predictions in windows
                if reshape_to_overlapping_windows:
                    try:
                        predictions_in_windows = neural_network_model(rri, mad)
                    except:
                        unpredictable_signals.append(data_dict["ID"]) # type: ignore
                        continue

                    """
                    Preparing Predicted Sleep Phases
                    """

                    predictions_in_windows = predictions_in_windows.cpu().numpy()

                    # reshape windows to original signal structure
                    # Lot of stuff happening below, so i explain the process:
                    # predictions_in_windows is a 2D array with shape (windows_per_signal, number_target_classes)
                    predictions_probability = np.empty((original_signal_length, 0))
                    for i in range(predictions_in_windows.shape[1]):
                        # get a list of probabilities for this target class
                        this_slp_stage_pred_probability = copy.deepcopy(predictions_in_windows[:, i])
                        # transform every probability to a list with the size of the SLP windows, with every entry 
                        # being the probability
                        pred_prob_expanded_to_windows = []
                        for pred_prob in this_slp_stage_pred_probability:
                            pred_prob_expanded_to_windows.append([pred_prob for _ in range(int(common_window_reshape_params["window_duration_seconds"]*slp_frequency))])
                        # if we now pass this list to the reverse window reshape function, using the frequency of the
                        # SLP signal, we get the probability for this sleep stage in the same sampling frequency as 
                        # the SLP signal
                        temp_original_structure = reverse_signal_to_windows_reshape(
                            signal_in_windows = pred_prob_expanded_to_windows,
                            target_frequency = slp_frequency,
                            original_signal_length = original_signal_length,
                            number_windows = common_window_reshape_params["windows_per_signal"],
                            window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                            overlap_seconds = common_window_reshape_params["overlap_seconds"],
                        )
                        temp_original_structure = np.array([[temp_val] for temp_val in temp_original_structure])
                        predictions_probability = np.append(predictions_probability, temp_original_structure, axis=1)
                    
                    # convert probabilities to sleep stages
                    predictions_original_structure = np.argmax(copy.deepcopy(predictions_probability), axis=1)

                    """
                    Saving Predicted (and Actual) Sleep Phases
                    """
                    
                    if actual_results_available:
                        # remove padding from signals with overlapping windows
                        predictions_in_windows = remove_padding_from_windows(
                            signal_in_windows = predictions_in_windows,
                            target_frequency = slp_frequency,
                            original_signal_length = original_signal_length,
                            window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                            overlap_seconds = common_window_reshape_params["overlap_seconds"],
                        )

                        slp = remove_padding_from_windows(
                            signal_in_windows = slp, # type: ignore
                            target_frequency = slp_frequency,
                            original_signal_length = original_signal_length,
                            window_duration_seconds = common_window_reshape_params["window_duration_seconds"],
                            overlap_seconds = common_window_reshape_params["overlap_seconds"],
                        )

                        # save results to new dictionary
                        results = {
                            "Predicted_Probabilities": predictions_probability,
                            "Predicted": predictions_original_structure,
                            "Actual": actual_original_structure,
                            "Predicted_in_windows": predictions_in_windows.argmax(1).flatten(),
                            "Actual_in_windows": slp
                        }
                    
                    else:
                        # save results to existing dictionary
                        results = copy.deepcopy(data_dict)
                        results["SLP_predicted_probability"] = predictions_probability
                        results["SLP_predicted"] = predictions_original_structure
                
                # predictions not in windows
                else:
                    try:
                        predictions_probability = neural_network_model(rri, mad)
                    except:
                        unpredictable_signals.append(data_dict["ID"]) # type: ignore
                        continue

                    predictions_probability = predictions_probability.cpu().numpy()
                    predicted = predictions_probability.argmax(1)

                    # expand to correct shape
                    predictions_probability = np.array([predictions_probability[0] for _ in range(original_signal_length)])
                    predicted = np.array([predicted[0] for _ in range(original_signal_length)])

                    if actual_results_available:
                        # save results to new dictionary
                        results = {
                            "Predicted_Probabilities": predictions_probability[:original_signal_length], # remove padding
                            "Predicted": predicted[:original_signal_length], # remove padding
                            "Actual": actual_original_structure,
                        }
                    else:
                        # save results to existing dictionary
                        results = copy.deepcopy(data_dict)
                        results["SLP_predicted_probability"] = predictions_probability
                        results["SLP_predicted"] = predicted
                
                error_occured = False
                pickle.dump(results, working_file)

            finally:
                working_file.close()

                if error_occured:
                    if os.path.exists(working_file_path):
                        os.remove(working_file_path)
                    
            # update progress
            progress_bar.update()

    # Remove the old file and rename the working file
    if actual_results_available:
        os.rename(working_file_path, path_to_save_results)
    else:
        if os.path.isfile(pid_file_path):
            os.remove(pid_file_path)

        os.rename(working_file_path, pid_file_path)
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def main_pipeline_SSG(
        project_configuration, 
        path_to_model_directory: str,
        neural_network_hyperparameters_shhs: dict,
        neural_network_hyperparameters_gif: dict,
        path_to_shhs_database: str,
        path_to_gif_database: str,
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
    create_directories_along_path(path_to_model_directory)


    """
    ==========================
    Set Project Configuration
    ==========================
    """

    check_project_configuration(project_configuration)

    if os.path.isfile(path_to_model_directory + project_configuration_file):
        os.remove(path_to_model_directory + project_configuration_file)
    save_to_pickle(project_configuration, path_to_model_directory + project_configuration_file)

    """
    ==============================
    Training Neural Network Model
    ==============================
    """

    newly_trained_model = False
    if not os.path.exists(path_to_model_directory + model_state_after_shhs_file):
        main_model_training(
            neural_network_hyperparameters = neural_network_hyperparameters_shhs,
            path_to_training_data_directory = path_to_shhs_database,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = None,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_file,
            paths_to_validation_data_directories = [path_to_shhs_database, path_to_gif_database],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_shhs_file,
        )
        newly_trained_model = True
    
    if not os.path.exists(path_to_model_directory + model_state_after_shhs_gif_file):
        main_model_training(
            neural_network_hyperparameters = neural_network_hyperparameters_gif,
            path_to_training_data_directory = path_to_gif_database,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = path_to_model_directory + model_state_after_shhs_file,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
            paths_to_validation_data_directories = [path_to_shhs_database, path_to_gif_database],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_gif_file,
        )
        newly_trained_model = True

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    if newly_trained_model:
        run_model_performance_evaluation_SSG(
            path_to_model_directory = path_to_model_directory,
            path_to_shhs_directory = path_to_shhs_database,
            path_to_gif_directory = path_to_gif_database,
        )


def main_pipeline_SAE(
        project_configuration, 
        path_to_model_directory: str,
        neural_network_hyperparameters_gif: dict,
        path_to_gif_database: str,
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
    create_directories_along_path(path_to_model_directory)


    """
    ==========================
    Set Project Configuration
    ==========================
    """

    check_project_configuration(project_configuration)

    if os.path.isfile(path_to_model_directory + project_configuration_file):
        os.remove(path_to_model_directory + project_configuration_file)
    save_to_pickle(project_configuration, path_to_model_directory + project_configuration_file)

    """
    ==============================
    Training Neural Network Model
    ==============================
    """

    newly_trained_model = False
    if not os.path.exists(path_to_model_directory + model_state_after_shhs_gif_file):
        main_model_training(
            neural_network_hyperparameters = neural_network_hyperparameters_gif,
            path_to_training_data_directory = path_to_gif_database,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = path_to_model_directory + model_state_after_shhs_file,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
            paths_to_validation_data_directories = [path_to_gif_database],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_gif_file,
        )
        newly_trained_model = True

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    if newly_trained_model:
        run_model_performance_evaluation_SAE(
            path_to_model_directory = path_to_model_directory,
            path_to_gif_directory = path_to_gif_database,
        )


def Reduced_Process_SHHS_SSG_Dataset(
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
    shhs_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    shhs_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations"]} # pid_distribution_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["shhs_min_duration_hours"] * 3600
    filter_ids = project_configuration["shhs_filter_ids"]

    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')
    
    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    shhs_target_classes = {"wake": [0], "LS": [1, 2], "DS": [3], "REM": [5], "artifact": ["other"]}

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    # check if patient ids are unique:
    shhs_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the SHHS dataset:")
    progress_bar = DynamicProgressBar(total = len(patients))

    # saving all data from SHHS dataset to the pickle file
    removed = 0
    for patient_id in patients:
        # filter data
        if patient_id in filter_ids:
            removed += 1
            continue
        if len(shhs_dataset["rri"][patient_id][:]) / shhs_dataset["rri"].attrs["freq"] < minimum_length_seconds: # type: ignore
            removed += 1
            continue

        new_datapoint = {
            "ID": patient_id,
            "RRI": shhs_dataset["rri"][patient_id][:], # type: ignore
            "SLP": shhs_dataset["slp"][patient_id][:], # type: ignore
            "RRI_frequency": shhs_dataset["rri"].attrs["freq"], # type: ignore
            "SLP_frequency": shhs_dataset["slp"].attrs["freq"], # type: ignore
            "target_classes": copy.deepcopy(shhs_target_classes)
        }

        shhs_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    shhs_data_manager.separate_train_test_validation(**distribution_params)
    print(len(patients)-removed, "of", len(patients), "datapoints were kept after filtering.")


def Reduced_Process_GIF_SSG_Dataset(
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
    gif_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    gif_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations"]} # pid_distribution_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["gif_min_duration_hours"] * 3600
    filter_ids = project_configuration["gif_filter_ids"]

    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"wake": [0], "LS": [1, 2], "DS": [3], "REM": [5], "artifact": ["other"]}

    gif_data_generator = load_from_pickle(path_to_gif_dataset)
    gif_length = 0
    for generator_entry in gif_data_generator:
        if generator_entry["ID"].split("_")[0] in filter_ids:
            continue
        if len(generator_entry["RRI"]) / generator_entry["RRI_frequency"] < minimum_length_seconds: # type: ignore
            continue
        gif_length += 1
    del gif_data_generator

    gif_data_generator = load_from_pickle(path_to_gif_dataset)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the GIF dataset:")
    progress_bar = DynamicProgressBar(total = gif_length)

    # saving all data from GIF dataset to the pickle file
    for generator_entry in gif_data_generator:
        # filter data
        if generator_entry["ID"].split("_")[0] in filter_ids:
            continue
        if len(generator_entry["RRI"]) / generator_entry["RRI_frequency"] < minimum_length_seconds: # type: ignore
            continue

        new_datapoint = {
            "ID": generator_entry["ID"],
            "RRI": generator_entry["RRI"],
            "MAD": generator_entry["MAD"],
            "SLP": np.array(generator_entry["SLP"]).astype(int),
            "RRI_frequency": generator_entry["RRI_frequency"],
            "MAD_frequency": generator_entry["MAD_frequency"],
            "SLP_frequency": generator_entry["SLP_frequency"],
            "target_classes": copy.deepcopy(gif_target_classes)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # Train-, Validation- and Test-Pid Distribution
    gif_data_manager.separate_train_test_validation(**distribution_params)


def Reduced_Process_GIF_SAE_Dataset(
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
    gif_data_manager = BigDataManager(directory_path = path_to_save_processed_data)

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access sampling frequency parameters
    freq_params = {key: project_configuration[key] for key in ["RRI_frequency", "MAD_frequency", "SLP_frequency"]} # sampling_frequency_parameters
    gif_data_manager.change_uniform_frequencies(freq_params)

    # access parameters used for distributing the data into train, validation, and test pids
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations"]} # pid_distribution_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["gif_min_duration_hours"] * 3600
    filter_ids = project_configuration["gif_filter_ids"]

    # define the sleep apnea event labels (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"Normal": [0], "Apnea": [1], "Obstructive Apnea": [2], "Central Apnea": [3], "Mixed Apnea": [4], "Hypopnea": [5], "Obstructive Hypopnea": [6], "Central Hypopnea": [7]}

    gif_data_generator = load_from_pickle(path_to_gif_dataset)
    gif_length = 0
    for generator_entry in gif_data_generator:
        if generator_entry["ID"].split("_")[0] in filter_ids:
            continue
        if len(generator_entry["RRI"]) / generator_entry["RRI_frequency"] < minimum_length_seconds: # type: ignore
            continue
        gif_length += 1
    del gif_data_generator

    gif_data_generator = load_from_pickle(path_to_gif_dataset)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the GIF dataset:")
    progress_bar = DynamicProgressBar(total = gif_length)

    # saving all data from GIF dataset to the pickle file
    for generator_entry in gif_data_generator:
        # filter data
        if generator_entry["ID"].split("_")[0] in filter_ids:
            continue
        if len(generator_entry["RRI"]) / generator_entry["RRI_frequency"] < minimum_length_seconds: # type: ignore
            continue

        new_datapoint = {
            "ID": generator_entry["ID"],
            "RRI": generator_entry["RRI"],
            "MAD": generator_entry["MAD"],
            "SLP": np.array(generator_entry["SAE"]).astype(int),
            "RRI_frequency": generator_entry["RRI_frequency"],
            "MAD_frequency": generator_entry["MAD_frequency"],
            "SLP_frequency": generator_entry["SAE_frequency"],
            "target_classes": copy.deepcopy(gif_target_classes)
        }

        gif_data_manager.save(new_datapoint, unique_id=True)
        progress_bar.update()

    # Train-, Validation- and Test-Pid Distribution
    gif_data_manager.separate_train_test_validation(**distribution_params)


def copy_and_split_default_database_SSG(
        path_to_default_shhs_database: str,
        path_to_default_gif_database: str,
        path_to_save_shhs_database: str,
        path_to_save_gif_database: str,
        project_configuration: dict
    ):
    """
    """
    """
    =======================
    Copy Default Databases
    =======================
    """

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "shift_length_seconds_interval"]} # signal_cropping_parameters
    del project_configuration

    # copy default SHHS and GIF databases
    if not os.path.exists(path_to_save_shhs_database):
        shutil.copytree(path_to_default_shhs_database, path_to_save_shhs_database)

        # process SHHS dataset
        shhs_data_manager = BigDataManager(directory_path = path_to_save_shhs_database)
        shhs_data_manager.crop_oversized_data(**signal_crop_params)

        del shhs_data_manager

    if not os.path.exists(path_to_save_gif_database):
        shutil.copytree(path_to_default_gif_database, path_to_save_gif_database)

        # process GIF dataset
        gif_data_manager = BigDataManager(directory_path = path_to_save_gif_database)
        gif_data_manager.crop_oversized_data(**signal_crop_params)

        del gif_data_manager

    del signal_crop_params


def copy_and_split_default_database_SAE(
        path_to_default_gif_database: str,
        path_to_save_gif_database: str,
        project_configuration: dict
    ):
    """
    """
    """
    =======================
    Copy Default Databases
    =======================
    """

    # access parameters used for cropping the data
    signal_crop_params = {key: project_configuration[key] for key in ["signal_length_seconds", "shift_length_seconds_interval"]} # signal_cropping_parameters
    del project_configuration

    if not os.path.exists(path_to_save_gif_database):
        shutil.copytree(path_to_default_gif_database, path_to_save_gif_database)

        # process GIF dataset
        gif_data_manager = BigDataManager(directory_path = path_to_save_gif_database)
        gif_data_manager.crop_oversized_data(**signal_crop_params)

        del gif_data_manager

    del signal_crop_params


def build_default_datasets_for_training_and_testing():
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
        "shhs_min_duration_hours": 0,
        "shhs_filter_ids": [],
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5
    }
    with open(limited_project_configuration_file, "wb") as file:
        pickle.dump(project_configuration, file)

    Reduced_Process_SHHS_SSG_Dataset(
        path_to_shhs_dataset = original_shhs_data_path,
        path_to_save_processed_data = default_complete_shhs_SSG_path,
        path_to_project_configuration = limited_project_configuration_file,
        )
    
    Reduced_Process_GIF_SSG_Dataset(
        path_to_gif_dataset = original_gif_ssg_data_path,
        path_to_save_processed_data = default_complete_gif_SSG_path,
        path_to_project_configuration = limited_project_configuration_file
        )
    
    os.remove(limited_project_configuration_file)

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
        "shhs_min_duration_hours": 7,
        "shhs_filter_ids": [],
        "gif_min_duration_hours": 7,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5
    }
    with open(limited_project_configuration_file, "wb") as file:
        pickle.dump(project_configuration, file)

    Reduced_Process_SHHS_SSG_Dataset(
        path_to_shhs_dataset = original_shhs_data_path,
        path_to_save_processed_data = default_reduced_shhs_SSG_path,
        path_to_project_configuration = limited_project_configuration_file,
        )
    
    Reduced_Process_GIF_SSG_Dataset(
        path_to_gif_dataset = original_gif_ssg_data_path,
        path_to_save_processed_data = default_reduced_gif_SSG_path,
        path_to_project_configuration = limited_project_configuration_file
        )
    
    os.remove(limited_project_configuration_file)

    limited_project_configuration_file = "default_project_configuration.pkl"
    project_configuration = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1,
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True,
        "shhs_min_duration_hours": 0,
        "shhs_filter_ids": [],
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]
    }
    with open(limited_project_configuration_file, "wb") as file:
        pickle.dump(project_configuration, file)
    
    Reduced_Process_GIF_SAE_Dataset(
        path_to_gif_dataset = original_gif_sae_data_path,
        path_to_save_processed_data = default_complete_gif_SAE_path,
        path_to_project_configuration = limited_project_configuration_file
        )
    
    os.remove(limited_project_configuration_file)

    limited_project_configuration_file = "default_project_configuration.pkl"
    project_configuration = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1,
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True,
        "shhs_min_duration_hours": 7,
        "shhs_filter_ids": [],
        "gif_min_duration_hours": 7,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]
    }
    with open(limited_project_configuration_file, "wb") as file:
        pickle.dump(project_configuration, file)
    
    Reduced_Process_GIF_SAE_Dataset(
        path_to_gif_dataset = original_gif_sae_data_path,
        path_to_save_processed_data = default_reduced_gif_SAE_path,
        path_to_project_configuration = limited_project_configuration_file
        )
    
    os.remove(limited_project_configuration_file)


def train_and_test_long_sequence_model_on_sleep_staging_data():

    """
    =======================================================
    Default Project Configuration for Long-Sequence Models
    =======================================================
    """

    sampling_frequency_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1/30,
    }

    signal_cropping_parameters = {
        "signal_length_seconds": 36000,
        "shift_length_seconds_interval": (3600, 7200),
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None), # (0.3, 2)
        "mad_inlier_interval": (None, None),
        "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
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
        "feature_transform": custom_transform,
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
        "neural_network_model": LongSequenceModel,
        "number_target_classes": 4,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "fully_connected_features": 128,
        "convolution_dilations": [2, 4, 8, 16, 32],
        #
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "windows_per_signal": window_reshape_parameters["windows_per_signal"],
    }

    neural_network_hyperparameters_shhs = {
        "batch_size": 8, # 80h for 10h data | 7K (6712) / 8 => 839 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    neural_network_hyperparameters_gif = {
        "batch_size": 4, # 40h for 10h data | 584 / 4 => 146 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    filter_shhs_data_parameters = {
        "shhs_min_duration_hours": 7,
        "shhs_filter_ids": []
    }

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 7,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5
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
    default_project_configuration.update(filter_shhs_data_parameters)
    default_project_configuration.update(filter_gif_data_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_shhs_data_parameters, filter_gif_data_parameters

    overlap_artifact_as_wake = {
        "number_target_classes": 4,
        "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
        "window_duration_seconds": 120,
        "windows_per_signal": 1197,
        "overlap_seconds": 90,
        "priority_order": [3, 2, 1, 0],
    }

    no_overlap_artifact_as_wake = {
        "number_target_classes": 4,
        "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
        "window_duration_seconds": 120,
        "windows_per_signal": 300,
        "overlap_seconds": 0,
        "priority_order": [3, 2, 1, 0],
    }
    
    overlap_full_class = {
        "number_target_classes": 5, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifact": 0},
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

    network_models = [LongSequenceModel, LongSequenceResidualModel]
    network_model_names = ["LSM", "LSM_Residual"]

    # all share same signal cropping parameters, so we need to create only one database to draw data from
    shhs_directory_path = "10h_SHHS_SSG_Data/"
    gif_directory_path = "10h_GIF_SSG_Data/"

    if not os.path.exists(shhs_directory_path) or not os.path.exists(gif_directory_path):
        copy_and_split_default_database_SSG(
            path_to_default_shhs_database = default_reduced_shhs_SSG_path,
            path_to_default_gif_database = default_reduced_gif_SSG_path,
            path_to_save_shhs_database = shhs_directory_path,
            path_to_save_gif_database = gif_directory_path,
            project_configuration = default_project_configuration
        )

    for clean_index in range(len(cleaning_adjustments)):
        for window_index in range(len(window_and_class_adjustments)):
            for model_index in range(len(network_models)):

                project_configuration = copy.deepcopy(default_project_configuration)
                project_configuration.update(cleaning_adjustments[clean_index])
                project_configuration.update(window_and_class_adjustments[window_index])
                project_configuration["neural_network_model"] = network_models[model_index]

                identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index]
                print_headline("Running " + identifier, "=")

                identifier += "/"

                main_pipeline_SSG(
                    project_configuration = project_configuration,
                    path_to_model_directory = identifier,
                    neural_network_hyperparameters_shhs = neural_network_hyperparameters_shhs,
                    neural_network_hyperparameters_gif = neural_network_hyperparameters_gif,
                    path_to_shhs_database = shhs_directory_path,
                    path_to_gif_database = gif_directory_path,
                )
    
    del project_configuration, default_project_configuration


def train_and_test_short_sequence_model_on_sleep_staging_data():

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
        "shift_length_seconds_interval": (30, 30),
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None),
        "mad_inlier_interval": (None, None),
        "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
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
        "feature_transform": custom_transform,
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
        "neural_network_model": ShortSequenceModel,
        "number_target_classes": 4,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "fully_connected_features": 128,
        "rri_datapoints": int(sampling_frequency_parameters["RRI_frequency"] * signal_cropping_parameters["signal_length_seconds"]),
        "mad_datapoints": int(sampling_frequency_parameters["MAD_frequency"] * signal_cropping_parameters["signal_length_seconds"]),
    }

    filter_shhs_data_parameters = {
        "shhs_min_duration_hours": 0,
        "shhs_filter_ids": []
    }

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5
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
    default_project_configuration.update(filter_shhs_data_parameters)
    default_project_configuration.update(filter_gif_data_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_shhs_data_parameters, filter_gif_data_parameters

    artifact_as_wake = {
        "number_target_classes": 4,
        "target_classes": {"wake": 0, "LS": 1, "DS": 2, "REM": 3, "artifact": 0},
    }
    
    full_class = {
        "number_target_classes": 5, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"wake": 1, "LS": 2, "DS": 3, "REM": 4, "artifact": 0},
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
        "shift_length_seconds_interval": (30, 30),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 30),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 30),
    }

    thirty_second_hyperparameters_shhs = {
        "batch_size": 128, # 64m for 30s data | 6M (5931923) / 128 => 46344 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    thirty_second_hyperparameters_gif = {
        "batch_size": 8, # 16m for 30s data | 350K (348524) / 32 => 10892 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    sixty_second_network = {
        "signal_length_seconds": 60,
        "shift_length_seconds_interval": (60, 60),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 60),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 60),
    }

    sixty_second_hyperparameters_shhs = {
        "batch_size": 128, # 2.1h for 60s data | 3M (2966296) / 128 => 23175 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    sixty_second_hyperparameters_gif = {
        "batch_size": 8, # 32m for 60s data | 175K (174374) / 32 => 5450 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    hundred_twenty_second_network = {
        "signal_length_seconds": 120,
        "shift_length_seconds_interval": (120, 120),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 120),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 120),
    }

    hundred_twenty_second_hyperparameters_shhs = {
        "batch_size": 128, # 4.2h for 120s data | 1.5M (1484839) / 128 => 11601 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    hundred_twenty_second_hyperparameters_gif = {
        "batch_size": 8, # 64m for 120s data | 90K (87221) / 32 => 2726 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
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

    # different networks have different signal cropping parameters, so we need to create a database for each network
    shhs_directory_paths = ["30s_SHHS_SSG_Data/", "60s_SHHS_SSG_Data/", "120s_SHHS_SSG_Data/"]
    gif_directory_paths = ["30s_GIF_SSG_Data/", "60s_GIF_SSG_Data/", "120s_GIF_SSG_Data/"]

    for net_adjust_index in range(len(network_adjustments)):
        copy_and_split_default_database_SSG(
            path_to_default_shhs_database = default_complete_shhs_SSG_path,
            path_to_default_gif_database = default_complete_gif_SSG_path,
            path_to_save_shhs_database = shhs_directory_paths[net_adjust_index],
            path_to_save_gif_database = gif_directory_paths[net_adjust_index],
            project_configuration = network_adjustments[net_adjust_index]
        )

    for clean_index in range(len(cleaning_adjustments)):
        project_configuration = copy.deepcopy(default_project_configuration)
        project_configuration.update(cleaning_adjustments[clean_index])

        for class_index in range(len(class_adjustments)):
            project_configuration.update(class_adjustments[class_index])
            
            for network_index in range(len(network_adjustments)):
                project_configuration.update(network_adjustments[network_index])

                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index]
                print_headline("Running " + identifier, "=")

                identifier += "/"

                main_pipeline_SSG(
                    project_configuration = project_configuration,
                    path_to_model_directory = identifier,
                    neural_network_hyperparameters_shhs = shhs_hyperparameter_adjustments[network_index],
                    neural_network_hyperparameters_gif = gif_hyperparameter_adjustments[network_index],
                    path_to_shhs_database = shhs_directory_paths[network_index],
                    path_to_gif_database = gif_directory_paths[network_index],
                )


def train_and_test_long_sequence_model_on_apnea_events():

    """
    =======================================================
    Default Project Configuration for Long-Sequence Models
    =======================================================
    """

    sampling_frequency_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1,
    }

    signal_cropping_parameters = {
        "signal_length_seconds": 36000,
        "shift_length_seconds_interval": (3600, 7200),
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None), # (0.3, 2)
        "mad_inlier_interval": (None, None),
        "target_classes": {"Normal": 0, "Apnea": 3, "Obstructive Apnea": 1, "Central Apnea": 2, "Mixed Apnea": 3, "Hypopnea": 4, "Obstructive Hypopnea": 4, "Central Hypopnea": 4},
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
        "feature_transform": custom_transform,
        "target_transform": None,
    }

    window_reshape_parameters = {
        "reshape_to_overlapping_windows": True,
        #
        "windows_per_signal": 4499,
        "window_duration_seconds": 16,
        "overlap_seconds": 8,
        "priority_order": [3, 2, 4, 1, 0],
    }

    signal_normalization_parameters = {
        "normalize_rri": False,
        "normalize_mad": False,
    }

    neural_network_model_parameters = {
        "neural_network_model": LongSequenceModel,
        "number_target_classes": 5,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "fully_connected_features": 128,
        "convolution_dilations": [2, 4, 8, 16, 32],
        #
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * window_reshape_parameters["window_duration_seconds"]),
        "windows_per_signal": window_reshape_parameters["windows_per_signal"],
    }

    neural_network_hyperparameters_gif = {
        "batch_size": 4, # 40h for 10h data | 584 / 4 => 146 steps per epoch
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 7,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]
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
    default_project_configuration.update(filter_gif_data_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_gif_data_parameters

    overlap = {
        "window_duration_seconds": 16,
        "windows_per_signal": 4499,
        "overlap_seconds": 8,
    }

    no_overlap_10 = {
        "window_duration_seconds": 10,
        "windows_per_signal": 3600,
        "overlap_seconds": 0,
    }

    no_overlap_16 = {
        "window_duration_seconds": 16,
        "windows_per_signal": 2250,
        "overlap_seconds": 0,
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

    window_and_class_adjustments = [overlap, no_overlap_10, no_overlap_16]
    window_and_class_names = ["Overlap_16", "No_Overlap_10", "No_Overlap_16"]

    cleaning_adjustments = [raw, cleaned, global_norm, local_norm]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]

    network_models = [LongSequenceModel, LongSequenceResidualModel]
    network_model_names = ["LSM", "LSM_Residual"]

    # all share same signal cropping parameters, so we need to create only one database to draw data from
    gif_directory_path = "10h_GIF_SAE_Data/"

    if not os.path.exists(gif_directory_path):
        copy_and_split_default_database_SAE(
            path_to_default_gif_database = default_complete_gif_SAE_path,
            path_to_save_gif_database = gif_directory_path,
            project_configuration = default_project_configuration
        )

    for clean_index in range(len(cleaning_adjustments)):
        for window_index in range(len(window_and_class_adjustments)):
            for model_index in range(len(network_models)):

                project_configuration = copy.deepcopy(default_project_configuration)
                project_configuration.update(cleaning_adjustments[clean_index])
                project_configuration.update(window_and_class_adjustments[window_index])
                project_configuration["neural_network_model"] = network_models[model_index]

                identifier = "SAE_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index]
                print_headline("Running " + identifier, "=")

                identifier += "/"

                main_pipeline_SAE(
                    project_configuration = project_configuration,
                    path_to_model_directory = identifier,
                    neural_network_hyperparameters_gif = neural_network_hyperparameters_gif,
                    path_to_gif_database = gif_directory_path,
                )
    
    del project_configuration, default_project_configuration


def train_and_test_short_sequence_model_on_apnea_events():

    """
    ==========================================================
    Default Project Configuration for Local Short-Time Models
    ==========================================================
    """

    sampling_frequency_parameters = {
        "RRI_frequency": 4,
        "MAD_frequency": 1,
        "SLP_frequency": 1,
    }

    signal_cropping_parameters = {
        "signal_length_seconds": 10,
        "shift_length_seconds_interval": (10, 10),
    }

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None),
        "mad_inlier_interval": (None, None),
        "target_classes": {"Normal": 0, "Apnea": 3, "Obstructive Apnea": 1, "Central Apnea": 2, "Mixed Apnea": 3, "Hypopnea": 4, "Obstructive Hypopnea": 4, "Central Hypopnea": 4},
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
        "feature_transform": custom_transform,
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
        "neural_network_model": ShortSequenceModel,
        "number_target_classes": 5,
        "rri_convolutional_channels": [1, 8, 16, 32, 64],
        "mad_convolutional_channels": [1, 8, 16, 32, 64],
        "max_pooling_layers": 5,
        "fully_connected_features": 128,
        "rri_datapoints": int(sampling_frequency_parameters["RRI_frequency"] * signal_cropping_parameters["signal_length_seconds"]),
        "mad_datapoints": int(sampling_frequency_parameters["MAD_frequency"] * signal_cropping_parameters["signal_length_seconds"]),
    }

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]
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
    default_project_configuration.update(filter_gif_data_parameters)

    del sampling_frequency_parameters, signal_cropping_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_gif_data_parameters

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

    one_second_network = {
        "signal_length_seconds": 1,
        "shift_length_seconds_interval": (1, 1),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 1),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 1),
    }

    one_second_hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    ten_second_network = {
        "signal_length_seconds": 10,
        "shift_length_seconds_interval": (10, 10),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 10),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 10),
    }

    ten_second_hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    sixteen_second_network = {
        "signal_length_seconds": 16,
        "shift_length_seconds_interval": (16, 16),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 16),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 16),
    }

    sixteen_second_hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 40,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    cleaning_adjustments = [raw, cleaned, norm]
    cleaning_names = ["RAW", "Cleaned", "Norm"]

    network_adjustments = [one_second_network, ten_second_network, sixteen_second_network]
    gif_hyperparameter_adjustments = [one_second_hyperparameters_gif, ten_second_hyperparameters_gif, sixteen_second_hyperparameters_gif]
    network_names = ["Local_1s", "Local_10s", "Local_16s"]

    # different networks have different signal cropping parameters, so we need to create a database for each network
    gif_directory_paths = ["1s_GIF_SAE_Data/", "10s_GIF_SAE_Data/", "16s_GIF_SAE_Data/"]

    for net_adjust_index in range(len(network_adjustments)):
        copy_and_split_default_database_SAE(
            path_to_default_gif_database = default_complete_gif_SSG_path,
            path_to_save_gif_database = gif_directory_paths[net_adjust_index],
            project_configuration = network_adjustments[net_adjust_index]
        )

    for clean_index in range(len(cleaning_adjustments)):
        project_configuration = copy.deepcopy(default_project_configuration)
        project_configuration.update(cleaning_adjustments[clean_index])
    
        for network_index in range(len(network_adjustments)):
            project_configuration.update(network_adjustments[network_index])

            identifier = "SAE_" + network_names[network_index] + "_" + cleaning_names[clean_index]
            print_headline("Running " + identifier, "=")

            identifier += "/"

            main_pipeline_SAE(
                project_configuration = project_configuration,
                path_to_model_directory = identifier,
                neural_network_hyperparameters_gif = gif_hyperparameter_adjustments[network_index],
                path_to_gif_database = gif_directory_paths[network_index],
            )


if __name__ == "__main__":
    build_default_datasets_for_training_and_testing()

    train_and_test_long_sequence_model_on_sleep_staging_data()
    train_and_test_short_sequence_model_on_sleep_staging_data()

    train_and_test_long_sequence_model_on_apnea_events()
    train_and_test_short_sequence_model_on_apnea_events()

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
        "shift_length_seconds_interval": (30, 30)
    }

    gif_data_manager.crop_oversized_data(**signal_cropping_parameters)
    shhs_data_manager.crop_oversized_data(**signal_cropping_parameters)

    print(len(shhs_data_manager), len(shhs_training_data_manager), len(shhs_validation_data_manager))
    print(len(gif_data_manager), len(gif_training_data_manager), len(gif_validation_data_manager))

    # -----------------------------------------------------------

    data_manager = SleepDataManager(directory_path = default_shhs_path, pid="train")
    count = 0
    for data in data_manager:
        count += 1

    print(count, data_manager.database_configuration["number_datapoints"])

    data_manager = SleepDataManager(directory_path = "10h_SHHS_Data/", pid="train")

    count = 0
    for data in data_manager:
        count += 1

    print(count, data_manager.database_configuration["number_datapoints"])

    raise SystemExit