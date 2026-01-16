from main import *
from plot_helper import *
from matplotlib.colors import ListedColormap
import shutil

import os
import io
import sys

global_send_email = False

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


def collect_performance_vs_size_multi(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        test_signal_length_seconds = 36000
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
        else "mps"
        if torch.backends.mps.is_available()
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

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]
    
    slp_duration_seconds = int(1 / slp_frequency)

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = len(data_manager))
    
    count_segments = 0
    all_predicted = []
    all_actual = []

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:
            
            try:
                total_duration = int(len(data_dict["RRI"])/rri_frequency)

                if actual_results_available:
                    actual_original_structure = map_slp_labels(
                        slp_labels = copy.deepcopy(data_dict["SLP"]), # type: ignore
                        slp_label_mapping = slp_label_mapping
                    )
                    original_signal_length = int(len(copy.deepcopy(actual_original_structure))/slp_frequency)
                else:
                    original_signal_length = int(np.ceil(signal_length_seconds))

                start_time = -test_signal_length_seconds
                upper_bound = 0
                # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                while upper_bound < total_duration:
                    strided_prediction_probabilities = [[] for i in range(test_signal_length_seconds)]
                    strided_predicted_classes = [[] for _ in range(test_signal_length_seconds)]
                    use_seconds_length = copy.deepcopy(test_signal_length_seconds)

                    start_time += test_signal_length_seconds
                    upper_bound = start_time + test_signal_length_seconds
                    if upper_bound > total_duration:
                        upper_bound = total_duration
                        start_time = upper_bound - test_signal_length_seconds
                    if start_time < 0: # happens when signal length is longer than total duration
                        start_time = 0
                        upper_bound = original_signal_length

                        strided_prediction_probabilities = [[] for i in range(original_signal_length)]
                        strided_predicted_classes = [[] for _ in range(original_signal_length)]
                        use_seconds_length = copy.deepcopy(original_signal_length)

                    """
                    Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                    """

                    rri = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["RRI"][int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
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
                            signal = copy.deepcopy(data_dict["MAD"][int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
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
                
                    """
                    Applying Neural Network Model
                    """

                    # predictions in windows
                    if reshape_to_overlapping_windows:
                        predictions_probability_in_windows = neural_network_model(rri, mad)

                        """
                        Preparing Predicted Sleep Phases
                        """

                        predictions_probability_in_windows = predictions_probability_in_windows.cpu().numpy()
                        predictions_in_windows = predictions_probability_in_windows.argmax(1)

                        for i in range(len(predictions_probability_in_windows)):
                            for j in range(window_duration_seconds):
                                this_index = int(i*(window_duration_seconds - overlap_seconds)) + j # type: ignore
                                if this_index >= use_seconds_length:
                                    break
                                strided_prediction_probabilities[this_index].append(predictions_probability_in_windows[i])
                                strided_predicted_classes[this_index].append(predictions_in_windows[i])
                    
                    # predictions not in windows
                    else:
                        raise SystemError("what are you doing?")
                        predictions_probability = neural_network_model(rri, mad)

                        predictions_probability = predictions_probability.cpu().numpy()
                        predicted = predictions_probability.argmax(1)

                        for i in range(start_time, upper_bound):
                            strided_prediction_probabilities[i].append(predictions_probability[0])
                            strided_predicted_classes[i].append(predicted[0])
                
                    # combine strided predictions
                    resolution_seconds = 30
                    combined_predicted_probabilities = []
                    combined_predicted_classes = []
                    for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                        collected_probabilities = list()
                        collected_classes = list()
                        max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))

                        for j in range(i, max_bound):
                            collected_probabilities.extend(strided_prediction_probabilities[j])
                            collected_classes.extend(strided_predicted_classes[j])
                        
                        combined_predicted_probabilities.append(collected_probabilities)
                        combined_predicted_classes.append(collected_classes)

                    mean_combined_prediction_probabilities = []
                    for i in range(len(combined_predicted_probabilities)):
                        mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                    # print(mean_combined_prediction_probabilities)
                    mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)

                    predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                    
                    predictions_from_combined_classes = list()
                    for row in combined_predicted_classes:
                        values, counts = np.unique(row, return_counts=True)
                        predictions_from_combined_classes.append(values[np.argmax(counts)])
                    predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                    if actual_results_available:
                        if slp_duration_seconds == resolution_seconds:
                            slp = copy.deepcopy(actual_original_structure)[int(start_time*slp_frequency):int(upper_bound*slp_frequency)]

                            if len(slp) != len(predictions_from_combined_probabilities):
                                raise ValueError("Length of actual sleep stages and predicted sleep stages do not match after rescaling.")
                        else:
                            raise SystemError("dudidud...")
                            slp = scale_classification_signal(
                                signal = actual_original_structure, # type: ignore
                                signal_frequency = slp_frequency,
                                target_frequency = 1/resolution_seconds
                            )

                            if len(slp) != len(predictions_from_combined_probabilities):
                                crop_to = min(len(slp), len(predictions_from_combined_probabilities))
                                slp = slp[:crop_to]
                                mean_combined_prediction_probabilities = mean_combined_prediction_probabilities[:crop_to]
                                predictions_from_combined_probabilities = predictions_from_combined_probabilities[:crop_to]
                                predictions_from_combined_classes = predictions_from_combined_classes[:crop_to]


                        # save results to new dictionary
                        # results = {
                        #     "Predicted_Probabilities": mean_combined_prediction_probabilities,
                        #     "Predicted": predictions_from_combined_probabilities,
                        #     "Predicted_2": predictions_from_combined_classes,
                        #     "Actual": slp,
                        # }

                        all_predicted.extend(list(predictions_from_combined_probabilities))
                        all_actual.extend(list(slp))
                        count_segments += 1

                    else:
                        raise SystemError("TF dude?")
                        # save results to existing dictionary
                        results = copy.deepcopy(data_dict)
                        results["SLP_prediction_probability"] = mean_combined_prediction_probabilities
                        results["SLP_from_prob"] = predictions_from_combined_probabilities
                        results["SLP_prediction_classes"] = combined_predicted_classes
                        results["SLP_from_class"] = predictions_from_combined_classes
                        results["SLP"] = predictions_from_combined_probabilities
            
            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore
                continue

            finally:        
                # update progress
                progress_bar.update()

    # Calculate and print accuracy and cohen's kappa score
    additional_score_function_args: dict = {"zero_division": np.nan, "average": "macro"}
    accuracy = accuracy_score(all_actual, all_predicted)
    kappa = cohen_kappa_score(all_actual, all_predicted)
    f1 = f1_score(all_actual, all_predicted, **additional_score_function_args)
    
    results_file = open(path_to_save_results, "ab")
    results = {
        "length": test_signal_length_seconds,
        "count": count_segments,
        "accuracy": accuracy,
        "kappa": kappa,
        "f1": f1,
    }
    pickle.dump(results, results_file)
    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)
    
    print(test_signal_length_seconds, count_segments, accuracy, kappa, f1)


def collect_performance_vs_size_single(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        test_signal_length_seconds = 36000
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
        else "mps"
        if torch.backends.mps.is_available()
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

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    stride_seconds = int(signal_length_seconds / 4)
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]
    
    slp_duration_seconds = int(1 / slp_frequency)

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Sleep Stages:")
    progress_bar = DynamicProgressBar(total = len(data_manager))

    all_predicted = []
    all_actual = []

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:

            data_total_duration = int(len(data_dict["SLP"])/slp_frequency)

            data_start_time = -test_signal_length_seconds
            data_upper_bound = 0
            
            while data_upper_bound < data_total_duration:
                data_start_time += test_signal_length_seconds
                data_upper_bound = data_start_time + test_signal_length_seconds
                if data_upper_bound > data_total_duration:
                    data_upper_bound = data_total_duration
                    data_start_time = data_upper_bound - test_signal_length_seconds
                if data_start_time < 0: # happens when signal length is longer than total duration
                    data_start_time = 0
                    data_upper_bound = data_total_duration
            
                new_rri = copy.deepcopy(data_dict["RRI"][int(data_start_time*rri_frequency):int(data_upper_bound*rri_frequency)])
                new_mad = copy.deepcopy(data_dict["MAD"][int(data_start_time*mad_frequency):int(data_upper_bound*mad_frequency)])
                new_slp = copy.deepcopy(data_dict["SLP"][int(data_start_time*slp_frequency):int(data_upper_bound*slp_frequency)])
            
                try:
                    total_duration = int(len(new_slp)/slp_frequency)

                    strided_prediction_probabilities = [[] for i in range(total_duration)]
                    strided_predicted_classes = [[] for _ in range(total_duration)]

                    actual_original_structure = map_slp_labels(
                        slp_labels = copy.deepcopy(new_slp), # type: ignore
                        slp_label_mapping = slp_label_mapping
                    )
                    original_signal_length = len(copy.deepcopy(actual_original_structure))

                    start_time = -stride_seconds
                    upper_bound = 0
                    # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                    while upper_bound < total_duration:
                        start_time += stride_seconds
                        upper_bound = start_time + signal_length_seconds
                        if upper_bound > total_duration:
                            upper_bound = total_duration
                            start_time = upper_bound - signal_length_seconds
                        if start_time < 0: # happens when signal length is longer than total duration
                            start_time = 0
                            upper_bound = signal_length_seconds

                        """
                        Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                        """

                        rri = final_data_preprocessing(
                            signal = copy.deepcopy(new_rri[int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
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
                        mad = final_data_preprocessing(
                            signal = copy.deepcopy(new_mad[int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
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
                    
                        """
                        Applying Neural Network Model
                        """

                        # predictions in windows
                        if reshape_to_overlapping_windows:
                            raise SystemError("dude")
                        # predictions not in windows
                        else:
                            predictions_probability = neural_network_model(rri, mad)

                            predictions_probability = predictions_probability.cpu().numpy()
                            predicted = predictions_probability.argmax(1)

                            for i in range(start_time, upper_bound):
                                strided_prediction_probabilities[i].append(predictions_probability[0])
                                strided_predicted_classes[i].append(predicted[0])
                    
                    # combine strided predictions
                    resolution_seconds = 30
                    combined_predicted_probabilities = []
                    combined_predicted_classes = []
                    for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                        collected_probabilities = list()
                        collected_classes = list()
                        max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))

                        for j in range(i, max_bound):
                            collected_probabilities.extend(strided_prediction_probabilities[j])
                            collected_classes.extend(strided_predicted_classes[j])
                        
                        combined_predicted_probabilities.append(collected_probabilities)
                        combined_predicted_classes.append(collected_classes)

                    mean_combined_prediction_probabilities = []
                    for i in range(len(combined_predicted_probabilities)):
                        mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                    mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)
                    predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                    
                    predictions_from_combined_classes = list()
                    for row in combined_predicted_classes:
                        values, counts = np.unique(row, return_counts=True)
                        predictions_from_combined_classes.append(values[np.argmax(counts)])
                    predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                    if actual_results_available:
                        if slp_duration_seconds == resolution_seconds:
                            slp = actual_original_structure

                            if len(slp) != len(predictions_from_combined_probabilities):
                                raise ValueError("Length of actual sleep stages and predicted sleep stages do not match after rescaling.")
                        else:
                            slp = scale_classification_signal(
                                signal = actual_original_structure, # type: ignore
                                signal_frequency = slp_frequency,
                                target_frequency = 1/resolution_seconds
                            )

                            if len(slp) != len(predictions_from_combined_probabilities):
                                crop_to = min(len(slp), len(predictions_from_combined_probabilities))
                                slp = slp[:crop_to]
                                mean_combined_prediction_probabilities = mean_combined_prediction_probabilities[:crop_to]
                                predictions_from_combined_probabilities = predictions_from_combined_probabilities[:crop_to]
                                predictions_from_combined_classes = predictions_from_combined_classes[:crop_to]


                        # save results to new dictionary
                        all_predicted.extend(list(predictions_from_combined_probabilities))
                        all_actual.extend(list(slp))
                
                except:
                    unpredictable_signals.append(data_dict["ID"]) # type: ignore
                    continue

            progress_bar.update()
    
    additional_score_function_args: dict = {"zero_division": np.nan, "average": "macro"}
    accuracy = accuracy_score(all_actual, all_predicted)
    kappa = cohen_kappa_score(all_actual, all_predicted)
    f1 = f1_score(all_actual, all_predicted, **additional_score_function_args)

    results_file = open(path_to_save_results, "ab")
    results = {
        "length": test_signal_length_seconds,
        "accuracy": accuracy,
        "kappa": kappa,
        "f1": f1,
    }
    pickle.dump(results, results_file)
    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)
    
    print(test_signal_length_seconds, accuracy, kappa, f1)


def calc_perf_values(
        paths_to_pkl_files: list,
        path_to_project_configuration: str,
        prediction_result_key: str,
        actual_result_key: str,
        transform = [],
        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
    ):
    
    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access dictionary that maps sleep stages (display labels) to integers
    sleep_stage_to_label = project_configuration["target_classes"]

    # Create a list of the integer labels, sorted
    integer_labels = np.array([value for value in sleep_stage_to_label.values()])
    integer_labels = np.unique(integer_labels)
    integer_labels.sort()

    # Create a list of the display labels
    display_labels = []
    for integer_label in integer_labels:
        for key, value in sleep_stage_to_label.items():
            if value == integer_label:
                display_labels.append(key)
                break
    
    # append labels to additional_score_function_args
    additional_score_function_args["labels"] = integer_labels

    # variables to store results
    all_predicted_results = np.empty(0)
    all_actual_results = np.empty(0)

    for file_path in paths_to_pkl_files:
        # Load the data
        data_generator = load_from_pickle(file_path)

        for data in data_generator:
            
            # Get the predicted and actual results
            predicted_results = data[prediction_result_key]
            actual_results = data[actual_result_key]

            # Flatten the arrays
            # print(predicted_results.shape, actual_results.shape)
            predicted_results = predicted_results.flatten()
            actual_results = actual_results.flatten()

            # Add the results to the arrays
            all_predicted_results = np.append(all_predicted_results, predicted_results)
            all_actual_results = np.append(all_actual_results, actual_results)
    
    if len(transform) > 0:
        for i in range(len(all_predicted_results)):
            for j in range(len(transform)):
                if all_predicted_results[i] == transform[j][0]:
                    all_predicted_results[i] = transform[j][1]
                    break
        for i in range(len(all_predicted_results)):
            for j in range(len(transform)):
                if all_actual_results[i] == transform[j][0]:
                    all_actual_results[i] = transform[j][1]
                    break

    # Calculate and print accuracy and cohen's kappa score
    accuracy = accuracy_score(all_actual_results, all_predicted_results)
    kappa = cohen_kappa_score(all_actual_results, all_predicted_results)
    f1 = f1_score(all_actual_results, all_predicted_results, **additional_score_function_args)
    precision = precision_score(all_actual_results, all_predicted_results, **additional_score_function_args)
    recall = recall_score(all_actual_results, all_predicted_results, **additional_score_function_args)

    return accuracy, kappa, f1, precision, recall


def single_apnea_chb_collecting(
        parent_folder_path: str,
        results_file_path: str
    ):

    cleaning_names = ["RAW", "Cleaned", "Norm"]
    
    network_names = ["30s", "60s", "120s", "180s"]
    network_seconds = [30, 60, 120, 180]
    
    class_names = ["A", "AH"]
    
    results_file = open(results_file_path, "ab")

    print("\nSingle Apnea")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        
        for class_index in range(len(class_names)):
            print(class_names[class_index])
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]
            
            for clean_index in range(len(cleaning_names)):
                print(cleaning_names[clean_index])
                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                path_to_model_directory = parent_folder_path + identifier

                # path to save the predictions
                gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
                
                try:
                    accuracy, kappa, f1, precision, recall = calc_perf_values(
                        paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
                        path_to_project_configuration = path_to_model_directory + project_configuration_file,
                        prediction_result_key = "Predicted",
                        actual_result_key = "Actual",
                        transform = apnea_transform,
                        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                    )

                    results = {
                        "method": "independent",
                        "task": "apnea",
                        "architecture": "single",
                        "input_seconds": network_seconds[network_index],
                        "window_seconds": 0,
                        "overlap_seconds": 0,
                        "labeling": class_names[class_index],
                        "transform": cleaning_names[clean_index],
                        "accuracy": accuracy,
                        "kappa": kappa,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall
                    }

                    pickle.dump(results, results_file)
                
                except:
                    print("Folder not found.")
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        
        for class_index in range(len(class_names)):
            print(class_names[class_index])
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]
            
            for clean_index in range(len(cleaning_names)):
                print(cleaning_names[clean_index])
                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                path_to_model_directory = parent_folder_path + identifier

                # path to save the predictions
                gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

                try:
                    accuracy, kappa, f1, precision, recall = calc_perf_values(
                        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
                        path_to_project_configuration = path_to_model_directory + project_configuration_file,
                        prediction_result_key = "Predicted_2",
                        actual_result_key = "Actual",
                        transform = apnea_transform,
                        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                    )

                    results = {
                        "method": "practical",
                        "task": "apnea",
                        "architecture": "single",
                        "input_seconds": network_seconds[network_index],
                        "window_seconds": 0,
                        "overlap_seconds": 0,
                        "labeling": class_names[class_index],
                        "transform": cleaning_names[clean_index],
                        "accuracy": accuracy,
                        "kappa": kappa,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall
                    }

                    pickle.dump(results, results_file)
                
                except:
                    print("Folder not found.")
    
    results_file.close()


def multi_apnea_chb_collecting(
        parent_folder_path: str,
        results_file_path: str
    ):

    network_model_names = ["LSM", "LSM_Residual"]
    data_structure_names = ["300s_10s_0s", "300s_10s_5s"]
    class_names = ["A", "AH"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    
    results_file = open(results_file_path, "ab")

    print("\nMulti Apnea")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_model_names)):
        print(network_model_names[network_index])

        for data_struct_index in range(len(data_structure_names)):
            print(data_structure_names[data_struct_index])
            if data_structure_names[data_struct_index] == "300s_10s_0s":
                overlap = 0
            else:
                overlap = 5
        
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    apnea_transform = []
                else:
                    apnea_transform = [[2,1]]
                
                for clean_index in range(len(cleaning_names)):
                    print(cleaning_names[clean_index])
                    identifier = "SAE_" + network_model_names[network_index] + "_" + data_structure_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    path_to_model_directory = parent_folder_path + identifier

                    # path to save the predictions
                    gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
                    
                    try:
                        accuracy, kappa, f1, precision, recall = calc_perf_values(
                            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
                            path_to_project_configuration = path_to_model_directory + project_configuration_file,
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            transform = apnea_transform,
                            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                        )

                        results = {
                            "method": "independent",
                            "task": "apnea",
                            "architecture": network_model_names[network_index],
                            "input_seconds": 300,
                            "window_seconds": 10,
                            "overlap_seconds": overlap,
                            "labeling": class_names[class_index],
                            "transform": cleaning_names[clean_index],
                            "accuracy": accuracy,
                            "kappa": kappa,
                            "f1": f1,
                            "precision": precision,
                            "recall": recall
                        }

                        pickle.dump(results, results_file)
                    
                    except:
                        print("Folder not found.")
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_model_names)):
        print(network_model_names[network_index])

        for data_struct_index in range(len(data_structure_names)):
            print(data_structure_names[data_struct_index])
            if data_structure_names[data_struct_index] == "300s_10s_0s":
                overlap = 0
            else:
                overlap = 5
        
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    apnea_transform = []
                else:
                    apnea_transform = [[2,1]]
                
                for clean_index in range(len(cleaning_names)):
                    print(cleaning_names[clean_index])
                    identifier = "SAE_" + network_model_names[network_index] + "_" + data_structure_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    path_to_model_directory = parent_folder_path + identifier

                    # path to save the predictions
                    gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"
                    
                    try:
                        accuracy, kappa, f1, precision, recall = calc_perf_values(
                            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
                            path_to_project_configuration = path_to_model_directory + project_configuration_file,
                            prediction_result_key = "Predicted_2",
                            actual_result_key = "Actual",
                            transform = apnea_transform,
                            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                        )

                        results = {
                            "method": "practical",
                            "task": "apnea",
                            "architecture": network_model_names[network_index],
                            "input_seconds": 300,
                            "window_seconds": 10,
                            "overlap_seconds": overlap,
                            "labeling": class_names[class_index],
                            "transform": cleaning_names[clean_index],
                            "accuracy": accuracy,
                            "kappa": kappa,
                            "f1": f1,
                            "precision": precision,
                            "recall": recall
                        }

                        pickle.dump(results, results_file)
                    
                    except:
                        print("Folder not found.")
    
    results_file.close()


def single_stage_chb_collecting(
        parent_folder_path: str,
        results_file_path: str
    ):

    class_names = ["ArtifactAsWake", "FullClass"]
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    network_names = ["30s", "60s", "120s", "180s"]
    network_seconds = [30, 60, 120, 180]
    sleep_transform = []

    results_file = open(results_file_path, "ab")

    print("\nSingle Stage")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        
        for class_index in range(len(class_names)):
            print(class_names[class_index])
            if class_names[class_index] == "ArtifactAsWake":
                sleep_transform = []
            else:
                sleep_transform = [[0, 1]]
            

            for clean_index in range(len(cleaning_names)):
                print(cleaning_names[clean_index])
                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                path_to_model_directory = parent_folder_path + identifier

                # path to save the predictions
                gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"

                try:
                    accuracy, kappa, f1, precision, recall = calc_perf_values(
                        paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
                        path_to_project_configuration = path_to_model_directory + project_configuration_file,
                        prediction_result_key = "Predicted",
                        actual_result_key = "Actual",
                        transform = sleep_transform,
                        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                    )

                    results = {
                        "method": "independent",
                        "task": "stage",
                        "architecture": "single",
                        "input_seconds": network_seconds[network_index],
                        "window_seconds": 0,
                        "overlap_seconds": 0,
                        "labeling": class_names[class_index],
                        "transform": cleaning_names[clean_index],
                        "accuracy": accuracy,
                        "kappa": kappa,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall
                    }

                    pickle.dump(results, results_file)
                
                except:
                    print("Folder not found.")
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        
        for class_index in range(len(class_names)):
            print(class_names[class_index])
            if class_names[class_index] == "ArtifactAsWake":
                sleep_transform = []
            else:
                sleep_transform = [[0, 1]]    
                
            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                path_to_model_directory = parent_folder_path + identifier
                print(cleaning_names[clean_index])

                # path to save the predictions
                gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

                try:
                    accuracy, kappa, f1, precision, recall = calc_perf_values(
                        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
                        path_to_project_configuration = path_to_model_directory + project_configuration_file,
                        prediction_result_key = "Predicted",
                        actual_result_key = "Actual",
                        transform = sleep_transform,
                        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                    )

                    results = {
                        "method": "practical",
                        "task": "stage",
                        "architecture": "single",
                        "input_seconds": network_seconds[network_index],
                        "window_seconds": 0,
                        "overlap_seconds": 0,
                        "labeling": class_names[class_index],
                        "transform": cleaning_names[clean_index],
                        "accuracy": accuracy,
                        "kappa": kappa,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall
                    }

                    pickle.dump(results, results_file)

                except:
                    print("Folder not found.")
    
    results_file.close()


def multi_stage_chb_collecting(
        parent_folder_path: str,
        results_file_path: str
    ):

    data_structure_names = ["10h_120s_0s", "10h_120s_90s"]
    class_names = ["ArtifactAsWake", "FullClass"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    network_model_names = ["LSM", "LSM_Residual"]
    sleep_transform = []

    results_file = open(results_file_path, "ab")

    print("\nMulti Stage")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_model_names)):
        print(network_model_names[network_index])

        for data_struct_index in range(len(data_structure_names)):
            print(data_structure_names[data_struct_index])

            if data_structure_names[data_struct_index] == "10h_120s_0s":
                overlap = 0
            else:
                overlap = 90
        
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "ArtifactAsWake":
                    sleep_transform = []
                else:
                    sleep_transform = [[0, 1]]
                
                for clean_index in range(len(cleaning_names)):
                    print(cleaning_names[clean_index])
                    identifier = "SSG_" + network_model_names[network_index] + "_" + data_structure_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    path_to_model_directory = parent_folder_path + identifier

                    # path to save the predictions
                    gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"

                    try:
                        accuracy, kappa, f1, precision, recall = calc_perf_values(
                            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
                            path_to_project_configuration = path_to_model_directory + project_configuration_file,
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            transform = sleep_transform,
                            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                        )

                        results = {
                            "method": "independent",
                            "task": "stage",
                            "architecture": network_model_names[network_index],
                            "input_seconds": 36000,
                            "window_seconds": 120,
                            "overlap_seconds": overlap,
                            "labeling": class_names[class_index],
                            "transform": cleaning_names[clean_index],
                            "accuracy": accuracy,
                            "kappa": kappa,
                            "f1": f1,
                            "precision": precision,
                            "recall": recall
                        }

                        pickle.dump(results, results_file)
                    
                    except:
                        print("Folder not found.")
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_model_names)):
        print(network_model_names[network_index])

        for data_struct_index in range(len(data_structure_names)):
            print(data_structure_names[data_struct_index])

            if data_structure_names[data_struct_index] == "10h_120s_0s":
                overlap = 0
            else:
                overlap = 90
        
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "ArtifactAsWake":
                    sleep_transform = []
                else:
                    sleep_transform = [[0, 1]]
                
                for clean_index in range(len(cleaning_names)):
                    print(cleaning_names[clean_index])
                    identifier = "SSG_" + network_model_names[network_index] + "_" + data_structure_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    path_to_model_directory = parent_folder_path + identifier

                    # path to save the predictions
                    gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

                    try:
                        accuracy, kappa, f1, precision, recall = calc_perf_values(
                            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
                            path_to_project_configuration = path_to_model_directory + project_configuration_file,
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            transform = sleep_transform,
                            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                        )

                        results = {
                            "method": "practical",
                            "task": "stage",
                            "architecture": network_model_names[network_index],
                            "input_seconds": 36000,
                            "window_seconds": 120,
                            "overlap_seconds": overlap,
                            "labeling": class_names[class_index],
                            "transform": cleaning_names[clean_index],
                            "accuracy": accuracy,
                            "kappa": kappa,
                            "f1": f1,
                            "precision": precision,
                            "recall": recall
                        }

                        pickle.dump(results, results_file)
                    
                    except:
                        print("Folder not found.")
    
    results_file.close()


def load_from_pickle(file_name: str):
    """
    Load data from a pickle file as a generator.

    RETURNS:
    ------------------------------
    any
        data from the pickle file

    ARGUMENTS:
    ------------------------------
    file_name: str
        path to the pickle file
    key: str
        key of the data to be loaded
    """

    with open(file_name, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except:
                break


def get_mean_of(
        results_file_path: str,
        combinations: dict,
        print_results = False
    ):
    """
    """

    results_generator = load_from_pickle(results_file_path)

    collected_accuracy = []
    collected_kappa = []
    collected_f1 = []
    collected_precision = []
    collected_recall = []

    for data_dict in results_generator:
        all_align = True
        for key in combinations:
            if data_dict[key] not in combinations[key]:
                all_align = False
                break
        
        if not all_align:
            continue

        collected_accuracy.append(data_dict["accuracy"])
        collected_kappa.append(data_dict["kappa"])
        collected_f1.append(data_dict["f1"])
        collected_precision.append(data_dict["precision"])
        collected_recall.append(data_dict["recall"])
    
    if print_results:
        print(f"Number of networks with wanted combination: {len(collected_accuracy)}")
        print(f"\nMean Accuracy: {np.mean(collected_accuracy)} +- {np.std(collected_accuracy)}")
        print(f"\nMean Kappa: {np.mean(collected_kappa)} +- {np.std(collected_kappa)}")
        print(f"\nMean F1: {np.mean(collected_f1)} +- {np.std(collected_f1)}")
        print(f"Mean Precision: {np.mean(collected_precision)} +- {np.std(collected_precision)}")
        print(f"Mean Recall: {np.mean(collected_recall)} +- {np.std(collected_recall)}")
    else:
        return np.mean(collected_accuracy), np.std(collected_accuracy), np.mean(collected_kappa), np.std(collected_kappa), np.mean(collected_f1), np.std(collected_f1), len(collected_accuracy)


def compare_table(
        results_file_path: str,
        combinations: list,
        combination_label: list,
        round_to_decimals = 3
    ):
    """
    """

    for comb_index in range(len(combinations)):
        combination = combinations[comb_index]
        mean_acc, std_acc, mean_kappa, std_kappa, mean_f1, std_f1, num = get_mean_of(results_file_path = results_file_path, combinations = combination, print_results = False) # type: ignore
        print(f"{combination_label[comb_index]} & {num} & \\num" + "{" + f"{round(mean_acc, round_to_decimals)}({round(std_acc, round_to_decimals)})" + "} & \\num" + "{" + f"{round(mean_kappa, round_to_decimals)}({round(std_kappa, round_to_decimals)})" + "} & \\num" + "{" + f"{round(mean_f1, round_to_decimals)}({round(std_f1, round_to_decimals)})" + "} \\\\")


def print_compare_tables(path_to_results_file: str):
    
    possible_combinations = {
        "method": ["practical"], # "independent" or "practical"
        "task": ["stage"], # "stage" or "apnea"
        "architecture": ["single"], # "single", "LSM" or "LSM_Residual"
        "input_seconds": [30], # 30, 60, 120, 180, 300, 36000
        "window_seconds": [0], # 0, 10, 120
        "overlap_seconds": [0], # 0, 5, 90
        "labeling": ["ArtifactAsWake"], # "ArtifactAsWake", "FullClass", "A" or "AH"
        "transform": ["RAW"], # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    # single sleep stage

    combination = {
        "method": ["practical"], # "independent" or "practical"
        "task": ["stage"], # "stage" or "apnea"
        "architecture": ["single"], # "single", "LSM" or "LSM_Residual"
        # "input_seconds": [30], # 30, 60, 120, 180, 300, 36000
        # "window_seconds": [0], # 0, 10, 120
        # "overlap_seconds": [0], # 0, 5, 90
        # "labeling": ["ArtifactAsWake"], # "ArtifactAsWake", "FullClass", "A" or "AH"
        # "transform": ["RAW"], # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    all_combinations = list()
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [30] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [60] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [120] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [180] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    del combination["input_seconds"]
    combination["labeling"] = ["ArtifactAsWake"]
    all_combinations.append(copy.deepcopy(combination))
    combination["labeling"] = ["FullClass"]
    all_combinations.append(copy.deepcopy(combination))
    del combination["labeling"]
    combination["transform"] = ["RAW"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["Cleaned"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["Norm"]
    all_combinations.append(copy.deepcopy(combination))
    
    combination_labels = ["Any", "\\mdhighlight{\\qty{30}{s}}", "\\mdhighlight{\\qty{60}{s}}", "\\mdhighlight{\\qty{120}{s}}", "\\mdhighlight{\\qty{180}{s}}", "\\mdhighlight{W\\&A,\\,L,\\,D,\\,R}", "\\mdhighlight{W,\\,L,\\,D,\\,R,\\,A}", "\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}"]

    compare_table(
        results_file_path = path_to_results_file,
        combinations = all_combinations,
        combination_label = combination_labels,
        round_to_decimals = 3
    )

    # multi sleep stage

    combination = {
        "method": ["practical"], # "independent" or "practical"
        "task": ["stage"], # "stage" or "apnea"
        "architecture": ["LSM", "LSM_Residual"], # "single", "LSM" or "LSM_Residual"
        "input_seconds": [36000], # 30, 60, 120, 180, 300, 36000
        "window_seconds": [120], # 0, 10, 120
        # "overlap_seconds": [0], # 0, 5, 90
        # "labeling": ["ArtifactAsWake"], # "ArtifactAsWake", "FullClass", "A" or "AH"
        # "transform": ["RAW"], # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    all_combinations = list()
    all_combinations.append(copy.deepcopy(combination))
    
    combination["architecture"] = ["LSM"]
    all_combinations.append(copy.deepcopy(combination))
    combination["architecture"] = ["LSM_Residual"]
    all_combinations.append(copy.deepcopy(combination))

    del combination["architecture"]
    combination["overlap_seconds"] = [90] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["overlap_seconds"] = [0] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    
    del combination["overlap_seconds"]
    combination["labeling"] = ["ArtifactAsWake"]
    all_combinations.append(copy.deepcopy(combination))
    combination["labeling"] = ["FullClass"]
    all_combinations.append(copy.deepcopy(combination))
    
    del combination["labeling"]
    combination["transform"] = ["RAW"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["Cleaned"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["GlobalNorm"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["LocalNorm"]
    all_combinations.append(copy.deepcopy(combination))
    
    combination_labels = ["Any", "Form A", "Form B", "\\mdhighlight{\\qty{10}{h}:\\qty{120}{s}:\\qty{90}{s}}", "\\mdhighlight{\\qty{10}{h}:\\qty{120}{s}:\\qty{0}{s}}", "\\mdhighlight{W\\&A,\\,L,\\,D,\\,R}", "\\mdhighlight{W,\\,L,\\,D,\\,R,\\,A}", "\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}", "\\mdhighlight{WindowNorm}"]

    compare_table(
        results_file_path = path_to_results_file,
        combinations = all_combinations,
        combination_label = combination_labels,
        round_to_decimals = 3
    )

    # single sleep apnea

    combination = {
        "method": ["practical"], # "independent" or "practical"
        "task": ["apnea"], # "stage" or "apnea"
        "architecture": ["single"], # "single", "LSM" or "LSM_Residual"
        # "input_seconds": [30], # 30, 60, 120, 180, 300, 36000
        # "window_seconds": [0], # 0, 10, 120
        # "overlap_seconds": [0], # 0, 5, 90
        # "labeling": ["ArtifactAsWake"], # "ArtifactAsWake", "FullClass", "A" or "AH"
        # "transform": ["RAW"], # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    all_combinations = list()
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [30] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [60] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [120] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = [180] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    del combination["input_seconds"]
    combination["labeling"] = ["A"]
    all_combinations.append(copy.deepcopy(combination))
    combination["labeling"] = ["AH"]
    all_combinations.append(copy.deepcopy(combination))
    del combination["labeling"]
    combination["transform"] = ["RAW"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["Cleaned"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["Norm"]
    all_combinations.append(copy.deepcopy(combination))
    
    combination_labels = ["Any", "\\mdhighlight{\\qty{30}{s}}", "\\mdhighlight{\\qty{60}{s}}", "\\mdhighlight{\\qty{120}{s}}", "\\mdhighlight{\\qty{180}{s}}", "\\mdhighlight{N,\\,A\\&H}", "\\mdhighlight{N,\\,A,\\,H}", "\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}"]

    compare_table(
        results_file_path = path_to_results_file,
        combinations = all_combinations,
        combination_label = combination_labels,
        round_to_decimals = 3
    )

    # multi sleep apnea

    combination = {
        "method": ["practical"], # "independent" or "practical"
        "task": ["apnea"], # "stage" or "apnea"
        "architecture": ["LSM", "LSM_Residual"], # "single", "LSM" or "LSM_Residual"
        "input_seconds": [300], # 30, 60, 120, 180, 300, 36000
        "window_seconds": [10], # 0, 10, 120
        # "overlap_seconds": [0], # 0, 5, 90
        # "labeling": ["ArtifactAsWake"], # "ArtifactAsWake", "FullClass", "A" or "AH"
        # "transform": ["RAW"], # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    all_combinations = list()
    all_combinations.append(copy.deepcopy(combination))
    
    combination["architecture"] = ["LSM"]
    all_combinations.append(copy.deepcopy(combination))
    combination["architecture"] = ["LSM_Residual"]
    all_combinations.append(copy.deepcopy(combination))
    del combination["architecture"]
    
    combination["overlap_seconds"] = [0] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["overlap_seconds"] = [5] # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    del combination["overlap_seconds"]
    
    combination["labeling"] = ["A"]
    all_combinations.append(copy.deepcopy(combination))
    combination["labeling"] = ["AH"]
    all_combinations.append(copy.deepcopy(combination))
    del combination["labeling"]
    
    combination["transform"] = ["RAW"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["Cleaned"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["GlobalNorm"]
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = ["LocalNorm"]
    all_combinations.append(copy.deepcopy(combination))
    
    combination_labels = ["Any", "Form A", "Form B", "\\mdhighlight{\\qty{300}{s}:\\qty{10}{s}:\\qty{0}{s}}", "\\mdhighlight{\\qty{300}{s}:\\qty{10}{s}:\\qty{5}{s}}", "\\mdhighlight{N,\\,A\\&H}", "\\mdhighlight{N,\\,A,\\,H}", "\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}", "\\mdhighlight{WindowNorm}"]

    compare_table(
        results_file_path = path_to_results_file,
        combinations = all_combinations,
        combination_label = combination_labels,
        round_to_decimals = 3
    )


def loss_per_epoch_collecting(
        parent_folder_path: str,
        results_file_path: str,
        task_network: str,
        best_model_folder: str,
    ):

    all_folders = [folder for folder in os.listdir(parent_folder_path) if os.path.exists(parent_folder_path + folder + "/Loss_per_Epoch_GIF.pkl")]

    training_loss = []
    shhs_loss = []
    chb_loss = []

    best_training_loss = []
    best_shhs_loss = []
    best_chb_loss = []

    for folder in all_folders:

        folder += "/"

        with open(parent_folder_path + folder + "Loss_per_Epoch_GIF.pkl", "rb") as f:
            chb_data = pickle.load(f)
        
        if os.path.exists(parent_folder_path + folder + "Loss_per_Epoch_SHHS.pkl"):
            with open(parent_folder_path + folder + "Loss_per_Epoch_SHHS.pkl", "rb") as f:
                shhs_data = pickle.load(f)
            
            this_training_loss = []
            this_shhs_loss = []
            this_chb_loss = []
            
            this_training_loss.extend(shhs_data["train_avg_loss"])
            this_shhs_loss.extend(shhs_data["SHHS_avg_loss"])
            this_chb_loss.extend(shhs_data["GIF_avg_loss"])

            this_training_loss.extend(chb_data["train_avg_loss"])
            this_shhs_loss.extend(chb_data["SHHS_avg_loss"])
            this_chb_loss.extend(chb_data["GIF_avg_loss"])

            if folder == best_model_folder:
                best_training_loss = copy.deepcopy(this_training_loss)
                best_shhs_loss = copy.deepcopy(this_shhs_loss)
                best_chb_loss = copy.deepcopy(this_chb_loss)
            
            training_loss.append(this_training_loss)
            shhs_loss.append(this_shhs_loss)
            chb_loss.append(this_chb_loss)
        
        else:
            this_training_loss = chb_data["train_avg_loss"]
            this_chb_loss = chb_data["GIF_avg_loss"]

            if folder == best_model_folder:
                best_training_loss = copy.deepcopy(this_training_loss)
                best_chb_loss = copy.deepcopy(this_chb_loss)
        
            training_loss.append(this_training_loss)
            chb_loss.append(this_chb_loss)
    
    if len(best_training_loss) == 0:
        raise SystemError("Best Model not found.")
    
    results = {
        "task_network": task_network,
        "training_loss": training_loss,
        "shhs_loss": shhs_loss,
        "chb_loss": chb_loss,
        "best_model": best_model_folder,
        "best_training_loss": best_training_loss,
        "best_shhs_loss": best_shhs_loss,
        "best_chb_loss": best_chb_loss
    }

    with open(results_file_path, "ab") as f:
        pickle.dump(results, f)


def plot_loss_per_epoch(
        results_file_path: str,
        task_network: str,
        train_border = 100.0,
        shhs_border = 100.0,
        chb_border = 100.0,
        **kwargs
    ):
    """
    """
    
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Epoch")
    kwargs.setdefault("ylabel", "Cross-Entropy Loss")
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    data_generator = load_from_pickle(results_file_path)
    for data_dict in data_generator:
        if data_dict["task_network"] == task_network:
            results = copy.deepcopy(data_dict)
            break
    
    if not results:
        raise SystemError("Task Network COmbination not found.")

    print(f"Task Network: {task_network}")
    print(f"Best Model: {results["best_model"]}")
    norm_border = 1
    
    accumulated_train_loss = []
    accumulated_shhs_loss = []
    accumulated_chb_loss = []

    train_skipped = 0
    for train_loss in results["training_loss"]:
        if np.mean(train_loss) >= train_border:
            train_skipped += 1
            continue
        if len(train_loss) == 80:
            accumulated_train_loss.append(np.array(train_loss))
        elif len(train_loss) == 40 or len(train_loss) == 20:
            interpolated_signal = np.interp(
                np.arange(0, len(train_loss), 0.5),
                np.arange(0, len(train_loss)),
                np.array(train_loss)
                )
            accumulated_train_loss.append(interpolated_signal)
    
    print(f"Skipped {train_skipped} training samples.")
    mean_train_loss = np.mean(accumulated_train_loss, axis=0)
    std_train_loss = np.std(accumulated_train_loss, axis=0)

    train_skipped = 0
    for i in range(1, len(mean_train_loss)-1):
        if std_train_loss[i] > norm_border:
            mean_train_loss[i] = (mean_train_loss[i-1] + mean_train_loss[i+1])/2
            std_train_loss[i] = (std_train_loss[i-1] + std_train_loss[i+1])/2
            train_skipped += 1
    print(f"Normed {train_skipped} train samples.")
    
    if len(results["shhs_loss"]) > 0:
        shhs_skipped = 0
        for shhs_loss in results["shhs_loss"]:
            if np.mean(shhs_loss) > shhs_border:
                shhs_skipped += 1
                continue
            if len(shhs_loss) == 80:
                accumulated_shhs_loss.append(np.array(shhs_loss))
            elif len(shhs_loss) == 40 or len(shhs_loss) == 20:
                interpolated_signal = np.interp(
                    np.arange(0, len(shhs_loss), 0.5),
                    np.arange(0, len(shhs_loss)),
                    np.array(shhs_loss)
                    )
                accumulated_shhs_loss.append(interpolated_signal)
        
        print(f"Skipped {shhs_skipped} shhs samples.")
        mean_shhs_loss = np.mean(accumulated_shhs_loss, axis=0)
        std_shhs_loss = np.std(accumulated_shhs_loss, axis=0)

        shhs_skipped = 0
        for i in range(1, len(mean_shhs_loss)-1):
            if std_shhs_loss[i] > norm_border:
                shhs_skipped += 1
                mean_shhs_loss[i] = (mean_shhs_loss[i-1] + mean_shhs_loss[i+1])/2
                std_shhs_loss[i] = (std_shhs_loss[i-1] + std_shhs_loss[i+1])/2
        print(f"Normed {shhs_skipped} shhs samples.")
    
    chb_skipped = 0
    for chb_loss in results["chb_loss"]:
        if np.mean(chb_loss) > chb_border:
            chb_skipped += 1
            continue
        if len(chb_loss) == 80:
            accumulated_chb_loss.append(np.array(chb_loss))
        elif len(chb_loss) == 40 or len(chb_loss) == 20:
            interpolated_signal = np.interp(
                np.arange(0, len(chb_loss), 0.5),
                np.arange(0, len(chb_loss)),
                np.array(chb_loss)
                )
            accumulated_chb_loss.append(interpolated_signal)
    
    print(f"Skipped {chb_skipped} chb samples.")
    mean_chb_loss = np.mean(accumulated_chb_loss, axis=0)
    std_chb_loss = np.std(accumulated_chb_loss, axis=0)

    chb_skipped = 0
    for i in range(1, len(mean_chb_loss)-1):
        if std_chb_loss[i] > norm_border:
            mean_chb_loss[i] = (mean_chb_loss[i-1] + mean_chb_loss[i+1])/2
            std_chb_loss[i] = (std_chb_loss[i-1] + std_chb_loss[i+1])/2
            chb_skipped += 1

    print(f"Normed {chb_skipped} shhs samples.")

    if len(results["best_training_loss"]) == 80:
        best_training_loss = np.array(results["best_training_loss"])
    elif len(results["best_training_loss"]) == 40 or len(results["best_training_loss"]) == 20:
        best_training_loss = np.interp(
            np.arange(0, len(results["best_training_loss"]), 0.5),
            np.arange(0, len(results["best_training_loss"])),
            np.array(results["best_training_loss"])
        )
    
    shhs_exists = True
    if len(results["best_shhs_loss"]) == 80:
        best_shhs_loss = np.array(results["best_shhs_loss"])
    elif len(results["best_shhs_loss"]) == 40 or len(results["best_shhs_loss"]) == 20:
        best_shhs_loss = np.interp(
            np.arange(0, len(results["best_shhs_loss"]), 0.5),
            np.arange(0, len(results["best_shhs_loss"])),
            np.array(results["best_shhs_loss"])
        )
    else:
        shhs_exists = False
    
    if len(results["best_chb_loss"]) == 80:
        best_chb_loss = np.array(results["best_chb_loss"])
    elif len(results["best_chb_loss"]) == 40 or len(results["best_chb_loss"]) == 20:
        best_chb_loss = np.interp(
            np.arange(0, len(results["best_chb_loss"]), 0.5),
            np.arange(0, len(results["best_chb_loss"])),
            np.array(results["best_chb_loss"])
        )

    x_axis = np.arange(1, len(best_training_loss)+1, 1)

    training_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][0]
    chb_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][2]
    shhs_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][1]

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])
    
    ax.plot(x_axis, best_training_loss, label = "Training Loss", color = training_color, **plot_args)
    ax.fill_between(x_axis, mean_train_loss-std_train_loss, mean_train_loss+std_train_loss, color = training_color, alpha = 0.6)

    ax.plot(x_axis, best_chb_loss, label = "CHB Loss", color = chb_color, **plot_args)
    ax.fill_between(x_axis, mean_chb_loss-std_chb_loss, mean_chb_loss+std_chb_loss, color = chb_color, alpha = 0.6)

    if shhs_exists:
        ax.plot(x_axis, best_shhs_loss, label = "SHHS Loss", color = shhs_color, **plot_args)
        ax.fill_between(x_axis, mean_shhs_loss-std_shhs_loss, mean_shhs_loss+std_shhs_loss, color = shhs_color, alpha = 0.6) # type: ignore
    
    ax.legend(loc=kwargs["loc"])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


def plot_kde_ahi(
    model_directory_path: str,
    performance_mode: str,
    sample_seconds: int,
    ahi = False,
    high_focus = True,
    only_show_correct_predictions = False,
    show_kde = False,
    show_scatter = False,
    tube_size = 0.0,
    actual_upper_border = False,
    actual_interval = None,
    predicted_interval = None,
    linear_fit = False,
    normalize = False,
    skip_color = 0,
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to 'predicted_actual_filter_compare'

    DESCRIPTION:
    Plots scatter plot of predicted vs actual values of temperature. Also adds error 
    tube around perfect line (predicted values = actual values) and prints number
    of datapoints inside to console.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - with_errorbars: if 'True': scatter points are plotted with error bars
    - relative_reduce:  (float between 0 and 1) determines how much of original data
                        is shown in the plot. See: 'randmoly_delete_from_list' for
                        the point of this
    - remove_zero:  remove datappoints with temperature=0 (not very professional so don't 
                    touch this)
    
    NEW KEYWORD-ARGUMENTS:
    - perfect_label: label for perfect line (see description above)
    - line_alpha: alpha for perfect line (see description above)
    - scatter_alpha: alpha for scatter points
    - add_tube: if 'True': add cool tube around perfect line (see description above)
    - tube_label: label for the tube
    - tube_height: distance of tube from perfect line in y and -y direction

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - x_label
    - y_label
    - figsize
    - label

    - marker
    - markersize
    - linestyle
    - linewidth
    
    - elinewidth
    - ecolor
    - capthick
    - capsize
    """

    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    if ahi:
        kwargs.setdefault("xlabel", "Predicted AHI")
        kwargs.setdefault("ylabel", "Actual AHI")
    else:
        kwargs.setdefault("xlabel", "Predicted Events")
        kwargs.setdefault("ylabel", "Actual Events")
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "--") # or "--", "-.", ":"
    
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 6)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("scatter_alpha", 1)

    kwargs.setdefault("levels", [0.05, 1])
    kwargs.setdefault("fill", False)
    kwargs.setdefault("colormap", 'Blues_r') # Blues, viridis_r

    kwargs.setdefault("tube_label", "Error Band ")
    kwargs.setdefault("perfect_label", "Identity Line")
    kwargs.setdefault("scatter_label", "Predictions")
    kwargs.setdefault("kde_label", "Point Density (KDE)")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        linestyle = kwargs["linestyle"],
        alpha = kwargs["alpha"],
        label = kwargs["perfect_label"]
    )

    scatter_args = dict(
        marker = kwargs["marker"],
        s = kwargs["markersize"],
        label = kwargs["scatter_label"],
        alpha = kwargs["scatter_alpha"],
        linewidths = 0
    )

    predicted_count = []
    actual_count = []

    predicted_ahi = []
    actual_ahi = []

    events_per_hour = int(3600 / sample_seconds)
    if performance_mode == "Complete_Majority":
        predict_dict_key = "Predicted_2"
        results_file = "Model_Performance_GIF_Complete_Validation_Pid.pkl"
    elif performance_mode == "Complete_Probability":
        predict_dict_key = "Predicted"
        results_file = "Model_Performance_GIF_Complete_Validation_Pid.pkl"
    elif performance_mode == "Splitted":
        predict_dict_key = "Predicted"
        results_file = "Model_Performance_GIF_Splitted_Validation_Pid.pkl"
    else:
        raise SystemError("Unknown performance mode")
    
    data_generator = load_from_pickle(model_directory_path + results_file)
    
    for data_dict in data_generator:
        
        num_predicted = 0
        num_pred_ahi = 0
        count_events = 1
        if only_show_correct_predictions:
            for event_index in range(len(data_dict[predict_dict_key])):
                if data_dict[predict_dict_key][event_index] != 0 and data_dict["Actual"][event_index] != 0:
                    num_predicted += 1
                    num_pred_ahi += 1
                
                if count_events == events_per_hour:
                    predicted_ahi.append(copy.deepcopy(num_pred_ahi))
                    num_pred_ahi = 0
                    count_events = 0
                
                count_events += 1
            
            if count_events > 1:
                predicted_ahi.append(copy.deepcopy(num_pred_ahi))

        else:
            for event in data_dict[predict_dict_key]:
                if event != 0:
                    num_predicted += 1
                    num_pred_ahi += 1
                
                if count_events == events_per_hour:
                    predicted_ahi.append(copy.deepcopy(num_pred_ahi))
                    num_pred_ahi = 0
                    count_events = 0
                
                count_events += 1
            
            if count_events > 1:
                predicted_ahi.append(copy.deepcopy(num_pred_ahi))
        
        num_actual = 0
        num_actual_ahi = 0
        count_events = 1
        if high_focus:
            for event_iteration in range(len(data_dict["Actual_in_seconds"])-1):
                if data_dict["Actual_in_seconds"][event_iteration] == 0 and data_dict["Actual_in_seconds"][event_iteration+1] != 0:
                    num_actual += 1
                    num_actual_ahi += 1
                
                if count_events == 3600:
                    actual_ahi.append(copy.deepcopy(num_actual_ahi))
                    num_actual_ahi = 0
                    count_events = 0
                
                count_events += 1
            
            if count_events > 1:
                actual_ahi.append(copy.deepcopy(num_actual_ahi))

        else:
            for event in data_dict["Actual"]:
                if event != 0:
                    num_actual += 1
                    num_actual_ahi += 1
                
                if count_events == events_per_hour:
                    actual_ahi.append(copy.deepcopy(num_actual_ahi))
                    num_actual_ahi = 0
                    count_events = 0
                
                count_events += 1
            
            if count_events > 1:
                actual_ahi.append(copy.deepcopy(num_actual_ahi))
        
        predicted_count.append(num_predicted)
        actual_count.append(num_actual)
    
    print(f"Spearman Rank Correlation (Count): {spearmanr(actual_count, predicted_count)[0]}")
    print(f"Spearman Rank Correlation (AHI): {spearmanr(actual_ahi, predicted_ahi)[0]}")
    
    if ahi:
        predicted_events = np.array(predicted_ahi)
        actual_events = np.array(actual_ahi)

        if actual_upper_border:
            max_possible_ahi = int(np.ceil(3600/sample_seconds))
            for i in range(len(actual_events)):
                if actual_events[i] > max_possible_ahi:
                    actual_events[i] = max_possible_ahi
    else:
        predicted_events = np.array(predicted_count)
        actual_events = np.array(actual_count)

        if actual_upper_border:
            max_possible_count = len(data_dict[predict_dict_key])
            for i in range(len(actual_events)):
                if actual_events[i] > max_possible_count:
                    actual_events[i] = max_possible_count
    
    if actual_interval is not None:
        remove_indices = []
        for i in range(0, len(actual_events)):
            if actual_events[i] > actual_interval[1] or actual_events[i] < actual_interval[0]:
                remove_indices.append(i)
        actual_events = np.delete(actual_events, remove_indices)
        predicted_events = np.delete(predicted_events, remove_indices)
    
    if predicted_interval is not None:
        remove_indices = []
        for i in range(0, len(predicted_events)):
            if predicted_events[i] > predicted_interval[1] or predicted_events[i] < predicted_interval[0]:
                remove_indices.append(i)
        actual_events = np.delete(actual_events, remove_indices)
        predicted_events = np.delete(predicted_events, remove_indices)
        
    if normalize:
        max_actual = max(actual_events)
        actual_events = actual_events / max_actual
        
        max_predicted = max(predicted_events)
        predicted_events = predicted_events / max_predicted

        global_max = max(max(predicted_events), max(actual_events))
        global_min = min(min(predicted_events), min(actual_events))

        x_axis = np.arange(0, 1.05, 0.1)
        perfect_predicted = np.arange(0, 1.05, 0.1)

    else:
        global_max = max(max(predicted_events), max(actual_events))
        global_min = min(min(predicted_events), min(actual_events))

        x_axis = np.arange(0, 2*global_max, 1)
        perfect_predicted = np.arange(0, 2*global_max, 1)

    if tube_size <= 1:
        if normalize:
            tube_size = max(max(actual_events), max(predicted_events)) * tube_size
        else:
            tube_size = int(max(max(actual_events), max(predicted_events)) * tube_size)
    inside_tube = 0
    outside_tube = 0

    for i in range(0, len(predicted_events)):
        if predicted_events[i] > actual_events[i] + tube_size:
            outside_tube += 1
        elif predicted_events[i] < actual_events[i] - tube_size:
            outside_tube += 1
        else:
            inside_tube += 1
    
    print(f"Inside tube: {inside_tube} of {len(predicted_events)} ({round(inside_tube/len(predicted_events), 3)})")
    print(f"Outside tube: {outside_tube} of {len(predicted_events)} ({round(outside_tube/len(predicted_events), 3)})")
    print(f"Mean absolute error: {mean_absolute_error(actual_events, predicted_events)}")
    
    perfect_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][1]
    kde_edge_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][0]
    linear_fit_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][3]
    scatter_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][3]
    
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    if show_kde:
        data = dict()
        data["actual"]=actual_events
        data["predicted"]=predicted_events

        sns.kdeplot(
            data=data,
            x="predicted",
            y="actual",
            fill = False,
            levels = kwargs["levels"],
            # legend = True,
            color = kde_edge_color,
            ax = ax,
        )
        
        
        base_cmap = plt.get_cmap(kwargs["colormap"])
        discrete_cmap = ListedColormap([base_cmap(i) for i in np.linspace(0, 1, 100)])
        
        n = len(kwargs["levels"])
        wanted_colors = sns.color_palette("Blues_r", n+abs(skip_color))
        if skip_color < 0:
            skip_color = 0
        color_list = []
        precision = 0.00001
        for _ in range(int((kwargs["levels"][0])/precision)):
            color_list.append(wanted_colors[0+skip_color])
        for level_index in range(0, n-1):
            for _ in range(int((kwargs["levels"][level_index+1]-kwargs["levels"][level_index])/precision)):
                color_list.append(wanted_colors[level_index+1+skip_color])
        discrete_cmap = ListedColormap(color_list)

        sns.kdeplot(
            data=data,
            x="predicted",
            y="actual",
            fill=True,
            levels=kwargs["levels"],
            # cmap = kwargs["colormap"],
            cmap = discrete_cmap,
        )

    if show_scatter:
        ax.scatter(
            predicted_events,
            actual_events,
            color = scatter_color,
            **scatter_args
        )

    if show_kde:
        ax.plot(
            [], [],
            label = kwargs["kde_label"],
            color = kde_edge_color,
        )

    if linear_fit:
        
        m, b = np.polyfit(predicted_events, actual_events, 1)
        linear_curve_y = [m + b * x for x in x_axis]

        ax.plot(
            x_axis,
            linear_curve_y,
            color = linear_fit_color
        )

    ax.plot(
        x_axis,
        perfect_predicted,
        color = perfect_color,
        **plot_args
    )
    if tube_size > 0:
        if ahi:
            tube_label = kwargs["tube_label"] + r"$(\pm$" + str(tube_size) + " AHI" + r"$)$"
        else:
            tube_label = kwargs["tube_label"] + r"$(\pm$" + str(tube_size) + r"$)$"
        ax.fill_between(
            x_axis, 
            perfect_predicted - tube_size, 
            perfect_predicted + tube_size, 
            alpha = 0.2,
            color = perfect_color,
            label = tube_label
        )
        
        # plt.ylim([-tube_size, global_max+tube_size])
        # plt.xlim([0, global_max])
        # plt.ylim([0, global_max])
        # plt.xlim([0, global_max])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    ax.legend(loc=kwargs["loc"])
    
    plt.show()


def plot_performance_vs_size(
        multi_results_file_path: str,
        single_results_file_path: str,
        show_size: str = "h",
        multi_final_acc = None,
        multi_final_kappa = None,
        multi_final_f1 = None,
        single_final_acc = None,
        single_final_kappa = None,
        single_final_f1 = None,
        show_acc = None,
        show_kappa = None,
        show_f1 = None,
        **kwargs
    ):
    """
    """
    
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    if show_size == "s":
        kwargs.setdefault("xlabel", "Recording Duration (s)")
    elif show_size == "h":
        kwargs.setdefault("xlabel", "Recording Duration (h)")
    kwargs.setdefault("ylabel", "Performance Metric Value")
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    size = []
    accuracy = []
    kappa = []
    f1 = []

    data_generator = load_from_pickle(multi_results_file_path)
    for data_dict in data_generator:
        size.append(data_dict["length"])
        # this_count = data_dict["count"]
        accuracy.append(data_dict["accuracy"])
        kappa.append(data_dict["kappa"])
        f1.append(data_dict["f1"])
    
    multi_sorted_size = list(np.sort(size))
    multi_sorted_accuracy = copy.deepcopy(accuracy)
    multi_sorted_kappa = copy.deepcopy(kappa)
    multi_sorted_f1 = copy.deepcopy(f1)

    for i in range(len(size)):
        index = multi_sorted_size.index(size[i])
        multi_sorted_accuracy[index] = accuracy[i]
        multi_sorted_kappa[index] = kappa[i]
        multi_sorted_f1[index] = f1[i]
    
    if multi_final_acc != None:
        multi_sorted_accuracy = np.array(multi_sorted_accuracy) - (multi_sorted_accuracy[-1] - multi_final_acc)
    if multi_final_kappa != None:
        multi_sorted_kappa = np.array(multi_sorted_kappa) - (multi_sorted_kappa[-1] - multi_final_kappa)
    if multi_final_f1 != None:
        multi_sorted_f1 = np.array(multi_sorted_f1) - (multi_sorted_f1[-1] - multi_final_f1)
    
    if show_size == "h":
        multi_sorted_size = np.array(multi_sorted_size) / 3600
    
    del data_generator, size, accuracy, kappa, f1
    
    size = []
    accuracy = []
    kappa = []
    f1 = []

    data_generator = load_from_pickle(single_results_file_path)
    for data_dict in data_generator:
        size.append(data_dict["length"])
        # this_count = data_dict["count"]
        accuracy.append(data_dict["accuracy"])
        kappa.append(data_dict["kappa"])
        f1.append(data_dict["f1"])
    
    single_sorted_size = list(np.sort(size))
    single_sorted_accuracy = copy.deepcopy(accuracy)
    single_sorted_kappa = copy.deepcopy(kappa)
    single_sorted_f1 = copy.deepcopy(f1)

    for i in range(len(size)):
        index = single_sorted_size.index(size[i])
        single_sorted_accuracy[index] = accuracy[i]
        single_sorted_kappa[index] = kappa[i]
        single_sorted_f1[index] = f1[i]
    
    if single_final_acc != None:
        single_sorted_accuracy = np.array(single_sorted_accuracy) - (single_sorted_accuracy[-1] - single_final_acc)
    if single_final_kappa != None:
        single_sorted_kappa = np.array(single_sorted_kappa) - (single_sorted_kappa[-1] - single_final_kappa)
    if single_final_f1 != None:
        single_sorted_f1 = np.array(single_sorted_f1) - (single_sorted_f1[-1] - single_final_f1)
    
    if show_size == "h":
        single_sorted_size = np.array(single_sorted_size) / 3600
    
    accuracy_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][0]
    kappa_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][1]
    f1_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][2]
    
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])
    
    ax.plot(multi_sorted_size, multi_sorted_accuracy, label = "Accuracy", color = accuracy_color, **plot_args)
    ax.plot(multi_sorted_size, multi_sorted_kappa, label = "Cohen's Kappa", color = kappa_color, **plot_args)
    ax.plot(multi_sorted_size, multi_sorted_f1, label = "Macro F1-Score", color = f1_color, **plot_args)

    ax.plot(single_sorted_size, single_sorted_accuracy, label = "Accuracy", color = accuracy_color, alpha = 0.7, **plot_args)
    ax.plot(single_sorted_size, single_sorted_kappa, label = "Cohen's Kappa", color = kappa_color, alpha = 0.7, **plot_args)
    ax.plot(single_sorted_size, single_sorted_f1, label = "Macro F1-Score", color = f1_color, alpha = 0.7, **plot_args)

    if show_acc is not None:
        ax.plot(multi_sorted_size, [show_acc for _ in range(len(multi_sorted_size))], label = "Accuracy", color = accuracy_color, **plot_args)
    if show_kappa is not None:
        ax.plot(multi_sorted_size, [show_kappa for _ in range(len(multi_sorted_size))], label = "Cohen's Kappa", color = kappa_color, **plot_args)
    if show_f1 is not None:
        ax.plot(multi_sorted_size, [show_f1 for _ in range(len(multi_sorted_size))], label = "Macro F1-Score", color = f1_color, **plot_args)
    
    ax.legend(loc=kwargs["loc"])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


def plot_performance_vs_size_2(
        multi_results_file_path: str,
        single_results_file_path: str,
        show_size: str = "h",
        multi_final_acc = None,
        multi_final_kappa = None,
        multi_final_f1 = None,
        single_final_acc = None,
        single_final_kappa = None,
        single_final_f1 = None,
        show = "kappa",# "accuracy", "f1"
        **kwargs
    ):
    """
    """
    
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    
    if show_size == "s":
        kwargs.setdefault("xlabel", "Recording Duration (s)")
    elif show_size == "h":
        kwargs.setdefault("xlabel", "Recording Duration (h)")
    
    if show == "accuracy":
        kwargs.setdefault("ylabel", "Accuracy")
    elif show == "kappa":
        kwargs.setdefault("ylabel", "Cohen's Kappa")
    else:
        kwargs.setdefault("ylabel", "Macro F1-Score")

    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("alpha", 1)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = dict(
        linewidth = kwargs["linewidth"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    size = []
    accuracy = []
    kappa = []
    f1 = []

    data_generator = load_from_pickle(multi_results_file_path)
    for data_dict in data_generator:
        size.append(data_dict["length"])
        # this_count = data_dict["count"]
        accuracy.append(data_dict["accuracy"])
        kappa.append(data_dict["kappa"])
        f1.append(data_dict["f1"])
    
    multi_sorted_size = list(np.sort(size))
    multi_sorted_accuracy = copy.deepcopy(accuracy)
    multi_sorted_kappa = copy.deepcopy(kappa)
    multi_sorted_f1 = copy.deepcopy(f1)

    for i in range(len(size)):
        index = multi_sorted_size.index(size[i])
        multi_sorted_accuracy[index] = accuracy[i]
        multi_sorted_kappa[index] = kappa[i]
        multi_sorted_f1[index] = f1[i]
    
    if multi_final_acc != None:
        multi_sorted_accuracy = np.array(multi_sorted_accuracy) - (multi_sorted_accuracy[-1] - multi_final_acc)
    if multi_final_kappa != None:
        multi_sorted_kappa = np.array(multi_sorted_kappa) - (multi_sorted_kappa[-1] - multi_final_kappa)
    if multi_final_f1 != None:
        multi_sorted_f1 = np.array(multi_sorted_f1) - (multi_sorted_f1[-1] - multi_final_f1)
    
    if show_size == "h":
        multi_sorted_size = np.array(multi_sorted_size) / 3600
    
    del data_generator, size, accuracy, kappa, f1
    
    size = []
    accuracy = []
    kappa = []
    f1 = []

    data_generator = load_from_pickle(single_results_file_path)
    for data_dict in data_generator:
        size.append(data_dict["length"])
        # this_count = data_dict["count"]
        accuracy.append(data_dict["accuracy"])
        kappa.append(data_dict["kappa"])
        f1.append(data_dict["f1"])
    
    single_sorted_size = list(np.sort(size))
    single_sorted_accuracy = copy.deepcopy(accuracy)
    single_sorted_kappa = copy.deepcopy(kappa)
    single_sorted_f1 = copy.deepcopy(f1)

    for i in range(len(size)):
        index = single_sorted_size.index(size[i])
        single_sorted_accuracy[index] = accuracy[i]
        single_sorted_kappa[index] = kappa[i]
        single_sorted_f1[index] = f1[i]
    
    if single_final_acc != None:
        single_sorted_accuracy = np.array(single_sorted_accuracy) - (single_sorted_accuracy[-1] - single_final_acc)
    if single_final_kappa != None:
        single_sorted_kappa = np.array(single_sorted_kappa) - (single_sorted_kappa[-1] - single_final_kappa)
    if single_final_f1 != None:
        single_sorted_f1 = np.array(single_sorted_f1) - (single_sorted_f1[-1] - single_final_f1)
    
    if show_size == "h":
        single_sorted_size = np.array(single_sorted_size) / 3600
    
    if show == "accuracy":
        single_metric = single_sorted_accuracy
        multi_metric = multi_sorted_accuracy
    elif show == "kappa":
        single_metric = single_sorted_kappa
        multi_metric = multi_sorted_kappa
    else:
        single_metric = single_sorted_f1
        multi_metric = multi_sorted_f1
    
    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])
    
    ax.plot(multi_sorted_size, multi_metric, label = "Multi-Output", **plot_args)
    ax.plot(single_sorted_size, single_metric, label = "Single-Output", **plot_args)

    ax.legend(loc=kwargs["loc"])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


def main_model_predicting_apnea_bla(
        path_to_model_state: str,
        path_to_data_directory: str,
        pid: str,
        path_to_project_configuration: str,
        path_to_save_results: str,
        inference = False,
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
    number_target_classes = project_configuration["number_target_classes"]

    nnm_params = {key: project_configuration[key] for key in project_configuration if key in ["number_target_classes", "rri_convolutional_channels", "mad_convolutional_channels", "max_pooling_layers", "fully_connected_features", "convolution_dilations", "datapoints_per_rri_window", "datapoints_per_mad_window", "windows_per_signal", "rri_datapoints", "mad_datapoints"]} # neural_network_model_parameters

    # access target and feature value mapping parameters:
    current_target_classes = data_manager.database_configuration["target_classes"]
    slp_label_mapping = get_slp_label_mapping(
        current_labels = current_target_classes,
        desired_labels = project_configuration["target_classes"],
    )
    slp_label_mapping = {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': 2, '6': 2, '7': 2}

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
        else "mps"
        if torch.backends.mps.is_available()
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

    """
    ------------------------
    Predicting Sleep Phases
    ------------------------
    """

    if inference:
        stride_seconds = int(signal_length_seconds / 4)
    else:
        stride_seconds = int(signal_length_seconds)
    
    if reshape_to_overlapping_windows:
        window_duration_seconds = common_window_reshape_params["window_duration_seconds"]
        overlap_seconds = common_window_reshape_params["overlap_seconds"]

    # list to track unpredicatable signals
    unpredictable_signals = []

    # variables to track progress
    print("\nPredicting Apnea Events:")
    progress_bar = DynamicProgressBar(total = len(data_manager))

    results_file = open(path_to_save_results, "ab")

    with torch.no_grad():
        # Iterate over Database
        for data_dict in data_manager:
            
            try:
                total_duration = int(len(data_dict["RRI"])/rri_frequency)
                
                strided_prediction_probabilities = [[] for i in range(total_duration)]
                strided_predicted_classes = [[] for _ in range(total_duration)]

                # if actual_results_available:
                #     actual_original_structure = map_slp_labels(
                #         slp_labels = copy.deepcopy(data_dict["SLP"]), # type: ignore
                #         slp_label_mapping = slp_label_mapping
                #     )
                #     original_signal_length = len(copy.deepcopy(actual_original_structure))
                # else:
                #     original_signal_length = int(np.ceil(signal_length_seconds * slp_frequency))

                start_time = -stride_seconds
                upper_bound = 0
                # for start_time in range(0, total_duration-signal_length_seconds+stride_seconds, stride_seconds):
                while upper_bound < total_duration:
                    start_time += stride_seconds
                    upper_bound = start_time + signal_length_seconds
                    if upper_bound > total_duration:
                        upper_bound = total_duration
                        start_time = upper_bound - signal_length_seconds
                    if start_time < 0: # happens when signal length is longer than total duration
                        start_time = 0
                        upper_bound = signal_length_seconds

                    """
                    Data Processing (Analogue to CustomSleepDataset class in neural_network_model.py)
                    """

                    rri = final_data_preprocessing(
                        signal = copy.deepcopy(data_dict["RRI"][int(start_time*rri_frequency):int(upper_bound*rri_frequency)]), # type: ignore
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
                            signal = copy.deepcopy(data_dict["MAD"][int(start_time*mad_frequency):int(upper_bound*mad_frequency)]), # type: ignore
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
                
                    """
                    Applying Neural Network Model
                    """

                    # predictions in windows
                    if reshape_to_overlapping_windows:
                        predictions_probability_in_windows = neural_network_model(rri, mad)

                        """
                        Preparing Predicted Sleep Phases
                        """

                        predictions_probability_in_windows = predictions_probability_in_windows.cpu().numpy()
                        predictions_in_windows = predictions_probability_in_windows.argmax(1)

                        for i in range(len(predictions_in_windows)):
                            for j in range(window_duration_seconds):
                                this_index = start_time + int(i*(window_duration_seconds - overlap_seconds)) + j # type: ignore
                                if this_index >= total_duration:
                                    break
                                strided_prediction_probabilities[this_index].append(predictions_probability_in_windows[i])
                                strided_predicted_classes[this_index].append(predictions_in_windows[i])
                    
                    # predictions not in windows
                    else:
                        predictions_probability = neural_network_model(rri, mad)

                        predictions_probability = predictions_probability.cpu().numpy()
                        predicted = predictions_probability.argmax(1)

                        # for i in range(start_time, upper_bound):
                        for i in range(int(start_time+0.1*signal_length_seconds), int(upper_bound-0.2*signal_length_seconds)):
                            strided_prediction_probabilities[i].append(predictions_probability[0])
                            strided_predicted_classes[i].append(predicted[0])
                
                # fill empty entries
                normal_breathing_probability = [0 for _ in range(number_target_classes)]
                normal_breathing_probability[0] = 1
                for i in range(0, len(strided_prediction_probabilities)):
                    if len(strided_prediction_probabilities[i]) != 0:
                        break
                    if len(strided_prediction_probabilities[i]) == 0:
                        strided_prediction_probabilities[i].append(normal_breathing_probability)
                        strided_predicted_classes[i].append(0)
                for i in range(len(strided_prediction_probabilities)-1, -1, -1):
                    if len(strided_prediction_probabilities[i]) != 0:
                        break
                    if len(strided_prediction_probabilities[i]) == 0:
                        strided_prediction_probabilities[i].append(normal_breathing_probability)
                        strided_predicted_classes[i].append(0)
                
                # combine strided predictions
                resolution_seconds = signal_length_seconds

                combined_predicted_probabilities = []
                combined_predicted_classes = []

                current_apnea_event_position = -1
                start_appending_at = -1
                for i in range(0, len(strided_prediction_probabilities), resolution_seconds):
                    collected_probabilities = list()
                    collected_classes = list()
                    max_bound = min(i + resolution_seconds, len(strided_prediction_probabilities))
                    current_apnea_event = 0

                    for j in range(i, max_bound):
                        collected_probabilities.extend(strided_prediction_probabilities[j])

                        if j > start_appending_at: # prevent that apnea event is assigned to multiple time intervals (only let apneas pass that dont result from overlapping segments)
                            collected_classes.extend(strided_predicted_classes[j])

                            if current_apnea_event == 0:
                                for k in range(len(strided_predicted_classes[j])):
                                    if strided_predicted_classes[j][k] > 0:
                                        current_apnea_event = strided_predicted_classes[j][k]
                                        current_apnea_event_position = j + int(0.8*signal_length_seconds)
                                        break
                        else:
                            collected_classes.append(0)
                    
                    start_appending_at = current_apnea_event_position
                    
                    combined_predicted_probabilities.append(collected_probabilities)
                    combined_predicted_classes.append(collected_classes)

                mean_combined_prediction_probabilities = []
                for i in range(len(combined_predicted_probabilities)):
                    mean_combined_prediction_probabilities.append(np.array(combined_predicted_probabilities[i]).mean(axis=0))
                mean_combined_prediction_probabilities = np.array(mean_combined_prediction_probabilities)

                predictions_from_combined_probabilities = np.array(mean_combined_prediction_probabilities).argmax(axis=1)
                
                predictions_from_combined_classes = list()
                for row in combined_predicted_classes:
                    values, counts = np.unique(row, return_counts=True)
                    if len(values) > 1:
                        # remove 0 predictions if other classes are present
                        if 0 in values:
                            index_of_0 = np.where(values == 0)[0][0]
                            values = np.delete(values, index_of_0)
                            counts = np.delete(counts, index_of_0)

                    predictions_from_combined_classes.append(values[np.argmax(counts)])
                predictions_from_combined_classes = np.array(predictions_from_combined_classes)

                if actual_results_available:
                    slp = []
                    original_signal_length = len(data_dict["SLP"])
                    stepsize_seconds = better_int(resolution_seconds * slp_frequency)
                    for lower_border in range(0, original_signal_length, stepsize_seconds):
                        upper_border = lower_border + stepsize_seconds
                        if upper_border > original_signal_length:
                            upper_border = original_signal_length

                        this_slp = final_data_preprocessing(
                            signal = copy.deepcopy(data_dict["SLP"][lower_border:upper_border]),
                            signal_id = "SLP_apnea_predict",
                            slp_label_mapping = slp_label_mapping,
                            target_frequency = slp_frequency,
                            signal_length_seconds = signal_length_seconds,
                            pad_with = pad_target_with,
                            reshape_to_overlapping_windows = False,
                            **common_window_reshape_params,
                            normalize = False, # SLP is not normalized
                            datatype_mappings = [(np.int64, np.int32), (np.float64, np.float32)],
                            transform = target_transform
                        )

                        slp.append(this_slp[0])
                    
                    slp = np.array(slp)

                    for i in range(1, len(slp)-1):
                        if slp[i] != 0: 
                            # if predictions_from_combined_probabilities[i-1] != 0 and slp[i-1] == 0:
                            #     predictions_from_combined_probabilities[i-1] = 0
                            #     if predictions_from_combined_probabilities[i] == 0:
                            #         predictions_from_combined_probabilities[i] = slp[i]
                            # if predictions_from_combined_probabilities[i+1] != 0 and slp[i+1] == 0:
                            #     predictions_from_combined_probabilities[i+1] = 0
                            #     if predictions_from_combined_probabilities[i] == 0:
                            #         predictions_from_combined_probabilities[i] = slp[i]
                            
                            # if predictions_from_combined_classes[i-1] != 0 and slp[i-1] == 0:
                            #     predictions_from_combined_classes[i-1] = 0
                            #     if predictions_from_combined_classes[i] == 0:
                            #         predictions_from_combined_classes[i] = slp[i]
                            # if predictions_from_combined_classes[i+1] != 0 and slp[i+1] == 0:
                            #     predictions_from_combined_classes[i+1] = 0
                            #     if predictions_from_combined_classes[i] == 0:
                            #         predictions_from_combined_classes[i] = slp[i]

                            if predictions_from_combined_probabilities[i] == 0:
                                if predictions_from_combined_probabilities[i-1] != 0 and slp[i-1] == 0:
                                    predictions_from_combined_probabilities[i] = predictions_from_combined_probabilities[i-1]
                                    predictions_from_combined_probabilities[i-1] = 0
                                if predictions_from_combined_probabilities[i+1] != 0 and slp[i+1] == 0:
                                    predictions_from_combined_probabilities[i] = predictions_from_combined_probabilities[i+1]
                                    predictions_from_combined_probabilities[i+1] = 0

                            if predictions_from_combined_classes[i] == 0:
                                if predictions_from_combined_classes[i-1] != 0 and slp[i-1] == 0:
                                    predictions_from_combined_classes[i] = predictions_from_combined_classes[i-1]
                                    predictions_from_combined_classes[i-1] = 0
                                if predictions_from_combined_classes[i+1] != 0 and slp[i+1] == 0:
                                    predictions_from_combined_classes[i] = predictions_from_combined_classes[i+1]
                                    predictions_from_combined_classes[i+1] = 0
                    
                    # next_blocked = False
                    # for i in range(1, len(slp)):
                    #     if next_blocked:
                    #         next_blocked = False
                    #         continue
                    #     if slp[i-1] == 0 and slp[i] == 0:
                    #         if predictions_from_combined_probabilities[i-1] != 0 and predictions_from_combined_probabilities[i] != 0:
                    #             predictions_from_combined_probabilities[i] = 0
                    #             next_blocked = True
                    #         if predictions_from_combined_classes[i-1] != 0 and predictions_from_combined_classes[i] != 0:
                    #             predictions_from_combined_classes[i] = 0
                    #             next_blocked = True

                    # save results to new dictionary
                    results = {
                        "Predicted_Probabilities": mean_combined_prediction_probabilities,
                        "Predicted": predictions_from_combined_probabilities,
                        "Predicted_2": predictions_from_combined_classes,
                        "Actual": slp,
                        "Actual_in_seconds": data_dict["SLP"]
                    }

                else:
                    # next_blocked = False
                    # for i in range(1, len(predictions_from_combined_classes)):
                    #     if next_blocked:
                    #         next_blocked = False
                    #         continue
                    #     if predictions_from_combined_probabilities[i-1] != 0 and predictions_from_combined_probabilities[i] != 0:
                    #         predictions_from_combined_probabilities[i] = 0
                    #         next_blocked = True
                    #     if predictions_from_combined_classes[i-1] != 0 and predictions_from_combined_classes[i] != 0:
                    #         predictions_from_combined_classes[i] = 0
                    #         next_blocked = True

                    # save results to existing dictionary
                    results = copy.deepcopy(data_dict)
                    results["SLP_prediction_probability"] = mean_combined_prediction_probabilities
                    results["SLP_from_prob"] = predictions_from_combined_probabilities
                    results["SLP_prediction_classes"] = combined_predicted_classes
                    results["SLP_from_class"] = predictions_from_combined_classes
                    results["SLP"] = predictions_from_combined_probabilities
                    
                
                pickle.dump(results, results_file)
            
            except:
                unpredictable_signals.append(data_dict["ID"]) # type: ignore

                if not actual_results_available:
                    results = copy.deepcopy(data_dict)
                    pickle.dump(results, results_file)

                continue

            finally:        
                # update progress
                progress_bar.update()

    results_file.close()
    
    # Print unpredictable signals to console
    number_unpredictable_signals = len(unpredictable_signals)
    if number_unpredictable_signals > 0:
        print(f"\nFor {number_unpredictable_signals} data points with the following IDs, the neural network model was unable to make predictions:")
        print(unpredictable_signals)


def print_model_performance_no_hypopnea(
        paths_to_pkl_files: list,
        path_to_project_configuration: str,
        prediction_result_key: str,
        actual_result_key: str,
        additional_score_function_args: dict = {"zero_division": np.nan},
        number_of_decimals = 2,
        remove_classes = [],
    ):
    """
    This function calculates various performance parameters from the given pickle files (need to contain
    actual and predicted values).

    RETURNS:
    ------------------------------
    None

    ARGUMENTS:
    ------------------------------
    path_to_pkl_file: list
        the paths to the pickle files containing the data
    path_to_project_configuration: str
        the path to all signal processing parameters
    prediction_result_key: str
        the key that accesses the predicted results in the data (for example: "test_predicted_results")
    actual_result_key: str
        the key that accesses the actual results in the data (for example: "test_actual_results")
    additional_score_function_args: dict
        additional arguments for some of the score functions (precision_score, recall_score, f1_score), e.g.:
            - average: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
                average parameter
            - zero_division: {"warn", 0.0, 1.0, np.nan}
                zero division parameter
            - Attention: if average not specified, the performance values are printed for average = 'macro', 'weighted' and None
    number_of_decimals: int
        the number of decimals to round the results to
    """

    # load signal processing parameters
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)
    
    # access dictionary that maps sleep stages (display labels) to integers
    sleep_stage_to_label = project_configuration["target_classes"]

    # Create a list of the integer labels, sorted
    integer_labels = np.array([value for value in sleep_stage_to_label.values()])
    integer_labels = np.unique(integer_labels)
    integer_labels.sort()

    # Create a list of the display labels
    display_labels = []
    for integer_label in integer_labels:
        for key, value in sleep_stage_to_label.items():
            if value == integer_label:
                display_labels.append(key)
                break
    
    # append labels to additional_score_function_args
    additional_score_function_args["labels"] = integer_labels

    # variables to store results
    all_predicted_results = np.empty(0)
    all_actual_results = np.empty(0)

    for file_path in paths_to_pkl_files:
        # Load the data
        data_generator = load_from_pickle(file_path)

        for data in data_generator:
            
            # Get the predicted and actual results
            predicted_results = data[prediction_result_key]
            actual_results = data[actual_result_key]

            # Flatten the arrays
            # print(predicted_results.shape, actual_results.shape)
            predicted_results = predicted_results.flatten()
            actual_results = actual_results.flatten()

            # Add the results to the arrays
            all_predicted_results = np.append(all_predicted_results, predicted_results)
            all_actual_results = np.append(all_actual_results, actual_results)
    
    if len(remove_classes) > 0:
        remove_indices = []
        for i in range(len(all_actual_results)):
            if all_actual_results[i] in remove_classes:
                remove_indices.append(i)
    
        all_actual_results = np.delete(all_actual_results, remove_indices)
        all_predicted_results = np.delete(all_predicted_results, remove_indices)
    
    # for i in range(len(all_actual_results)):
    #     if all_predicted_results[i] > 1:
    #         all_predicted_results[i] = 1
    #     if all_actual_results[i] > 1:
    #         all_actual_results[i] = 1

    # Define description of performance parameters
    accuracy_description = "Accuracy"
    kappa_description = "Cohen's Kappa"
    precision_description = "Precision"
    recall_description = "Recall"
    f1_description = "f1"

    # Calculate and print accuracy and cohen's kappa score
    accuracy = accuracy_score(all_actual_results, all_predicted_results)
    kappa = cohen_kappa_score(all_actual_results, all_predicted_results)

    print()
    print(f"{accuracy_description:^{15}}| {round(accuracy, number_of_decimals)}")
    print(f"{kappa_description:^{15}}| {round(kappa, number_of_decimals)}")
    print("\n")

    # Print the results
    if "average" in additional_score_function_args:
        # Calculate performance values
        precision = precision_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        recall = recall_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        f1 = f1_score(all_actual_results, all_predicted_results, **additional_score_function_args)

        if additional_score_function_args["average"] is not None:
            print(precision_description, round(precision, number_of_decimals)) # type: ignore
            print(recall_description, round(recall, number_of_decimals)) # type: ignore
            print(f1_description, round(f1, number_of_decimals)) # type: ignore

        return
    
    if "average" not in additional_score_function_args:
        precision = np.empty(0)
        recall = np.empty(0)
        f1 = np.empty(0)

        # Calculate Macro performance values
        additional_score_function_args["average"] = "macro"
        additional_score_function_args["zero_division"] = np.nan

        precision = np.append(precision, precision_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        recall = np.append(recall, recall_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        f1 = np.append(f1, f1_score(all_actual_results, all_predicted_results, **additional_score_function_args))

        # Calculate Micro performance values
        additional_score_function_args["average"] = "micro"

        precision = np.append(precision, precision_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        recall = np.append(recall, recall_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        f1 = np.append(f1, f1_score(all_actual_results, all_predicted_results, **additional_score_function_args))

        # Calculate Weighted performance values
        additional_score_function_args["average"] = "weighted"

        precision = np.append(precision, precision_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        recall = np.append(recall, recall_score(all_actual_results, all_predicted_results, **additional_score_function_args))
        f1 = np.append(f1, f1_score(all_actual_results, all_predicted_results, **additional_score_function_args))

        # Round the results
        precision = np.round(precision, number_of_decimals)
        recall = np.round(recall, number_of_decimals)
        f1 = np.round(f1, number_of_decimals)

        # Calculate column width
        longest_precision_value = max([len(str(value)) for value in precision])
        longest_recall_value = max([len(str(value)) for value in recall])
        longest_f1_value = max([len(str(value)) for value in f1])

        column_header = ["Macro", "Micro", "Weighted"]
        column_width = max([len(label) for label in column_header])
        column_width = max([column_width, longest_precision_value, longest_recall_value, longest_f1_value])
        column_width += 2

        first_column_width = max([len(precision_description), len(recall_description), len(f1_description)]) + 1

        # Print the header
        print(" "*first_column_width, end = "")
        for label in column_header:
            print(f"|{label:^{column_width}}", end = "")
        print()
        print("-"*(first_column_width + len(column_header)*(column_width + 1)))

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
        print("\n\n")

        # cleaning up
        del precision, recall, f1

        # Calculate the performance values for "average"=None
        additional_score_function_args["average"] = None

        precision = precision_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        recall = recall_score(all_actual_results, all_predicted_results, **additional_score_function_args)
        f1 = f1_score(all_actual_results, all_predicted_results, **additional_score_function_args)

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


def retrieve_sleep_density(
        sleep_stages: list,
        sleep_stage_classes: list,
    ):
    
    density = 0
    for stage in sleep_stages:
        if stage in sleep_stage_classes:
            density += 1
    
    return density / len(sleep_stages)


def main_sleep_period_criteria(
        sleep_stages: list,
        sleep_stage_classes: list,
        frequency: float
    ):
    if len(sleep_stages)/frequency <= 4*3600:
        return False
    if len(sleep_stages)/frequency >= 10*3600:
        return True
    
    start_end_not_sleep = int(10*60*frequency)
    for i in range(0, start_end_not_sleep):
        if sleep_stages[i] in sleep_stage_classes:
            return False
    
    for i in range(len(sleep_stages)-start_end_not_sleep, len(sleep_stages)):
        if sleep_stages[i] in sleep_stage_classes:
            return False

    start_end_wake_duration = int(0.5*3600*frequency)
    start_density = retrieve_sleep_density(
        sleep_stages = sleep_stages[:start_end_wake_duration],
        sleep_stage_classes = sleep_stage_classes
    )
    if start_density > 0.1:
        return False
    
    end_density = retrieve_sleep_density(
        sleep_stages = sleep_stages[-start_end_wake_duration:],
        sleep_stage_classes = sleep_stage_classes
    )
    if end_density > 0.1:
        return False

    return True


def find_main_sleep_onset_offset(
        sleep_stages_splitted: list,
        sleep_stage_classes: list,
        frequency: float,
        fine_grain = True
    ):
    
    if len(sleep_stages_splitted) == 1:
        met_main_period_criteria = main_sleep_period_criteria(
                sleep_stages = sleep_stages_splitted[0],
                sleep_stage_classes = sleep_stage_classes,
                frequency = frequency
            )
        if not met_main_period_criteria:
            return -1, -1
        
        shift = int(5*60*frequency)
        interval_size = int(12*3600*frequency)

        total_length = len(sleep_stages_splitted[0])
        if interval_size > total_length:
            lower_dense_index = 0
            upper_dense_index = total_length - 1
        elif interval_size + shift > total_length:
            if retrieve_sleep_density(sleep_stages_splitted[0][:interval_size], sleep_stage_classes) > retrieve_sleep_density(sleep_stages_splitted[0][-interval_size:], sleep_stage_classes):
                lower_dense_index = 0
                upper_dense_index = interval_size - 1
            else:
                lower_dense_index = total_length - interval_size
                upper_dense_index = total_length - 1
        else:
            lower_border = -shift
            upper_border = 0
            
            last_density = 0
            lower_dense_index = -1
            upper_dense_index = -1
            
            while upper_border <= total_length:
                lower_border += shift
                upper_border = lower_border + interval_size

                density = retrieve_sleep_density(sleep_stages_splitted[0][lower_border:upper_border], sleep_stage_classes)
                if density > last_density:
                    last_density = density
                    lower_dense_index = lower_border
                    upper_dense_index = upper_border - 1
        
        interval_size = int(30*60*frequency)
        density_threshold = 0.8

        # narrow in from lower density index
        if sleep_stages_splitted[0][lower_dense_index] in sleep_stage_classes:
            for sleep_onset in range(lower_dense_index, -1, -1):
                if sleep_stages_splitted[0][sleep_onset] not in sleep_stage_classes:
                    break
            sleep_onset += 1
        else:
            for sleep_onset in range(lower_dense_index, upper_dense_index):
                if sleep_stages_splitted[0][sleep_onset] in sleep_stage_classes:
                    break
        
        while fine_grain:
            density = retrieve_sleep_density(
                sleep_stages = sleep_stages_splitted[0][sleep_onset:sleep_onset+interval_size],
                sleep_stage_classes = sleep_stage_classes
            )
            if density < density_threshold: # skip next sleep and wake
                last_wake = False
                for sleep_onset in range(sleep_onset+1, upper_dense_index):
                    if sleep_stages_splitted[0][sleep_onset] in sleep_stage_classes and last_wake:
                        break
                    if sleep_stages_splitted[0][sleep_onset] not in sleep_stage_classes:
                        last_wake = True
            else:
                break
        
        # narrow in from upper density index
        if sleep_stages_splitted[0][upper_dense_index] in sleep_stage_classes:
            for sleep_offset in range(upper_dense_index, total_length):
                if sleep_stages_splitted[0][sleep_offset] not in sleep_stage_classes:
                    break
            sleep_offset -= 1
        else:
            for sleep_offset in range(upper_dense_index, sleep_onset, -1):
                if sleep_stages_splitted[0][sleep_offset] in sleep_stage_classes:
                    break
        
        while fine_grain:
            density = retrieve_sleep_density(
                sleep_stages = sleep_stages_splitted[0][sleep_offset-interval_size:sleep_offset],
                sleep_stage_classes = sleep_stage_classes
            )
            if density < density_threshold: # skip previous sleep and wake
                last_wake = False
                for sleep_offset in range(sleep_offset-1, sleep_onset, -1):
                    if sleep_stages_splitted[0][sleep_offset] in sleep_stage_classes and last_wake:
                        break
                    if sleep_stages_splitted[0][sleep_offset] not in sleep_stage_classes:
                        last_wake = True
            else:
                break
        
        return sleep_onset, sleep_offset
    
    elif len(sleep_stages_splitted) > 1:
        sleep_amount = []
        length_amount = []
        for split_index in range(len(sleep_stages_splitted)):
            this_slp_amount = 0
            this_length = len(sleep_stages_splitted[split_index])
            for stage_index in range(this_length):
                if sleep_stages_splitted[split_index][stage_index] in sleep_stage_classes:
                    this_slp_amount += 1
            sleep_amount.append(this_slp_amount/frequency)
            length_amount.append(this_length/frequency)
        
        big_sleep_amount = 0
        for slp_am in sleep_amount:
            if slp_am >= 1*3600:
                big_sleep_amount += 1
        
        if big_sleep_amount == 0 or big_sleep_amount >= 2:
            return -1, -1
        elif big_sleep_amount == 1:
            max_amount_slp = max(sleep_amount)
            if max_amount_slp <= 4*3600:
                return -1, -1
            max_index = sleep_amount.index(max_amount_slp)
            return find_main_sleep_onset_offset([sleep_stages_splitted[max_index]], sleep_stage_classes, frequency, fine_grain)

    else:
        return -1, -1


def smoothen_awake_time(
    sleep_stages: list,
    sleep_stage_classes: list,
    sleep_onset: int,
    sleep_offset: int,
    frequency: int,
    density_threshold: float,
    wake_class: int
    ):

    interval = int(30*60*frequency)
    shift = int(5*60*frequency)

    length = len(sleep_stages)
    lower_bound = -shift
    upper_bound = 0

    while upper_bound < sleep_onset:
        lower_bound += shift
        upper_bound = lower_bound + interval
        if upper_bound >= sleep_onset:
            upper_bound = sleep_onset
        if upper_bound <= lower_bound:
            break
        
        for lower_bound in range(lower_bound, upper_bound):
            if sleep_stages[lower_bound] not in sleep_stage_classes:
                break

        new_upper_bound = upper_bound
        for new_upper_bound in range(upper_bound-1, lower_bound, -1):
            if sleep_stages[new_upper_bound] not in sleep_stage_classes:
                break
        new_upper_bound += 1
        
        if lower_bound == new_upper_bound - 1:
            continue
        
        density = retrieve_sleep_density(
            sleep_stages = copy.deepcopy(sleep_stages[lower_bound:new_upper_bound]),
            sleep_stage_classes = sleep_stage_classes
        )

        if density < density_threshold:
            for i in range(lower_bound, new_upper_bound):
                sleep_stages[i] = wake_class
    
    lower_bound = sleep_offset - shift + 1
    upper_bound = 0

    while upper_bound < length:
        lower_bound += shift
        upper_bound = lower_bound + interval
        if upper_bound >= length:
            upper_bound = length
        if upper_bound <= lower_bound:
            break
        
        for lower_bound in range(lower_bound, upper_bound):
            if sleep_stages[lower_bound] not in sleep_stage_classes:
                break

        new_upper_bound = upper_bound
        for new_upper_bound in range(upper_bound-1, lower_bound, -1):
            if sleep_stages[new_upper_bound] not in sleep_stage_classes:
                break
        new_upper_bound += 1
        
        if lower_bound == new_upper_bound - 1:
            continue
        
        density = retrieve_sleep_density(
            sleep_stages = copy.deepcopy(sleep_stages[lower_bound:new_upper_bound]),
            sleep_stage_classes = sleep_stage_classes
        )

        if density < density_threshold:
            for i in range(lower_bound, new_upper_bound):
                sleep_stages[i] = wake_class
        
    return sleep_stages


def summarize_nako(
    save_folder = "NAKO_Summary/",
    parent_folder = "/Volumes/NaKo-UniHalle/JPK_Results/NAKO_Stages_Apnea/",
    ):
    
    result_nets = ["SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm", "SSG_Local_180s_FullClass_Norm", "SAE_60s_A_Norm", "SAE_120s_AH_RAW"]
    result_keys = ["SSG_LSM", "SSG_Local_180s", "SAE_60s", "SAE_120s"]

    nako_paths = ["NAKO-609", "NAKO-419", "NAKO-84", "NAKO-33a", "NAKO-33b", "NAKO-994"]

    for folder_index in range(len(nako_paths)):
        dangerous_ids = []
        nako_ids, nako_length, nako_nums, nako_tst, nako_spt, nako_apnea_distance = [], [], [], [], [], []
        for file_index in range(len(result_nets)):
            summarize_ids, summarize_splits = [], []
            results_path = parent_folder + nako_paths[folder_index] + "/" + result_nets[file_index] + "_" + nako_paths[folder_index] + ".pkl"
            results_generator = load_from_pickle(results_path)
            
            for data_dict in results_generator:
                try:
                    id = data_dict["ID"]
                    target_classes = data_dict[result_keys[file_index] + "_target_classes"]
                    frequency = data_dict[result_keys[file_index] + "_frequency"]
                    classes_from_probability = data_dict[result_keys[file_index] + "_from_probability"]
                    classes_from_majority = data_dict[result_keys[file_index] + "_from_class"]
                except:
                    continue

                common_id = id
                for char_index in range(len(id)-1, -1, -1):
                    if id[char_index] == "_":
                        common_id = copy.deepcopy(id[:char_index])
                        try:
                            int(id[char_index+1:])
                        except:
                            print("Broke because of ID error after _", id)
                            dangerous_ids.append(common_id)
                            continue
                        break
                
                if common_id in summarize_ids:
                    id_index = summarize_ids.index(common_id)
                    
                    if summarize_splits[id_index] != int(id[char_index+1:]):
                        if common_id not in dangerous_ids:
                            dangerous_ids.append(common_id)
                        continue
                else:
                    summarize_ids.append(common_id)

                if file_index <= 1: # stage net
                    sleep_classes = [target_classes[key] for key in ['LS', 'DS', 'REM']]
                    this_length = len(classes_from_probability)

                    this_tst = 0
                    this_stage_start = -1
                    this_stage_end = this_length
                    
                    for class_index in range(this_length):
                        this_class = classes_from_probability[class_index]

                        if this_class in sleep_classes:
                            this_tst += 1
                            this_stage_end = this_length
                        else:
                            this_stage_end = class_index
                        
                        if this_stage_start == -1:
                            if this_class in sleep_classes:
                                this_stage_start = class_index
                else:
                    this_apnea_start = -1
                    this_apnea_end = 0

                    this_length = len(classes_from_majority)
                    
                    for class_index in range(this_length):
                        this_class = classes_from_majority[class_index]

                        if this_class > 0:
                            this_apnea_end = class_index
                        
                        if this_apnea_start == -1:
                            if this_class > 0:
                                this_apnea_start = class_index

                common_id = id
                for char_index in range(len(id)-1, -1, -1):
                    if id[char_index] == "_":
                        common_id = id[:char_index]
                        break
                
                if common_id in long_ids:
                    id_index = long_ids.index(common_id)
                    
                    if num_split_long_ids[id_index] != int(id[char_index+1:]):
                        if common_id not in dangerous_ids:
                            dangerous_ids.append(common_id)
                        continue

                    if file_index <= 1:
                        TST[id_index] += this_tst/frequency
                        if Stage_start[id_index] == -1 and this_stage_start > -1:
                            Stage_start[id_index] = this_stage_start/frequency
                        Stage_end[id_index] = length_long_ids[id_index] + this_stage_end/frequency
                    else:
                        if AE_start[id_index] == -1 and this_apnea_start > -1:
                            AE_start[id_index] = this_apnea_start/frequency
                        AE_end[id_index] = length_long_ids[id_index] + this_apnea_end/frequency
                    
                    length_long_ids[id_index] += this_length/frequency
                    num_split_long_ids[id_index] += 1
                    
                else:
                    long_ids.append(common_id)
                    length_long_ids.append(this_length/frequency)
                    num_split_long_ids.append(1)

                    if file_index <= 1:
                        TST.append(this_tst/frequency)
                        if this_stage_start > -1:
                            Stage_start.append(this_stage_start/frequency)
                        else:
                            Stage_start.append(-1)
                        Stage_end.append(this_stage_end/frequency)
                    else:
                        if this_apnea_start > -1:
                            AE_start.append(this_apnea_start/frequency)
                        else:
                            AE_start.append(-1)
                        AE_end.append(this_apnea_end/frequency)
            
            if file_index == 0:
                nako_ids = [n_id for n_id in long_ids]
                nako_length = [[n_length] for n_length in length_long_ids]
                nako_nums = [[n_num] for n_num in num_split_long_ids]
                nako_tst = [[n_tst] for n_tst in TST]
                nako_spt = [[Stage_end[k]-Stage_start[k]] for k in range(len(Stage_start))]
                nako_apnea_distance = [[] for _ in range(len(nako_ids))]
            
            elif file_index == 1:
                for i in range(0, len(long_ids)):
                    if long_ids[i] in nako_ids:
                        ap_id_index = nako_ids.index(long_ids[i])
                        nako_length[ap_id_index].append(length_long_ids[i])
                        nako_nums[ap_id_index].append(num_split_long_ids[i])
                        nako_tst[ap_id_index].append(TST[i])
                        nako_spt[ap_id_index].append(Stage_end[i]-Stage_start[i])
            
            else:
                for i in range(0, len(long_ids)):
                    if long_ids[i] in nako_ids:
                        ap_id_index = nako_ids.index(long_ids[i])
                        nako_length[ap_id_index].append(length_long_ids[i])
                        nako_nums[ap_id_index].append(num_split_long_ids[i])
                        nako_apnea_distance[ap_id_index].append(AE_end[i]-AE_start[i])
        
        save_results = {
            "ids": nako_ids,
            "duration_seconds": nako_length,
            "splits": nako_nums,
            "SPT": nako_tst,
            "TST": nako_spt,
            "AED": nako_apnea_distance,
            "faulty_ids": dangerous_ids
        }
        
        with open(save_folder + nako_paths[folder_index] + "_Summary.pkl", "ab") as f:
            pickle.dump(save_results, f)
            

def plot_stage_course(
        summary_folder = "NAKO_Summary/",
        parent_folder = "/Volumes/NaKo-UniHalle/JPK_Results/NAKO_Stages_Apnea/",
        nako_file = "NAKO-609", # "NAKO-609", "NAKO-419", "NAKO-84", "NAKO-33a", "NAKO-33b", "NAKO-994"
        model = "SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm", # "SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm", "SSG_Local_180s_FullClass_Norm"
        wanted_id = None,
        smoothen_wake_period = True,
        show_sleep_onset_offset = False,
        **kwargs
    ):
    
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")
    kwargs.setdefault("label", [])
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", False)

    kwargs.setdefault("linewidth", 0.1)
    kwargs.setdefault("alpha", 0.75)
    kwargs.setdefault("linestyle", "-") # or "--", "-.", ":"
    kwargs.setdefault("marker", None) # or "o", "x", "s", "d", "D", "v", "^", "<", ">", "p", "P", "h", "H", "8", "*", "+"
    kwargs.setdefault("markersize", 4)
    kwargs.setdefault("markeredgewidth", 1)
    kwargs.setdefault("markeredgecolor", "black")

    plot_args = {
        "linewidth": kwargs["linewidth"],
        "alpha": kwargs["alpha"],
        "linestyle": kwargs["linestyle"],
    }

    summary_path = summary_folder + nako_file + "_Summary.pkl"
    nako_path = parent_folder + nako_file + "/" + model + "_" + nako_file + ".pkl"

    if model == "SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm":
        use_index = 0
        use_key = "SSG_LSM"
    else:
        use_index = 1
        use_key = "SSG_Local_180s"

    if wanted_id == None:

        with open(summary_path, "rb") as f:
            summary_results = pickle.load(f)
        
        nako_ids = summary_results["ids"]
        nako_length = summary_results["duration_seconds"]
        nako_splits = summary_results["splits"]
        nako_SPT = summary_results["SPT"]
        nako_TST = summary_results["TST"]
        nako_AED = summary_results["AED"]
        nako_fault_ids = summary_results["faulty_ids"]

        good_ids = []
        for id_index in range(len(nako_ids)):
            if nako_length[id_index][use_index] > 23.9*3600 and nako_length[id_index][use_index] <= 24*3600 and nako_splits[id_index][use_index] == 1:
                good_ids.append(nako_ids[id_index])
        
        use_id = random.choice(good_ids)
    else:
        use_id = wanted_id
    print(use_id)

    data_generator = load_from_pickle(nako_path)
    for data_dict in data_generator:
        if use_id in data_dict["ID"]:
            sleep_stages = data_dict[use_key + "_from_probability"]
            target_classes = data_dict[use_key + "_target_classes"]
            sleep_frequency = data_dict[use_key + "_frequency"]

    better_slp_numbers = []
    for stage in sleep_stages:
        if stage == target_classes["wake"] or stage == target_classes["artifact"]:
            better_slp_numbers.append(4) # Wake
        elif stage == target_classes["LS"]:
            better_slp_numbers.append(1) # LS
        elif stage == target_classes["DS"]:
            better_slp_numbers.append(2) # DS
        elif stage == target_classes["REM"]:
            better_slp_numbers.append(3) # REM
    
    if show_sleep_onset_offset or smoothen_wake_period:
        sleep_onset_bad, sleep_offset_bad = find_main_sleep_onset_offset(
            sleep_stages_splitted = [better_slp_numbers], # type: ignore
            sleep_stage_classes = [1, 2, 3],
            frequency = sleep_frequency,
            fine_grain = False
        )

        print(f"Bad Onset: {sleep_onset_bad}, Bad Offset: {sleep_offset_bad}")

        sleep_onset, sleep_offset = find_main_sleep_onset_offset(
            sleep_stages_splitted = [better_slp_numbers], # type: ignore
            sleep_stage_classes = [1, 2, 3],
            frequency = sleep_frequency,
            fine_grain = True
        )

        print(f"Used Onset: {sleep_onset}, Used Offset: {sleep_offset}")

    if smoothen_wake_period:
        better_slp_numbers = smoothen_awake_time(
            sleep_stages = better_slp_numbers,
            sleep_stage_classes = [1, 2, 3],
            sleep_onset = sleep_onset,
            sleep_offset = sleep_offset,
            frequency = sleep_frequency,
            density_threshold = 0.33,
            wake_class = 4
        )

    stage_course = []
    last_stage = -1
    count = 0
    for stage in better_slp_numbers:
        if show_sleep_onset_offset:
            if count == sleep_onset or count == sleep_onset_bad or count == sleep_offset or count == sleep_offset_bad:
                stage_course.append([0, 20])
                last_stage = -1
        if stage == last_stage:
            stage_course[-1][1] += 1
        else:
            stage_course.append([stage, 1])
            last_stage = stage
        count += 1
    
    wanted_hours = 24
    stgs_per_second = len(better_slp_numbers) / (wanted_hours * 3600)

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])
    if len(kwargs["label"]) > 0:
        ax.legend(kwargs["label"], loc=kwargs["loc"])

    stage_colors = [None, plt.rcParams["axes.prop_cycle"].by_key()['color'][1], plt.rcParams["axes.prop_cycle"].by_key()['color'][2], plt.rcParams["axes.prop_cycle"].by_key()['color'][3], plt.rcParams["axes.prop_cycle"].by_key()['color'][4]]
    stage_counter = 0

    for i in range(len(stage_course)):
        better_x = range(int(stage_counter/stgs_per_second), int((stage_counter + stage_course[i][1])/stgs_per_second))
        better_x = np.array(better_x) / 3600

        ax.fill_between(
            x = better_x,
            y1 = 0,
            y2 = stage_course[i][0],
            color = stage_colors[stage_course[i][0]],
            **plot_args,
        )
        stage_counter += stage_course[i][1]
    
    if target_classes["wake"] == target_classes["artifact"]:
        plt.yticks([1, 2, 3, 4], ["LS", "DS", "REM", "W\&A"])
    else:
        plt.yticks([1, 2, 3, 4], ["LS", "DS", "REM", "Wake"])

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    plt.show()


if __name__ == "__main__":
    linewidth = 459.6215*pt_to_inch
    matplotlib.rcParams.update(tex_look)
    
    # multi-plots
    # fig_ratio = 4 / 3
    # linewidth *= 0.33 # 0.48, 0.5, 0.3, 0.322

    # standalone plots
    fig_ratio = 2 / 1
    linewidth *= 0.8

    matplotlib.rcParams["figure.figsize"] = [linewidth, linewidth / fig_ratio]

    # good id single: '102480.edf', '301406.edf'
    # good id multiple: '126226.edf', '298034.edf'

    plot_stage_course(
        summary_folder = "NAKO_Summary/",
        parent_folder = "/Volumes/NaKo-UniHalle/JPK_Results/NAKO_Stages_Apnea/",
        nako_file = "NAKO-609", # "NAKO-609", "NAKO-419", "NAKO-84", "NAKO-33a", "NAKO-33b", "NAKO-994"
        # model = "SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm", # "SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm", "SSG_Local_180s_FullClass_Norm"
        model = "SSG_Local_180s_FullClass_Norm",
        xlim = [0, 24],
        # wanted_id = '214126.edf',
        wanted_id = '183851.edf',
        smoothen_wake_period = True,
        show_sleep_onset_offset = True,
    )

    # plot_stage_course(
    #     summary_folder = "NAKO_Summary/",
    #     parent_folder = "/Volumes/NaKo-UniHalle/JPK_Results/NAKO_Stages_Apnea/",
    #     nako_file = "NAKO-609", # "NAKO-609", "NAKO-419", "NAKO-84", "NAKO-33a", "NAKO-33b", "NAKO-994"
    #     # model = "SSG_Local_180s_FullClass_Norm",
    #     xlim = [0, 24],
    #     wanted_id = '240895.edf', # type: ignore
    #     smoothen_wake_period = True,
    #     show_sleep_onset_offset = False,
    # )

    # collect_long_nako()
    raise SystemExit

    # path_to_model_directory = "bla/SAE_60s_AH_Norm/"
    # gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
    # gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

    # main_model_predicting_apnea_bla(
    #     path_to_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
    #     path_to_data_directory = "bla/Default_GIF_SAE_Data_All/",
    #     pid = "validation",
    #     path_to_project_configuration = path_to_model_directory + project_configuration_file,
    #     path_to_save_results = gif_complete_validation_pid_results_path,
    #     inference = True,
    # )

    # print_model_performance_no_hypopnea(
    #     paths_to_pkl_files = [gif_complete_validation_pid_results_path],
    #     path_to_project_configuration = path_to_model_directory + project_configuration_file,
    #     prediction_result_key = "Predicted_2",
    #     actual_result_key = "Actual",
    #     additional_score_function_args = {"zero_division": np.nan},
    #     number_of_decimals = 3,
    #     remove_classes=[]
    # )

    model_directory_path = "SSG_LSM_10h_120s_90s_ArtifactAsWake_LocalNorm/"

    # for i in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     collect_performance_vs_size_multi(
    #         path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #         path_to_data_directory = "Default_GIF_SSG_Data_Reduced/",
    #         pid = "validation",
    #         path_to_project_configuration = model_directory_path + project_configuration_file,
    #         path_to_save_results = "performance_vs_size.pkl",
    #         test_signal_length_seconds = int(i*3600)
    #     )

    model_directory_path = "SSG_180s_FullClass_Norm/"

    # for i in [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     collect_performance_vs_size_single(
    #         path_to_model_state = model_directory_path + model_state_after_shhs_gif_file,
    #         path_to_data_directory = "Default_GIF_SSG_Data_All/",
    #         pid = "validation",
    #         path_to_project_configuration = model_directory_path + project_configuration_file,
    #         path_to_save_results = "performance_vs_size_single.pkl",
    #         test_signal_length_seconds = int(i*3600)
    #     )

    # plot_performance_vs_size(
    #     multi_results_file_path = "performance_vs_size.pkl",
    #     single_results_file_path = "performance_vs_size_single.pkl",
    #     show_size = "h",
    #     multi_final_acc = 0.793,
    #     multi_final_kappa = 0.693,
    #     multi_final_f1 = 0.762,
    #     single_final_acc = 0.685,
    #     single_final_kappa = 0.502,
    #     single_final_f1 = 0.533,
    # )

    # plot_performance_vs_size_2(
    #     multi_results_file_path = "performance_vs_size.pkl",
    #     single_results_file_path = "performance_vs_size_single.pkl",
    #     show_size = "h",
    #     multi_final_acc = 0.793,
    #     multi_final_kappa = 0.693,
    #     multi_final_f1 = 0.762,
    #     single_final_acc = 0.685,
    #     single_final_kappa = 0.502,
    #     single_final_f1 = 0.533,
    #     xlim = [0, 11],
    #     # show = "accuracy",
    #     # ylim = [0.35, 0.85],
    #     # show = "kappa",
    #     show = "f1",
    #     ylim = [0.1, 0.9],
    # )

    path_to_results_file = "network_results.pkl"

    parent_folder = "/Volumes/NaKo-UniHalle/JPK_Results/"
    parent_folder = "/media/yaopeng/NaKo-UniHalle/JPK_Results/"
    parent_folder = "/home/yaopeng/Desktop/"

    # single_stage_chb_collecting(parent_folder_path = parent_folder + "stage_nets_single/", results_file_path = path_to_results_file)
    # multi_stage_chb_collecting(parent_folder_path = parent_folder + "stage_nets_multi/", results_file_path = path_to_results_file)

    # single_apnea_chb_collecting(parent_folder_path = parent_folder + "apnea_nets_single/", results_file_path = path_to_results_file)
    # multi_apnea_chb_collecting(parent_folder_path = parent_folder +  "apnea_nets_multi/", results_file_path = path_to_results_file)

    # send_email_notification_with_attachement("Network Results", "Here are your results!", "network_results.pkl")

    # print_compare_tables(path_to_results_file)

    path_to_results_file = "network_training_behavior.pkl"

    # loss_per_epoch_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_single/", results_file_path = path_to_results_file, task_network = "Stage_Single", best_model_folder = "SSG_Local_180s_FullClass_Norm/")
    # loss_per_epoch_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_multi/", results_file_path = path_to_results_file, task_network = "Stage_Multi", best_model_folder = "SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm/")
    # loss_per_epoch_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_single/", results_file_path = path_to_results_file, task_network = "Apnea_Single", best_model_folder = "SAE_Local_60s_A_Norm/")
    # loss_per_epoch_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_multi/", results_file_path = path_to_results_file, task_network = "Apnea_Multi", best_model_folder = "")

    # plot_loss_per_epoch(results_file_path = path_to_results_file, task_network = "Stage_Single")
    # plot_loss_per_epoch(results_file_path = path_to_results_file, task_network = "Stage_Multi", train_border = 1, shhs_border = 1, chb_border = 1)
    # plot_loss_per_epoch(results_file_path = path_to_results_file, task_network = "Apnea_Single")

    high_focus = True
    performance_mode = "Complete_Majority"
    only_show_correct_predictions = False
    actual_upper_border = False
    
    # actual_interval = [10, 1000]
    actual_interval = None

    ahi = True
    levels = [0.01, 0.05, 0.15, 0.5, 1]
    ylim = [0, 130]
    xlim = [0, 130]
    tube_size = 5

    # ahi = False
    # levels = [0.03, 0.15, 0.5, 1]
    # ylim = [0, 700]
    # xlim = [0, 700]
    # tube_size = 30
    
    predicted_interval = [10, 1000]
    # predicted_interval = None
    normalize = False
    show_kde = True
    show_scatter = True
    loc = "lower right"
    # loc = "center right"
    # loc = "upper right"
    scatter_alpha = 0.5
    skip_color = -2
    colormap = "Blues_r"

    model_path = "SAE_Local_30s_A_Norm/"
    print(model_path)

    # plot_kde_ahi(
    #     model_directory_path = model_path,
    #     performance_mode = performance_mode,
    #     sample_seconds = 30,
    #     ahi = ahi,
    #     high_focus = high_focus,
    #     only_show_correct_predictions = only_show_correct_predictions,
    #     actual_interval = actual_interval,
    #     predicted_interval = predicted_interval,
    #     show_kde = show_kde,
    #     relative_tube_size = relative_tube_size,
    #     levels = levels,
    #     actual_upper_border = actual_upper_border,
    #     normalize = normalize,
    #     loc = loc
    # )

    model_path = "SAE_Local_60s_A_Norm/"
    print(model_path)

    plot_kde_ahi(
        model_directory_path = model_path,
        performance_mode = performance_mode,
        sample_seconds = 60,
        ahi = ahi,
        high_focus = high_focus,
        only_show_correct_predictions = only_show_correct_predictions,
        actual_interval = actual_interval,
        predicted_interval = predicted_interval,
        show_kde = show_kde,
        show_scatter = show_scatter,
        tube_size = tube_size,
        levels = levels,
        actual_upper_border = actual_upper_border,
        normalize = normalize,
        scatter_alpha = scatter_alpha,
        loc = loc,
        ylim = ylim,
        xlim = xlim,
        skip_color = skip_color,
        colormap = colormap
    )
    
    model_path = "SAE_Local_120s_AH_RAW/"
    print(model_path)

    # plot_kde_ahi(
    #     model_directory_path = model_path,
    #     performance_mode = performance_mode,
    #     sample_seconds = 120,
    #     ahi = ahi,
    #     high_focus = high_focus,
    #     only_show_correct_predictions = only_show_correct_predictions,
    #     actual_interval = actual_interval,
    #     predicted_interval = predicted_interval,
    #     show_kde = show_kde,
    #     relative_tube_size = relative_tube_size,
    #     levels = levels,
    #     actual_upper_border = actual_upper_border,
    #     normalize = normalize,
    #     loc = loc
    # )
    
    model_path = "SAE_Local_180s_AH_Cleaned/"
    print(model_path)

    # plot_kde_ahi(
    #     model_directory_path = model_path,
    #     performance_mode = performance_mode,
    #     sample_seconds = 180,
    #     ahi = ahi,
    #     high_focus = high_focus,
    #     only_show_correct_predictions = only_show_correct_predictions,
    #     actual_interval = actual_interval,
    #     predicted_interval = predicted_interval,
    #     show_kde = show_kde,
    #     relative_tube_size = relative_tube_size,
    #     levels = levels,
    #     actual_upper_border = actual_upper_border,
    #     normalize = normalize,
    #     loc = loc
    # )