"""
Author: Johannes Peter Knoll

This file is unnecessary for the user. I used this to test different data processing configurations and 
neural network models.
"""

from main import *
from plot_helper import *
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
            this_project_configuration["neural_network_model"] = LongSequenceModel
        elif "Yao" == directory[:3]:
            with open(directory + "/" + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            this_project_configuration["neural_network_model"] = LongSequenceResidualModel
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


def fix_project_configuration_3(directory = ""):
    """
    """
    if directory == "":
        directory = os.getcwd() + "/"
    all_files = os.listdir(directory)
    for file in all_files:
        if os.path.isdir(file):
            fix_project_configuration_3(directory + file + "/")
        if file == project_configuration_file:
            print("Fixing project configuration in", directory + project_configuration_file)
            with open(directory + project_configuration_file, "rb") as file:
                this_project_configuration = pickle.load(file)
            if "number_windows" in this_project_configuration:
                del this_project_configuration["number_windows"]
            if "nn_signal_duration_seconds" in this_project_configuration:
                del this_project_configuration["nn_signal_duration_seconds"]
            if not this_project_configuration["normalize_rri"] and not this_project_configuration["normalize_mad"]:
                for key in ["normalization_max", "normalization_min", "normalization_mode"]:
                    if key in this_project_configuration:
                        del this_project_configuration[key]
            this_project_configuration["reshape_to_overlapping_windows"] = True
            this_project_configuration["max_pooling_layers"] = 5
            if "transform" in this_project_configuration:
                this_project_configuration["feature_transform"] = this_project_configuration["transform"]
                del this_project_configuration["transform"]
        else:
            continue

        os.remove(directory + "/" + project_configuration_file)
        save_to_pickle(this_project_configuration, directory + "/" + project_configuration_file)


def fix_file_info(path):
    """
    """
    # Create temporary file to save data in progress
    working_file_path = os.path.split(copy.deepcopy(path))[0] + "/save_in_progress"
    working_file_path = find_non_existing_path(path_without_file_type = working_file_path, file_type = "pkl")

    gen = load_from_pickle(path)
    file_info = next(gen)
    if "SLP_expected_predicted_frequency" in file_info:
        del file_info["SLP_expected_predicted_frequency"]
    file_info["SLP_predicted_frequency"] = 1/30

    save_to_pickle(file_info, working_file_path)

    # Save the rest of the data
    for data in gen:
        append_to_pickle(data, working_file_path)

    os.remove(path)
    os.rename(working_file_path, path)


"""
===============
Model Training
===============
"""


def main_pipeline_SSG(
        project_configuration, 
        path_to_model_directory: str,
        neural_network_hyperparameters_shhs: dict,
        neural_network_hyperparameters_gif: dict,
        path_to_shhs_database: str,
        path_to_default_shhs_database: str,
        path_to_gif_database: str,
        path_to_default_gif_database: str,
        send_email: bool = False,
        email_subject: str = ""
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

    if not os.path.exists(path_to_model_directory + model_state_after_shhs_file):
        main_model_training_stage(
            neural_network_hyperparameters = neural_network_hyperparameters_shhs,
            path_to_training_data_directory = path_to_shhs_database,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = None,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_file,
            paths_to_validation_data_directories = [path_to_shhs_database, path_to_gif_database],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_shhs_file,
        )
    
    if not os.path.exists(path_to_model_directory + model_state_after_shhs_gif_file):
        main_model_training_stage(
            neural_network_hyperparameters = neural_network_hyperparameters_gif,
            path_to_training_data_directory = path_to_gif_database,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = path_to_model_directory + model_state_after_shhs_file,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
            paths_to_validation_data_directories = [path_to_shhs_database, path_to_gif_database],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_gif_file,
        )

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    run_model_performance_evaluation_SSG(
        path_to_model_directory = path_to_model_directory,
        path_to_splitted_shhs_directory = path_to_shhs_database,
        path_to_complete_shhs_directory = path_to_default_shhs_database,
        path_to_splitted_gif_directory = path_to_gif_database,
        path_to_complete_gif_directory = path_to_default_gif_database,
    )

    if send_email:

        buffer = io.StringIO()
        sys.stdout = buffer

        """
        -------------------------------
        Print Performance on SHHS Data
        -------------------------------
        """

        # path to save the predictions
        shhs_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_SHHS_Splitted_Validation_Pid.pkl"
        shhs_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_SHHS_Complete_Validation_Pid.pkl"

        # calculate and print performance results for splitted data
        print_headline("Performance on Splitted SHHS Data:", symbol_sequence="=")
        print("   - Reflecting performance on single data segments of required signal length")

        print_model_performance(
            paths_to_pkl_files = [shhs_splitted_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        # calculate and print performance results for complete data
        print_headline("Performance on Complete SHHS Data:", symbol_sequence="=")
        print("   - Reflecting performance on data segments of variable lengths, as they occur in practice")
        print("   - Predictions were made using inference, combining results from overlapping data segments")
        
        print_headline("Performance calculated from combined Probability:", symbol_sequence="-")

        print_model_performance(
            paths_to_pkl_files = [shhs_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        print_headline("Performance calculated from majority vote:", symbol_sequence="-")

        print_model_performance(
            paths_to_pkl_files = [shhs_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted_2",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )


        """
        ------------------------------
        Print Performance on GIF Data
        ------------------------------
        """
        
        # path to save the predictions
        gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
        gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

        # calculate and print performance results for splitted data
        print_headline("Performance on Splitted GIF Data:", symbol_sequence="=")
        print("   - Reflecting performance on single data segments of required signal length")

        print_model_performance(
            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        # calculate and print performance results for complete data
        print_headline("Performance on Complete GIF Data:", symbol_sequence="=")
        print("   - Reflecting performance on data segments of variable lengths, as they occur in practice")
        print("   - Predictions were made using inference, combining results from overlapping data segments")
        
        print_headline("Performance calculated from combined Probability:", symbol_sequence="-")

        print_model_performance(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        print_headline("Performance calculated from majority vote:", symbol_sequence="-")

        print_model_performance(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted_2",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        try:
            send_email_notification(
                email_subject=email_subject,
                email_body=buffer.getvalue()
            )
        
        except:
            pass

        buffer.truncate(0)
        buffer.seek(0)

        sys.stdout = sys.__stdout__
        del buffer


def main_pipeline_SAE(
        project_configuration, 
        path_to_model_directory: str,
        neural_network_hyperparameters_gif: dict,
        path_to_gif_database: str,
        path_to_default_gif_database: str,
        send_email: bool = False,
        email_subject: str = "",
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

    if not os.path.exists(path_to_model_directory + model_state_after_shhs_gif_file):
        main_model_training_apnea(
            neural_network_hyperparameters = neural_network_hyperparameters_gif,
            path_to_training_data_directory = path_to_gif_database,
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            path_to_model_state = None,
            path_to_updated_model_state = path_to_model_directory + model_state_after_shhs_gif_file,
            paths_to_validation_data_directories = [path_to_gif_database],
            path_to_loss_per_epoch = path_to_model_directory + loss_per_epoch_gif_file,
        )

    """
    ===========================
    Evaluate Model Performance
    ===========================
    """

    run_model_performance_evaluation_SAE(
        path_to_model_directory = path_to_model_directory,
        path_to_splitted_gif_directory = path_to_gif_database,
        path_to_complete_gif_directory = path_to_default_gif_database,
    )

    if send_email:

        buffer = io.StringIO()
        sys.stdout = buffer

        """
        ------------------------------
        Print Performance on GIF Data
        ------------------------------
        """
        
        # path to save the predictions
        gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
        gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

        # calculate and print performance results for splitted data
        print_headline("Performance on Splitted GIF Data:", symbol_sequence="=")
        print("   - Reflecting performance on single data segments of required signal length")

        print_model_performance(
            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        # calculate and print performance results for complete data
        print_headline("Performance on Complete GIF Data:", symbol_sequence="=")
        print("   - Reflecting performance on data segments of variable lengths, as they occur in practice")
        print("   - Predictions were made using inference, combining results from overlapping data segments")
        
        print_headline("Performance calculated from combined Probability:", symbol_sequence="-")

        print_model_performance(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        print_headline("Performance calculated from majority vote:", symbol_sequence="-")

        print_model_performance(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted_2",
            actual_result_key = "Actual",
            additional_score_function_args = {"zero_division": np.nan},
            number_of_decimals = 3
        )

        try:
            send_email_notification(
                email_subject=email_subject,
                email_body=buffer.getvalue()
            )
        
        except:
            pass

        buffer.truncate(0)
        buffer.seek(0)

        sys.stdout = sys.__stdout__
        del buffer
    

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
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["shhs_min_duration_hours"] * 3600
    filter_ids = project_configuration["shhs_filter_ids"]

    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')
    
    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    shhs_target_classes = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifact": ["other"]}

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
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["gif_min_duration_hours"] * 3600
    filter_ids = project_configuration["gif_filter_ids"]

    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifact": ["other"]}

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


def Reduced_Process_GIF_SSG_Dataset_h5(
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

    Others: See 'Process_SHHS_Sleep_Dataset' function
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
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

    # access parameters used for filtering the data
    minimum_length_seconds = project_configuration["gif_min_duration_hours"] * 3600
    filter_ids = project_configuration["gif_filter_ids"]

    # define the sleep stage labels (attention: a different dataset will most likely have different labels)
    gif_target_classes = {"wake": [0, 1], "LS": [2], "DS": [3], "REM": [5], "artifact": ["other"]}

    # access the GIF dataset
    gif_dataset = h5py.File(path_to_gif_dataset, 'r')

    # accessing patient ids:
    patients = list(gif_dataset['stage'].keys()) # type: ignore

    # check if patient ids are unique:
    gif_data_manager.check_if_ids_are_unique(patients)

    # showing progress bar
    print("\nEnsuring sampling frequency uniformity in the datapoints from the GIF dataset:")
    progress_bar = DynamicProgressBar(total = len(patients))

    # saving all data from GIF dataset to the pickle file
    for patient_id in patients:
        if patient_id[:5] in filter_ids:
            continue
        if len(gif_dataset["rri"][patient_id][:]) / gif_dataset["rri"].attrs["freq"] < minimum_length_seconds: # type: ignore
            continue

        new_datapoint = {
            "ID": patient_id,
            "RRI": gif_dataset["rri"][patient_id][:], # type: ignore
            "MAD": gif_dataset["mad"][patient_id][:], # type: ignore
            "SLP": np.array(gif_dataset["stage"][patient_id][:]).astype(int), # type: ignore
            "RRI_frequency": gif_dataset["rri"].attrs["freq"], # type: ignore
            "MAD_frequency": gif_dataset["mad"].attrs["freq"], # type: ignore
            "SLP_frequency": 1/30, # type: ignore
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
    distribution_params = {key: project_configuration[key] for key in ["train_size", "validation_size", "test_size", "random_state", "shuffle", "join_splitted_parts", "equally_distribute_signal_durations", "stratify_by_target", "consider_targets_for_stratification"]} # pid_distribution_parameters

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

        sae_signal = np.array(generator_entry["SAE"]).astype(int)
        lower_length = int(generator_entry["SAE_frequency"] * 0) #2
        upper_length = int(generator_entry["SAE_frequency"] * 2) #10
        i = 0
        while i < len(sae_signal)-1:
            if sae_signal[i] != 0 and sae_signal[i+1] == 0:
                current_event = sae_signal[i]

                lower_border = max(0, i-lower_length)
                upper_border = min(len(sae_signal), i+upper_length)
                
                for j in range(lower_border, upper_border):
                    sae_signal[j] = current_event
                i += upper_length
            else:
                sae_signal[i] = 0
                i += 1

        new_datapoint = {
            "ID": generator_entry["ID"],
            "RRI": generator_entry["RRI"],
            "MAD": generator_entry["MAD"],
            "SLP": sae_signal,
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


def build_default_datasets_for_training_and_testing_SSG():
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
        "stratify_by_target": False,
        "consider_targets_for_stratification": [],
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
        "stratify_by_target": False,
        "consider_targets_for_stratification": [],
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


def build_default_datasets_for_training_and_testing_SAE():
    """
    =======================
    Build Default Database
    =======================
    """

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
        "stratify_by_target": True,
        "consider_targets_for_stratification": ["Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"],
        "shhs_min_duration_hours": 0,
        "shhs_filter_ids": [],
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]# + ["SL432", "SL427", "SL422", "SL359", "SL165", "SL428", "SL007", "SL431"]
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
        "stratify_by_target": True,
        "consider_targets_for_stratification": ["Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"],
        "shhs_min_duration_hours": 7,
        "shhs_filter_ids": [],
        "gif_min_duration_hours": 7,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]# + ["SL432", "SL427", "SL422", "SL359", "SL165", "SL428", "SL007", "SL431"]
    }
    with open(limited_project_configuration_file, "wb") as file:
        pickle.dump(project_configuration, file)
    
    # Reduced_Process_GIF_SAE_Dataset(
    #     path_to_gif_dataset = original_gif_sae_data_path,
    #     path_to_save_processed_data = default_reduced_gif_SAE_path,
    #     path_to_project_configuration = limited_project_configuration_file
    #     )
    
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
        "equally_distribute_signal_durations": True,
        "stratify_by_target": False,
        "consider_targets_for_stratification": [],
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
                    path_to_default_shhs_database = default_reduced_shhs_SSG_path,
                    path_to_gif_database = gif_directory_path,
                    path_to_default_gif_database = default_reduced_gif_SSG_path,
                    send_email = global_send_email,
                    email_subject = identifier
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
        "equally_distribute_signal_durations": True,
        "stratify_by_target": False,
        "consider_targets_for_stratification": [],
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

    sixty_second_network = {
        "signal_length_seconds": 60,
        "shift_length_seconds_interval": (60, 60),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 60),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 60),
    }

    hundred_twenty_second_network = {
        "signal_length_seconds": 120,
        "shift_length_seconds_interval": (120, 120),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 120),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 120),
    }

    hundred_eighty_second_network = {
        "signal_length_seconds": 180,
        "shift_length_seconds_interval": (180, 180),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 180),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 180),
    }

    hyperparameters_shhs = {
        "batch_size": 128, # 4.2h for 120s data | 1.5M (1484839) / 128 => 11601 steps per epoch
        "number_epochs": 20,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 2,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 20,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 2,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    class_adjustments = [artifact_as_wake, full_class]
    class_names = ["ArtifactAsWake", "FullClass"]

    cleaning_adjustments = [raw, cleaned, norm]
    cleaning_names = ["RAW", "Cleaned", "Norm"]

    network_adjustments = [hundred_eighty_second_network, hundred_twenty_second_network, sixty_second_network, thirty_second_network]
    network_names = ["Local_180s", "Local_120s", "Local_60s", "Local_30s"]

    # different networks have different signal cropping parameters, so we need to create a database for each network
    shhs_directory_paths = ["180s_SHHS_SSG_Data/", "120s_SHHS_SSG_Data/", "60s_SHHS_SSG_Data/", "30s_SHHS_SSG_Data/"]
    gif_directory_paths = ["180s_GIF_SSG_Data/", "120s_GIF_SSG_Data/", "60s_GIF_SSG_Data/", "30s_GIF_SSG_Data/"]
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
                    neural_network_hyperparameters_shhs = hyperparameters_shhs,
                    neural_network_hyperparameters_gif = hyperparameters_gif,
                    path_to_shhs_database = shhs_directory_paths[network_index],
                    path_to_default_shhs_database = default_complete_shhs_SSG_path,
                    path_to_gif_database = gif_directory_paths[network_index],
                    path_to_default_gif_database = default_complete_gif_SSG_path,
                    send_email = global_send_email,
                    email_subject = identifier
                )


def train_and_test_long_sequence_model_varying_duration_on_apnea_events():

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

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None), # (0.3, 2)
        "mad_inlier_interval": (None, None),
        "target_classes": {"Normal": 0, "Mixed Apnea": 3, "Apnea": 3, "Obstructive Apnea": 1, "Central Apnea": 2, "Hypopnea": 4, "Obstructive Hypopnea": 4, "Central Hypopnea": 4},
    }

    pid_distribution_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True,
        "stratify_by_target": True,
        "consider_targets_for_stratification": ["Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"],
    }

    dataset_class_transform_parameters = {
        "feature_transform": custom_transform,
        "target_transform": None,
    }

    window_reshape_parameters = {
        "reshape_to_overlapping_windows": True,
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
    }

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]
    }

    default_project_configuration = dict()
    default_project_configuration.update(sampling_frequency_parameters)
    default_project_configuration.update(padding_parameters)
    default_project_configuration.update(value_mapping_parameters)
    default_project_configuration.update(pid_distribution_parameters)
    default_project_configuration.update(window_reshape_parameters)
    default_project_configuration.update(signal_normalization_parameters)
    default_project_configuration.update(dataset_class_transform_parameters)
    default_project_configuration.update(neural_network_model_parameters)
    default_project_configuration.update(filter_gif_data_parameters)

    apnea_hypopnea_type = {
        "number_target_classes": 5, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"Normal": 0, "Mixed Apnea": 3, "Apnea": 3, "Obstructive Apnea": 1, "Central Apnea": 2, "Hypopnea": 4, "Obstructive Hypopnea": 4, "Central Hypopnea": 4},
    }

    apnea_hypopnea = {
        "number_target_classes": 3, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"Normal": 0, "Apnea": 1, "Obstructive Apnea": 1, "Central Apnea": 1, "Mixed Apnea": 1, "Hypopnea": 2, "Obstructive Hypopnea": 2, "Central Hypopnea": 2},
    }

    apnea_event = {
        "number_target_classes": 2, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"Normal": 0, "Apnea": 1, "Obstructive Apnea": 1, "Central Apnea": 1, "Mixed Apnea": 1, "Hypopnea": 1, "Obstructive Hypopnea": 1, "Central Hypopnea": 1},
    }

    # class_adjustments = [apnea_hypopnea_type, apnea_hypopnea, apnea_event]
    # class_names = ["AHT", "AH", "AE"]
    class_adjustments = [apnea_event, apnea_hypopnea]
    class_names = ["A", "AH"]

    two_minute_network = {
        "signal_length_seconds": 120,
        "shift_length_seconds_interval": (30, 30),
        "windows_per_signal": 23,
        "window_duration_seconds": 10,
        "overlap_seconds": 5,
        "priority_order": [3, 2, 4, 1, 0],
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * 10),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * 10),
    }

    five_minute_network = {
        "signal_length_seconds": 300,
        "shift_length_seconds_interval": (75, 75),
        "windows_per_signal": 59,
        "window_duration_seconds": 10,
        "overlap_seconds": 5,
        "priority_order": [3, 2, 4, 1, 0],
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * 10),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * 10),
    }

    ten_minute_network = {
        "signal_length_seconds": 600,
        "shift_length_seconds_interval": (600, 600),
        "windows_per_signal": 119,
        "window_duration_seconds": 10,
        "overlap_seconds": 5,
        "priority_order": [3, 2, 4, 1, 0],
        "datapoints_per_rri_window": int(sampling_frequency_parameters["RRI_frequency"] * 10),
        "datapoints_per_mad_window": int(sampling_frequency_parameters["MAD_frequency"] * 10),
    }

    hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 20,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 4,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    del sampling_frequency_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_gif_data_parameters

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

    cleaning_adjustments = [raw, cleaned, global_norm, local_norm]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]

    network_models = [LongSequenceModel, LongSequenceResidualModel]
    network_model_names = ["LSM", "LSM_Residual"]

    # network_adjustments = [two_minute_network, five_minute_network, ten_minute_network]
    # network_names = ["Multiple_2min", "Multiple_5min", "Multiple_10min"]
    network_adjustments = [five_minute_network]
    network_names = ["Multiple_5min"]

    # different networks have different signal cropping parameters, so we need to create a database for each network
    # gif_directory_paths = ["2min_GIF_SAE_Data/", "5min_GIF_SAE_Data/", "10min_GIF_SAE_Data/"]
    gif_directory_paths = ["5min_GIF_SAE_Data/"]
    for net_adjust_index in range(len(network_adjustments)):
        copy_and_split_default_database_SAE(
            path_to_default_gif_database = default_complete_gif_SAE_path,
            path_to_save_gif_database = gif_directory_paths[net_adjust_index],
            project_configuration = network_adjustments[net_adjust_index]
        )
    
    # here you could remove some data without apnea event
    # i hoped it would provide better results, as the model would train on more equal amounts of apnea vs. normal breathing data
    # however, this does not really help, as you get better recall but worse precision
    if False:
        for path_index in range(len(gif_directory_paths)):
            train_file_path = gif_directory_paths[path_index] + "training_pid.pkl"
            train_data_offset_path = gif_directory_paths[path_index] + ".training_pid_offset.pkl"
            configuration_path = gif_directory_paths[path_index] + "configuration.pkl"
            
            train_file_generator = load_from_pickle(train_file_path)
            collect_some_data = list()

            count_high_proportion = 0
            for data_point in train_file_generator:
                if count_high_proportion == 10:
                    count_high_proportion = 0

                sae_signal = np.array(data_point["SLP"])
                proportion_normal_breathing = (sae_signal == 0).sum() / len(sae_signal)
                if sae_signal.sum() == 0:# and random.random() < 0.8:
                    count_high_proportion += 1
                    if count_high_proportion > 3:
                        continue

                collect_some_data.append(data_point)
            
            os.remove(train_file_path)
            os.remove(train_data_offset_path)
            new_file = open(train_file_path, "ab")

            train_byte_offsets = []
            for data_point in collect_some_data:
                train_byte_offsets.append(new_file.tell())
                pickle.dump(data_point, new_file)
            
            with open(train_data_offset_path, "wb") as f:
                pickle.dump(train_byte_offsets, f)
            
            with open(configuration_path, "rb") as f:
                database_configuration = pickle.load(f)
            
            database_configuration["number_datapoints"][1] = len(collect_some_data)

            with open(configuration_path, "wb") as f:
                pickle.dump(database_configuration, f)

    for clean_index in range(len(cleaning_adjustments)):
        for class_index in range(len(class_adjustments)):
            for model_index in range(len(network_models)):
                for network_index in range(len(network_adjustments)):

                    project_configuration = copy.deepcopy(default_project_configuration)
                    project_configuration.update(cleaning_adjustments[clean_index])
                    project_configuration.update(class_adjustments[class_index])
                    project_configuration["neural_network_model"] = network_models[model_index]
                    project_configuration.update(network_adjustments[network_index])

                    identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index]
                    print_headline("Running " + identifier, "=")

                    identifier += "/"

                    main_pipeline_SAE(
                        project_configuration = project_configuration,
                        path_to_model_directory = identifier,
                        neural_network_hyperparameters_gif = hyperparameters_gif,
                        path_to_gif_database = gif_directory_paths[network_index],
                        path_to_default_gif_database = default_complete_gif_SAE_path,
                        send_email = global_send_email,
                        email_subject = identifier
                    )
    
    del project_configuration, default_project_configuration # type: ignore


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

    padding_parameters = {
        "pad_feature_with": 0,
        "pad_target_with": 0
    }

    value_mapping_parameters = {
        "rri_inlier_interval": (None, None),
        "mad_inlier_interval": (None, None),
        "target_classes": {"Normal": 0, "Mixed Apnea": 1, "Apnea": 1, "Obstructive Apnea": 1, "Central Apnea": 1, "Hypopnea": 2, "Obstructive Hypopnea": 2, "Central Hypopnea": 2},
    }

    pid_distribution_parameters = {
        "train_size": 0.8,
        "validation_size": 0.2,
        "test_size": None,
        "random_state": None,
        "shuffle": True,
        "join_splitted_parts": True,
        "equally_distribute_signal_durations": True,
        "stratify_by_target": True,
        "consider_targets_for_stratification": ["Mixed Apnea", "Hypopnea"],
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
    }

    filter_gif_data_parameters = {
        "gif_min_duration_hours": 0,
        "gif_filter_ids": gif_error_code_4 + gif_error_code_5 + ["SL067"]
    }

    default_project_configuration = dict()
    default_project_configuration.update(sampling_frequency_parameters)
    default_project_configuration.update(padding_parameters)
    default_project_configuration.update(value_mapping_parameters)
    default_project_configuration.update(pid_distribution_parameters)
    default_project_configuration.update(window_reshape_parameters)
    default_project_configuration.update(signal_normalization_parameters)
    default_project_configuration.update(dataset_class_transform_parameters)
    default_project_configuration.update(neural_network_model_parameters)
    default_project_configuration.update(filter_gif_data_parameters)

    del sampling_frequency_parameters, padding_parameters, value_mapping_parameters, pid_distribution_parameters, dataset_class_transform_parameters, window_reshape_parameters, signal_normalization_parameters, neural_network_model_parameters, filter_gif_data_parameters

    apnea_hypopnea_type = {
        "number_target_classes": 5, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"Normal": 0, "Mixed Apnea": 3, "Apnea": 3, "Obstructive Apnea": 1, "Central Apnea": 2, "Hypopnea": 4, "Obstructive Hypopnea": 4, "Central Hypopnea": 4},
    }

    apnea_hypopnea = {
        "number_target_classes": 3, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"Normal": 0, "Apnea": 1, "Obstructive Apnea": 1, "Central Apnea": 1, "Mixed Apnea": 1, "Hypopnea": 2, "Obstructive Hypopnea": 2, "Central Hypopnea": 2},
    }

    apnea_event = {
        "number_target_classes": 2, # 5 sleep stages: wake, LS, DS, REM, artifact
        "target_classes": {"Normal": 0, "Apnea": 1, "Obstructive Apnea": 1, "Central Apnea": 1, "Mixed Apnea": 1, "Hypopnea": 1, "Obstructive Hypopnea": 1, "Central Hypopnea": 1},
    }

    class_adjustments = [apnea_event, apnea_hypopnea]
    class_names = ["A", "AH"]

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

    ten_second_network = {
        "signal_length_seconds": 10,
        "shift_length_seconds_interval": (10, 10),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 10),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 10),
    }

    sixteen_second_network = {
        "signal_length_seconds": 16,
        "shift_length_seconds_interval": (16, 16),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 16),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 16),
    }

    thirty_second_network = {
        "signal_length_seconds": 30,
        "shift_length_seconds_interval": (10, 10),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 30),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 30),
    }

    sixty_second_network = {
        "signal_length_seconds": 60,
        "shift_length_seconds_interval": (15, 15),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 60),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 60),
    }

    hundred_twenty_second_network = {
        "signal_length_seconds": 120,
        "shift_length_seconds_interval": (30, 30),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 120),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 120),
    }

    hundred_eighty_second_network = {
        "signal_length_seconds": 180,
        "shift_length_seconds_interval": (45, 45),
        "rri_datapoints": int(default_project_configuration["RRI_frequency"] * 180),
        "mad_datapoints": int(default_project_configuration["MAD_frequency"] * 180),
    }

    hyperparameters_gif = {
        "batch_size": 8,
        "number_epochs": 20,
        "lr_scheduler_parameters": {
            "number_updates_to_max_lr": 2,
            "start_learning_rate": 1 * 1e-5,
            "max_learning_rate": 1 * 1e-3,
            "end_learning_rate": 1 * 1e-6
        }
    }

    cleaning_adjustments = [raw, cleaned, norm]
    cleaning_names = ["RAW", "Cleaned", "Norm"]

    network_adjustments = [hundred_eighty_second_network, hundred_twenty_second_network, sixty_second_network, thirty_second_network]
    network_names = ["Local_180s", "Local_120s", "Local_60s", "Local_30s"]
    # network_adjustments = [hundred_eighty_second_network]
    # network_names = ["Local_180s"]

    # different networks have different signal cropping parameters, so we need to create a database for each network
    gif_directory_paths = ["180s_GIF_SAE_Data/", "120s_GIF_SAE_Data/", "60s_GIF_SAE_Data/", "30s_GIF_SAE_Data/"]
    # gif_directory_paths = ["180s_GIF_SAE_Data/"]

    for net_adjust_index in range(len(network_adjustments)):
        copy_and_split_default_database_SAE(
            path_to_default_gif_database = default_complete_gif_SAE_path,
            path_to_save_gif_database = gif_directory_paths[net_adjust_index],
            project_configuration = network_adjustments[net_adjust_index]
        )
    
    # here you could remove some data without apnea event
    # i hoped it would provide better results, as the model would train on more equal amounts of apnea vs. normal breathing data
    # however, this does not really help, as you get better recall but worse precision
    if False:
        remove_training_data_from_paths = []
        for path in gif_directory_paths:
            if not os.path.exists(path):
                remove_training_data_from_paths.append(path)
        
        for path_index in range(len(remove_training_data_from_paths)):
            train_file_path = remove_training_data_from_paths[path_index] + "training_pid.pkl"
            train_data_offset_path = remove_training_data_from_paths[path_index] + ".training_pid_offset.pkl"
            configuration_path = remove_training_data_from_paths[path_index] + "configuration.pkl"
            
            train_file_generator = load_from_pickle(train_file_path)
            collect_some_data = list()

            count_high_proportion = 0
            for data_point in train_file_generator:
                if count_high_proportion == 10:
                    count_high_proportion = 0

                sae_signal = np.array(data_point["SLP"])
                proportion_normal_breathing = (sae_signal == 0).sum() / len(sae_signal)
                if sae_signal.sum() == 0:# and random.random() < 0.8:
                    count_high_proportion += 1
                    # if count_high_proportion > 10:
                    #     continue

                collect_some_data.append(data_point)
            
            os.remove(train_file_path)
            os.remove(train_data_offset_path)
            new_file = open(train_file_path, "ab")

            train_byte_offsets = []
            for data_point in collect_some_data:
                train_byte_offsets.append(new_file.tell())
                pickle.dump(data_point, new_file)
            
            with open(train_data_offset_path, "wb") as f:
                pickle.dump(train_byte_offsets, f)
            
            with open(configuration_path, "rb") as f:
                database_configuration = pickle.load(f)
            
            database_configuration["number_datapoints"][1] = len(collect_some_data)

            with open(configuration_path, "wb") as f:
                pickle.dump(database_configuration, f)

    for clean_index in range(len(cleaning_adjustments)):
        for class_index in range(len(class_adjustments)):
            for network_index in range(len(network_adjustments)):
                project_configuration = copy.deepcopy(default_project_configuration)
                project_configuration.update(cleaning_adjustments[clean_index])
                project_configuration.update(class_adjustments[class_index])
                project_configuration.update(network_adjustments[network_index])

                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index]
                print_headline("Running " + identifier, "=")

                identifier += "/"

                main_pipeline_SAE(
                    project_configuration = project_configuration,
                    path_to_model_directory = identifier,
                    neural_network_hyperparameters_gif = hyperparameters_gif,
                    path_to_gif_database = gif_directory_paths[network_index],
                    path_to_default_gif_database = default_complete_gif_SAE_path,
                    send_email = global_send_email,
                    email_subject = identifier
                )


if __name__ == "__main__":

    # run_model_performance_evaluation_SAE(
    #     path_to_model_directory = "SAE_Local_120s_A_RAW/",
    #     path_to_splitted_gif_directory = "120s_GIF_SAE_Data/",
    #     path_to_complete_gif_directory = "Default_GIF_SAE_Data_All/",
    # )

    # build_default_datasets_for_training_and_testing_SSG()
    # build_default_datasets_for_training_and_testing_SAE()

    try:
        # train_and_test_long_sequence_model_on_sleep_staging_data()
        pass
    except:
        pass

    try:
        # train_and_test_short_sequence_model_on_sleep_staging_data()
        pass
    except:
        pass

    try:
        # train_and_test_long_sequence_model_on_apnea_events()
        pass
    except:
        pass

    try:
        # train_and_test_long_sequence_model_varying_duration_on_apnea_events()
        pass
    except:
        pass

    try:
        # train_and_test_short_sequence_model_on_apnea_events()
        pass
    except:
        pass

    if True:
        nako_directory = "Processed_NAKO/"
        nako_paths = [nako_directory + "NAKO-994.pkl", nako_directory + "NAKO-609.pkl", nako_directory + "NAKO-419.pkl", nako_directory + "NAKO-84.pkl", nako_directory + "NAKO-33a.pkl", nako_directory + "NAKO-33b.pkl"]
        nako_size = [37059, 267752, 223486, 255086, 7365, 9691]

        stage_prediction_paths = ["SSG_LSM_Residual_Overlap_ArtifactAsWake_LocalNorm/", "SSG_Local_180s_FullClass_Norm/", "SSG_Local_120s_ArtifactAsWake_Cleaned/"]
        stage_prediction_keys = ["SSG_LSM", "SSG_Local_180s", "SSG_Local_120s"]
        
        apnea_prediction_paths = ["SAE_Local_60s_A_Norm/", "SAE_Local_120s_AH_RAW/"]
        apnea_prediction_keys = ["SAE_Local_60s", "SAE_Local_120s"]

        for nako_path_index in range(len(nako_paths)):
            path = nako_paths[nako_path_index]

            print_headline(f"Predicting Sleep Stages and Apnea Events within: {path}")

            for stage_path_index in range(len(stage_prediction_paths)):
                main_model_predicting_stage_inference(
                    path_to_model_state = stage_prediction_paths[stage_path_index] + model_state_after_shhs_gif_file,
                    path_to_data_directory = path,
                    path_to_project_configuration = stage_prediction_paths[stage_path_index] + project_configuration_file,
                    path_to_save_results = path,
                    inference = True,
                    results_key = stage_prediction_keys[stage_path_index],
                    data_length = nako_size[nako_path_index]
                )

            for apnea_path_index in range(len(apnea_prediction_paths)):
                main_model_predicting_apnea_inference(
                    path_to_model_state = apnea_prediction_paths[apnea_path_index] + model_state_after_shhs_gif_file,
                    path_to_data_directory = path,
                    path_to_project_configuration = apnea_prediction_paths[apnea_path_index] + project_configuration_file,
                    path_to_save_results = path,
                    inference = True,
                    results_key = apnea_prediction_keys[apnea_path_index],
                    data_length = nako_size[nako_path_index]
                )


if False:
    build_default_datasets_for_training_and_testing()

    buffer = io.StringIO()
    sys.stdout = buffer

    # sys.stdout = sys.__stdout__

    success = 0

    try:
        train_and_test_long_sequence_model_on_sleep_staging_data()

        print(buffer.getvalue())

        send_email_notification(
            email_subject="LSM on Sleep Staging - Completed",
            email_body=buffer.getvalue()
        )

        success += 1
    
    except:
        pass

    finally:
        buffer.truncate(0)
        buffer.seek(0)

    try:
        train_and_test_short_sequence_model_on_sleep_staging_data()

        print(buffer.getvalue())

        send_email_notification(
            email_subject="SSM on Sleep Staging - Completed",
            email_body=buffer.getvalue()
        )

        success += 1
    
    except:
        pass

    finally:
        buffer.truncate(0)
        buffer.seek(0)

    try:
        train_and_test_long_sequence_model_on_apnea_events()

        print(buffer.getvalue())

        send_email_notification(
            email_subject="LSM on Apnea Events - Completed",
            email_body=buffer.getvalue()
        )

        success += 1
    
    except:
        pass

    finally:
        buffer.truncate(0)
        buffer.seek(0)

    try:
        train_and_test_long_sequence_model_varying_duration_on_apnea_events()

        print(buffer.getvalue())

        send_email_notification(
            email_subject="Varying LSM on Apnea Events - Completed",
            email_body=buffer.getvalue()
        )

        success += 1
    
    except:
        pass

    finally:
        buffer.truncate(0)
        buffer.seek(0)

    try:
        train_and_test_short_sequence_model_on_apnea_events()

        print(buffer.getvalue())

        send_email_notification(
            email_subject="SSM on Apnea Events - Completed",
            email_body=buffer.getvalue()
        )

        success += 1
    
    except:
        pass

    finally:
        buffer.truncate(0)
        buffer.seek(0)
    
    send_email_notification(
        email_subject="All Training and Testing Completed",
        email_body=f"All training and testing pipelines have been completed successfully. ({success}/5)"
    )


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