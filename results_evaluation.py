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
    
    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
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

    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    network_model_names = ["LSM", "LSM_Residual"]
    class_names = ["A", "AH"]

    results_file = open(results_file_path, "ab")

    print("\nMulti Apnea")
    print("\nSplitted Table Rows")
    for model_index in range(len(network_model_names)):
        print(network_model_names[model_index])
        
        for class_index in range(len(class_names)):
            print(class_names[class_index])
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]

            for clean_index in range(len(cleaning_names)):
                print(cleaning_names[clean_index])
                identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"
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
                        "architecture": network_model_names[model_index],
                        "input_seconds": 300,
                        "window_seconds": 10,
                        "overlap_seconds": 5,
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
    for model_index in range(len(network_model_names)):
        print(network_model_names[model_index])
        
        for class_index in range(len(class_names)):
            print(class_names[class_index])
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]    

            for clean_index in range(len(cleaning_names)):
                print(cleaning_names[clean_index])
                identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"
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
                        "architecture": network_model_names[model_index],
                        "input_seconds": 300,
                        "window_seconds": 10,
                        "overlap_seconds": 5,
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
    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
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

    window_and_class_names = ["Overlap_ArtifactAsWake", "NoOverlap_ArtifactAsWake", "Overlap_FullClass"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    network_model_names = ["LSM", "LSM_Residual"]

    results_file = open(results_file_path, "ab")

    print("\nMulti Stage")
    print("\nSplitted Table Rows")
    for model_index in range(len(network_model_names)):
        print(network_model_names[model_index])
        
        for window_index in range(len(window_and_class_names)):
            print(window_and_class_names[window_index])
            if window_and_class_names[window_index] == "Overlap_FullClass":
                stage_transform = [[0, 1]]
                class_name = "FullCLass"
            else:
                stage_transform = []
                class_name = "ArtifactAsWake"
            
            if window_and_class_names[window_index] == "NoOverlap_ArtifactAsWake":
                overlap = 0
            else:
                overlap = 90
            
            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"
                path_to_model_directory = parent_folder_path + identifier
                print(cleaning_names[clean_index])

                # path to save the predictions
                gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
                
                try:
                    accuracy, kappa, f1, precision, recall = calc_perf_values(
                        paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
                        path_to_project_configuration = path_to_model_directory + project_configuration_file,
                        prediction_result_key = "Predicted",
                        actual_result_key = "Actual",
                        transform = stage_transform,
                        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                    )

                    results = {
                        "method": "independent",
                        "task": "stage",
                        "architecture": network_model_names[model_index],
                        "input_seconds": 36000,
                        "window_seconds": 120,
                        "overlap_seconds": overlap,
                        "labeling": class_name,
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
    for model_index in range(len(network_model_names)):
        print(network_model_names[model_index])
        
        for window_index in range(len(window_and_class_names)):
            print(window_and_class_names[window_index])
            if window_and_class_names[window_index] == "Overlap_FullClass":
                stage_transform = [[0, 1]]
                class_name = "FullCLass"
            else:
                stage_transform = []
                class_name = "ArtifactAsWake"
            
            if window_and_class_names[window_index] == "NoOverlap_ArtifactAsWake":
                overlap = 0
            else:
                overlap = 90
            
            for clean_index in range(len(cleaning_names)):
                print(cleaning_names[clean_index])
                identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"
                path_to_model_directory = parent_folder_path + identifier

                # path to save the predictions
                gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"
                
                try:
                    accuracy, kappa, f1, precision, recall = calc_perf_values(
                        paths_to_pkl_files = [gif_complete_validation_pid_results_path],
                        path_to_project_configuration = path_to_model_directory + project_configuration_file,
                        prediction_result_key = "Predicted",
                        actual_result_key = "Actual",
                        transform = stage_transform,
                        additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
                    )

                    results = {
                        "method": "practical",
                        "task": "stage",
                        "architecture": network_model_names[model_index],
                        "input_seconds": 36000,
                        "window_seconds": 120,
                        "overlap_seconds": overlap,
                        "labeling": class_name,
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
            if data_dict[key] != combinations[key]:
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



if __name__ == "__main__":
    path_to_results_file = "network_results.pkl"

    # single_stage_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_single/", results_file_path = path_to_results_file)
    # multi_stage_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_multi/", results_file_path = path_to_results_file)

    # single_apnea_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_single/", results_file_path = path_to_results_file)
    # multi_apnea_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_multi/", results_file_path = path_to_results_file)

    possible_combinations = {
        "method": "practical", # "independent" or "practical"
        "task": "stage", # "stage" or "apnea"
        "architecture": "single", # "single", "LSM" or "LSM_Residual"
        "input_seconds": 30, # 30, 60, 120, 180, 300, 36000
        "window_seconds": 0, # 0, 10, 120
        "overlap_seconds": 0, # 0, 5, 90
        "labeling": "ArtifactAsWake", # "ArtifactAsWake", "FullClass", "A" or "AH"
        "transform": "RAW", # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    # single sleep stage

    combination = {
        "method": "practical", # "independent" or "practical"
        "task": "stage", # "stage" or "apnea"
        "architecture": "single", # "single", "LSM" or "LSM_Residual"
        # "input_seconds": 30, # 30, 60, 120, 180, 300, 36000
        # "window_seconds": 0, # 0, 10, 120
        # "overlap_seconds": 0, # 0, 5, 90
        # "labeling": "ArtifactAsWake", # "ArtifactAsWake", "FullClass", "A" or "AH"
        # "transform": "RAW", # "RAW", "Cleaned", "GlobalNorm", "LocalNorm" or "Norm"
    }

    all_combinations = list()
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = 30 # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = 60 # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = 120 # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    combination["input_seconds"] = 180 # type: ignore
    all_combinations.append(copy.deepcopy(combination))
    del combination["input_seconds"]
    combination["labeling"] = "ArtifactAsWake"
    all_combinations.append(copy.deepcopy(combination))
    combination["labeling"] = "FullClass"
    all_combinations.append(copy.deepcopy(combination))
    del combination["labeling"]
    combination["transform"] = "RAW"
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = "Cleaned"
    all_combinations.append(copy.deepcopy(combination))
    combination["transform"] = "Norm"
    all_combinations.append(copy.deepcopy(combination))
    
    combination_labels = ["Any", "\\mdhighlight{\\qty{30}{s}}", "\\mdhighlight{\\qty{60}{s}}", "\\mdhighlight{\\qty{120}{s}}", "\\mdhighlight{\\qty{180}{s}}", "\\mdhighlight{W\\&A,\\,L,\\,D,\\,R}", "\\mdhighlight{W,\\,L,\\,D,\\,R,\\,A}", "\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}"]

    compare_table(
        results_file_path = path_to_results_file,
        combinations = all_combinations,
        combination_label = combination_labels,
        round_to_decimals = 3
    )