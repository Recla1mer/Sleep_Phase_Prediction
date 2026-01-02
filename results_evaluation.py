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
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    
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
    cleaning_names = ["RAW", "Cleaned", "Norm"]
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
    tube_size = 0,
    actual_upper_border = None,
    linear_fit = False,
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
    kwargs.setdefault("markersize", 4)

    kwargs.setdefault("levels", [0.05, 1])
    kwargs.setdefault("fill", False)
    kwargs.setdefault("colormap", 'Blues_r') # Blues, viridis_r

    kwargs.setdefault("tube_label", "error tube: ")
    kwargs.setdefault("perfect_label", "Predicted = Actual")
    kwargs.setdefault("scatter_label", "data")

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
        predicted_events = predicted_ahi
        actual_events = actual_ahi
    else:
        predicted_events = predicted_count
        actual_events = actual_count
    
    if actual_upper_border is not None:
        for i in range(len(actual_events)):
            if actual_events[i] > actual_upper_border:
                actual_events[i] = actual_upper_border
    
    global_max = max(max(predicted_events), max(actual_events))
    global_min = min(min(predicted_events), min(actual_events))

    x_axis = np.arange(0, global_max, 1)
    perfect_predicted = np.arange(0, global_max, 1)

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
    
    perfect_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][1]
    scatter_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][0]
    linear_fit_color = matplotlib.rcParams["axes.prop_cycle"].by_key()['color'][3]
    
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
            fill=False,
            levels=kwargs["levels"],
            legend=True
        )
        sns.kdeplot(
            data=data,
            x="predicted",
            y="actual",
            fill=True,
            levels=kwargs["levels"],
            cmap = kwargs["colormap"]
        )

    else:
        ax.scatter(
            predicted_events,
            actual_events,
            color = scatter_color,
            **scatter_args
        )
    if linear_fit:
        max_actual = int(3600/sample_seconds)
        cropped_actual = copy.deepcopy(actual_events)
        # for i in range(len(cropped_actual)):
        #     if cropped_actual[i] > max_actual:
        #         cropped_actual[i] = max_actual
        for i in range(len(cropped_actual)-1, -1, -1):
            if cropped_actual[i] <= 5:
                del cropped_actual[i]
                del predicted_events[i]

        m, b = np.polyfit(predicted_events, cropped_actual, 1)
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
        ax.fill_between(
            x_axis, 
            perfect_predicted - tube_size, 
            perfect_predicted + tube_size, 
            alpha=0.2, 
            color = perfect_color,
            label=kwargs["tube_label"] + str(tube_size)
        )
        
        # plt.ylim([-tube_size, global_max+tube_size])
        # plt.xlim([0, global_max])
        plt.ylim([0, global_max])
        plt.xlim([0, global_max])
    
    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    ax.legend(loc=kwargs["loc"])
    
    plt.show()


if __name__ == "__main__":
    path_to_results_file = "network_results.pkl"

    # single_stage_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_single/", results_file_path = path_to_results_file)
    # multi_stage_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_multi/", results_file_path = path_to_results_file)

    # single_apnea_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_single/", results_file_path = path_to_results_file)
    multi_apnea_chb_collecting(parent_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_multi/", results_file_path = path_to_results_file)

    # print_compare_tables(path_to_results_file)

    raise SystemExit

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
    ahi = True
    show_kde = False
    levels = [0.01, 0.05, 0.2, 0.5, 0.7, 1]
    loc = "lower right"

    model_path = "SAE_Local_30s_A_Norm/"
    print(model_path)

    plot_kde_ahi(
        model_directory_path = model_path,
        performance_mode = performance_mode,
        sample_seconds = 30,
        ahi = ahi,
        high_focus = high_focus,
        only_show_correct_predictions = only_show_correct_predictions,
        show_kde = show_kde,
        tube_size = 5,
        levels = levels,
        # actual_upper_border = 120,
        loc = loc
    )

    model_path = "SAE_Local_60s_A_Norm/"
    print(model_path)

    plot_kde_ahi(
        model_directory_path = model_path,
        performance_mode = performance_mode,
        sample_seconds = 60,
        ahi = ahi,
        high_focus = high_focus,
        only_show_correct_predictions = only_show_correct_predictions,
        show_kde = show_kde,
        tube_size = 5,
        levels = levels,
        # actual_upper_border = 60,
        loc = loc
    )
    
    model_path = "SAE_Local_120s_AH_RAW/"
    print(model_path)

    plot_kde_ahi(
        model_directory_path = model_path,
        performance_mode = performance_mode,
        sample_seconds = 120,
        ahi = ahi,
        high_focus = high_focus,
        only_show_correct_predictions = only_show_correct_predictions,
        show_kde = show_kde,
        tube_size = 5,
        levels = levels,
        # actual_upper_border = 30,
        loc = loc
    )
    
    model_path = "SAE_Local_180s_AH_Cleaned/"
    print(model_path)

    plot_kde_ahi(
        model_directory_path = model_path,
        performance_mode = performance_mode,
        sample_seconds = 180,
        ahi = ahi,
        high_focus = high_focus,
        only_show_correct_predictions = only_show_correct_predictions,
        show_kde = show_kde,
        tube_size = 5,
        levels = levels,
        # actual_upper_border = 20,
        loc = loc
    )