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


def single_apnea_confusion_plot(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
        values_format = 'd', #'d', '.2g'
        normalize = None,
        save_folder = "/Users/propeter/Desktop/master_thesis/confusion_matrices/"
):

    network_names = ["30s", "60s", "120s", "180s"]
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    class_names = ["A", "AH"]

    print("\nSingle Apnea")

    if run_splitted:
        print("\nSplitted Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Splitted",
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            remove_classes = [],
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Complete",
                            prediction_result_key = "Predicted_2",
                            actual_result_key = "Actual",
                            remove_classes = [],
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")


def multi_apnea_confusion_plot(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
        values_format = 'd', #'d', '.2g'
        normalize = None,
        save_folder = "/Users/propeter/Desktop/master_thesis/confusion_matrices/"
):

    data_strucutre_names = ["300s_10s_5s", "300s_10s_0s"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    class_names = ["A", "AH"]
    network_model_names = ["LSM", "LSM_Residual"]

    print("\nMulti Apnea")

    if run_splitted:
        print("\nSplitted Table Rows")
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Splitted",
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            remove_classes = [],
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Complete",
                            prediction_result_key = "Predicted_2",
                            actual_result_key = "Actual",
                            remove_classes = [],
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")

def single_stage_confusion_plot(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
        values_format = 'd', #'d', '.2g'
        normalize = None,
        save_folder = "/Users/propeter/Desktop/master_thesis/confusion_matrices/"
    ):

    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    class_names = ["ArtifactAsWake", "FullClass"]

    print("\nSingle Stage")

    if run_splitted:
        print("\nSplitted Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "ArtifactAsWake":
                    map_classes = [["Wake", "W\&A"]]
                    remove_classes = []
                elif class_names[class_index] == "FullClass":
                    map_classes = []
                    remove_classes = ["artifact"]
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Splitted",
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            remove_classes = remove_classes,
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "ArtifactAsWake":
                    map_classes = [["Wake", "W\&A"]]
                    remove_classes = []
                elif class_names[class_index] == "FullClass":
                    map_classes = []
                    remove_classes = ["artifact"]
                else:
                    raise SystemError("Unknown class name")
                    
                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Complete",
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            remove_classes = remove_classes,
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")


def multi_stage_confusion_plot(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
        values_format = 'd', #'d', '.2g'
        normalize = None,
        save_folder = "/Users/propeter/Desktop/master_thesis/confusion_matrices/"
    ):

    window_and_class_names = ["Overlap_ArtifactAsWake", "Overlap_FullClass", "NoOverlap_ArtifactAsWake"]
    network_model_names = ["LSM", "LSM_Residual"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]

    print("\nMulti Stage")

    if run_splitted:
        print("\nSplitted Table Rows")
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for window_index in range(len(window_and_class_names)):
                print(window_and_class_names[window_index])
                if "ArtifactAsWake" in window_and_class_names[window_index]:
                    map_classes = [["Wake", "W\&A"]]
                    remove_classes = []
                elif "FullClass" in window_and_class_names[window_index]:
                    map_classes = []
                    remove_classes = ["artifact"]
                else:
                    raise SystemError("Unknown class name")
                
                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Splitted",
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            remove_classes = remove_classes,
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for window_index in range(len(window_and_class_names)):
                print(window_and_class_names[window_index])
                if "ArtifactAsWake" in window_and_class_names[window_index]:
                    map_classes = [["Wake", "W\&A"]]
                    remove_classes = []
                elif "FullClass" in window_and_class_names[window_index]:
                    map_classes = []
                    remove_classes = ["artifact"]
                else:
                    raise SystemError("Unknown class name")
                
                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"

                    try:
                        print(identifier[:-1])
                        plot_confusion_matrix(
                            path_to_model_directory = parent_folder_path + identifier,
                            dataset = "GIF_Complete",
                            prediction_result_key = "Predicted",
                            actual_result_key = "Actual",
                            remove_classes = remove_classes,
                            values_format = values_format,
                            normalize = normalize,
                            map_classes = map_classes,
                            save_path = save_folder + identifier[:-1] + ".pdf"
                        )
                    except:
                        print("Folder not found.")


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


def create_latex_table_row(
        path_to_model_directory: str,
        performance_of: str, # "Splitted", "Complete_Probability", "Complete_Majority"
        transform = [],
        round_to_decimals = 3,
        ):
    """
    """

    # path to save the predictions
    gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
    gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

    if performance_of == "Splitted":
        accuracy, kappa, f1, _, _ = calc_perf_values(
            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            transform = transform,
            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
        )

    elif performance_of == "Complete_Probability":
        accuracy, kappa, f1, _, _ = calc_perf_values(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            transform = transform,
            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
        )

    elif performance_of == "Complete_Majority":
        accuracy, kappa, f1, _, _ = calc_perf_values(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted_2",
            actual_result_key = "Actual",
            transform = transform,
            additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
        )
    else:
        raise ValueError("Unknown \'performance_of\' value.")
    
    accuracy = round(accuracy, round_to_decimals)
    kappa = round(kappa, round_to_decimals)
    f1 = round(f1, round_to_decimals) # type: ignore
    
    return "\\num{" + str(accuracy) + "} & \\num{" + str(kappa) + "} & \\num{" + str(f1) + "} \\\\"


def single_apnea_table():

    cleaning_names = ["RAW", "Cleaned", "Norm"]
    latex_cleaning = ["\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}"]
    network_names = ["30s", "60s", "120s", "180s"]
    class_names = ["A", "AH"]

    print("\nSingle Apnea")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        for class_index in range(len(class_names)):
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]
            
            print(class_names[class_index])

            for clean_index in range(len(cleaning_names)):
                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + " & "
                try:
                    text += create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Splitted", #"Complete_Probability", "Complete_Majority"
                                transform = apnea_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                
                print(text)
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        for class_index in range(len(class_names)):
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]
            
            print(class_names[class_index])

            for clean_index in range(len(cleaning_names)):
                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + " & "
                try:
                    text += create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Complete_Majority",
                                transform = apnea_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                
                print(text)


def multi_apnea_table():

    data_strucutre_names = ["300s_10s_0s", "300s_10s_5s"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    latex_cleaning = ["\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}", "\\mdhighlight{WindowNorm}"]
    network_model_names = ["LSM", "LSM_Residual"]
    class_names = ["A", "AH"]

    print("\nMulti Apnea")
    print("\nSplitted Table Rows")
    for model_index in range(len(network_model_names)):
        print(network_model_names[model_index])
        for data_struct_index in range(len(data_strucutre_names)):
            print(data_strucutre_names[data_struct_index])
            for class_index in range(len(class_names)):
                if class_names[class_index] == "A":
                    apnea_transform = []
                else:
                    apnea_transform = [[2,1]]
                
                print(class_names[class_index])

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + network_model_names[model_index] + "_" + data_strucutre_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    text = "& " + latex_cleaning[clean_index] + " & "
                    try:
                        text += create_latex_table_row(
                                    path_to_model_directory = identifier,
                                    performance_of = "Splitted",
                                    transform = apnea_transform,
                                    round_to_decimals = 3,
                                )
                    except:
                        text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                    
                    print(text)
    
    print("\nCombined Table Rows")
    for model_index in range(len(network_model_names)):
        print(network_model_names[model_index])
        for data_struct_index in range(len(data_strucutre_names)):
            print(data_strucutre_names[data_struct_index])
            for class_index in range(len(class_names)):
                if class_names[class_index] == "A":
                    apnea_transform = []
                else:
                    apnea_transform = [[2,1]]
                
                print(class_names[class_index])

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + network_model_names[model_index] + "_" + data_strucutre_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    text = "& " + latex_cleaning[clean_index] + " & "
                    try:
                        text += create_latex_table_row(
                                    path_to_model_directory = identifier,
                                    performance_of = "Complete_Majority",
                                    transform = apnea_transform,
                                    round_to_decimals = 3,
                                )
                    except:
                        text += "\\num{0} & \\num{0} & \\num{0} \\\\"

                    print(text)

def single_stage_table():

    class_names = ["ArtifactAsWake", "FullClass"]
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    latex_cleaning = ["\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}"]
    network_names = ["30s", "60s", "120s", "180s"]
    sleep_transform = []

    print("\nSingle Stage")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        for class_index in range(len(class_names)):
            if class_names[class_index] == "ArtifactAsWake":
                sleep_transform = []
            else:
                sleep_transform = [[0, 1]]
            
            print(class_names[class_index])

            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + " & "
                try:
                    text += create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Splitted",
                                transform = sleep_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                
                print(text)
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_names)):
        print(network_names[network_index])
        for class_index in range(len(class_names)):
            if class_names[class_index] == "ArtifactAsWake":
                sleep_transform = []
            else:
                sleep_transform = [[0, 1]]
            
            print(class_names[class_index])
                
            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + " & "
                try:
                    text += create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Complete_Probability",
                                transform = sleep_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"

                print(text)


def multi_stage_table():

    data_strucutre_names = ["10h_120s_0s", "10h_120s_90s"]
    class_names = ["ArtifactAsWake", "FullClass"]
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    latex_cleaning = ["\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}"]
    network_model_names = ["LSM", "LSM_Residual"]
    sleep_transform = []

    print("\nMulti Stage")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_model_names)):
        print(network_model_names[network_index])
        for data_struct_index in range(len(data_strucutre_names)):
            print(data_strucutre_names[data_struct_index])
            for class_index in range(len(class_names)):
                if class_names[class_index] == "ArtifactAsWake":
                    sleep_transform = []
                else:
                    sleep_transform = [[0, 1]]
                
                print(class_names[class_index])

                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_model_names[network_index] + "_" + data_strucutre_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    text = "& " + latex_cleaning[clean_index] + " & "
                    try:
                        text += create_latex_table_row(
                                    path_to_model_directory = identifier,
                                    performance_of = "Splitted",
                                    transform = sleep_transform,
                                    round_to_decimals = 3,
                                )
                    except:
                        text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                    
                    print(text)
    
    print("\nCombined Table Rows")
    for network_index in range(len(network_model_names)):
        print(network_model_names[network_index])
        for data_struct_index in range(len(data_strucutre_names)):
            print(data_strucutre_names[data_struct_index])
            for class_index in range(len(class_names)):
                if class_names[class_index] == "ArtifactAsWake":
                    sleep_transform = []
                else:
                    sleep_transform = [[0, 1]]
                
                print(class_names[class_index])

                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_model_names[network_index] + "_" + data_strucutre_names[data_struct_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                    text = "& " + latex_cleaning[clean_index] + " & "
                    try:
                        text += create_latex_table_row(
                                    path_to_model_directory = identifier,
                                    performance_of = "Complete_Probability",
                                    transform = sleep_transform,
                                    round_to_decimals = 3,
                                )
                    except:
                        text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                    
                    print(text)


def create_performance_box(
        path_to_model_directory: str,
        performance_of: str, # "Splitted", "Complete_Probability", "Complete_Majority"
        transform = [],
        round_to_decimals = 3,
        order = [],
        modeling_task = "",
        network_architecture = "",
        sample_structure = "",
        signal_transformation = "",
        labeling_strategy = "",
        file_name = "",
        map_classes = []
        ):
    """
    """

    # path to save the predictions
    gif_splitted_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Splitted_Validation_Pid.pkl"
    gif_complete_validation_pid_results_path = path_to_model_directory + model_performance_file[:-4] + "_GIF_Complete_Validation_Pid.pkl"

    if performance_of == "Splitted":
        accuracy, kappa, f1, precision, recall = calc_perf_values(
            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            transform = transform,
            additional_score_function_args = {"zero_division": np.nan, "average": None}
        )

    elif performance_of == "Complete_Probability":
        accuracy, kappa, f1, precision, recall = calc_perf_values(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            transform = transform,
            additional_score_function_args = {"zero_division": np.nan, "average": None}
        )

    elif performance_of == "Complete_Majority":
        accuracy, kappa, f1, precision, recall = calc_perf_values(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted_2",
            actual_result_key = "Actual",
            transform = transform,
            additional_score_function_args = {"zero_division": np.nan, "average": None}
        )
    else:
        raise ValueError("Unknown \'performance_of\' value.")
    
    accuracy = round(accuracy, round_to_decimals)
    kappa = round(kappa, round_to_decimals)

    # Round the results
    f1 = np.round(f1, round_to_decimals) # type: ignore
    precision = np.round(precision, round_to_decimals) # type: ignore
    recall = np.round(recall, round_to_decimals) # type: ignore

    # load signal processing parameters
    path_to_project_configuration = path_to_model_directory + project_configuration_file
    with open(path_to_project_configuration, "rb") as f:
        project_configuration = pickle.load(f)

    # access dictionary that maps sleep stages (display labels) to integers
    sleep_stage_to_label = project_configuration["target_classes"]

    # Create a list of the integer labels, sorted
    integer_labels = np.array([value for value in sleep_stage_to_label.values()])
    integer_labels = np.unique(integer_labels)
    integer_labels.sort()

    if len(order) != len(integer_labels):
        order = integer_labels

    # Create a list of the display labels
    display_labels = []
    for integer_label in integer_labels:
        for key, value in sleep_stage_to_label.items():
            if value == integer_label:
                display_labels.append(key)
                break
    
    for label_index in range(len(display_labels)):
        if display_labels[label_index] == "artifact":
            display_labels[label_index] = "Art"
        elif display_labels[label_index] == "wake":
            display_labels[label_index] = "Wake"
    
    for label_index in range(len(display_labels)):
        for map_index in range(len(map_classes)):
            if display_labels[label_index] == map_classes[map_index][0]:
                display_labels[label_index] = map_classes[map_index][1]

    text_output = ""
    text_output += "\\performancebox{\n"
    text_output += "\t\\centering\n"
    text_output += "\t\\footnotesize\n"
    text_output += f"\tModeling Task: {modeling_task} \\hfill Network Architecture: {network_architecture}\n\n"
	
    text_output += "\t\\smallskip\n"
    text_output += "\t\\normalsize\n"
    text_output += "\t\\textbf{Preprocessing Configuration:} \\hfill \\mdhighlight{"
    text_output += sample_structure # type: ignore
    text_output += "} \\hfill \\mdhighlight{"
    text_output += signal_transformation
    text_output += "} \\hfill \\mdhighlight{"
    text_output += labeling_strategy
    text_output += "}\n\n"
	
    text_output += "\t\\smallskip\n"
    text_output += "\t\\begin{minipage}{0.55\\linewidth}\n"
    text_output += "\t\tAccuracy: \\num{"
    text_output += str(accuracy)
    text_output += "} \\hfill Cohen's Kappa: \\num{"
    text_output += str(kappa)
    text_output += "}\n\n"

    text_output += "\t\t\\bigskip\n"
    text_output += "\t\t\\footnotesize\n"
    text_output += "\t\t\\captionof*{table}{Per-Class Metrics:}\n"
    text_output += "\t\t\\vspace{-7pt}\n"
    text_output += "\t\t\\begin{tabularx}{\\linewidth}{c|" + "N"*len(f1) + "}\n"
    text_output += "\t\t\t\\toprule\n"
    text_output += "\t\t\t\\multirow{2}{*}{\\normalsize Metric} & \\multicolumn{" + str(len(f1)) + "}{c}{\\normalsize Class} \\\\\n"
    text_output += "\t\t\t"
    for index in order:
        text_output += "& " + display_labels[index] + " "
    text_output += "\\\\\n"
    text_output += "\t\t\t\\midrule\n"
    text_output += "\t\t\tPrecision "
    for index in order:
        text_output += "& \\num{" + str(precision[index]) + "} "
    text_output += "\\\\\n"
    text_output += "\t\t\tRecall "
    for index in order:
        text_output += "& \\num{" + str(recall[index]) + "} "
    text_output += "\\\\\n"
    text_output += "\t\t\tF1-Score "
    for index in order:
        text_output += "& \\num{" + str(f1[index]) + "} "
    text_output += "\\\\\n"
    text_output += "\t\t\t\\bottomrule\n"
    text_output += "\t\t\\end{tabularx}\n"
    text_output += "\t\\end{minipage}%\n"
    text_output += "\t\\hfill\n"
    text_output += "\t\\begin{minipage}{0.45\\linewidth}\n"
    text_output += "\t\t\\centering\n"
    text_output += "\t\t\\captionof*{figure}{Confusion Matrix:}\n"
    text_output += "\t\t\\vspace{-5pt}\n"
    text_output += "\t\t\\includegraphics{confusion_matrices/" + file_name + "}\n"
    text_output += "\t\\end{minipage}\n"
    text_output += "}"

    return text_output


def single_apnea_detailed_performance(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
):

    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
    latex_network = ["\\qty{30}{s}", "\\qty{60}{s}", "\\qty{120}{s}", "\\qty{180}{s}"]
    
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    latex_cleaning = ["Raw", "Cleaned", "SampleNorm"]
    
    class_names = ["A", "AH"]
    latex_class = ["N,\\,A\\&H", "N,\\,A,\\,H"]

    print("\nSingle Apnea")

    if run_splitted:
        print("\nSplitted Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Splitted",# "Complete_Probability", "Complete_Majority"
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Apnea Detection",
                            network_architecture = "Single-Output",
                            sample_structure = latex_network[network_index],
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class[class_index],
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Complete_Majority",
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Apnea Detection",
                            network_architecture = "Single-Output",
                            sample_structure = latex_network[network_index],
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class[class_index],
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")


def multi_apnea_detailed_performance(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
):

    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    latex_cleaning = ["Raw", "Cleaned", "SampleNorm", "WindowNorm"]
    
    class_names = ["A", "AH"]
    latex_class = ["N,\\,A\\&H", "N,\\,A,\\,H"]

    network_model_names = ["LSM", "LSM_Residual"]
    latex_model = ["(Form A)", "(Form B)"]

    print("\nMulti Apnea")

    if run_splitted:
        print("\nSplitted Table Rows")
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Splitted",# "Complete_Probability", "Complete_Majority"
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Apnea Detection",
                            network_architecture = "Multi-Output " + latex_model[model_index],
                            sample_structure = "\\qty{300}{s}:\\qty{10}{s}:\\qty{5}{s}",
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class[class_index],
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "A":
                    map_classes = [["Apnea", "A\\&H"]]
                elif class_names[class_index] == "AH":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Complete_Majority",
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Apnea Detection",
                            network_architecture = "Multi-Output " + latex_model[model_index],
                            sample_structure = "\\qty{300}{s}:\\qty{10}{s}:\\qty{5}{s}",
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class[class_index],
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")

def single_stage_detailed_performance(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
    ):

    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
    latex_network = ["\\qty{30}{s}", "\\qty{60}{s}", "\\qty{120}{s}", "\\qty{180}{s}"]
    
    cleaning_names = ["RAW", "Cleaned", "Norm"]
    latex_cleaning = ["Raw", "Cleaned", "SampleNorm"]
    
    class_names = ["ArtifactAsWake", "FullClass"]
    latex_class = ["W\\&A,\\,L,\\,D,\\,R", "W,\\,L,\\,D,\\,R,\\,A"]

    print("\nSingle Stage")

    if run_splitted:
        print("\nSplitted Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "ArtifactAsWake":
                    map_classes = [["Wake", "W\\&A"]]
                elif class_names[class_index] == "FullClass":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")

                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Splitted",# "Complete_Probability", "Complete_Majority"
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Stage Classification",
                            network_architecture = "Single-Output",
                            sample_structure = latex_network[network_index],
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class[class_index],
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        for network_index in range(len(network_names)):
            print(network_names[network_index])
            
            for class_index in range(len(class_names)):
                print(class_names[class_index])
                if class_names[class_index] == "ArtifactAsWake":
                    map_classes = [["Wake", "W\\&A"]]
                elif class_names[class_index] == "FullClass":
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")
                    
                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Complete_Probability",
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Stage Classification",
                            network_architecture = "Single-Output",
                            sample_structure = latex_network[network_index],
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class[class_index],
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")


def multi_stage_detailed_performance(
        parent_folder_path = "",
        run_splitted = False,
        run_combined = True,
    ):

    window_and_class_names = ["Overlap_ArtifactAsWake", "Overlap_FullClass", "NoOverlap_ArtifactAsWake"]
    
    network_model_names = ["LSM", "LSM_Residual"]
    latex_network = ["(Form A)", "(Form B)"]

    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    latex_cleaning = ["Raw", "Cleaned", "SampleNorm", "WindowNorm"]

    print("\nMulti Stage")

    if run_splitted:
        print("\nSplitted Table Rows")
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for window_index in range(len(window_and_class_names)):
                print(window_and_class_names[window_index])
                if window_and_class_names[window_index] == "Overlap_FullClass":
                    latex_class = "W,\\,L,\\,D,\\,R,\\,A"
                    sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{90}{s}"
                elif window_and_class_names[window_index] == "NoOverlap_ArtifactAsWake":
                    latex_class = "W\\&A,\\,L,\\,D,\\,R"
                    sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{0}{s}"
                elif window_and_class_names[window_index] == "Overlap_ArtifactAsWake":
                    latex_class = "W\\&A,\\,L,\\,D,\\,R"
                    sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{90}{s}"
                else:
                    raise SystemError("unknown window and class names")

                if "ArtifactAsWake" in window_and_class_names[window_index]:
                    map_classes = [["Wake", "W\\&A"]]
                elif "FullClass" in window_and_class_names[window_index]:
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")
                
                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Splitted",# "Complete_Probability", "Complete_Majority"
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Stage Classification",
                            network_architecture = "Multi-Output " + latex_network[model_index],
                            sample_structure = sample_structure,
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class,
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")
    
    if run_combined:
        print("\nCombined Table Rows")
        
        for model_index in range(len(network_model_names)):
            print(network_model_names[model_index])
            
            for window_index in range(len(window_and_class_names)):
                print(window_and_class_names[window_index])
                if window_and_class_names[window_index] == "Overlap_FullClass":
                    latex_class = "W,\\,L,\\,D,\\,R,\\,A"
                    sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{90}{s}"
                elif window_and_class_names[window_index] == "NoOverlap_ArtifactAsWake":
                    latex_class = "W\\&A,\\,L,\\,D,\\,R"
                    sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{0}{s}"
                elif window_and_class_names[window_index] == "Overlap_ArtifactAsWake":
                    latex_class = "W\\&A,\\,L,\\,D,\\,R"
                    sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{90}{s}"
                else:
                    raise SystemError("unknown window and class names")
                
                if "ArtifactAsWake" in window_and_class_names[window_index]:
                    map_classes = [["Wake", "W\\&A"]]
                elif "FullClass" in window_and_class_names[window_index]:
                    map_classes = []
                else:
                    raise SystemError("Unknown class name")
                
                for clean_index in range(len(cleaning_names)):
                    identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"
                    print(cleaning_names[clean_index])
                    print(identifier)

                    try:
                        print(create_performance_box(
                            path_to_model_directory = parent_folder_path + identifier,
                            performance_of = "Complete_Probability",
                            transform = [],
                            round_to_decimals = 3,
                            order = [],
                            modeling_task = "Sleep Stage Classification",
                            network_architecture = "Multi-Output " + latex_network[model_index],
                            sample_structure = sample_structure,
                            signal_transformation = latex_cleaning[clean_index],
                            labeling_strategy = latex_class,
                            file_name = identifier[:-1] + ".pdf",
                            map_classes = map_classes
                        ))
                    except:
                        print("Folder not found.")


def rename_single_stage():
    all_directories = os.listdir()
    for dir in all_directories:
        if "Local" in dir:
            for char_pos in range(0, len(dir)):
                if dir[char_pos:char_pos+5] == "Local":
                    new_name = dir[:char_pos] + dir[char_pos+6:]

            print(dir, "        ", new_name)
            # os.rename(dir, new_name)


def rename_multi_apnea():
    all_directories = os.listdir()
    for dir in all_directories:
        if "5min" in dir:
            new_name = "SAE_"
            
            if "LSM_Residual" in dir:
                new_name += "LSM_Residual_"
            else:
                new_name += "LSM_"
            
            new_name += "300s_10s_5s_"

            if "_AH_" in dir:
                new_name += "AH_"
            else:
                new_name += "A_"

            stuffs = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
            
            for stuff in stuffs:
                if stuff in dir:
                    new_name += stuff
                    break
            
            print(dir, "        ", new_name)
            # os.rename(dir, new_name)


def rename_multi_stage():
    all_directories = os.listdir()
    for dir in all_directories:
        if "SSG_LSM_" in dir:
            new_name = "SSG_"
            
            if "LSM_Residual" in dir:
                new_name += "LSM_Residual_"
            else:
                new_name += "LSM_"
            
            if "NoOverlap" in dir:
                new_name += "10h_120s_0s_"
            else:
                new_name += "10h_120s_90s_"

            if "FullClass" in dir:
                new_name += "FullClass_"
            else:
                new_name += "ArtifactAsWake_"

            stuffs = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
            
            for stuff in stuffs:
                if stuff in dir:
                    new_name += stuff
                    break
            
            print(dir, "        ", new_name)
            # os.rename(dir, new_name)



if __name__ == "__main__":
    # single_stage_table()
    # multi_stage_table()

    # single_apnea_table()
    # multi_apnea_table()
    
    rename_single_stage()
    rename_multi_stage()
    rename_multi_apnea()

    raise SystemExit

    matplotlib.rcParams.update(tex_look)
    
    # multi-plots
    fig_ratio = 4 / 3
    linewidth *= 0.45 # 0.48, 0.5, 0.3, 0.322
    linewidth = 193.1246 * pt_to_inch

    # standalone plots
    # fig_ratio = 3 / 2
    # fig_ratio = 2 / 1
    # linewidth *= 0.8
    matplotlib.rcParams["figure.figsize"] = [linewidth, linewidth / fig_ratio]

    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_single/"
    # single_stage_confusion_plot(parent_folder_path = parent_network_folder_path)
    
    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_multi/"
    # multi_stage_confusion_plot(parent_folder_path = parent_network_folder_path)

    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_single/"
    # single_apnea_confusion_plot(parent_folder_path = parent_network_folder_path)
    
    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_multi/"
    # multi_apnea_confusion_plot(parent_folder_path = parent_network_folder_path)





    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_single/"
    # single_stage_detailed_performance(parent_folder_path = parent_network_folder_path)
    
    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/slp_nets_multi/"
    # multi_stage_detailed_performance(parent_folder_path = parent_network_folder_path)

    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_single/"
    # single_apnea_detailed_performance(parent_folder_path = parent_network_folder_path)
    
    parent_network_folder_path = "/Volumes/NaKo-UniHalle/JPK_Results/sae_nets_multi/"
    # multi_apnea_detailed_performance(parent_folder_path = parent_network_folder_path)

    # a = create_performance_box(path_to_model_directory = "SSG_Local_120s_FullClass_RAW/",
    #     performance_of = "Complete_Probability",
    #     transform = [],
    #     round_to_decimals = 3,
    #     order = [1, 2, 3, 4, 0],
    #     modeling_task = "Sleep Phase Prediction",
    #     network_architecture = "Single-Output",
    #     sample_structure = "\\qty{10}{h}:\\qty{120}{s}:\\qty{90}{s}",
    #     signal_transformation = "WindowNorm",
    #     labeling_strategy = "W\\&ALDR",
    # )

    # print(a)