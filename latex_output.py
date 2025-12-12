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
        transform = []
    ):

    additional_score_function_args = {"zero_division": np.nan, "average": "macro"}
    
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
                    all_actual_results = transform[j][1]
                    break

    # Calculate and print accuracy and cohen's kappa score
    accuracy = accuracy_score(all_actual_results, all_predicted_results)
    kappa = cohen_kappa_score(all_actual_results, all_predicted_results)
    f1 = f1_score(all_actual_results, all_predicted_results, **additional_score_function_args)

    return accuracy, kappa, f1


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
        accuracy, kappa, f1 = calc_perf_values(
            paths_to_pkl_files = [gif_splitted_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            transform = transform
        )

    elif performance_of == "Complete_Probability":
        accuracy, kappa, f1 = calc_perf_values(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted",
            actual_result_key = "Actual",
            transform = transform
        )

    elif performance_of == "Complete_Majority":
        accuracy, kappa, f1 = calc_perf_values(
            paths_to_pkl_files = [gif_complete_validation_pid_results_path],
            path_to_project_configuration = path_to_model_directory + project_configuration_file,
            prediction_result_key = "Predicted_2",
            actual_result_key = "Actual",
            transform = transform
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
    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
    class_names = ["A", "AH"]

    print("\nSingle Apnea")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_names)):
        for class_index in range(len(class_names)):
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]

            for clean_index in range(len(cleaning_names)):
                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
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
        for class_index in range(len(class_names)):
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]

            for clean_index in range(len(cleaning_names)):
                identifier = "SAE_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Complete_Majority",
                                transform = apnea_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                
                print(text)


def multi_apnea_table():

    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    latex_cleaning = ["\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}", "\\mdhighlight{WindowNorm}"]
    network_model_names = ["LSM", "LSM_Residual"]
    class_names = ["A", "AH"]

    print("\nMulti Apnea")
    print("\nSplitted Table Rows")
    for model_index in range(len(network_model_names)):
        for class_index in range(len(class_names)):
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]

            for clean_index in range(len(cleaning_names)):
                identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
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
        for class_index in range(len(class_names)):
            if class_names[class_index] == "A":
                apnea_transform = []
            else:
                apnea_transform = [[2,1]]

            for clean_index in range(len(cleaning_names)):
                identifier = "SAE_" + "Multiple_5min" + "_" + class_names[class_index] + "_" + network_model_names[model_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
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
    network_names = ["Local_30s", "Local_60s", "Local_120s", "Local_180s"]
    sleep_transform = []

    print("\nSingle Stage")
    print("\nSplitted Table Rows")
    for network_index in range(len(network_names)):
        for class_index in range(len(class_names)):
            if class_names[class_index] == "ArtifactAsWake":
                sleep_transform = []
            else:
                sleep_transform = [[0, 1]]

            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
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
        for class_index in range(len(class_names)):
            if class_names[class_index] == "ArtifactAsWake":
                sleep_transform = []
            else:
                sleep_transform = [[0, 1]]
                
            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_names[network_index] + "_" + class_names[class_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Complete_Probability",
                                transform = sleep_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"

                print(text)


def multi_stage_table():

    window_and_class_names = ["Overlap_ArtifactAsWake", "NoOverlap_ArtifactAsWake", "Overlap_FullClass"]
    cleaning_names = ["RAW", "Cleaned", "GlobalNorm", "LocalNorm"]
    latex_cleaning = ["\\mdhighlight{Raw}", "\\mdhighlight{Cleaned}", "\\mdhighlight{SampleNorm}", "\\mdhighlight{WindowNorm}"]
    network_model_names = ["LSM", "LSM_Residual"]

    print("\nMulti Stage")
    print("\nSplitted Table Rows")
    for model_index in range(len(network_model_names)):
        for window_index in range(len(window_and_class_names)):
            if window_and_class_names[window_index] == "Overlap_FullClass":
                stage_transform = [[0, 1]]
            else:
                stage_transform = []
            
            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Splitted",
                                transform = stage_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"

                print(text)
    
    print("\nCombined Table Rows")
    for model_index in range(len(network_model_names)):
        for window_index in range(len(window_and_class_names)):
            if window_and_class_names[window_index] == "Overlap_FullClass":
                stage_transform = [[0, 1]]
            else:
                stage_transform = []
            
            for clean_index in range(len(cleaning_names)):
                identifier = "SSG_" + network_model_names[model_index] + "_" + window_and_class_names[window_index] + "_" + cleaning_names[clean_index] + "/"

                text = "& " + latex_cleaning[clean_index] + "&"
                try:
                    text = create_latex_table_row(
                                path_to_model_directory = identifier,
                                performance_of = "Complete_Probability",
                                transform = stage_transform,
                                round_to_decimals = 3,
                            )
                except:
                    text += "\\num{0} & \\num{0} & \\num{0} \\\\"
                    
                print(text)


if __name__ == "__main__":
    single_stage_table()
    multi_stage_table()

    single_apnea_table()
    multi_apnea_table()
