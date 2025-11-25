from main import *

# IMPORTS
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns

import bitsandbobs as bnb

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler( # type: ignore
    "color", bnb.plt.get_default_colors()
) 
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 2  # padding between text and the tick
matplotlib.rcParams["ytick.major.pad"] = 2  # default 3.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)  # transparent figure bg
matplotlib.rcParams["axes.facecolor"] = (1.0, 0.0, 0.0, 0.0)

chb_error_code_1 = ["SL007", "SL010", "SL012", "SL014", "SL022", "SL026", "SL039", "SL044", "SL049", "SL064", "SL070", "SL146", "SL150", "SL261", "SL266", "SL296", "SL303", "SL306", "SL342", "SL350", "SL410", "SL411", "SL416"]
chb_error_code_2 = ["SL032", "SL037", "SL079", "SL088", "SL114", "SL186", "SL255", "SL328", "SL336", "SL341", "SL344", "SL424"]
chb_error_code_3 = ["SL001", "SL004", "SL011", "SL025", "SL027", "SL034", "SL055", "SL057", "SL073", "SL075", "SL076", "SL083", "SL085", "SL087", "SL089", "SL096", "SL111", "SL116", "SL126", "SL132", "SL138", "SL141", "SL151", "SL157", "SL159", "SL166", "SL173", "SL174", "SL176", "SL178", "SL179", "SL203", "SL207", "SL208", "SL210", "SL211", "SL214", "SL217", "SL218", "SL221", "SL228", "SL229", "SL236", "SL237", "SL240", "SL245", "SL250", "SL252", "SL269", "SL286", "SL293", "SL294", "SL315", "SL348", "SL382", "SL384", "SL386", "SL389", "SL397", "SL406", "SL408", "SL418", "SL422", "SL428"]
chb_error_code_4 = ["SL061", "SL066", "SL091", "SL105", "SL202", "SL204", "SL205", "SL216", "SL305", "SL333", "SL349", "SL430", "SL439", "SL440"]
chb_error_code_5 = ['SL016', 'SL040', 'SL145', 'SL199', 'SL246', 'SL268', 'SL290', 'SL316', 'SL332', 'SL365', 'SL392', 'SL426', 'SL433', 'SL438']

def alter_gif():
    for path in ["Raw_Data/gif_separated.pkl", "Raw_Data/gif_5min.pkl"]:
        gif_generator = load_from_pickle(path)
        split_original_ids = []
        all_gif_data = []

        split_ids = []
        split_ids_length = []

        with open(path[:-4] + "_only_max.pkl", "wb") as f:
            for data_dict in gif_generator:
                # print(data_dict["ID"])
                # print(data_dict.keys())
                if "_" in data_dict["ID"]:
                    if data_dict["ID"].split("_")[0] in chb_error_code_4 or data_dict["ID"].split("_")[0] in chb_error_code_5:
                        continue
                else:
                    if data_dict["ID"] in chb_error_code_4 or data_dict["ID"] in chb_error_code_5:
                        continue
                all_gif_data.append(data_dict)
                if "_" in data_dict["ID"]:
                    original_id = data_dict["ID"].split("_")[0]
                    if original_id not in split_original_ids:
                        split_original_ids.append(original_id)
            
            max_length_id = ["" for _ in split_original_ids]
            max_length_original = [0 for _ in split_original_ids]
            all_length_original = [[] for _ in split_original_ids]

            time_points = [[] for _ in split_original_ids]
            time_point_id = [[] for _ in split_original_ids]

            last_end = 0
            for original_id in split_original_ids:
                for data_dict in all_gif_data:
                    if data_dict["ID"].startswith(original_id):
                        time_point_id[split_original_ids.index(original_id)].append(data_dict["ID"])
                        time_points[split_original_ids.index(original_id)].append(data_dict["start_time_psg"] - last_end)
                        last_end = data_dict["end_time_psg"]
                        split_ids.append(data_dict["ID"])
                        split_ids_length.append(len(data_dict["SLP"]))
            
            # print(time_point_id)
            # print(time_points)

            for i in range(len(split_original_ids)):
                original_id = split_original_ids[i]
                for id_index in range(len(split_ids)):
                    id = split_ids[id_index]
                    if id.startswith(original_id):
                        all_length_original[i].append(split_ids_length[id_index])
                        if max_length_original[i] < split_ids_length[id_index]:
                            max_length_original[i] = split_ids_length[id_index]
                            max_length_id[i] = id

            for data_dict in all_gif_data:
                if data_dict["ID"] not in split_ids:
                    pickle.dump(data_dict, f)
                else:
                    if data_dict["ID"] in max_length_id:
                        if "_" in data_dict["ID"]:
                            data_dict["ID"] = data_dict["ID"].split("_")[0]
                        pickle.dump(data_dict, f)


def plot_learning_rate_scheduler(
        scheduler,
        **kwargs
    ):
    """
    Plot the learning rate scheduler.
    """

    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Epoch")
    kwargs.setdefault("ylabel", "Learning Rate")
    kwargs.setdefault("yscale", "linear") # or "log"
    kwargs.setdefault("xticks", None)
    kwargs.setdefault("yticks", None)
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
        alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])

    number_updates = scheduler.number_updates_total

    x_data = np.arange(1, number_updates+1)
    y_data = np.array([scheduler(update) for update in x_data])
    print(y_data[0], y_data[-1], np.max(y_data))
    
    ax.plot(
        x_data,
        y_data,
        **plot_args
    )

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    if kwargs["xticks"] is not None:
        ax.set_xticks(kwargs["xticks"])
    if kwargs["yticks"] is not None:
        ax.set_yticks(kwargs["yticks"])

    plt.show()


def plot_crop_shift_length(
        signal_duration_limit = (10, 15), # in hours
        **kwargs
    ):
    # Default values
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "")
    kwargs.setdefault("ylabel", "")
    kwargs.setdefault("xticks", None)
    kwargs.setdefault("yticks", None)
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
        alpha = kwargs["alpha"],
        linestyle = kwargs["linestyle"],
        marker = kwargs["marker"],
        markersize = kwargs["markersize"],
        # markeredgewidth = kwargs["markeredgewidth"],
        # markeredgecolor = kwargs["markeredgecolor"],
    )

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax.set(title=kwargs["title"], xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.grid(kwargs["grid"])

    x_data = np.arange(signal_duration_limit[0]*3600, signal_duration_limit[1]*3600 + 120, 120)  # in seconds
    x_data = x_data[x_data > 36000]  # only hours above desired length

    ax.plot(
        np.array(x_data)/3600,
        np.array([calculate_optimal_shift_length(signal_length_seconds=duration, desired_length_seconds=36000, shift_length_seconds_interval=(3600, 7200), all_signal_frequencies=[4,1,1/30]) for duration in x_data])/3600,
        **plot_args
    )

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])

    if kwargs["xticks"] is not None:
        ax.set_xticks(kwargs["xticks"])
    if kwargs["yticks"] is not None:
        ax.set_yticks(kwargs["yticks"])

    plt.show()


def plot_length_distribution(
    pickle_name = "shhs_gif_plot.pkl",
    include_shhs = True,
    include_gifs = ["gif_5min", "gif_separated", "GIF_dataset"],
    label_gifs = ["Gaps Fused", "Gaps Separated", "Yao"],
    hour_thresholds = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    **kwargs
    ):
    
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Recording Length (h)")
    kwargs.setdefault("ylabel", "Count")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bins", 'auto')
    kwargs.setdefault("binwidth", None)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer") # “layer”, “dodge”, “stack”, “fill”
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)
    kwargs.setdefault("legend", True)

    sns_args = dict(
        kde=kwargs["kde"],
        bins=kwargs["bins"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"],
        legend=kwargs["legend"]
    )

    with open(pickle_name, "rb") as f:
        pickle_data_loaded = pickle.load(f)

    SHHS = np.array(pickle_data_loaded["SHHS_signal_length"])
    GIF = list()
    for name in include_gifs:
        GIF.append(np.array(pickle_data_loaded[f"{name}_signal_length"]))

    print(f"SHHS min length: {np.min(SHHS) / 3600:.2f} hours")
    print(f"SHHS max length: {np.max(SHHS) / 3600:.2f} hours")
    for i in range(len(GIF)):
        print(f"{label_gifs[i]} min length: {np.min(GIF[i]) / 3600:.2f} hours")
        print(f"{label_gifs[i]} max length: {np.max(GIF[i]) / 3600:.2f} hours")
    print(f"SHHS mean length: {np.mean(SHHS) / 3600:.2f} hours, std: {np.std(SHHS) / 3600:.2f} hours")

    for threshold in np.array(hour_thresholds) * 3600:
        print(f"SHHS length > {threshold/3600} hours: {np.sum(SHHS > threshold)} / {len(SHHS)} ({np.sum(SHHS > threshold) / len(SHHS) * 100:.2f}%)")
        for i in range(len(GIF)):
            print(f"{label_gifs[i]} length > {threshold/3600} hours: {np.sum(GIF[i] > threshold)} / {len(GIF[i])} ({np.sum(GIF[i] > threshold) / len(GIF[i]) * 100:.2f}%)")
    for i in range(len(GIF)):
        print(f"{label_gifs[i]} mean length: {np.mean(GIF[i]) / 3600:.2f} hours, std: {np.std(GIF[i]) / 3600:.2f} hours")

    if not include_shhs:
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.rcParams["axes.prop_cycle"][1:]
        SHHS = np.array([])

    if len(include_gifs) == 0:
        GIF = np.array([[]])
    
    lengths = np.concatenate((SHHS, GIF[0])) / 3600  # convert to hours
    for i in range(1, len(GIF)):
        lengths = np.concatenate((lengths, GIF[i] / 3600))  # convert to hours
    
    label = np.concatenate((np.array(["SHHS" for _ in range(len(SHHS))]), np.array([label_gifs[0] for _ in range(len(GIF[0]))])))
    for i in range(1, len(GIF)):
        label = np.concatenate((label, np.array([label_gifs[i] for _ in range(len(GIF[i]))])))

    pd_dataframe = pd.DataFrame({
        "lengths": lengths,
        "Dataset": label
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.histplot(data=pd_dataframe, x="lengths", hue="Dataset", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()

    if not include_shhs:
        matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler( # type: ignore
            "color", bnb.plt.get_default_colors()
        )


def plot_length_distribution_nako(
    pickle_name = "nako_information.pkl",
    **kwargs
    ):
    
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Recording Length (h)")
    kwargs.setdefault("ylabel", "Count")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bins", 'auto')
    kwargs.setdefault("binwidth", None)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer") # “layer”, “dodge”, “stack”, “fill”
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)

    sns_args = dict(
        kde=kwargs["kde"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"]
    )

    with open(pickle_name, "rb") as f:
        pickle_data_loaded = pickle.load(f)

    baseline = np.array(pickle_data_loaded["baseline_file_durations"])
    baseline_channels = pickle_data_loaded["baseline_channels"]
    for i in range(len(baseline_channels)):
        if baseline_channels[i] != 4:
            baseline_channels[i] = 0
    baseline_channels = np.array(baseline_channels)
    
    follow_up = np.array(pickle_data_loaded["follow_up_file_durations"])
    follow_up_channels = pickle_data_loaded["follow_up_channels"]
    for i in range(len(follow_up_channels)):
        if follow_up_channels[i] != 4:
            follow_up_channels[i] = 0
    follow_up_channels = np.array(follow_up_channels)

    # remove file durations where channels != 4
    baseline = baseline[baseline_channels == 4]
    follow_up = follow_up[follow_up_channels == 4]

    for threshold in np.array([27, 25, 24, 23, 22, 10, 5, 1, 0]) * 3600:
        print(f"Baseline length > {threshold/3600} hours: {np.sum(baseline > threshold)} / {len(baseline)} ({np.sum(baseline > threshold) / len(baseline) * 100:.2f}%)")
        print(f"Follow-up length > {threshold/3600} hours: {np.sum(follow_up > threshold)} / {len(follow_up)} ({np.sum(follow_up > threshold) / len(follow_up) * 100:.2f}%)")

    lengths = np.concatenate((baseline, follow_up)) / 3600  # convert to hours
    for i in range(len(lengths)-1, -1, -1):
        if lengths[i] > 27:
            continue
            lengths = np.delete(lengths, i)
    print(f"GNC mean length: {np.mean(lengths):.2f} hours, std: {np.std(lengths):.2f} hours")
    print(f"GNC min length: {np.min(lengths):.2f} hours")
    print(f"GNC max length: {np.max(lengths):.2f} hours")

    label = np.concatenate((np.array(["Baseline" for _ in range(len(baseline))]), np.array(["Follow-up" for _ in range(len(follow_up))])))
    pd_dataframe = pd.DataFrame({
        "lengths": lengths,
        "Dataset": label[:len(lengths)]
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.histplot(data=pd_dataframe, x="lengths", hue="Dataset", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()


def remap_stag_count(considered_stages, original_stages, original_stages_count, mapping):
    considered_stages = list(considered_stages)
    for i in range(len(original_stages)):
        for j in range(len(mapping)):
            if original_stages[i] == mapping[j][0]:
                original_stages[i] = mapping[j][1]
    
    num_stages = [0 for _ in considered_stages]
    for i in range(len(original_stages_count)):
        stage = original_stages[i]
        if stage in considered_stages:
            num_stages[considered_stages.index(stage)] += original_stages_count[i]
    
    return num_stages


def plot_sleep_stages_distribution(
    results_file_path: str,
    include_gifs = ["gif_5min", "gif_separated", "GIF_dataset"],
    label_gifs = ["Gaps Fused", "Gaps Separated", "Yao"],
    sleep_labels = ["Wake", "LS", "DS", "REM", "Artifact"],
    sleep_mapping = [[0,0], [1,1], [2,1], [3,2], [4,4], [5,3], [6,4]],
    stat = "percentage", # "count" or "percentage"
    **kwargs
    ):

    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Classes")
    kwargs.setdefault("ylabel", "Count" if stat == "count" else "Relative Frequency (\%)") # type: ignore
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)

    sns_args = dict(
        edgecolor=kwargs["edgecolor"],
        alpha=kwargs["alpha"]
    )

    considered_stages = []
    for i in range(len(sleep_mapping)):
        considered_stages.append(sleep_mapping[i][1])
    considered_stages.sort()
    considered_stages = np.unique(considered_stages)

    if len(considered_stages) != len(sleep_labels):
        raise ValueError(f"Inconsistent sleep stage mapping and labels: {len(considered_stages)} vs {len(sleep_labels)}")
    else:
        print("Current Mapping:")
        print(considered_stages)
        print(sleep_labels)

    with open(results_file_path, "rb") as f:
        pickle_data_loaded = pickle.load(f)
    
    print(pickle_data_loaded.keys())

    original_shhs_num_stages = pickle_data_loaded["SHHS_slp_stages_count"]
    original_shhs_stages = pickle_data_loaded["SHHS_slp_stages"]
    print(original_shhs_stages, original_shhs_num_stages)
    shhs_num_stages = remap_stag_count(considered_stages, original_shhs_stages, original_shhs_num_stages, sleep_mapping)

    original_gif_stages = list()
    original_gif_num_stages = list()
    for name in include_gifs:
        original_gif_num_stages.append(pickle_data_loaded[f"{name}_slp_stages_count"])
        original_gif_stages.append(pickle_data_loaded[f"{name}_slp_stages"])

    gif_num_stages = list()
    for stage_index in range(len(original_gif_stages)):
        gif_num_stages.append(remap_stag_count(considered_stages, original_gif_stages[stage_index], original_gif_num_stages[stage_index], sleep_mapping))

    print(f"SHHS stages: {considered_stages}, counts: {shhs_num_stages}, relative: {[count/sum(shhs_num_stages) * 100 for count in shhs_num_stages]}")

    for i in range(len(gif_num_stages)):
        print(f"{label_gifs[i]} stages: {considered_stages}, counts: {gif_num_stages[i]}, relative: {[count/sum(gif_num_stages[i]) * 100 for count in gif_num_stages[i]]}")

    if stat == "count":
        lengths = np.concatenate((shhs_num_stages, gif_num_stages[0]))
        for i in range(1, len(gif_num_stages)):
            lengths = np.concatenate((lengths, gif_num_stages[i]))
    elif stat == "percentage":
        total_shhs = sum(shhs_num_stages)
        total_gif = [sum(gif) for gif in gif_num_stages]

        lengths = np.concatenate((np.array(shhs_num_stages) / total_shhs * 100, np.array(gif_num_stages[0]) / total_gif[0] * 100))
        for i in range(1, len(gif_num_stages)):
            lengths = np.concatenate((lengths, np.array(gif_num_stages[i]) / total_gif[i] * 100))

    stages = np.concatenate((sleep_labels, sleep_labels))
    for i in range(1, len(original_gif_stages)):
        stages = np.concatenate((stages, sleep_labels))

    label = np.concatenate((["SHHS" for _ in range(len(shhs_num_stages))], [label_gifs[0] for _ in range(len(gif_num_stages[0]))]))
    for i in range(1, len(gif_num_stages)):
        label = np.concatenate((label, [label_gifs[i] for _ in range(len(gif_num_stages[i]))]))
    
    print(label)

    pd_dataframe = pd.DataFrame({
        "lengths": lengths,
        "slp_stage": stages,
        "Dataset": label
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.barplot(data=pd_dataframe, x="slp_stage", y="lengths", hue="Dataset", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    print(f"Y-limits: {kwargs['ylim']}, X-limits: {kwargs['xlim']}")
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()


def data_shhs_distribution(results_file_path: str, path_to_shhs_dataset: str, path_to_gif_pkl_datasets: list, path_to_gif_h5: str, include_aso_only: bool = False):
    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    # saving all data from SHHS dataset to the pickle file
    signal_length = []
    slp_frequency = shhs_dataset["slp"].attrs["freq"]
    
    slp_stage = []
    slp_stage_count = []

    for i in range(7):
        slp_stage.append(i)
        slp_stage_count.append(0)

    tie = 0
    tie_at_transition = 0
    total = 0
    count_patient = 0

    start_counting = True
    if include_aso_only:
        start_counting = False
    
    for patient_id in patients:
        data = shhs_dataset["slp"][patient_id][:] # type: ignore

        signal_length.append(len(data)/slp_frequency) # type: ignore

        for i in range(len(data)-1): # type: ignore
            upper_bound = i + 4
            if upper_bound >= len(data): # type: ignore
                upper_bound = len(data) - 1 # type: ignore
            
            total += 1
            window = data[i:upper_bound] # type: ignore
            unique, counts = np.unique(window, return_counts=True) # type: ignore
            max_count = np.max(counts)
            appeared = False
            for count in counts:
                if count == max_count and appeared:
                    tie += 1
                    
                    upper_bound = i + 4
                    if upper_bound >= len(data): # type: ignore
                        upper_bound = len(data) - 1 # type: ignore
                    
                    lower_bound = i - 1
                    if lower_bound < 0: # type: ignore
                        lower_bound = 0 # type: ignore
                    
                    if data[lower_bound] == window[0] and data[upper_bound] == window[-1]: # type: ignore
                        tie_at_transition += 1

                    break
                if count == max_count:
                    appeared = True
        
        stop_counting_at = 0
        for i in range(len(data)-1, -1, -1): # type: ignore
            if data[i] != 0: # type: ignore
                stop_counting_at = i
                break
        
        count_iteration = 0
        for stage in data: # type: ignore
            if include_aso_only and count_iteration > stop_counting_at:
                break
            if stage != 0:
                start_counting = True
            if not start_counting:
                continue
            if stage not in slp_stage:
                slp_stage.append(stage)
                slp_stage_count.append(1)
            else:
                index = slp_stage.index(stage)
                slp_stage_count[index] += 1
            
            count_iteration += 1
        
        count_patient += 1
        print(count_patient, end = "\r")

    print(f"SHHS Tie: {tie}, At transition: {tie_at_transition}, Total: {total}, Tie Ratio: {tie/total if total > 0 else 0}, Transition Ratio: {tie_at_transition/tie if tie > 0 else 0}")
    print(slp_stage, slp_stage_count)

    save_dict = {
        "SHHS_slp_stages": slp_stage,
        "SHHS_slp_stages_count": slp_stage_count,
        "SHHS_signal_length": signal_length,
    }

    # access the GIF h5 dataset
    gif_dataset = h5py.File(path_to_gif_h5, 'r')

    # accessing patient ids:
    patients = list(gif_dataset['stage'].keys()) # type: ignore

    # saving all data from SHHS dataset to the pickle file
    gif_signal_length = []
    slp_frequency = 1/30
    
    gif_slp_stage = []
    gif_slp_stage_count = []

    for i in range(7):
        gif_slp_stage.append(i)
        gif_slp_stage_count.append(0)

    gif_tie = 0
    gif_tie_at_transition = 0
    gif_total = 0
    count_patient = 0

    start_counting = True
    if include_aso_only:
        start_counting = False

    # saving all data from GIF dataset to the pickle file
    for patient_id in patients:
        if patient_id in chb_error_code_4 or patient_id in chb_error_code_5:
            continue
        data = np.array(gif_dataset["stage"][patient_id][:]).astype(int) # type: ignore

        gif_signal_length.append(len(data)/slp_frequency) # type: ignore

        for i in range(len(data)-1): # type: ignore
            upper_bound = i + 4
            if upper_bound >= len(data): # type: ignore
                upper_bound = len(data) - 1 # type: ignore

            gif_total += 1
            window = data[i:upper_bound] # type: ignore
            unique, counts = np.unique(window, return_counts=True) # type: ignore
            max_count = np.max(counts)
            appeared = False
            for count in counts:
                if count == max_count and appeared:
                    gif_tie += 1

                    upper_bound = i + 4
                    if upper_bound >= len(data): # type: ignore
                        upper_bound = len(data) - 1 # type: ignore
                    
                    lower_bound = i - 1
                    if lower_bound < 0: # type: ignore
                        lower_bound = 0 # type: ignore
                    
                    if data[lower_bound] == window[0] and data[upper_bound] == window[-1]: # type: ignore
                        gif_tie_at_transition += 1

                    break
                if count == max_count:
                    appeared = True
        
        if include_aso_only:
            for i in range(len(data)):
                if data[i] != 0:
                    break

            data = data[i:] # remove everything before the first sleep stage

            for i in range(len(data)-1, -1, -1):
                if data[i] != 0:
                    break

            data = data[:i+1] # remove everything after the last sleep stage
        
        for stage in data: # type: ignore
            gif_slp_stage_count[stage] += 1

        count_patient += 1
        print(count_patient, end = "\r")

    print("")
    print(path_to_gif_h5)
    print(f"GIF Tie: {gif_tie}, At transition: {gif_tie_at_transition}, Total: {gif_total}, Tie Ratio: {gif_tie/gif_total if gif_total > 0 else 0}, Transition Ratio: {gif_tie_at_transition/gif_tie if gif_tie > 0 else 0}")
    print(gif_slp_stage, gif_slp_stage_count)

    name = path_to_gif_h5.split("/")[-1].split(".")[0]

    save_dict[name + "_slp_stages"] = gif_slp_stage
    save_dict[name + "_slp_stages_count"] = gif_slp_stage_count
    save_dict[name + "_signal_length"] = gif_signal_length

    # access the GIF dataset
    for path in path_to_gif_pkl_datasets:
        gif_generator = load_from_pickle(path)

        # saving all data from GIF dataset to the pickle file
        gif_signal_length = []
        slp_frequency = 1/30
        
        gif_slp_stage = []
        gif_slp_stage_count = []

        for i in range(7):
            gif_slp_stage.append(i)
            gif_slp_stage_count.append(0)

        gif_tie = 0
        gif_tie_at_transition = 0
        gif_total = 0
        count_patient = 0

        start_counting = True
        if include_aso_only:
            start_counting = False

        # saving all data from GIF dataset to the pickle file
        for data_dict in gif_generator:
            check_id = data_dict["ID"]
            if "_" in data_dict["ID"]:
                check_id = data_dict["ID"].split("_")[0]
            if check_id in chb_error_code_4 or check_id in chb_error_code_5:
                continue
            data = np.array(data_dict["SLP"]).astype(int) # type: ignore

            gif_signal_length.append(len(data)/slp_frequency) # type: ignore

            for i in range(len(data)-1): # type: ignore
                upper_bound = i + 4
                if upper_bound >= len(data): # type: ignore
                    upper_bound = len(data) - 1 # type: ignore

                gif_total += 1
                window = data[i:upper_bound] # type: ignore
                unique, counts = np.unique(window, return_counts=True) # type: ignore
                max_count = np.max(counts)
                appeared = False
                for count in counts:
                    if count == max_count and appeared:
                        gif_tie += 1

                        upper_bound = i + 4
                        if upper_bound >= len(data): # type: ignore
                            upper_bound = len(data) - 1 # type: ignore
                        
                        lower_bound = i - 1
                        if lower_bound < 0: # type: ignore
                            lower_bound = 0 # type: ignore
                        
                        if data[lower_bound] == window[0] and data[upper_bound] == window[-1]: # type: ignore
                            gif_tie_at_transition += 1

                        break
                    if count == max_count:
                        appeared = True
            
            if include_aso_only:
                for i in range(len(data)):
                    if data[i] != 0:
                        break

                data = data[i:] # remove everything before the first sleep stage

                for i in range(len(data)-1, -1, -1):
                    if data[i] != 0:
                        break

                data = data[:i+1] # remove everything after the last sleep stage

            for stage in data: # type: ignore
                gif_slp_stage_count[stage] += 1

            count_patient += 1
            print(count_patient, end = "\r")

        print("")
        print(path)
        print(f"GIF Tie: {gif_tie}, At transition: {gif_tie_at_transition}, Total: {gif_total}, Tie Ratio: {gif_tie/gif_total if gif_total > 0 else 0}, Transition Ratio: {gif_tie_at_transition/gif_tie if gif_tie > 0 else 0}")
        print(gif_slp_stage, gif_slp_stage_count)

        name = path.split("/")[-1].split(".")[0]

        save_dict[name + "_slp_stages"] = gif_slp_stage
        save_dict[name + "_slp_stages_count"] = gif_slp_stage_count
        save_dict[name + "_signal_length"] = gif_signal_length

    with open(results_file_path, "wb") as f:
        pickle.dump(save_dict, f)


def nako_info():
    dataframe = pd.read_csv("datasets_plot_info/report-alter.csv", sep=";")
    dataframe_2 = pd.read_csv("datasets_plot_info/report-alter-1.csv", sep=";")
    print(dataframe.head())
    print(dataframe_2.head())
    total = np.sum(dataframe["m"]) + np.sum(dataframe_2["m"]) + np.sum(dataframe["f"]) + np.sum(dataframe_2["f"])
    male = np.sum(dataframe["m"]) + np.sum(dataframe_2["m"])
    female = np.sum(dataframe["f"]) + np.sum(dataframe_2["f"])
    print(male, female, male/female, male/total, female/total)
    age = []
    for i in range(len(dataframe)):
        try:
            for _ in range(int(dataframe["m"][i])):
                age.append(dataframe["age"][i])
        except:
            continue

    for i in range(len(dataframe_2)):
        try:
            for _ in range(int(dataframe_2["m"][i])):
                age.append(dataframe_2["age"][i])
        except:
            continue

    print(np.mean(age), np.std(age), np.min(age), np.max(age))


def apnea_info(
    data_path: str = "RAW_Data/gif_sleep_apnea_events.pkl",
    results_file_path: str = "gif_apnea_info.pkl",
):
    """
    """
    gif_generator = load_from_pickle(data_path)

    apnea_classes = [i for i in range(0, 8)]
    apnea_classes_count = [1 for _ in apnea_classes]

    for data_dict in gif_generator:
        apnea_events = data_dict["SAE"]
        for i in range(len(apnea_events)):
            index = apnea_classes.index(apnea_events[i])
            apnea_classes_count[index] += 1

    with open(results_file_path, "rb") as f:
        current_dict = pickle.load(f)
    
    current_dict["apnea_classes_for_count"] = apnea_classes
    current_dict["apnea_classes_count"] = apnea_classes_count

    with open(results_file_path, "wb") as f:
        pickle.dump(current_dict, f)


def apnea_info_2(
    data_path: str = "RAW_Data/gif_sleep_apnea_events.pkl",
    results_file_path: str = "gif_apnea_info.pkl",
):
    """
    """
    gif_generator = load_from_pickle(data_path)

    apnea_classes = [i for i in range(1, 8)]
    all_count = [[] for _ in range(len(apnea_classes))]


    for data_dict in gif_generator:
        event_count = 0
        apnea_events = data_dict["SAE"]
        apnea_classes_count = [0 for _ in apnea_classes]
        for i in range(len(apnea_events)-1):
            if apnea_events[i] != 0 and apnea_events[i+1] == 0:
                index = apnea_classes.index(apnea_events[i])
                apnea_classes_count[index] += 1
                event_count += 1
        
        if apnea_events[-1] != 0:
            index = apnea_classes.index(apnea_events[-1])
            apnea_classes_count[index] += 1
            event_count += 1
        
        for i in range(len(apnea_classes)):
            all_count[i].append(apnea_classes_count[i])
        
        if event_count > 50:
            print(f"High apnea event count: {event_count} in file {data_dict['ID']}")

    # with open(results_file_path, "rb") as f:
    #     current_dict = pickle.load(f)
    
    # current_dict["apnea_classes_for_occurance"] = apnea_classes
    # current_dict["apnea_classes_occurance"] = all_count

    # with open(results_file_path, "wb") as f:
    #     pickle.dump(current_dict, f)


def apnea_info_3(
    data_path: str = "RAW_Data/gif_sleep_apnea_events.pkl",
    results_file_path: str = "gif_apnea_info.pkl",
):
    """
    """
    gif_generator = load_from_pickle(data_path)

    apnea_classes = [i for i in range(1, 8)]
    all_count = [[] for _ in range(len(apnea_classes))]

    ids = []
    ids_apnea_classes_count = []
    ids_lengths = []


    for data_dict in gif_generator:
        
        id = data_dict["ID"][:5]
        apnea_events = data_dict["SAE"]

        if id in ids:
            id_index = ids.index(id)
            apnea_classes_count = ids_apnea_classes_count[id_index]
            for i in range(len(apnea_events)-1):
                if apnea_events[i] != 0:
                    index = apnea_classes.index(apnea_events[i])
                    apnea_classes_count[index] += 1
            ids_apnea_classes_count[id_index] = apnea_classes_count
            ids_lengths[id_index] += len(apnea_events)
            
        else:
            apnea_classes_count = [0 for _ in apnea_classes]
            for i in range(len(apnea_events)-1):
                if apnea_events[i] != 0:
                    index = apnea_classes.index(apnea_events[i])
                    apnea_classes_count[index] += 1
            ids.append(id)
            ids_lengths.append(len(apnea_events))
            ids_apnea_classes_count.append(apnea_classes_count)

    for id_index in range(len(ids_apnea_classes_count)):
        apnea_classes_count = ids_apnea_classes_count[id_index]
        apnea_events_length = ids_lengths[id_index]
        for i in range(len(apnea_classes)):
            all_count[i].append(int((apnea_classes_count[i] / apnea_events_length) * 100))

    with open(results_file_path, "rb") as f:
        current_dict = pickle.load(f)
    
    # current_dict["apnea_classes_for_occurance"] = apnea_classes
    current_dict["apnea_classes_relative_time"] = all_count

    with open(results_file_path, "wb") as f:
        pickle.dump(current_dict, f)


def slp_info_2(
    data_path_gif: str = "RAW_Data/gif_sleep_stages.pkl",
    data_path_shhs: str = "RAW_Data/SHHS_dataset.h5",
    results_file_path: str = "slp_stage_info.pkl",
):
    """
    """
    gif_generator = load_from_pickle(data_path_gif)
    apnea_classes = [0, 1, 2, 3]
    gif_count = [[] for _ in range(len(apnea_classes))]
    all_count = [[] for _ in range(len(apnea_classes))]

    ids = []
    ids_apnea_classes_count = []

    for data_dict in gif_generator:
        id = data_dict["ID"][:5]
        apnea_events = data_dict["SLP"]

        if id in ids:
            id_index = ids.index(id)
            apnea_classes_count = ids_apnea_classes_count[id_index]
            for i in range(len(apnea_events)):
                current_event = apnea_events[i]
                if current_event == 2:
                    current_event = 1
                elif current_event == 3:
                    current_event = 2
                elif current_event == 5:
                    current_event = 3
                elif current_event == 4 or current_event > 5:
                    continue
                index = apnea_classes.index(current_event)
                apnea_classes_count[index] += 1
            ids_apnea_classes_count[id_index] = apnea_classes_count
        else:
            apnea_classes_count = [0 for _ in apnea_classes]
            start_counting = False
            for i in range(len(apnea_events)):
                current_event = apnea_events[i]
                if current_event != 0:
                    start_counting = True
                if not start_counting:
                    continue
                if current_event == 2:
                    current_event = 1
                elif current_event == 3:
                    current_event = 2
                elif current_event == 5:
                    current_event = 3
                elif current_event == 4 or current_event > 5:
                    continue
                index = apnea_classes.index(current_event)
                apnea_classes_count[index] += 1
            
            if sum(apnea_classes_count) != 0:
                ids.append(id)
                ids_apnea_classes_count.append(apnea_classes_count)
    
    for apnea_classes_count in ids_apnea_classes_count:
        for i in range(len(apnea_classes)):
            gif_count[i].append(apnea_classes_count[i])
            all_count[i].append(apnea_classes_count[i])
    
    # access the SHHS dataset
    shhs_dataset = h5py.File(data_path_shhs, 'r')
    shhs_count = [[] for _ in range(len(apnea_classes))]

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    for patient_id in patients:
        apnea_events = shhs_dataset["slp"][patient_id][:] # type: ignore
        time = len(apnea_events) * 30 # type: ignore
        apnea_classes_count = [0 for _ in apnea_classes]
        for i in range(len(apnea_events)): # type: ignore
            current_event = apnea_events[i] # type: ignore
            if current_event == 2:
                current_event = 1
            elif current_event == 3:
                current_event = 2
            elif current_event == 5:
                current_event = 3
            elif current_event == 4 or current_event > 5: # type: ignore
                continue
            index = apnea_classes.index(current_event) # type: ignore
            apnea_classes_count[index] += 1
        
        for i in range(len(apnea_classes)):
            shhs_count[i].append(apnea_classes_count[i])
            all_count[i].append(apnea_classes_count[i])
        
    current_dict = dict()
    current_dict["slp_classes"] = apnea_classes
    current_dict["shhs_occurance"] = shhs_count
    current_dict["gif_occurance"] = gif_count
    current_dict["all_occurance"] = all_count

    with open(results_file_path, "wb") as f:
        pickle.dump(current_dict, f)


def plot_slp_occurance_distribution(
    pickle_name = "slp_stage_info.pkl",
    dataset_name = "all", # "shhs", "gif", or "all"
    sleep_labels = ["Wake", "LS", "DS", "REM"],
    sleep_mapping = [[0,0], [1,1], [2,2], [3,3]],
    **kwargs
    ):
    
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", r"Relative Class Time (\%)")
    kwargs.setdefault("ylabel", "Count")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bins", 'auto')
    kwargs.setdefault("binwidth", None)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer") # “layer”, “dodge”, “stack”, “fill”
    kwargs.setdefault("linewidth", 1)
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)
    kwargs.setdefault("legend", True)

    sns_args = dict(
        kde=kwargs["kde"],
        bins=kwargs["bins"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"],
        legend=kwargs["legend"],
        # linewidth=kwargs["linewidth"]
    )

    with open(pickle_name, "rb") as f:
        pickle_data_loaded = pickle.load(f)
    
    original_classes = pickle_data_loaded["slp_classes"]
    original_occurance = pickle_data_loaded[f"{dataset_name}_occurance"]

    remove_rows = []

    for i in range(len(original_occurance[0])):
        total = np.sum([original_occurance[j][i] for j in range(len(original_occurance))])
        
        time = total * 30 / 3600  # in hours
        # if time > 6.75 or time > 7.25:
        #     remove_rows.append(i)

        for j in range(len(original_occurance)):
            original_occurance[j][i] = int(original_occurance[j][i] / total * 100)
    
    original_occurance = np.array(original_occurance)
    # original_occurance = np.delete(original_occurance, remove_rows, axis=1)

    considered_stages = []
    considered_occurances = []

    for i in range(len(original_classes)):
        current_class = original_classes[i]
        class_found = False
        for j in range(len(sleep_mapping)):
            source_class = sleep_mapping[j][0]
            target_class = sleep_mapping[j][1]
            if source_class == current_class:
                class_found = True
                break
        
        if not class_found:
            continue
        
        if target_class not in considered_stages:
            considered_stages.append(target_class)
            considered_occurances.append(np.array(original_occurance[i]))
        else:
            index = considered_stages.index(target_class)
            considered_occurances[index] = considered_occurances[index] + np.array(original_occurance[i])
    
    if len(considered_stages) != len(sleep_labels):
        raise ValueError(f"Inconsistent sleep stage mapping and labels: {len(considered_stages)} vs {len(sleep_labels)}")
    else:
        print("Current Mapping:")
        print(considered_stages)
        print([sleep_labels[i] for i in considered_stages])
        where_zero = [len([considered_occurances[stage][i] for i in range(len(considered_occurances[stage])) if considered_occurances[stage][i] == 0]) for stage in considered_stages]
        ratio_where_zero = [where_zero[i] / len(considered_occurances[i]) * 100 for i in range(len(considered_stages))]
        print("Zero occurrences:", where_zero)
        print("Ratio of zero occurrences (%):", ratio_where_zero)
        print(np.mean(considered_occurances, axis=1))
        print(np.std(considered_occurances, axis=1))
    
    collected_stages = []
    collected_occurances = []

    for i in range(len(considered_stages)):
        stage = considered_stages[i]
        occurances = considered_occurances[i]
        label = sleep_labels[considered_stages.index(stage)]
        for j in range(len(occurances)):
            if occurances[j] > -1:
                collected_stages.append(label)
                collected_occurances.append(occurances[j])

    pd_dataframe = pd.DataFrame({
        "durations": collected_occurances,
        "Class": collected_stages
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.histplot(data=pd_dataframe, x="durations", hue="Class", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()


def plot_apnea_duration_distribution(
    pickle_name = "gif_apnea_info.pkl",
    transform = [['Hypopnea', 'Hypopnea'], ['Obstructive Apnea', 'Obstructive Apnea'], ['Mixed Apnea', 'Mixed Apnea'], ['Central Apnea', 'Central Apnea'], ['Central Hypopnea', 'Central Hypopnea'], ['Obstructive Hypopnea', 'Obstructive Hypopnea'], ['Apnea', 'Apnea']],
    include_labels = ['Hypopnea', 'Obstructive Apnea', 'Mixed Apnea', 'Central Apnea', 'Central Hypopnea', 'Obstructive Hypopnea', 'Apnea'],
    check_times = [5, 10, 15, 16, 17, 20, 25, 30, 60, 120, 200],
    crop_outside = [0, 24*3600],
    **kwargs
    ):
    
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Apnea Event Duration (s)")
    kwargs.setdefault("ylabel", "Count")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bins", 'auto')
    kwargs.setdefault("binwidth", None)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer") # “layer”, “dodge”, “stack”, “fill”
    kwargs.setdefault("linewidth", 1)
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)
    kwargs.setdefault("legend", True)

    sns_args = dict(
        kde=kwargs["kde"],
        bins=kwargs["bins"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"],
        legend=kwargs["legend"],
        # linewidth=kwargs["linewidth"]
    )

    with open(pickle_name, "rb") as f:
        pickle_data_loaded = pickle.load(f)
    
    original_classes = pickle_data_loaded["apnea_classes"]
    original_durations = pickle_data_loaded["durations"]

    collected_durations = []
    collected_classes = []

    for i in range(len(include_labels)):
        sae_class = include_labels[i]
        
        for j in range(len(transform)):
            target_class = transform[j][1]
            if target_class == sae_class:
                source_class = transform[j][0]
                for k in range(len(original_classes)):
                    if original_classes[k] == source_class:
                        collected_durations.extend(original_durations[k])
                        collected_classes.extend([sae_class for _ in original_durations[k]])
    
    for time in check_times:
        count = 0
        for duration in collected_durations:
            if duration >= time:
                count += 1
        print(f"Apnea events with duration >= {time} seconds: {count} / {len(collected_durations)} ({count / len(collected_durations) * 100:.2f}%)")
    
    for i in range(len(collected_durations)-1, -1, -1):
        if collected_durations[i] < crop_outside[0] or collected_durations[i] > crop_outside[1]:
            del collected_durations[i]
            del collected_classes[i]

    pd_dataframe = pd.DataFrame({
        "durations": collected_durations,
        "Apnea Event": collected_classes
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.histplot(data=pd_dataframe, x="durations", hue="Apnea Event", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()

# {"Normal": [0], "Apnea": [1], "Obstructive Apnea": [2], "Central Apnea": [3], "Mixed Apnea": [4], "Hypopnea": [5], "Obstructive Hypopnea": [6], "Central Hypopnea": [7]}
def plot_apnea_frequency(
    results_file_path: str = "gif_apnea_info.pkl",
    sleep_labels = ["Normal", "Apnea", "Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea", "Obstructive Hypopnea", "Central Hypopnea"],
    sleep_mapping = [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7]],
    stat = "percentage", # "count" or "percentage"
    **kwargs
    ):

    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Classes")
    kwargs.setdefault("ylabel", "Count" if stat == "count" else "Relative Recording Time (\%)") # type: ignore
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)

    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.rcParams["axes.prop_cycle"][1:]

    sns_args = dict(
        edgecolor=kwargs["edgecolor"],
        alpha=kwargs["alpha"]
    )

    considered_stages = []
    for i in range(len(sleep_mapping)):
        considered_stages.append(sleep_mapping[i][1])
    considered_stages.sort()
    considered_stages = np.unique(considered_stages)

    if len(considered_stages) != len(sleep_labels):
        raise ValueError(f"Inconsistent sleep stage mapping and labels: {len(considered_stages)} vs {len(sleep_labels)}")
    else:
        print("Current Mapping:")
        print(considered_stages)
        print(sleep_labels)

    with open(results_file_path, "rb") as f:
        pickle_data_loaded = pickle.load(f)

    original_stages = pickle_data_loaded["apnea_classes_for_count"]
    original_num_stages = pickle_data_loaded["apnea_classes_count"]
    print(original_stages, original_num_stages)
    num_stages = remap_stag_count(considered_stages, original_stages, original_num_stages, sleep_mapping)

    print(f"Apnea stages: {considered_stages}, counts: {num_stages}, relative: {[count/sum(num_stages) * 100 for count in num_stages]}")

    if stat == "count":
        lengths = num_stages
    elif stat == "percentage":
        total = sum(num_stages)
        lengths = np.array(num_stages) / total * 100

    stages = sleep_labels
    
    label = ["CHB" for _ in range(len(num_stages))]

    pd_dataframe = pd.DataFrame({
        "lengths": lengths,
        "slp_stage": stages,
        "Dataset": label
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.barplot(data=pd_dataframe, x="slp_stage", y="lengths", hue="Dataset", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    print(f"Y-limits: {kwargs['ylim']}, X-limits: {kwargs['xlim']}")
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()

    matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler( # type: ignore
            "color", bnb.plt.get_default_colors()
        )


def plot_apnea_occurance_distribution(
    pickle_name = "gif_apnea_info.pkl",
    sleep_labels = ["Normal", "Apnea", "Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea", "Obstructive Hypopnea", "Central Hypopnea"],
    sleep_mapping = [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7]],
    **kwargs
    ):
    
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", r"Total Apnea Event Duration (\%)")
    kwargs.setdefault("ylabel", "Count")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bins", 'auto')
    kwargs.setdefault("binwidth", None)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer") # “layer”, “dodge”, “stack”, “fill”
    kwargs.setdefault("linewidth", 1)
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", matplotlib.rcParams["figure.figsize"])
    kwargs.setdefault("yscale", "linear")
    kwargs.setdefault("grid", True)
    kwargs.setdefault("legend", True)

    sns_args = dict(
        kde=kwargs["kde"],
        bins=kwargs["bins"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"],
        legend=kwargs["legend"],
        # linewidth=kwargs["linewidth"]
    )

    with open(pickle_name, "rb") as f:
        pickle_data_loaded = pickle.load(f)
    
    original_classes = pickle_data_loaded["apnea_classes_for_occurance"]
    # original_occurance = pickle_data_loaded["apnea_classes_occurance"]
    original_occurance = pickle_data_loaded["apnea_classes_relative_time"]

    considered_stages = []
    considered_occurances = []

    for i in range(len(original_classes)):
        current_class = original_classes[i]
        class_found = False
        for j in range(len(sleep_mapping)):
            source_class = sleep_mapping[j][0]
            target_class = sleep_mapping[j][1]
            if source_class == current_class:
                class_found = True
                break
        
        if not class_found:
            continue
        
        if target_class not in considered_stages:
            considered_stages.append(target_class)
            considered_occurances.append(np.array(original_occurance[i]))
        else:
            index = considered_stages.index(target_class)
            considered_occurances[index] = considered_occurances[index] + np.array(original_occurance[i])
    
    considered_occurances = np.array(considered_occurances)
    considered_stages = [int(stage)-1 for stage in considered_stages]
    
    if len(considered_stages) != len(sleep_labels):
        raise ValueError(f"Inconsistent sleep stage mapping and labels: {len(considered_stages)} vs {len(sleep_labels)}")
    else:
        print("Current Mapping:")
        print(considered_stages)
        print([sleep_labels[i] for i in considered_stages])
        print(np.mean(considered_occurances, axis=1))
        print(np.std(considered_occurances, axis=1))
    
    collected_stages = []
    collected_occurances = []

    for i in range(len(considered_stages)):
        stage = considered_stages[i]
        occurances = considered_occurances[i]
        label = sleep_labels[considered_stages.index(stage)]
        for j in range(len(occurances)):
            if occurances[j] > -1:
                collected_stages.append(label)
                collected_occurances.append(occurances[j])

    pd_dataframe = pd.DataFrame({
        "durations": collected_occurances,
        "Apnea Event": collected_stages
    })

    fig, ax = plt.subplots(figsize=kwargs["figsize"], constrained_layout=True)
    ax = sns.histplot(data=pd_dataframe, x="durations", hue="Apnea Event", **sns_args)
    ax.set(xlabel=kwargs["xlabel"], ylabel=kwargs["ylabel"])
    ax.set_yscale(kwargs["yscale"])
    ax.grid(kwargs["grid"])
    ax.set_axisbelow(True)

    kwargs.setdefault("ylim", plt.ylim())
    kwargs.setdefault("xlim", plt.xlim())
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()


tex_correction = 0.5
tex_look = {
    "text.usetex": True,
    # "text.latex.preamble": \usepackage{amsmath}\usepackage{amssymb},
    "font.family": "serif",
    "font.serif": "Computer Modern",
    #
    "legend.fontsize": 10-tex_correction,
    "xtick.labelsize": 10-tex_correction,
    "ytick.labelsize": 10-tex_correction,
    "font.size": 12-tex_correction,
    "axes.titlesize": 12-tex_correction,
    "axes.labelsize": 12-tex_correction,
    #
    "savefig.format": "pdf",
    #
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.dpi": 600,
}

python_correction = 0
python_look = {
    "legend.fontsize": 8+python_correction,
    "xtick.labelsize": 8+python_correction,
    "ytick.labelsize": 8+python_correction,
    "font.size": 10+python_correction,
    "axes.titlesize": 10+python_correction,
    "axes.labelsize": 10+python_correction,
    #
    "savefig.format": "pdf",
    #
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.dpi": 600,
}

pt_to_inch = 1./72.27
cm_to_inch = 1/2.54

# linewidth = 16.2*cm_to_inch
linewidth = 459.6215*pt_to_inch

# fig_ratio = 3.4 / 2.7

if __name__ == "__main__":
    matplotlib.rcParams.update(tex_look)
    
    # multi-plots
    fig_ratio = 4 / 3
    linewidth *= 0.322 # 0.48, 0.5, 0.3

    # standalone plots
    # fig_ratio = 3 / 2
    # fig_ratio = 2 / 1
    # linewidth *= 0.8
    matplotlib.rcParams["figure.figsize"] = [linewidth, linewidth / fig_ratio]

    # MA Plots
    if True:
        # slp_info_2()
        # apnea_info_3()

        # plot_slp_occurance_distribution(
        #     dataset_name="gif",
        #     bins = np.arange(0, 100, 2),
        # )
        # plot_slp_occurance_distribution(
        #     dataset_name="shhs",
        #     bins = np.arange(0, 100, 2),
        # )

        plot_apnea_occurance_distribution(
            pickle_name = "gif_apnea_info.pkl",
            # xlabel = "",
            # ylabel = "",
            # sleep_labels = ["Apnea", "Hypopnea"],
            # sleep_mapping = [[1,2], [2,1], [3,1], [4,1], [5,2], [6,2], [7,2]],
            sleep_labels = ["Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"],
            sleep_mapping = [[1,1], [2,1], [3,2], [4,3], [5,4], [6,4], [7,4]],
            # ylim = [0, 20],
            # xlim = [0, 100],
            binwidth = 1,
            # legend = False
        )

        plot_apnea_occurance_distribution(
            pickle_name = "gif_apnea_info.pkl",
            xlabel = "",
            ylabel = "",
            # sleep_labels = ["Apnea", "Hypopnea"],
            # sleep_mapping = [[1,2], [2,1], [3,1], [4,1], [5,2], [6,2], [7,2]],
            sleep_labels = ["Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"],
            sleep_mapping = [[1,1], [2,1], [3,2], [4,3], [5,4], [6,4], [7,4]],
            ylim = [0, 20],
            # xlim = [0, 100],
            binwidth = 1,
            legend = False
        )

    if False:

        plot_apnea_occurance_distribution(
            pickle_name = "gif_apnea_info.pkl",
            # sleep_labels = ["Apnea", "Hypopnea"],
            # sleep_mapping = [[1,2], [2,1], [3,1], [4,1], [5,2], [6,2], [7,2]],
            sleep_labels = ["Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"],
            sleep_mapping = [[1,1], [2,1], [3,2], [4,3], [5,4], [6,4], [7,4]],
            # ylim = [0, 100],
            # xlim = [0, 100],
            binwidth = 5,
        )
        
        # plot_apnea_frequency(
        #     sleep_labels = ["Normal Breathing", "Apnea", "Hypopnea"],
        #     sleep_mapping = [[0,0], [1,1], [2,1], [3,1], [4,1], [5,2], [6,2], [7,2]],
        # )

        # plot_apnea_frequency(
        #     sleep_labels = ["A", "OA", "CA", "MA", "H", "OH", "CH"],
        #     sleep_mapping = [[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7]],
        # )
        
        # plot_apnea_duration_distribution(
        #     pickle_name = "gif_apnea_info.pkl",
        #     transform = [['Hypopnea', 'Hypopnea'], ['Obstructive Apnea', 'Obstructive Apnea'], ['Mixed Apnea', 'Mixed Apnea'], ['Central Apnea', 'Central Apnea'], ['Central Hypopnea', 'Central Hypopnea'], ['Obstructive Hypopnea', 'Obstructive Hypopnea'], ['Apnea', 'Apnea']],
        #     # include_labels = ['Apnea', 'Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea', 'Obstructive Hypopnea', 'Central Hypopnea'],
        #     include_labels = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea', 'Obstructive Hypopnea', 'Central Hypopnea'],
        #     xlim = [0, 100],
        #     bins = np.arange(0, 301, 2),
        # )

        plot_apnea_duration_distribution(
            pickle_name = "gif_apnea_info.pkl",
            transform = [['Hypopnea', 'Hypopnea'], ['Obstructive Apnea', 'Obstructive Apnea'], ['Mixed Apnea', 'Mixed Apnea'], ['Central Apnea', 'Central Apnea'], ['Central Hypopnea', 'Hypopnea'], ['Obstructive Hypopnea', 'Hypopnea'], ['Apnea', 'Apnea']],
            # include_labels = ['Apnea', 'Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea', 'Obstructive Hypopnea', 'Central Hypopnea'],
            include_labels = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea', 'Hypopnea'],
            xlim = [0, 100],
            bins = np.arange(0, 301, 2),
        )

    if False:

        # nako_info()

        # plot_crop_shift_length()

        # yao
        # plot_learning_rate_scheduler(
        #     scheduler=CosineScheduler(
        #         number_updates_total = 40,
        #         number_updates_to_max_lr = 10,
        #         start_learning_rate = 2.5 * 1e-5,
        #         max_learning_rate = 1e-4,
        #         end_learning_rate = 5 * 1e-5,
        #     ))
        
        # mine
        plot_learning_rate_scheduler(
            scheduler=CosineScheduler(
                number_updates_total = 40,
                number_updates_to_max_lr = 4,
                start_learning_rate = 1e-5,
                max_learning_rate = 1e-3,
                end_learning_rate = 1e-6,
            )
        )

        # data_shhs_distribution(
        #     results_file_path="slp_overview_all_removed_error_4.pkl",
        #     path_to_shhs_dataset="Raw_Data/SHHS_dataset.h5",
        #     path_to_gif_pkl_datasets=["Raw_Data/gif_separated_only_max.pkl", "Raw_Data/gif_5min_only_max.pkl"],
        #     path_to_gif_h5="Raw_Data/GIF_dataset.h5",
        #     include_aso_only=False
        # )
        # data_shhs_distribution(
        #     results_file_path="slp_overview_all_removed_error_4.pkl",
        #     path_to_shhs_dataset="Raw_Data/SHHS_dataset.h5",
        #     path_to_gif_pkl_datasets=["Raw_Data/gif_separated.pkl"],
        #     path_to_gif_h5="Raw_Data/GIF_dataset.h5",
        #     include_aso_only=True
        # )

        # plot_length_distribution_nako(xlim=[0,48])

        plot_length_distribution(
            pickle_name = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_shhs = True,
            include_gifs = ["gif_separated", "GIF_dataset"],
            label_gifs = ["CHB (sep)", "CHB (Ma et al.)"],
            hour_thresholds = [7],
            binwidth = 0.5
        )

        plot_length_distribution(
            pickle_name = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_shhs = True,
            include_gifs = ["gif_separated"],
            label_gifs = ["CHB"],
            hour_thresholds = [7],
            bins = np.arange(0, 14, 0.5),
            xlim = [-0.7, 13.2]
        )

        plot_length_distribution(
            pickle_name = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_shhs = True,
            include_gifs = ["gif_separated"],
            label_gifs = ["CHB"],
            hour_thresholds = [7],
            bins = np.arange(0, 14, 0.5),
            xlim = [-0.7, 13.2],
            ylim = [0, 225],
            legend = False,
            ylabel = "",
            xlabel = ""
        )

        plot_length_distribution(
            pickle_name = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_shhs = False,
            include_gifs = ["gif_separated"],
            label_gifs = ["CHB"],
            hour_thresholds = [2, 3, 4, 5, 6, 7, 8, 9],
            bins = np.arange(0, 14, 0.5),
            xlim = [-0.7, 13.2]
        )

        plot_length_distribution(
            pickle_name = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_shhs = True,
            include_gifs = [],
            label_gifs = [],
            hour_thresholds = [7],
            bins = np.arange(0, 14, 0.5),
            xlim = [-0.7, 13.2]
        )

        # plot_length_distribution(
        #     pickle_name = "slp_overview.pkl",
        #     include_gifs = ["gif_5min", "gif_separated", "GIF_dataset"],
        #     label_gifs = ["Gaps Fused", "Gaps Separated", "Yao"],
        #     yscale = "log",
        #     ylim = [1, 10000],
        #     binwidth = 0.5,
        #     xlim = [0, 16]
        # )

        plot_sleep_stages_distribution(
            results_file_path = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_gifs=["gif_separated", "GIF_dataset"],
            label_gifs=["Gaps Separated", "Yao"],
            sleep_labels = ["Wake", "N1", "N2", "DS", "REM", "Artifact"],
            sleep_mapping = [[0,0], [1,1], [2,2], [3,3], [4,5], [5,4], [6,5]],
            stat="percentage",  # "count" or "percentage"
            yscale="linear",
        )

        plot_sleep_stages_distribution(
            results_file_path = "datasets_plot_info/slp_overview_all_removed_error_4.pkl",
            include_gifs=["gif_separated"],
            label_gifs=["CHB"],
            sleep_labels = ["Wake", "LS", "DS", "REM", "Artifact"],
            sleep_mapping = [[0,0], [1,1], [2,1], [3,2], [4,4], [5,3], [6,4]],
            stat="percentage",  # "count" or "percentage"
            yscale="linear",
        )

        plot_sleep_stages_distribution(
            results_file_path = "datasets_plot_info/slp_overview_aso_all_removed_error_4.pkl",
            include_gifs=["gif_separated", "GIF_dataset"],
            label_gifs=["Gaps Separated", "Yao"],
            sleep_labels = ["Wake", "N1", "N2", "DS", "REM", "Artifact"],
            sleep_mapping = [[0,0], [1,1], [2,2], [3,3], [4,5], [5,4], [6,5]],
            stat="percentage",  # "count" or "percentage"
            yscale="linear",
        )

        plot_sleep_stages_distribution(
            results_file_path = "datasets_plot_info/slp_overview_aso_all_removed_error_4.pkl",
            include_gifs=["gif_separated"],
            label_gifs=["CHB"],
            sleep_labels = ["Wake", "LS", "DS", "REM", "Artifact"],
            sleep_mapping = [[0,0], [1,1], [2,1], [3,2], [4,4], [5,3], [6,4]],
            stat="percentage",  # "count" or "percentage"
            yscale="linear",
        )

        # plot_length_distribution(binwidth = 0.25)
        
        # plot_length_distribution(yscale = "log", ylim = [1, 10000], binwidth = 0.5, xlim = [0, 16])

    # Nako Plots
    if False:

        with open("nako_information.pkl", "rb") as f:
            pickle_data_loaded = pickle.load(f)

        baseline_channels = pickle_data_loaded["baseline_channels"]
        baseline_only_ecg = 0
        baseline_only_act = 0
        baseline_none = 0
        baseline_missing_channels = 0
        
        for i in range(len(baseline_channels)):
            if baseline_channels[i] != 4:
                baseline_missing_channels += 1
                if 'ECG' in baseline_channels[i]:
                    baseline_only_ecg += 1
                elif 'X' in baseline_channels[i] and 'Y' in baseline_channels[i] and 'Z' in baseline_channels[i]:
                    baseline_only_act += 1
                else:
                    baseline_none += 1

        follow_up_channels = pickle_data_loaded["follow_up_channels"]
        follow_up_only_ecg = 0
        follow_up_only_act = 0
        follow_up_none = 0
        follow_up_missing_channels = 0

        for i in range(len(follow_up_channels)):
            if follow_up_channels[i] != 4:
                follow_up_missing_channels += 1
                if 'ECG' in follow_up_channels[i]:
                    follow_up_only_ecg += 1
                elif 'X' in follow_up_channels[i] and 'Y' in follow_up_channels[i] and 'Z' in follow_up_channels[i]:
                    follow_up_only_act += 1
                else:
                    follow_up_none += 1

        print(f"Baseline total files: {pickle_data_loaded['baseline_total_files']}")
        print(f"Baseline error Files: {pickle_data_loaded['baseline_error_files']}")
        print(f"Baseline missing channels: {baseline_missing_channels}")
        print(f"Baseline only ECG: {baseline_only_ecg}")
        print(f"Baseline only Actigraphy: {baseline_only_act}")
        print(f"Baseline none: {baseline_none}")

        print(f"Follow-up total files: {pickle_data_loaded['follow_up_total_files']}")
        print(f"Follow-up error Files: {pickle_data_loaded['follow_up_error_files']}")
        print(f"Follow-up missing channels: {follow_up_missing_channels}")
        print(f"Follow-up only ECG: {follow_up_only_ecg}")
        print(f"Follow-up only Actigraphy: {follow_up_only_act}")
        print(f"Follow-up none: {follow_up_none}")

        plot_length_distribution_nako(
            pickle_name="nako_information.pkl",
            binwidth = 0.1,
            xlim = [23, 25],
            ylim = [0, 100],
            # yscale = "log"
        )