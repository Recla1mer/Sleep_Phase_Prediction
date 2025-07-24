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

    number_updates = scheduler.number_updates_total

    x_data = np.arange(1, number_updates)
    y_data = np.array([scheduler(update) for update in x_data])

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
    include_gifs = ["gif_5min", "gif_separated", "GIF_dataset"],
    label_gifs = ["Gaps Fused", "Gaps Separated", "Yao"],
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

    SHHS = np.array(pickle_data_loaded["SHHS_signal_length"])
    GIF = list()
    for name in include_gifs:
        GIF.append(np.array(pickle_data_loaded[f"{name}_signal_length"]))

    threshold = 10 * 3600  # 10 hours in seconds
    print(f"SHHS length > {threshold/3600} hours: {np.sum(SHHS > threshold)} / {len(SHHS)} ({np.sum(SHHS > threshold) / len(SHHS) * 100:.2f}%)")
    for i in range(len(GIF)):
        print(f"{label_gifs[i]} length > {threshold/3600} hours: {np.sum(GIF[i] > threshold)} / {len(GIF[i])} ({np.sum(GIF[i] > threshold) / len(GIF[i]) * 100:.2f}%)")

    threshold = 8 * 3600 # about 4 hours in seconds
    print(f"SHHS length > {threshold/3600} hours: {np.sum(SHHS > threshold)} / {len(SHHS)} ({np.sum(SHHS > threshold) / len(SHHS) * 100:.2f}%)")
    for i in range(len(GIF)):
        print(f"{label_gifs[i]} length > {threshold/3600} hours: {np.sum(GIF[i] > threshold)} / {len(GIF[i])} ({np.sum(GIF[i] > threshold) / len(GIF[i]) * 100:.2f}%)")

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


def plot_sleep_stages_distribution(
    results_file_path: str,
    include_gifs = ["gif_5min", "gif_separated", "GIF_dataset"],
    label_gifs = ["Gaps Fused", "Gaps Separated", "Yao"],
    sleep_labels = ["Wake", "LS", "DS", "REM", "Artifact"],
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

    with open(results_file_path, "rb") as f:
        pickle_data_loaded = pickle.load(f)
    
    print(pickle_data_loaded.keys())

    shhs_num_stages = pickle_data_loaded["SHHS_slp_stages_count"]
    gif_num_stages = list()
    for name in include_gifs:
        gif_num_stages.append(pickle_data_loaded[f"{name}_slp_stages_count"])

    shhs_stages = pickle_data_loaded["SHHS_slp_stages"]
    shhs_labels = [sleep_labels[stage] for stage in shhs_stages]

    gif_stages = list()
    for name in include_gifs:
        gif_stages.append(pickle_data_loaded[f"{name}_slp_stages"])
    
    gif_labels = list()
    for stages in gif_stages:
        gif_labels.append([sleep_labels[stage] for stage in stages])

    print(f"SHHS stages: {shhs_stages}, counts: {shhs_num_stages}, relative: {[count/sum(shhs_num_stages) * 100 for count in shhs_num_stages]}")

    for i, stages in enumerate(gif_stages):
        print(f"{label_gifs[i]} stages: {stages}, counts: {gif_num_stages[i]}, relative: {[count/sum(gif_num_stages[i]) * 100 for count in gif_num_stages[i]]}")

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

    stages = np.concatenate((shhs_labels, gif_labels[0]))
    for i in range(1, len(gif_labels)):
        stages = np.concatenate((stages, gif_labels[i]))

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

    for i in range(5):
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
        for i in range(len(data)): # type: ignore
            if data[i] == 0: # type: ignore
                data[i] = 0 # type: ignore
            elif data[i] == 1: # type: ignore
                data[i] = 1 # type: ignore
            elif data[i] == 2: # type: ignore
                data[i] = 1 # type: ignore
            elif data[i] == 3: # type: ignore
                data[i] = 2 # type: ignore
            elif data[i] == 5: # type: ignore
                data[i] = 3 # type: ignore
            else: # type: ignore
                data[i] = 4 # type: ignore

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
            if data[i] in [1, 2, 3]: # type: ignore
                stop_counting_at = i
                break
        
        count_iteration = 0
        for stage in data: # type: ignore
            if include_aso_only and count_iteration > stop_counting_at:
                break
            if stage in [1,2,3]:
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

    for i in range(5):
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
        data = np.array(gif_dataset["stage"][patient_id][:]).astype(int) # type: ignore
        for i in range(len(data)): # type: ignore
            if data[i] == 0: # type: ignore
                data[i] = 0 # type: ignore
            elif data[i] == 1: # type: ignore
                data[i] = 1 # type: ignore
            elif data[i] == 2: # type: ignore
                data[i] = 1 # type: ignore
            elif data[i] == 3: # type: ignore
                data[i] = 2 # type: ignore
            elif data[i] == 5: # type: ignore
                data[i] = 3 # type: ignore
            else: # type: ignore
                data[i] = 4 # type: ignore

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
                if data[i] in [1, 2, 3]:
                    break

            data = data[i:] # remove everything before the first sleep stage

            for i in range(len(data)-1, -1, -1):
                if data[i] in [1, 2, 3]:
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

        for i in range(5):
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
            data = np.array(data_dict["SLP"]).astype(int) # type: ignore
            for i in range(len(data)): # type: ignore
                if data[i] == 0: # type: ignore
                    data[i] = 0 # type: ignore
                elif data[i] == 1: # type: ignore
                    data[i] = 1 # type: ignore
                elif data[i] == 2: # type: ignore
                    data[i] = 1 # type: ignore
                elif data[i] == 3: # type: ignore
                    data[i] = 2 # type: ignore
                elif data[i] == 5: # type: ignore
                    data[i] = 3 # type: ignore
                else: # type: ignore
                    data[i] = 4 # type: ignore

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
                    if data[i] in [1, 2, 3]:
                        break

                data = data[i:] # remove everything before the first sleep stage

                for i in range(len(data)-1, -1, -1):
                    if data[i] in [1, 2, 3]:
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
fig_ratio = 4 / 3

if __name__ == "__main__":
    matplotlib.rcParams.update(tex_look)
    # linewidth*=0.3
    linewidth*=0.48
    # linewidth*=0.5
    matplotlib.rcParams["figure.figsize"] = [linewidth, linewidth / fig_ratio]

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
    
    # shhs
    # plot_learning_rate_scheduler(
    #     scheduler=CosineScheduler(
    #         number_updates_total = 40,
    #         number_updates_to_max_lr = 4,
    #         start_learning_rate = 1e-5,
    #         max_learning_rate = 1e-3,
    #         end_learning_rate = 1e-6,
    #     ))
    
    # gif
    # plot_learning_rate_scheduler(
    #     scheduler=CosineScheduler(
    #         number_updates_total = 100,
    #         number_updates_to_max_lr = 10,
    #         start_learning_rate = 1e-5,
    #         max_learning_rate = 1e-3,
    #         end_learning_rate = 1e-6,
    #     ))

    # data_shhs_distribution(
    #     results_file_path="slp_overview.pkl",
    #     path_to_shhs_dataset="Raw_Data/SHHS_dataset.h5",
    #     path_to_gif_pkl_datasets=["Raw_Data/gif_separated.pkl", "Raw_Data/gif_5min.pkl"],
    #     path_to_gif_h5="Raw_Data/GIF_dataset.h5",
    #     include_aso_only=False
    # )
    # data_shhs_distribution(
    #     results_file_path="slp_overview_aso.pkl",
    #     path_to_shhs_dataset="Raw_Data/SHHS_dataset.h5",
    #     path_to_gif_pkl_datasets=["Raw_Data/gif_separated.pkl", "Raw_Data/gif_5min.pkl"],
    #     path_to_gif_h5="Raw_Data/GIF_dataset.h5",
    #     include_aso_only=True
    # )

    plot_length_distribution(
        pickle_name = "slp_overview.pkl",
        include_gifs = ["gif_5min", "gif_separated", "GIF_dataset"],
        label_gifs = ["Gaps Fused", "Gaps Separated", "Yao"],
        binwidth = 0.25
    )

    plot_length_distribution(
        pickle_name = "slp_overview.pkl",
        include_gifs = ["gif_separated"],
        label_gifs = ["Gaps Separated"],
        binwidth = 0.25
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

    # plot_sleep_stages_distribution(
    #     results_file_path = "slp_overview.pkl",
    #     include_gifs=["gif_5min", "gif_separated", "GIF_dataset"],
    #     label_gifs=["Gaps Fused", "Gaps Separated", "Yao"],
    #     sleep_labels=["Wake", "LS", "DS", "REM", "Artifact"],
    #     stat="percentage",  # "count" or "percentage"
    #     yscale="linear",
    # )

    # plot_sleep_stages_distribution(
    #     results_file_path = "slp_overview_aso.pkl",
    #     include_gifs=["gif_5min", "gif_separated", "GIF_dataset"],
    #     label_gifs=["Gaps Fused", "Gaps Separated", "Yao"],
    #     sleep_labels=["Wake", "LS", "DS", "REM", "Artifact"],
    #     stat="percentage",  # "count" or "percentage"
    #     yscale="linear",
    # )

    # plot_length_distribution(binwidth = 0.25)
    
    # plot_length_distribution(yscale = "log", ylim = [1, 10000], binwidth = 0.5, xlim = [0, 16])