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


def plot_length_distribution(
    pickle_name = "shhs_gif_plot.pkl",
    **kwargs
    ):
    
    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Recording length (hours)")
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
    GIF = np.array(pickle_data_loaded["GIF_signal_length"])

    threshold = 10 * 3600  # 10 hours in seconds
    print(f"SHHS length > {threshold/3600} hours: {np.sum(SHHS > threshold)} / {len(SHHS)} ({np.sum(SHHS > threshold) / len(SHHS) * 100:.2f}%)")
    print(f"GIF length > {threshold/3600} hours: {np.sum(GIF > threshold)} / {len(GIF)} ({np.sum(GIF > threshold) / len(GIF) * 100:.2f}%)")

    threshold = 11.5 * 3600 # about 11.5 hours in seconds
    print(f"SHHS length > {threshold/3600} hours: {np.sum(SHHS > threshold)} / {len(SHHS)} ({np.sum(SHHS > threshold) / len(SHHS) * 100:.2f}%)")
    print(f"GIF length > {threshold/3600} hours: {np.sum(GIF > threshold)} / {len(GIF)} ({np.sum(GIF > threshold) / len(GIF) * 100:.2f}%)")

    lengths = np.concatenate((SHHS, GIF)) / 3600  # convert to hours
    shhs_label = np.array(["SHHS" for _ in range(len(SHHS))])
    gif_label = np.array(["GIF" for _ in range(len(GIF))])
    label = np.concatenate((shhs_label, gif_label))

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
    pickle_name = "shhs_gif_plot.pkl",
    sleep_labels = ["Wake", "LS", "DS", "REM", "Artefect"],
    stat = "percentage", # "count" or "percentage"
    **kwargs
    ):

    kwargs.setdefault("title", "")
    kwargs.setdefault("xlabel", "Sleep stage")
    kwargs.setdefault("ylabel", "Count" if stat == "count" else "Relative Count (\%)") # type: ignore
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

    with open(pickle_name, "rb") as f:
        pickle_data_loaded = pickle.load(f)

    shhs_num_stages = pickle_data_loaded["SHHS_slp_stages_count"]
    gif_num_stages = pickle_data_loaded["GIF_slp_stages_count"]

    shhs_stages = pickle_data_loaded["SHHS_slp_stages"]
    shhs_labels = [sleep_labels[stage] for stage in shhs_stages]
    gif_stages = pickle_data_loaded["GIF_slp_stages"]
    gif_labels = [sleep_labels[stage] for stage in gif_stages]

    print(f"SHHS stages: {shhs_stages}, counts: {shhs_num_stages}, relative: {[count/sum(shhs_num_stages) * 100 for count in shhs_num_stages]}")
    print(f"GIF stages: {gif_stages}, counts: {gif_num_stages}, relative: {[count/sum(gif_num_stages) * 100 for count in gif_num_stages]}")

    if stat == "count":
        lengths = np.concatenate((shhs_num_stages, gif_num_stages))
    elif stat == "percentage":
        total_shhs = sum(shhs_num_stages)
        total_gif = sum(gif_num_stages)
        lengths = np.concatenate((np.array(shhs_num_stages) / total_shhs * 100, np.array(gif_num_stages) / total_gif * 100))
    stages = np.concatenate((shhs_labels, gif_labels))
    label = np.concatenate((["SHHS" for _ in range(len(shhs_num_stages))], ["GIF" for _ in range(len(gif_num_stages))]))

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
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    
    ax.set_title(kwargs["title"])
    plt.show()



def data_shhs_distribution(path_to_shhs_dataset: str, path_to_gif_dataset: str):
    # access the SHHS dataset
    shhs_dataset = h5py.File(path_to_shhs_dataset, 'r')

    # accessing patient ids:
    patients = list(shhs_dataset['slp'].keys()) # type: ignore

    # saving all data from SHHS dataset to the pickle file
    signal_length = []
    slp_frequency = shhs_dataset["slp"].attrs["freq"]
    
    slp_stage = []
    slp_stage_count = []

    tie = 0
    total = 0
    count_patient = 0
    
    for patient_id in patients:
        data = shhs_dataset["slp"][patient_id][:] # type: ignore
        for i in range(len(data)): # type: ignore
            if data[i] == 0: # type: ignore
                data[i] = 0 # type: ignore
            elif data[i] == 1: # type: ignore
                data[i] = 0 # type: ignore
            elif data[i] == 2: # type: ignore
                data[i] = 1 # type: ignore
            elif data[i] == 3: # type: ignore
                data[i] = 2 # type: ignore
            elif data[i] == 4: # type: ignore
                data[i] = 4 # type: ignore
            elif data[i] == 5: # type: ignore
                data[i] = 3 # type: ignore
            elif data[i] == 6: # type: ignore
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
                    break
                if count == max_count:
                    appeared = True
            
        for stage in data: # type: ignore
            if stage not in slp_stage:
                slp_stage.append(stage)
                slp_stage_count.append(1)
            else:
                index = slp_stage.index(stage)
                slp_stage_count[index] += 1
        
        count_patient += 1
        print(count_patient, end = "\r")

    print(f"SHHS Tie: {tie}, Total: {total}, Ratio: {tie/total if total > 0 else 0}")
    print(slp_stage, slp_stage_count)

    # access the GIF dataset
    gif_dataset = h5py.File(path_to_gif_dataset, 'r')

    # accessing patient ids:
    patients = list(gif_dataset['stage'].keys()) # type: ignore

    # saving all data from SHHS dataset to the pickle file
    gif_signal_length = []
    slp_frequency = 1/30
    
    gif_slp_stage = []
    gif_slp_stage_count = []

    gif_tie = 0
    gif_total = 0
    count_patient = 0

    # saving all data from GIF dataset to the pickle file
    for patient_id in patients:
        data = np.array(gif_dataset["stage"][patient_id][:]).astype(int) # type: ignore
        for i in range(len(data)): # type: ignore
            if data[i] == 0: # type: ignore
                data[i] = 0 # type: ignore
            elif data[i] == 1: # type: ignore
                data[i] = 0 # type: ignore
            elif data[i] == 2: # type: ignore
                data[i] = 1 # type: ignore
            elif data[i] == 3: # type: ignore
                data[i] = 2 # type: ignore
            elif data[i] == 4: # type: ignore
                data[i] = 4 # type: ignore
            elif data[i] == 5: # type: ignore
                data[i] = 3 # type: ignore
            elif data[i] == 6: # type: ignore
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
                    break
                if count == max_count:
                    appeared = True
            
        for stage in data: # type: ignore
            if stage not in gif_slp_stage:
                gif_slp_stage.append(stage)
                gif_slp_stage_count.append(1)
            else:
                index = gif_slp_stage.index(stage)
                gif_slp_stage_count[index] += 1

        count_patient += 1
        print(count_patient, end = "\r")

    print(f"GIF Tie: {gif_tie}, Total: {gif_total}, Ratio: {gif_tie/gif_total if gif_total > 0 else 0}")
    print(gif_slp_stage, gif_slp_stage_count)

    dict = {
        "SHHS_slp_stages": slp_stage,
        "SHHS_slp_stages_count": slp_stage_count,
        "SHHS_signal_length": signal_length,
        "GIF_slp_stages": gif_slp_stage,
        "GIF_slp_stages_count": gif_slp_stage_count,
        "GIF_signal_length": gif_signal_length,
    }

    with open("shhs_gif_plot.pkl", "wb") as f:
        pickle.dump(dict, f)


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

    # data_shhs_distribution("Raw_Data/SHHS_dataset.h5", "Raw_Data/GIF_dataset.h5")
    plot_length_distribution(binwidth = 0.5)
    
    # plot_length_distribution(yscale = "log", ylim = [1, 10000], binwidth = 0.5, xlim = [0, 16])

    plot_sleep_stages_distribution(stat="percentage", yscale="linear")