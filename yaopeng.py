"""
NOT MY CODE!
SOURCE: https://github.com/AlexMa123/DCNN-SHHS

Project also predicts sleep stages using same features as me. Applying it to some data and will compare our
predictions.
"""


import torch
from torch.nn.functional import softmax
from torch import nn
from scipy.interpolate import interp1d

import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

import time
import datetime
import pyedflib 
from numba import jit


def padding_same(kernel_size=3, dilation=1, stride=1):
    # stride * Lout = Lin + 2padding - dilation (kernel_size - 1) - 1 + stride
    # (stride - 1) * Lout + dilation (kernel_size - 1) + 1 - stride = 2 padding
    padding = dilation * (kernel_size - 1) + 1 - stride
    return padding // 2


class DCNN_classifier_input(nn.Module):
    """DCNN network for sleep stage classification

    """
    def __init__(self,
                 epoch_length=(512, 128),
                 num_windows_features=128,
                 num_channels_rri=[8, 16, 32, 64],
                 num_channels_mad=[8, 16, 32, 64],
                 dilations=[2, 4, 8, 16, 32],
                 n_classes=4):
        """
        DCNN network for sleep stage classification based on RRI and MAD

        Parameters
        ----------
        epoch_length : tuple, optional
            length of the input signals in one epoch.
            For example, 128s RRI (4Hz) and 128s MAD (1Hz) will be (512, 128), by default (512, 128)
        num_windows_features : int, optional
            how many features learned from each window, by default 128
        num_channels_rri : list, optional
            number of channels to process RRI signal by 1d-convolution, by default [8, 16, 32, 64]
        num_channels_mad : list, optional
            number of channels to process MAD signal, by default [8, 16, 32, 64]
        dilations : list, optional
            for DCNN temporal network, by default [2, 4, 8, 16, 32]
        n_classes : int, optional
            number of predict sleep stages., by default 4
        """
        super(DCNN_classifier_input, self).__init__()
        self.n_classes = n_classes
        self.w1, self.w2 = epoch_length
        self.num_pooling_rri = len(num_channels_rri) - 1
        self.num_pooling_mad = len(num_channels_mad) - 1
        # ======================================================================================
        # Windows learning part
        # signal feature learning part
        # RRI
        self.input_convolution_rri = nn.Conv1d(1, num_channels_rri[0], 1)
        cnn_layers = []
        cnn_layers.append(nn.LeakyReLU(0.15))
        cnn_layers.append(nn.BatchNorm1d(num_channels_rri[0]))

        for i in range(0, len(num_channels_rri) - 1):
            cnn_layers.append(nn.Conv1d(num_channels_rri[i],
                                        num_channels_rri[i + 1],
                                        3, padding=padding_same(3)))
            cnn_layers.append(nn.MaxPool1d(2),)
            cnn_layers.append(nn.LeakyReLU(0.15))
            cnn_layers.append(nn.BatchNorm1d(num_channels_rri[i+1]))

        self.signallearning_rri = nn.Sequential(*cnn_layers)
        # MAD
        self.input_convolution_mad = nn.Conv1d(1, num_channels_mad[0], 1)
        cnn_layers = []
        cnn_layers.append(nn.LeakyReLU(0.15))
        cnn_layers.append(nn.BatchNorm1d(num_channels_mad[0]))

        for i in range(0, len(num_channels_mad) - 1):
            cnn_layers.append(nn.Conv1d(num_channels_mad[i],
                                        num_channels_mad[i + 1],
                                        3, padding=padding_same(3)))
            cnn_layers.append(nn.MaxPool1d(2))
            cnn_layers.append(nn.LeakyReLU(0.15))
            cnn_layers.append(nn.BatchNorm1d(num_channels_mad[i+1]))
        self.signallearning_mad = nn.Sequential(*cnn_layers)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((epoch_length[0] // (2 ** self.num_pooling_rri) * num_channels_rri[-1] +
                                 epoch_length[1] // (2 ** self.num_pooling_mad) * num_channels_mad[-1]),
                                num_windows_features)
        # Windows feature learning END
        # =====================================================================================
        self.classfier = nn.Sequential(
            ResBlock(num_windows_features, 7, dilations),
            ResBlock(num_windows_features, 7, dilations),
            nn.Conv1d(num_windows_features, n_classes, 1)
        )

    def forward(self, rri, mad=None):
        batchsize, Nwindows, w1 = rri.shape
        assert w1 == self.w1, 'the input length of RRI is not the same with your configuration of NN'
        rri = rri.reshape(-1, 1, w1)
        out = self.input_convolution_rri(rri)
        out = self.signallearning_rri(out)
        if mad is not None:
            _, _, w2 = mad.shape
            assert w2 == self.w2, 'the input length of MAD is not the same with your configuration of NN'
            mad = mad.reshape(-1, 1, w2)
            out_mad = self.input_convolution_mad(mad)
            out_mad = self.signallearning_mad(out_mad)
        else:
            out_mad = torch.zeros((out.shape[0], out.shape[1],
                                   self.w2 // (2 ** self.num_pooling_mad)), device=rri.device)
        out = torch.cat([out, out_mad], dim=-1)

        out = self.flatten(out)
        out = self.linear(out)
        out = out.reshape(batchsize, Nwindows, -1)
        out = out.transpose(1, 2).contiguous()
        out = self.classfier(out)
        out = out.transpose(1, 2).contiguous().reshape(batchsize, Nwindows, -1)
        return out


class ResBlock(nn.Module):
    def __init__(self, num_windows_features=128, kernel_size=7,
                 dilations=[2, 4, 8, 16, 32]):
        super(ResBlock, self).__init__()
        cnn_list = []
        for d in dilations:
            cnn_list.append(nn.LeakyReLU(0.15))
            cnn_list.append(nn.Conv1d(
                num_windows_features, num_windows_features,
                kernel_size=kernel_size,
                dilation=d,
                padding=padding_same(kernel_size, d)
            ))
            cnn_list.append(nn.Dropout(0.2))
        self.cnn = nn.Sequential(
            *cnn_list
        )

    def forward(self, x):
        out = self.cnn(x)
        out = x + out
        return out


def cal_mad(data, points):
    """
    Calculate MAD values. (by Johannes Zschocke)
    This routine calculates the MAD values based on the number of points
    (given by points) on the given 3D data.
    Parameters
    ----------
    data : 3D numpy array
        contains 3D data.
    points : integer
        number of datapoints to use for mad calculation.
    Returns
    -------
    mad : numpy array
        contains the calculated mad values.
    """
    # 'allocate' arrays and variables
    # the following variables are used as described and introduced by VahaYpya
    # 2015 DOI: 10.1111/cpf.12127

    # array for mad values
    mad = np.zeros(int(len(data[0]) / points))

    # array for r_i values
    r_i_array = np.empty(points)
    r_i_array[:] = np.nan

    # R_ave value
    R_ave = 0
    i_mad = 0

    # iterate over all values in data
    i = 0
    for (x, y, z) in zip(data[0], data[1], data[2]):
        r_i = np.sqrt(x**2 + y**2 + z**2)
        r_i_array[i] = r_i
        i += 1
        if (i == points):
            R_ave = np.nanmean(r_i_array)
            s = 0
            for ri in r_i_array:
                s += np.abs(ri - R_ave)

            s = s / points
            mad[i_mad] = s
            i_mad += 1
            r_i_array[:] = np.nan
            i = 0

    return mad


def read_rri(rri_file: str):
    """
    load rri file, return rri, start time (datetime), and frequency
    """
    f = open(rri_file, 'r')
    f.readline()
    f.readline()
    """
    Get start time of this rri file
    """
    rr_startdate = f.readline()
    index_start = rr_startdate.find("=") + 1
    rr_startdate = rr_startdate[index_start:-1]
    rr_starttime = f.readline()
    index_start = rr_starttime.find("=") + 1
    rr_starttime = rr_starttime[index_start:-1]
    file_starttime = rr_startdate + " " + rr_starttime

    try:
        file_starttime = datetime.datetime.strptime(file_starttime,
                                                    "%d.%m.%Y %H:%M:%S")
    except ValueError:
        file_starttime = datetime.datetime.strptime(file_starttime,
                                                    "%Y-%m-%d %H:%M:%S")
    """
    get frequency of rri
    """
    line = f.readline()
    index_start = line.find("=") + 1
    freq = int(line[index_start:])
    skiprows = 5
    while not line.startswith("----"):
        line = f.readline()
        skiprows += 1

    f.close()
    df = pd.read_csv(rri_file,
                     skiprows=skiprows,
                     sep="\t",)
    # df = df[df.type == "N"]
    return df.iloc[:, 0].values, freq, file_starttime


def rr_interval(rposition, signalfreq, resample_freq=4):
    r_t = np.empty(rposition.size + 2)
    r_t[0] = (rposition[0] * 2 - rposition[1]) / signalfreq
    r_t[-1] = (rposition[-1] * 2 - rposition[-2]) / signalfreq
    r_t[1:-1] = rposition / signalfreq
    rri = np.diff(r_t)
    mean_rri = np.mean(rri)

    rri[rri > 2.0] = mean_rri
    rri[rri < 0.2] = mean_rri

    t = np.arange(0, int(r_t[-2]), 1. / resample_freq)
    f = interp1d(r_t[:-1],
                 rri,
                 kind='linear',
                 fill_value="extrapolate")
    rri = f(t)
    return t, rri


def get_mad(edffilename: str, duration: int = 2, channel_name=["X", "Y", "Z"]):
    """ get mad value from a edffile

    Parameters
    ----------
    edffilename :
        edffile's name
    duration : int
        num of secondes used to cal mad value, by default 2
    channel_name : list, optional
        by default ["X", "Y", "Z"]
    """
    edffile = pyedflib.EdfReader(edffilename)
    starttime = edffile.getStartdatetime()
    labels = edffile.getSignalLabels()
    assert len(channel_name) == 3, "length of channel_name list should be 3"
    indexs = [labels.index(channel_name[i]) for i in range(3)]
    signals = [edffile.readSignal(i) for i in indexs]
    signals = np.array(signals)
    freq = edffile.getSampleFrequency(indexs[0])
    mad = cal_mad(signals, int(freq * duration))
    edffile.close()
    return starttime, mad


def get_sleep_stage(filename):
    df = pd.read_csv(filename,
                     sep=";",
                     usecols=[1, 3, 31],
                     names=["time", "stage", "apnea"])
    starttime = df.iloc[0, 0][1:]
    return starttime, df.stage.values, df.apnea.values


def get_second(starttime):
    second = 0
    if isinstance(starttime, str):
        second = int(starttime[:2]) * 3600 + int(starttime[3:5]) * 60 + int(
            starttime[6:])
    elif isinstance(starttime, datetime.datetime):
        second = starttime.hour * 3600 + starttime.minute * 60 + starttime.second
    else:
        raise TypeError("Starttime should be datetime or str")
    return second if second > 12 * 3600 else second + 24 * 3600


def reshape_signal(signal, freq, num_windows=1200, windows_size=128, overlap=98, segment_step=240):
    """
    Reshape a signal whose shape is (n * freq) to (num_windows, windows_size)

    If the signal's size is smaller than num_windows * (windows_size - overlap) + overlap,
    then the signal will be padded with zeros.

    If the signal's size is larger than num_windows * (windows_size - overlap) + overlap,
    then the signal will be splitted into multiple segments with step = 1 hour
    Parameters
    ----------
    signal
        signal
    freq
        sampling frequency
    num_windows
        number of windows
    windows_size
        size of each window / in seconds
    overlap
        overlap between two windows / in seconds
    segment_step
        step between two segments / in windows
    """
    step = windows_size - overlap
    left = (windows_size - step)
    num_droped = step - left % step
    if num_droped != 0 and num_droped != 30:
        signal = signal[int(num_droped * freq): - int(num_droped * freq)]
    signal = signal.unfold(0, int(windows_size * freq), int(step * freq))
    if signal.shape[0] < num_windows:
        zero_start = signal.shape[0]
        signal = torch.cat([signal, torch.zeros(num_windows - signal.shape[0],
                                                int(windows_size * freq), device=signal.device) - 1], dim=0)
        return signal.reshape(1, signal.shape[0], signal.shape[1]), zero_start + 1
    else:
        zero_start = -1
        signals = []
        segment_overlap = num_windows - segment_step
        num_segments = (signal.shape[0] - segment_overlap) / segment_step
        for i in range(ceil(num_segments)):
            if i * segment_step + num_windows > signal.shape[0]:
                zero_start = signal.shape[0] - i * segment_step
                signals.append(torch.cat([signal[i * segment_step:],
                                          torch.zeros(num_windows - zero_start,
                                                      int(windows_size * freq), device=signal.device) - 1],
                                         dim=0))

            else:
                signals.append(signal[i * segment_step: i * segment_step + num_windows])

        return torch.stack(signals), zero_start


def predict_stage(net, rri, mad=None, out_prob=False, out_energy=False):
    net.eval()
    reshaped_rri, zero_start = reshape_signal(rri, 4)

    if mad is not None:
        reshaped_mad, zero_start = reshape_signal(mad, 1)
        reshaped_mad = reshaped_mad.to(device)
        if reshaped_mad.shape[0] > reshaped_rri.shape[0]:
            shape_diff = reshaped_mad.shape[0] - reshaped_rri.shape[0]
            reshaped_rri = torch.cat([reshaped_rri, torch.zeros(shape_diff, reshaped_rri.shape[1], reshaped_rri.shape[2],
                                                                device=reshaped_rri.device, dtype=reshaped_rri.dtype)])
    else:
        reshaped_mad = None

    reshaped_rri = reshaped_rri.to(device)
    with torch.no_grad():
        predict_score = net(reshaped_rri, reshaped_mad)
    result = []
    predict_result = torch.argmax(predict_score, dim=-1)
    if out_energy or out_prob:
        exp_values = torch.exp(predict_score)
    if out_energy:
        energy = - torch.log(exp_values.sum(axis=-1))
        result_energy = []
    else:
        result_energy = None
    if out_prob:
        predict_prob = exp_values / exp_values.sum(axis=-1, keepdim=True)
        result_prob = []
    else:
        result_prob = None

    if predict_result.shape[0] == 1:
        result.append(predict_result[0][:zero_start])
        if out_prob:
            result_prob.append(predict_prob[0][:zero_start])
        if out_energy:
            result_energy.append(energy[0][:zero_start])
    else:
        for i in range(predict_result.shape[0]):
            if i == predict_result.shape[0] - 1:
                result.append(predict_result[i][-240:zero_start])
            elif i == 0:
                result.append(predict_result[i])
            else:
                result.append(predict_result[i][-240:])
            if out_prob:
                if i == predict_result.shape[0] - 1:
                    result_prob.append(predict_prob[i][-240:zero_start])
                elif i == 0:
                    result_prob.append(predict_prob[i])
                else:
                    result_prob.append(predict_prob[i][-240:])
            if out_energy:
                if i == predict_result.shape[0] - 1:
                    result_energy.append(energy[i][-240:zero_start])
                elif i == 0:
                    result_energy.append(energy[i])
                else:
                    result_energy.append(energy[i][-240:])

    if out_prob:
        result_prob = torch.cat(result_prob)
    if out_energy:
        result_energy = torch.cat(result_energy)
    return torch.cat(result), result_prob, result_energy


def predict_stage_fullp(net, rri, mad=None):
    net.eval()
    reshaped_rri, zero_start = reshape_signal(rri, 4)

    if mad is not None:
        reshaped_mad, zero_start = reshape_signal(mad, 1)
        if reshaped_mad.shape[0] > reshaped_rri.shape[0]:
            shape_diff = reshaped_mad.shape[0] - reshaped_rri.shape[0]
            reshaped_rri = torch.cat([reshaped_rri, torch.zeros(shape_diff, reshaped_rri.shape[1], reshaped_rri.shape[2],
                                                                device=reshaped_rri.device, dtype=reshaped_rri.dtype)])
        reshaped_mad = reshaped_mad.to(device)
    else:
        reshaped_mad = None
    reshaped_rri = reshaped_rri.to(device)
    with torch.no_grad():
        predict_score = net(reshaped_rri, reshaped_mad)
    predict_result = torch.argmax(predict_score, dim=-1)
    exp_values = torch.exp(predict_score)
    predict_energy = - torch.log(exp_values.sum(axis=-1))
    predict_prob = exp_values / exp_values.sum(axis=-1, keepdim=True)
    # predict_prob = softmax(predict_score, dim=-1)
    total_length = predict_result.shape[0] * 240 + 1200 - 240
    results = np.full((total_length, 5 * (2 + net.n_classes) + 1), np.nan)
    for i in range(predict_prob.shape[0]):
        j = i % 5
        results[i * 240: i * 240 + 1200, j + 1] = predict_result[i].cpu().numpy()
        results[i * 240: i * 240 + 1200,
                5 + j + 1] = predict_energy[i].cpu().numpy()
        results[i * 240: i * 240 + 1200, 
                10 + j * net.n_classes + 1: 10 + net.n_classes * (j + 1) + 1] = predict_prob[i].cpu().numpy()
        if i == 0:
            results[:1200, 0] = predict_result[i].cpu().numpy()
        else:
            results[1200 + 240 * (i - 1): 240 * i + 1200, 0] = predict_result[i][-240:].cpu().numpy()
    results = results[:zero_start - 1200 - 1]
    return results


def predict_stage_fromfile(net, rrifile, madfile, out_prob=False, with_energy=False):
    net.eval()
    rpeak, freq, starttime = read_rri(rrifile)
    _, rri = rr_interval(rpeak, freq, 4)
    rri = rri[11 * 4:]
    rri = torch.from_numpy(rri).float()
    # rri = rri - rri.mean()
    if madfile is not None:
        madfile = np.load(madfile, allow_pickle=True)
        mad_starttime = madfile['starttime']
        mad = madfile['mad']
        mad = mad[11:]
        mad = torch.from_numpy(mad).float()
    else:
        mad = None
    predicted_stage, stage_prob, stage_energy = predict_stage(net, rri, 
                                                              mad, out_prob,
                                                              with_energy)
    if mad_starttime != starttime:
        print("Warning: starttime of MAD file is different from starttime of RRI file", madfile)
    if out_prob:
        predicted_stage = predicted_stage
        stage_prob = stage_prob.cpu().numpy()
    if with_energy:
        stage_energy = stage_energy.cpu().numpy()
    starttime = starttime + datetime.timedelta(0, 49 + 11)
    time_index = pd.date_range(start=starttime, periods=predicted_stage.shape[0], freq='30S')
    t = time_index.hour + time_index.minute / 60 + time_index.second / 3600
    t = t + (time_index.day - time_index.day[0]) * 24
    # result = pd.Series(predicted_stage, index=t)
    result = {
        'stage': predicted_stage.cpu().numpy(),
    }
    if with_energy:
        result['energy'] = stage_energy
    if out_prob:
        for j in range(net.n_classes):
            result[f'P{j}'] = stage_prob[:, j]
        # result['Pwake'] = stage_prob[:, 0]
        # result['Plight'] = stage_prob[:, 1]
        # result['Pdeep'] = stage_prob[:, 2]
        # result['Prem'] = stage_prob[:, 3]
    result = pd.DataFrame(result, index=t)
    return result


def predict_stage_fromfile_fullp(net, rrifile, madfile):
    net.eval()
    rpeak, freq, starttime = read_rri(rrifile)
    _, rri = rr_interval(rpeak, freq, 4)
    rri = rri[11 * 4:]
    rri = torch.from_numpy(rri).float()
    # rri = rri - rri.mean()
    if madfile is not None:
        mad = np.load(madfile)['mad']
        mad = mad[11:]
        mad = torch.from_numpy(mad).float()
    else:
        mad = None
    predict_result = predict_stage_fullp(net, rri, mad)
    starttime = starttime + datetime.timedelta(0, 49 + 11)
    time_index = pd.date_range(start=starttime, periods=predict_result.shape[0], freq='30S')
    t = time_index.hour + time_index.minute / 60 + time_index.second / 3600
    t = t + (time_index.day - time_index.day[0]) * 24
    headers = ['predict_stage']
    headers = headers + [f'stage{i+1}' for i in range(5)]
    headers = headers + [f'energy{i+1}' for i in range(5)]
    for i in range(5):
        for j in range(net.n_classes):
            headers.append(f'P{j}_{i+1}')
            # headers += [f'P{j}_{i+1}', f'P{j}_{i+1}', f'P{j}_{i+1}', f'P{j}_{i+1}']
    df = pd.DataFrame(predict_result, index=t, columns=headers)
    df['predict_stage'] = df['predict_stage'].astype('Int32')
    for i in range(5):
        df[f'stage{i+1}'] = df[f'stage{i+1}'].astype('Int32')
    return df


from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score

def sort_yao_files(my_files, yao_files):

    for file_index in range(len(my_files)):
        ids = []
        working_file_path = os.path.split(yao_files[file_index])[0] + "/working_file.pkl"

        my_gen = load_from_pickle(my_files[file_index])
        next(my_gen)
        yao_gen = load_from_pickle(yao_files[file_index])

        for data in my_gen:
            ids.append(data["ID"])

        all_paths = ["" for i in range(len(ids))]
        
        for data in yao_gen:
            try:
                position = ids.index(data["ID"])
            except:
                continue
            new_path = os.path.split(yao_files[file_index])[0] + "/split" + str(position) + ".pkl"
            all_paths[position] = new_path
            
            save_to_pickle(data, new_path)
        
        for path_index in range(len(all_paths)):
            path = all_paths[path_index]
            if path == "":
                this_data = {
                    "ID": ids[path_index],
                    "SLP_predicted": []
                }
            else:
                this_data = next(load_from_pickle(path))
                os.remove(path)

            append_to_pickle(this_data, working_file_path)
        
        os.remove(yao_files[file_index])
        os.rename(working_file_path, yao_files[file_index])
        

def compare_predictions(my_files, yao_files):
    my_results = []
    yao_results = []

    number_data = 0
    
    for file_index in range(len(my_files)):
        my_gen = load_from_pickle(my_files[file_index])
        next(my_gen)
        yao_gen = load_from_pickle(yao_files[file_index])

        while True:
            try:
                my_data = next(my_gen)
                yao_data = next(yao_gen)

                if my_data["ID"] != yao_data["ID"]:
                    print("ID mismatch")
                    break

                min_length = min(len(my_data["SLP_predicted"]), len(yao_data["SLP_predicted"]))
                if len(my_data["SLP_predicted"]) != len(yao_data["SLP_predicted"]):
                    print("Length mismatch: ", len(my_data["SLP_predicted"]), len(yao_data["SLP_predicted"]))
                my_results.extend(my_data["SLP_predicted"][:min_length])
                yao_results.extend(yao_data["SLP_predicted"][:min_length])
                number_data += 1
            except:
                break
    
    print("Number of data:", number_data)
    print("Number of predictions:", len(my_results))
    print(accuracy_score(my_results, yao_results))


def quick_yao_predict(unknown_dataset_paths: list):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    net = DCNN_classifier_input()
    net.load_state_dict(torch.load(directory + "yao_net.pt", map_location=torch.device(device), weights_only=True))
    net.to(device)

    unpreditable_data = []

    # predict sleep stages for unknown data
    for unknown_dataset_path in unknown_dataset_paths:

        data_generator = load_from_pickle(unknown_dataset_path)
        results_path = directory + os.path.split(unknown_dataset_path)[1].split(".")[0] + "_yao.pkl"

        count = 0
        for data_dict in data_generator:
            try:
                predicted_results = predict_stage(net, torch.from_numpy(np.array(copy.deepcopy(data_dict["RRI"]))).float(), torch.from_numpy(np.array(copy.deepcopy(data_dict["MAD"]))).float())
                new_result = scale_classification_signal(predicted_results[0].cpu().numpy(), len(predicted_results[0].cpu().numpy())/len(data_dict["MAD"]), 1/30)
            except:
                unpreditable_data.append([unknown_dataset_path, data_dict["ID"]])
                continue
            
            # print(len(data_dict["MAD"]), len(predicted_results[0].cpu().numpy())*30, len(new_result)*30)

            results = {
                "ID": data_dict["ID"],
                "SLP_predicted": new_result,
            }
            append_to_pickle(results, results_path)
            count += 1
            print(count, end="\r")
        
        print(unpreditable_data)
        print(len(unpreditable_data))


def append_all_results_to_one_file(my_files, yao_files, new_path):
    for file_index in range(len(my_files)):
        my_gen = load_from_pickle(my_files[file_index])
        next(my_gen)
        yao_gen = load_from_pickle(yao_files[file_index])

        while True:
            try:
                my_data = next(my_gen)
                my_slp = summarize_predicted_signal(predicted_signal = my_data["SLP_predicted_probability"], mode = "probability")
                yao_data = next(yao_gen)

                if my_data["ID"] != yao_data["ID"]:
                    print("ID mismatch")
                    break

                min_length = min(len(my_data["SLP_predicted"]), len(yao_data["SLP_predicted"]))
                if len(my_data["SLP_predicted"]) != len(yao_data["SLP_predicted"]):
                    print("Length mismatch: ", len(my_data["SLP_predicted"]), len(yao_data["SLP_predicted"]))
                
                this_data = {
                    "ID": my_data["ID"],
                    "my_SLP": my_slp[:min_length],
                    "yao_SLP": yao_data["SLP_predicted"][:min_length]
                }
                append_to_pickle(this_data, new_path)
            except:
                break


from main import *
from plot_helper import *

if __name__ == "__main__":

    directory = "yao_net_compare/"
    model_directory_path = "Yao_Original/"

    unknown_dataset_paths = ["/Volumes/NaKo-UniHalle/RRI_and_MAD/NAKO-33a.pkl"]
    # unknown_dataset_paths = ["RRI_and_MAD/NAKO-33a.pkl", "RRI_and_MAD/NAKO-33b.pkl", "RRI_and_MAD/NAKO-84.pkl", "RRI_and_MAD/NAKO-419.pkl", "RRI_and_MAD/NAKO-609.pkl"]

    quick_yao_predict(unknown_dataset_paths)

    my_files = ["Processed_NAKO/NAKO-33a.pkl", "Processed_NAKO/NAKO-33b.pkl", "Processed_NAKO/NAKO-84.pkl", "Processed_NAKO/NAKO-419.pkl", "Processed_NAKO/NAKO-609.pkl"]
    yao_files = [directory + "NAKO-33a_yao.pkl", directory + "NAKO-33b_yao.pkl", directory + "NAKO-84_yao.pkl", directory + "NAKO-419_yao.pkl", directory + "NAKO-609_yao.pkl"]

    sort_yao_files(my_files, yao_files)
    # compare_predictions(my_files, yao_files)

    all_results_path = directory + "all_results.pkl"
    append_all_results_to_one_file(my_files, yao_files, all_results_path)

    print_model_performance(
        paths_to_pkl_files = [all_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "my_SLP",
        actual_result_key = "yao_SLP",
        additional_score_function_args = {"average": None, "zero_division": np.nan},
        number_of_decimals = 3
    )

    plot_distribution_of_score(
        paths_to_pkl_files = [all_results_path],
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "my_SLP",
        actual_result_key = "yao_SLP",
        score_function = metrics.accuracy_score, # metrics.cohen_kappa_score
        combine_file_predictions = False,
        title = "Distribution of Accuracy",
        xlabel = "Accuracy",
        label = ["Training Data"],
        binrange = (0, 1),
        binwidth = 0.05,
        xlim = (0.6, 1.01),
    )

    plot_confusion_matrix(
        path_to_pkl_file = all_results_path,
        path_to_project_configuration = model_directory_path + project_configuration_file,
        prediction_result_key = "my_SLP",
        actual_result_key = "yao_SLP",
        title = "Confusion Matrix of Neural Network",
        xlabel = "my stage",
        ylabel = "yao stage",
        normalize = None, # 'true', 'pred', 'all'
        values_format = None, # 'd', 'f', '.1%'
    )