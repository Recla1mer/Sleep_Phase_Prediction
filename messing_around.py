import h5py
import os

import numpy as np
import torch
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn

if os.path.exists('../Coding_Tutorials/Training_Data/SHHS_dataset.h5'):
    print("File exists")

# open file
# shhs_dataset = h5py.File("Training_Data/SHHS_dataset.h5", 'r')

# print(list(shhs_dataset.keys()))
# #print(list(shhs_dataset['rri'].keys()))
# print(len(shhs_dataset['rri']['205764_1'][:]))
# print(len(shhs_dataset['slp']['205764_1'][:]))
# # where the RRI are saved
# shhs_dataset['rri']
# # where the sleep stages are saved
# shhs_dataset['slp']

# print(shhs_dataset['slp'].attrs['freq'])


def reshape_signal(signal, frequency, combine=False, number_windows=300, window_size=120):
    """
    Reshape signal to windows
    """
    samples_in_window = int(window_size * frequency)

    if combine:
        windows = np.empty((0), int)
    else:
        windows = np.empty((0, samples_in_window), float)

    for i in range(0, number_windows):
        if (i + 1) * samples_in_window <= len(signal):
            upper_border = (i + 1) * samples_in_window
        else:
            upper_border = len(signal)
        
        if i*samples_in_window < len(signal):
            lower_border = i * samples_in_window
            this_window = signal[lower_border:upper_border]

            if len(this_window) < samples_in_window:
                this_window = np.append(this_window, [0 for i in range(samples_in_window - len(this_window))])
        else:
            this_window = [0 for i in range(samples_in_window)]
        
        if combine:
            this_mean = round(np.mean(this_window))
            if this_mean == 4:
                this_mean = 3
            if this_mean == 5:
                this_mean = 4
            windows = np.append(windows, this_mean)
        else:
            windows = np.append(windows, [this_window], axis=0) #type: ignore
    
    return windows


# print(reshape_signal(shhs_dataset['slp']['205764_1'][:], shhs_dataset['slp'].attrs['freq']))

target_transform = Lambda(lambda y: torch.zeros(
    5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


class CustomSleepDataset(Dataset):
    def __init__(self, h5file, feature_key, target_key, transform=ToTensor(), target_transform=target_transform):
        self.h5file = h5file
        self.feature_key = feature_key
        self.target_key = target_key
        self.files = list(h5file[feature_key].keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patient_id = self.files[idx]
        feature = reshape_signal(self.h5file[self.feature_key][patient_id][:], self.h5file[self.feature_key].attrs['freq'])
        label = reshape_signal(self.h5file[self.target_key][patient_id][:], self.h5file[self.target_key].attrs['freq'], combine=True)

        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label


# shhs_dataset = h5py.File("Training_Data/SHHS_dataset.h5", 'r')
# training_data = CustomSleepDataset(shhs_dataset, 'rri', 'slp')

# print(training_data[0])

# train_dataloader = DataLoader(training_data, batch_size=3)

# # for batch, (X, y) in enumerate(train_dataloader):
# #     # Compute prediction and loss
# #     print(batch)
# #     print(len(X), len(X[0]), len(X[0][0]))
# #     print(len(y), len(y[0]), len(y[0][0]))
# test_a_tensor = torch.rand([4,1,2,3])
# print(test_a_tensor.shape)
# print(test_a_tensor)

# a = np.array([[[[1,1,1],[2,2,2]]],[[[1,1,1],[2,2,2]]],[[[1,1,1],[2,2,2]]],[[[1,1,1],[2,2,2]]]])
# a_tensor = torch.tensor(a)
# print(a_tensor.shape)
# print(a_tensor)

# trans_a = a_tensor.view(4*2, 1, 3)
# print(trans_a.shape)
# print(trans_a)

# test_arr = torch.rand([4,1,20,128])
# #ecg = ecg.view(batch_size * num_windows, 1, samples_in_window)
# test_arr = test_arr.view(4*20, 1, 128)
# print(test_arr.shape)
# conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
# out = conv1(test_arr)
# print(out.shape)

"""
with torch.no_grad():
    rri = torch.rand([2, 5, 20])
    print(rri.shape)
    rri = rri.reshape(-1, 1, 20)
    print(rri.shape)
    conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
    out = conv(rri)
    print(out.shape)

    out = torch.cat([out, out], dim=-1)
    print("cat", out.shape)

    flatten = nn.Flatten()
    out = flatten(out)
    print("flatten", out.shape)

    linear = nn.Linear(120, 30)
    out = linear(out)
    print("linear", out.shape)

    out = out.reshape(2, 5, -1)
    print("reshape", out.shape)

    out = out.transpose(1, 2).contiguous()
    print("transpose", out.shape)


# out = out.reshape(2, 1200, -1)
# out = out.transpose(1, 2).contiguous()
# out = self.classfier(out)
# out = out.transpose(1, 2).contiguous().reshape(2 * 1200, -1)

# test_arr = torch.rand([4,1,20,128])
# print(test_arr.shape)
# test_arr = test_arr.reshape(-1,1,128)
# print(test_arr.shape)

print("view")

rand_tens = torch.rand([4, 1200, 128])
print(rand_tens.shape)
rand_tens = rand_tens.view(4*1200, 1, 128)
print(rand_tens.shape)

print("")

rand_tens = torch.rand([4, 1, 1200, 128])
print(rand_tens.shape)
rand_tens = rand_tens.view(4*1200, 1, 128)
print(rand_tens.shape)

print("reshape")

rand_tens = torch.rand([4, 1200, 128])
print(rand_tens.shape)
rand_tens = rand_tens.reshape(-1, 1, 128)
print(rand_tens.shape)

print("")

rand_tens = torch.rand([4, 1, 1200, 128])
print(rand_tens.shape)
rand_tens = rand_tens.reshape(-1, 1, 128)
print(rand_tens.shape)

print("bla")

rand_tens = torch.rand([4800, 6, 128])
print(rand_tens.shape)
rand_tens = rand_tens.view(4, 1200, -1)
print(rand_tens.shape)

print("bla")
rand_tens = torch.rand([1200, 2, 512])
print(rand_tens.shape)
batch_norm = nn.BatchNorm1d(num_features=2)
pool = nn.MaxPool1d(kernel_size=3)
dropout = nn.Dropout(0.2)
#rand_tens = batch_norm(rand_tens)
#rand_tens = pool(rand_tens)
rand_tens = dropout(rand_tens)
print(rand_tens.shape)

a = []
a.append(nn.Dropout(0.2))
a.append(nn.Dropout(0.4))

# unpack a twice in nn.Sequential:
b = nn.Sequential(nn.Dropout(0.1))
b = nn.Sequential(nn.Dropout(0.1), *a, *a)
print(b)


print(torch.zeros(2,1,3,4))
"""
batchsize = 1
Nwindows = 1200

out_2 = torch.rand([1200, 128])
out = torch.rand([1200,128])
print("out", out.shape, out_2.shape)

out = out.reshape(batchsize, Nwindows, -1)
out_2 = out_2.view(batchsize, Nwindows, -1)
print("reshape", out.shape, out_2.shape)

out = out.transpose(1, 2).contiguous()
print("transpose", out.shape)

out = torch.rand([1, 4, 1200])
print("classifier", out.shape)

out = out.transpose(1, 2).contiguous().reshape(batchsize * Nwindows, -1)
print("reshape", out.shape)

def padding_same(kernel_size=3, dilation=1, stride=1):
    # stride * Lout = Lin + 2padding - dilation (kernel_size - 1) - 1 + stride
    # (stride - 1) * Lout + dilation (kernel_size - 1) + 1 - stride = 2 padding
    padding = dilation * (kernel_size - 1) + 1 - stride
    return padding // 2

conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, dilation=5, padding='same')
conv2 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, dilation=5, padding=padding_same(5, 5, 1))
print(padding_same(5, 1, 1))

test_arr = torch.rand([1200,1,512])
print(test_arr.shape)

out = conv1(test_arr)

test_arr = torch.rand([1200,1,512])
out_2 = conv2(test_arr)

print(out.shape, out_2.shape)

test_arr = torch.rand([4,128,1200])
print(test_arr.shape)

conv = nn.Conv1d(
                in_channels = 128, 
                out_channels = 4, 
                kernel_size = 9,
                padding='same'
                )

out = conv(test_arr)
print(out.shape)