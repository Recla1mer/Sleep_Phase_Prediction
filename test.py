import numpy as np

arr = np.empty((0, 3), int)

a = [1, 2, 3]
b = [5,5,5]

print(np.append(arr, [a], axis=0))
print(np.append(a, b))
#print(np.vstack([b, a]))


# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor

# class CustomArrayDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         labels = self.labels[idx]

#         if self.transform:
#             sample = self.transform(sample)

#         return sample, labels

# # Example usage
# if __name__ == "__main__":
#     # Sample data: 3D array (e.g., 5 samples, each with a 2D array of shape (2, 3))
#     data = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
#                          [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
#                          [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
#                          [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
#                          [[25.0, 26.0, 27.0], [28.0, 29.0, 30.0]]])

#     # Corresponding labels for each sample (e.g., 5 samples, each with 2 labels)
#     labels = torch.tensor([[0, 1],
#                            [1, 0],
#                            [0, 1],
#                            [1, 0],
#                            [0, 1]])

#     # Create dataset
#     dataset = CustomArrayDataset(data, labels, transform=ToTensor())

#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#     # Iterate over DataLoader
#     for batch, (X,y) in enumerate(dataloader):
#         print(batch)
#         print(len(X), len(X[0]), len(X[0][0]), len(X[0][0][0]))
#         print(len(y), len(y[0]))



# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor
# import numpy as np

# class CustomArrayDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         labels = self.labels[idx]

#         if self.transform:
#             sample = self.transform(sample)

#         return sample, labels

# # Example usage
# if __name__ == "__main__":
#     # Sample data: 4D array (e.g., 2 nights, each with 5 windows, each window with a 2D array of shape (2, 3))
#     data = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
#                       [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
#                       [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
#                       [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
#                       [[25.0, 26.0, 27.0], [28.0, 29.0, 30.0]]],
#                      [[[31.0, 32.0, 33.0], [34.0, 35.0, 36.0]],
#                       [[37.0, 38.0, 39.0], [40.0, 41.0, 42.0]],
#                       [[43.0, 44.0, 45.0], [46.0, 47.0, 48.0]],
#                       [[49.0, 50.0, 51.0], [52.0, 53.0, 54.0]],
#                       [[55.0, 56.0, 57.0], [58.0, 59.0, 60.0]]]])

#     # Corresponding labels for each window (e.g., 2 nights, each with 5 windows, each window with 1 label)
#     labels = torch.tensor([[0, 1, 0, 1, 0],
#                            [1, 0, 1, 0, 1]])

#     # Create dataset
#     dataset = CustomArrayDataset(data, labels, transform=ToTensor())

#     # Create DataLoader
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     # Iterate over DataLoader
#     for batch, (X, y) in enumerate(dataloader):
#         print(f"Batch {batch}:")
#         print("Data:", X)
#         print("Labels:", y)



"""

Simplest case

"""


"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np

class CustomArrayDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, labels

# Example usage
if __name__ == "__main__":
    # Sample data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (6,))
    data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                      [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                      [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
                     [[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                      [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                      [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                      [49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                      [55.0, 56.0, 57.0, 58.0, 59.0, 60.0]]])

    # Corresponding labels for each window (e.g., 2 nights, each with 5 windows, each window with 1 label)
    labels = torch.tensor([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1]])

    # Create dataset
    dataset = CustomArrayDataset(data, labels, transform=ToTensor())

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate over DataLoader
    for batch, (X, y) in enumerate(dataloader):
        # print shape of data and labels:
        print(f"Batch {batch}:")
        print("Data:", X)
        print("Labels:", y)
        print("Data shape:", X.shape)
        print("Labels shape:", y.shape)

"""


"""

added MAD data

"""

"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np

class CustomArrayDataset(Dataset):
    def __init__(self, ecg_data, mad_data, labels, transform=None):
        self.ecg_data = ecg_data
        self.mad_data = mad_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_sample = self.ecg_data[idx]
        mad_sample = self.mad_data[idx]
        labels = self.labels[idx]

        if self.transform:
            ecg_sample = self.transform(ecg_sample)
            mad_sample = self.transform(mad_sample)

        return ecg_sample, mad_sample, labels

# Example usage
if __name__ == "__main__":
    # Sample ECG data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (6,))
    ecg_data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                          [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                          [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
                         [[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                          [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                          [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                          [49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                          [55.0, 56.0, 57.0, 58.0, 59.0, 60.0]]])

    # Sample MAD data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (3,))
    mad_data = np.array([[[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9],
                          [1.0, 1.1, 1.2],
                          [1.3, 1.4, 1.5]],
                         [[1.6, 1.7, 1.8],
                          [1.9, 2.0, 2.1],
                          [2.2, 2.3, 2.4],
                          [2.5, 2.6, 2.7],
                          [2.8, 2.9, 3.0]]])

    # Corresponding labels for each window (e.g., 2 nights, each with 5 windows, each window with 1 label)
    labels = torch.tensor([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1]])

    # Create dataset
    dataset = CustomArrayDataset(ecg_data, mad_data, labels, transform=ToTensor())

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate over DataLoader
    for batch, (ecg, mad, y) in enumerate(dataloader):
        # print shape of data and labels:
        print(f"Batch {batch}:")
        print("ECG Data:", ecg)
        print("MAD Data:", mad)
        print("Labels:", y)
        print("ECG Data shape:", ecg.shape)
        print("MAD Data shape:", mad.shape)
        print("Labels shape:", y.shape)
"""

"""

added model

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np

class CustomArrayDataset(Dataset):
    def __init__(self, ecg_data, mad_data, labels, transform=None):
        self.ecg_data = ecg_data
        self.mad_data = mad_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ecg_data)

    def __getitem__(self, idx):
        ecg_sample = self.ecg_data[idx]
        mad_sample = self.mad_data[idx]
        labels = self.labels[idx]

        if self.transform:
            ecg_sample = self.transform(ecg_sample)
            mad_sample = self.transform(mad_sample)

        return ecg_sample, mad_sample, labels

class SleepStageModel(nn.Module):
    def __init__(self):
        super(SleepStageModel, self).__init__()
        # ECG branch
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # MAD branch
        self.mad_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 + 32 * 6, 128),  # Adjust input size based on concatenated features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Assuming 5 sleep stages
        )

    def forward(self, ecg, mad):
        # Process ECG data
        ecg = ecg.unsqueeze(1)  # Add channel dimension
        ecg_features = self.ecg_branch(ecg)
        ecg_features = ecg_features.view(ecg_features.size(0), -1)  # Flatten

        # Process MAD data
        mad = mad.unsqueeze(1)  # Add channel dimension
        mad_features = self.mad_branch(mad)
        mad_features = mad_features.view(mad_features.size(0), -1)  # Flatten

        # Concatenate features
        combined_features = torch.cat((ecg_features, mad_features), dim=1)

        # Fully connected layers
        output = self.fc(combined_features)
        return output


class SleepStageModel2(nn.Module):
    def __init__(self):
        super(SleepStageModel2, self).__init__()
        # ECG branch
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # MAD branch
        self.mad_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 + 32 * 6, 128),  # Adjust input size based on concatenated features
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1),  # New Conv1d layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Assuming 5 sleep stages
        )

    def forward(self, ecg, mad):
        # ecg_out = self.ecg_branch(ecg_sample)
        # mad_out = self.mad_branch(mad_sample)
        # combined_features = torch.cat((ecg_out, mad_out), dim=2)
        # combined_features = combined_features.view(combined_features.size(0), 1, -1)  # Reshape for Conv1d
        # output = self.fc(combined_features)
        # output = output.view(output.size(0), -1)  # Flatten before final linear layer
        # return output

        #maybe rather like this?
        # Process ECG data
        ecg_features = self.ecg_branch(ecg)
        ecg_features = ecg_features.view(ecg_features.size(0), -1)  # Flatten

        # Process MAD data
        mad_features = self.mad_branch(mad)
        mad_features = mad_features.view(mad_features.size(0), -1)  # Flatten

        # Concatenate features
        combined_features = torch.cat((ecg_features, mad_features), dim=1)

        # Reshape for Conv1d layer
        combined_features = combined_features.unsqueeze(1)  # Add channel dimension

        # Fully connected layers
        output = self.fc(combined_features)
        output = output.view(output.size(0), -1)  # Flatten before final linear layer
        return output

# Example usage
if __name__ == "__main__":
    # Sample ECG and MAD data
    ecg_sample = torch.randn(10, 5, 6)  # Batch size of 10, 1 channel, 6 samples
    mad_sample = torch.randn(10, 5, 3)  # Batch size of 10, 1 channel, 3 samples

    # Initialize the model
    model = SleepStageModel()

    # Forward pass
    output = model(ecg_sample, mad_sample)
    print(output)


class SleepStageModel3(nn.Module):
    def __init__(self):
        super(SleepStageModel3, self).__init__()
        # ECG branch
        self.ecg_branch = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # MAD branch
        self.mad_branch = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        # Fully connected layers after concatenation
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 + 32 * 5, 128),  # Adjust input size based on concatenated features
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1),  # New Conv1d layer
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Assuming 5 sleep stages
        )

    def forward(self, ecg, mad):
        batch_size, _, num_windows, _ = ecg.size()

        # Process ECG data
        ecg = ecg.view(batch_size * num_windows, 6)  # Combine batch and windows dimensions
        ecg_features = self.ecg_branch(ecg)
        ecg_features = ecg_features.view(batch_size, num_windows, -1)  # Separate batch and windows dimensions

        # Process MAD data
        mad = mad.view(batch_size * num_windows, 3)  # Combine batch and windows dimensions
        mad_features = self.mad_branch(mad)
        mad_features = mad_features.view(batch_size, num_windows, -1)  # Separate batch and windows dimensions

        # Concatenate features
        combined_features = torch.cat((ecg_features, mad_features), dim=2)

        # Reshape for Conv1d layer
        combined_features = combined_features.view(batch_size, 1, -1)  # Combine windows and features dimensions

        # Fully connected layers
        output = self.fc(combined_features)
        output = output.view(output.size(0), -1)  # Flatten before final linear layer
        return output

# Example usage
if __name__ == "__main__":
    # Sample ECG and MAD data
    ecg_sample = torch.randn(10, 1, 5, 6)  # Batch size of 10, 1 channel, 5 windows, 6 samples per window
    mad_sample = torch.randn(10, 1, 5, 3)  # Batch size of 10, 1 channel, 5 windows, 3 samples per window

    # Initialize the model
    model = SleepStageModel3()

    # Forward pass
    output = model(ecg_sample, mad_sample)
    print(output)

# code to stop here
exit()


# Example usage
if __name__ == "__main__":
    # Sample ECG data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (6,))
    ecg_data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                          [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                          [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
                         [[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                          [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                          [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                          [49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                          [55.0, 56.0, 57.0, 58.0, 59.0, 60.0]],
                          [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                          [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                          [25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
                         [[31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
                          [37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
                          [43.0, 44.0, 45.0, 46.0, 47.0, 48.0],
                          [49.0, 50.0, 51.0, 52.0, 53.0, 54.0],
                          [55.0, 56.0, 57.0, 58.0, 59.0, 60.0]]])

    # Sample MAD data: 3D array (e.g., 2 nights, each with 5 windows, each window with a 1D array of shape (3,))
    mad_data = np.array([[[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9],
                          [1.0, 1.1, 1.2],
                          [1.3, 1.4, 1.5]],
                         [[1.6, 1.7, 1.8],
                          [1.9, 2.0, 2.1],
                          [2.2, 2.3, 2.4],
                          [2.5, 2.6, 2.7],
                          [2.8, 2.9, 3.0]],
                          [[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6],
                          [0.7, 0.8, 0.9],
                          [1.0, 1.1, 1.2],
                          [1.3, 1.4, 1.5]],
                         [[1.6, 1.7, 1.8],
                          [1.9, 2.0, 2.1],
                          [2.2, 2.3, 2.4],
                          [2.5, 2.6, 2.7],
                          [2.8, 2.9, 3.0]]])

    # Corresponding labels for each window (e.g., 2 nights, each with 5 windows, each window with 1 label)
    labels = torch.tensor([[0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1],
                           [0, 1, 0, 1, 0],
                           [1, 0, 1, 0, 1]])

    # Create dataset
    dataset = CustomArrayDataset(ecg_data, mad_data, labels, transform=ToTensor())

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SleepStageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for batch, (ecg, mad, y) in enumerate(dataloader):
        # print shape of data and labels:
        print(f"Batch {batch}:")
        print("ECG Data:", ecg)
        print("MAD Data:", mad)
        print("Labels:", y)
        print("ECG Data shape:", ecg.shape)
        print("MAD Data shape:", mad.shape)
        print("Labels shape:", y.shape)


    # Training loop
    """
    num_epochs = 10
    for epoch in range(num_epochs):
        for batch, (ecg, mad, y) in enumerate(dataloader):
            # Flatten the labels to match the output shape
            y = y.view(-1)

            # Forward pass
            outputs = model(ecg.float(), mad.float())
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch+1}], Loss: {loss.item()}")
    """