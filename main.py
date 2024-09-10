"""
Author: Johannes Peter Knoll

"""

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *

"""
------------------------
Looping over the dataset
------------------------
"""

# TRAINING LOOP
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step() # updates the model parameters based on the gradients computed during the backward pass
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# TESTING LOOP
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


"""
--------------------------------
Training and Testing the Model
--------------------------------
"""

if __name__ == "__main__":

    # SET DEVICE
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # HYPERPARAMETERS
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    input_size = 28*28
    num_classes = 10

    # INITIALIZE NETWORK
    model = SleepStageModel(input_size, num_classes).to(device)

    # Initialize the loss and optimizer functions
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #edf_dataset = EDF_Dataset()
    #edf_dataloader = DataLoader(edf_dataset, batch_size=64)