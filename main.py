"""
Author: Johannes Peter Knoll

"""

# LOCAL IMPORTS
from dataset_processing import *
from neural_network_model import *


if __name__ == "__main__":

    """
    -----------------------------------
    Preprocess Data (uncomment if done)
    -----------------------------------
    """
    processed_shhs_path = "Processed_Data/shhs_data.pkl"
    processed_gif_path = "Processed_Data/gif_data.pkl"

    Process_SHHS_Dataset(path_to_shhs_dataset = "Raw_Data/SHHS_dataset.h5", path_to_save_processed_data = processed_shhs_path)
    Process_GIF_Dataset(path_to_gif_dataset = "Raw_Data/GIF_dataset.h5", path_to_save_processed_data = processed_gif_path)

    """
    -----------------------------------
    Train-, Validation- and Test-Split
    -----------------------------------
    """
    shhs_data_manager = SleepDataManager(processed_shhs_path)

    shhs_data_manager.separate_train_test_validation(
        train_size = 0.8, 
        validation_size = 0.1, 
        test_size = 0.1, 
        random_state = None, 
        shuffle = True
    )

    gif_data_manager = SleepDataManager(processed_gif_path)

    gif_data_manager.separate_train_test_validation(
        train_size = 0.8, 
        validation_size = 0.1, 
        test_size = 0.1, 
        random_state = None, 
        shuffle = True
    )

    """
    -----------------------------------
    Neural Network Training
    -----------------------------------
    """

    # SET DEVICE
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nUsing {device} device")

    # HYPERPARAMETERS
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    input_size = 28*28
    num_classes = 10

    # INITIALIZE NETWORK
    model = SleepStageModel(input_size, num_classes).to(device)

    # LOSS FUNCTION
    loss_fn = nn.CrossEntropyLoss()

    # OPTIMIZER
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(net.parameters(), lr=lr_scheduler(epoch))

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)