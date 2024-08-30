import pickle
import os

def save_to_pickle(data, file_name):
    """
    Save data to a pickle file, overwriting the file if it already exists.

    ARGUMENTS:
    --------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    None
    """
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def append_to_pickle(data, file_name):
    """
    Append data to a pickle file, without deleting previous data.

    ARGUMENTS:
    --------------------------------
    data: any
        data to be saved
    file_name: str
        path to the pickle file
    
    RETURNS:
    --------------------------------
    None
    """
    with open(file_name, "ab") as f:
        pickle.dump(data, f)


def load_from_pickle(file_name: str):
    """
    Load data from a pickle file as a generator.

    ARGUMENTS:
    --------------------------------
    file_name: str
        path to the pickle file
    key: str
        key of the data to be loaded
    
    RETURNS:
    --------------------------------
    any
        data from the pickle file
    """
    # with open(file_name, "rb") as f:
    #     data = pickle.load(f)
    # return data
    with open(file_name, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def find_non_existing_path(path_without_file_type: str, file_type: str = "pkl"):
    """
    Find a path that does not exist yet by adding a number to the end of the path.

    ARGUMENTS:
    --------------------------------
    path_without_file_type: str
        path without the file type
    file_type: str
        file type of the file to be saved
    
    RETURNS:
    --------------------------------
    str
        path that does not exist yet
    """
    if not os.path.exists(f"{path_without_file_type}.{file_type}"):
        return f"{path_without_file_type}.{file_type}"
    i = 0
    while os.path.exists(f"{path_without_file_type}_{i}.{file_type}"):
        i += 1
    return f"{path_without_file_type}_{i}.{file_type}"


class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path

        self.id_key = "ID"
        self.data_point_keys = ["ID", "RRI", "MAD"]

        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pass
    

    def save_data(self, data, overwrite_id=True):
        if not isinstance(data, dict):
            raise ValueError("Data point must be a dictionary")
        if self.id_key not in data:
            raise ValueError("Data point must have an ID key")
        for key in data:
            if key not in self.data_point_keys:
                raise ValueError(f"Unknown key in data point: {key}")
        
        working_file_path = "save_in_progress.pkl"

        file_generator = load_from_pickle(self.file_path)

        for data_point in file_generator:
            if data_point[self.id_key] == data[self.id_key]:
                if overwrite_id:
                    raise ValueError("ID already exists in the file")
                else:
                    raise ValueError("ID already exists in the file")

    def get_data_points(self):
        return self.data_points

    def save_to_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.data_points, f)

    def load_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

# Example usage:
# manager = DataManager()
# manager.add_data_point({'key1': [1, 2, 3], 'key2': 42})
# manager.save_to_file('data.pkl')
# for data_point in manager.load_from_file('data.pkl'):
#     print(data_point)