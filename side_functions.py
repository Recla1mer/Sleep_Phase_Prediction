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
    

    def save_data(self, new_data, overwrite_id = True):
        # Check if new_data is a dictionary
        if not isinstance(new_data, dict):
            raise ValueError("Data point must be a dictionary")
        # Check if new_data has the ID key
        if self.id_key not in new_data:
            raise ValueError("Data point must have an ID key")
        # Check if ID key is nothing misleading
        if new_data[self.id_key] in self.data_point_keys:
            raise ValueError("ID must not be the same as a key in the data point")
        # Check if key in new_data is unknown
        for new_data_key in new_data:
            if new_data_key not in self.data_point_keys:
                raise ValueError(f"Unknown key in data point: {new_data_key}")
        
        # Create temporary file to save data in progress
        working_file_path = find_non_existing_path(path_without_file_type = "save_in_progress", file_type = "pkl")

        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        overwrite_denied = False
        not_appended = True

        # Check if ID already exists in the data file, then overwrite keys if allowed
        for data_point in file_generator:
            if data_point[self.id_key] == new_data[self.id_key]:
                not_appended = False
                if overwrite_id:
                    new_data_point = dict()
                    for possible_key in self.data_point_keys:
                        if possible_key in new_data:
                            new_data_point[possible_key] = new_data[possible_key]
                        elif possible_key in data_point:
                            new_data_point[possible_key] = data_point[possible_key]
                else:
                    new_data_point = data_point
                    overwrite_denied = True
            else:
                new_data_point = data_point
            
            # Append data point to the working file
            append_to_pickle(data = new_data_point, file_name = working_file_path)
        
        if not_appended:
            append_to_pickle(data = new_data, file_name = working_file_path)
        
        # Remove the old file and rename the working file
        try:
            os.remove(self.file_path)
        except:
            pass

        os.rename(working_file_path, self.file_path)

        if overwrite_denied:
            raise ValueError("ID already existed in the data file and Overwrite was denied. Data was not saved.")


    def load_data(self, id):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        id_found = False
        for data_point in file_generator:
            if data_point[self.id_key] == id:
                id_found = True
                return data_point
        
        del file_generator
        
        if not id_found:
            raise ValueError(f"ID {id} not found in the data file")
    

    def __len__(self):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        count = 0
        for _ in file_generator:
            count += 1
        
        del file_generator

        return count
    

    def __contains__(self, id):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        id_found = False
        for data_point in file_generator:
            if data_point[self.id_key] == id:
                id_found = True
                break
        
        del file_generator

        return id_found
    

    def __iter__(self):
        # Load data generator from the file
        file_generator = load_from_pickle(self.file_path)

        for data_point in file_generator:
            yield data_point
        
        del file_generator
    

    def __getitem__(self, key):        

        if key in self.data_point_keys:
            # Load data generator from the file
            file_generator = load_from_pickle(self.file_path)

            values_for_key_from_all_data_points = list()
            count_data_points_missing_key = 0

            for data_point in file_generator:
                if key in data_point:
                    values_for_key_from_all_data_points.append(data_point[key])
                else:
                    count_data_points_missing_key += 1
            
            del file_generator

            if count_data_points_missing_key > 0:
                print(f"Attention: {count_data_points_missing_key} data points are missing the key {key}")
            
            return values_for_key_from_all_data_points


shhs_data_manager = DataManager(file_path = "messing_around.pkl")

# shhs_data_manager.save_data(new_data = {"ID": 1, "RRI": [1, 2, 3], "MAD": [5, 6, 7]}, overwrite_id = True)
# shhs_data_manager.save_data(new_data = {"ID": 2, "RRI": [2, 3, 4], "MAD": [6, 7, 8]}, overwrite_id = True)
# shhs_data_manager.save_data(new_data = {"ID": 3, "RRI": [3, 4, 5], "MAD": [7, 8, 9]}, overwrite_id = True)

print(len(shhs_data_manager))

for data_point in shhs_data_manager:
    print(data_point)

print(shhs_data_manager.load_data(id = 2))

print(shhs_data_manager["RRI"])

print(shhs_data_manager['1'])

# let getitem retrieve also ids
# change so that sampling frequency must always be the same
# maybe create first dictionary that holds information like sampling frequency, etc.
# which is not transformable by the user