### Import Packages
######################################################################################################
import os
import pandas as pd
import sklearn

### Define Functions
######################################################################################################

def create_folder_if_not_exist(folder_path : str, print_creation = True):
    """
    Create a folder in the operating system if it does not already exist
    Args:
        folder_path (str): folder path in file system (e.g. 'C:/Users/name/my_folder/')
        print_creation (bool): if True, print newly created folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        if print_creation:
            print(f"Creating folder '{folder_path}'")
        


def load_dataset_from_config(config_dict : dict, test_size = 0.2, random_seed = 912):
    """
    Load .csv dataset based on configured dictionary in dataset_configuration.py,
    splitting into train, test, and validation dataframes
    Args:
        config_dict (dict): configuration dictionary with keys:
            > 'y_col', 'id_col', 'categ_x_cols', 'contin_x_cols',
              'x_cols', 'interaction_cols', 'polynomial_col_dict',
              'folder_path', 'train_file_name', 'dataset_name', 'result_folder'
        test_size (float): portion of training set to allocate to test and validation combined
        random_seed (int): integer to use for random seed in dataset split
    Returns
        df, train, test, validation (pandas.DataFrame)
    """
    df = pd.read_csv(f"{config_dict.get('folder_path')}{config_dict.get('train_file_name')}")
    train, test = sklearn.model_selection.train_test_split(df, test_size = test_size, random_state = random_seed)
    test, valid = sklearn.model_selection.train_test_split(test, test_size = 0.5, random_state = random_seed)
    return df, train, test, valid
