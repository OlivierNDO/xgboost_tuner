### Import Packages
######################################################################################################
import os


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
        


