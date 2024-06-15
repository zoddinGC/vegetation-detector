from os import listdir, makedirs

def check_folder_existence(folder_path:str):
    """
    Check if there's a folder in the given folder path. If not, try to create.

    :param folder_path: Relative path to check the folder
    """
    try:
        try:
            # Check if folder exists
            listdir(folder_path)
        except:
            # Create of not
            makedirs(folder_path)
    except ValueError as e:
        print(f'Not possible to create folder in {folder_path}. Error: {e}')