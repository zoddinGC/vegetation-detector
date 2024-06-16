import os
from shutil import rmtree

def check_folder_existence(folder_path:str):
    """
    Check if there's a folder in the given folder path. If not, try to create.

    :param folder_path: Relative path to check the folder
    """
    try:
        try:
            # Check if folder exists
            os.listdir(folder_path)
        except:
            # Create of not
            os.makedirs(folder_path)
    except ValueError as e:
        print(f'Not possible to create folder in {folder_path}. Error: {e}')

def clean_folder(folders:str or list, main_folder:bool=False):
    """
    Delete all files and folders in the specified directory.

    :param folder: Path to the folder to be cleaned.
    """
    if isinstance(folders, str):
        folders = [folders]

    # For each file, try to delete
    for folder in folders:
        # Check if folder exists
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist.")
            return
        
        for filename in os.listdir(folder):
            if filename == '__init__.py':
                continue
            
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        if main_folder:
            rmtree(folder)
