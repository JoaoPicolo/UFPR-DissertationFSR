import os
import shutil

def create_new_dir(dir_path: str):
    """
    Removes dir if exists to create a new one
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)