import os
import shutil

from PIL import Image

from .common import create_new_dir
from .preprocessing import crop_and_align


def quiscampi_to_lr_hr_unstructured(dataset_path: str):
    """
    Adjusts the Quiscampi dataset with the images
    captured by the outdoor camera.
    
    The following tree is implemented:
    - dir/
    ---- LR/
    ---- HR/
    """
    # Creates dir to save manipulated images
    base_dir = "./quiscampi-lr-hr-real/"
    create_new_dir(base_dir)

    # Creates HR dir and copy original folder content
    hr_dir = f"{base_dir}/HR/"
    os.makedirs(hr_dir)
    subjects = os.listdir(dataset_path)
    for subject in subjects:
        if subject.isdigit():
            subject_path = f"{dataset_path}/{subject}/"
            images = os.listdir(subject_path)
            for image_name in images:
                if "high-quality" in image_name:
                    crop_and_align(f"{subject_path}/{image_name}", hr_dir, f"subject_{subject}")

    # Creates LR dir and copy original data from folders
    lr_dir = f"{base_dir}/LR/"
    os.makedirs(lr_dir)
    subjects = os.listdir(dataset_path)
    for subject in subjects:
        if subject.isdigit():
            subject_path = f"{dataset_path}/{subject}/"
            images = os.listdir(subject_path)
            for image_name in images:
                if "high-quality" not in image_name:
                    crop_and_align(f"{subject_path}/{image_name}", lr_dir, f"subject_{subject}")

    # Zips created folder
    shutil.make_archive(base_dir, "zip", base_dir)


def quiscampi_to_structured(dataset_path: str):
    """
    Adjusts the Quiscampi dataset with the images
    captured by the outdoor camera.
    
    The following tree is implemented:
    - dir/
    ---- LR/
    -------- Subject1/
    ---- HR/
    -------- Subject1/

    "Original" is the HR/* and LR/* tree implemented above.
    """
    # Checks if dir to save data exists
    base_dir = f"{dataset_path}-structured/"
    create_new_dir(base_dir)

    options = ["HR", "LR"]
    for option in options:
        new_base_path = f"{base_dir}/{option}"
        os.makedirs(new_base_path)

        original_base_path = f"{dataset_path}/{option}"
        original_images = os.listdir(original_base_path)
        for image_name in original_images:
            subject = image_name.split('_')[1]
            subject_path = f"{new_base_path}/{subject}"

            if not os.path.isdir(subject_path):
                os.makedirs(subject_path)

            image = Image.open(f"{original_base_path}/{image_name}")
            image.save(f"{subject_path}/{image_name}")

    # Zips created folder
    shutil.make_archive(base_dir, "zip", base_dir)