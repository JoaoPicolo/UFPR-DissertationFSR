import os
import shutil

from PIL import Image

from .common import create_new_dir
from .preprocessing import crop_and_align


def scface_to_lr_hr_unstructured(dataset_path: str):
    """
    Adjusts the SCFace dataset with the images
    captured by the low resolution cameras.
    
    The following tree is implemented:
    - dir/
    ---- LR/
    ---- HR/
    """
    # Creates dir to save manipulated images
    base_dir = "./scface-lr-hr-real/"
    create_new_dir(base_dir)

    # Creates HR dir and copy original folder content
    hr_dir = f"{base_dir}/HR/"
    os.makedirs(hr_dir)
    images = os.listdir(f"{dataset_path}/mugshot_frontal_cropped_all/")
    for file_name in images:
        crop_and_align(f"{dataset_path}/mugshot_frontal_cropped_all/{file_name}", hr_dir)

    # Creates LR dir and copy original data from folders
    lr_dir = f"{base_dir}/LR/"
    os.makedirs(lr_dir)
    # Will copy one by one, since won't use data from cams 6, 7 and 8
    not_used_cams = ["cam6", "cam7", "cam8"]
    files = os.listdir(f"{dataset_path}/surveillance_cameras_all")
    for file_name in files:
        if not any(cam in file_name for cam in not_used_cams):
            crop_and_align(f"{dataset_path}/surveillance_cameras_all/{file_name}", lr_dir)

    # Zips created folder
    shutil.make_archive(base_dir, "zip", base_dir)


def scface_to_structured(dataset_path: str):
    """
    Adjusts the SCFace dataset with the images
    captured by the low resolution cameras.
    
    The following tree is implemented:
    - dir/
    ---- LR/
    -------- Subject1/
    ---- HR/
    -------- Subject1/

    "Original" is the HR/* and LR/* tree implemented above
    """
    # Creates dir to save manipulated images
    base_dir = f"{dataset_path}-structured/"
    create_new_dir(base_dir)

    options = ["HR", "LR"]
    for option in options:
        new_base_path = f"{base_dir}/{option}"
        os.makedirs(new_base_path)

        original_base_path = f"{dataset_path}/{option}"
        original_images = os.listdir(original_base_path)
        for image_name in original_images:
            subject = image_name.split('_')[0]
            subject_path = f"{new_base_path}/{subject}"

            if not os.path.isdir(subject_path):
                os.makedirs(subject_path)

            image = Image.open(f"{original_base_path}/{image_name}")
            image.save(f"{subject_path}/{image_name}")

    # Zips created folder
    shutil.make_archive(base_dir, "zip", base_dir)
