import os
from typing import List

from PIL import Image


def resize_structured_images(structured_path: str, new_lr_sizes: List[str] = ["LR-32", "LR-64"]):
    structured_path = "./scface-lr-hr-real-structured/"

    for dir in new_lr_sizes:
        os.makedirs(f"{structured_path}/{dir}/")

    subjects_path = f"{structured_path}/LR/"
    subjects = os.listdir(subjects_path)
    for subject in subjects:
        for dir in new_lr_sizes:
            subject_dir = f"{structured_path}/{dir}/{subject}"
            if not os.path.isdir(subject_dir):
                os.makedirs(subject_dir)

        images = os.listdir(f"{subjects_path}/{subject}")
        for image_name in images:
            image = Image.open(f"{subjects_path}/{subject}/{image_name}")
            for dir in new_lr_sizes:
                subject_dir = f"{structured_path}/{dir}/{subject}"
                new_size = int(dir.split('-')[1])
                new_image = image.resize((new_size,new_size), Image.BICUBIC)
                new_image.save(f"{subject_dir}/{image_name}")
                print(f"{subjects_path}/{subject}/{image_name} -> {subject_dir}/{image_name}")


def resize_unstructured_images(unstructured_path: str, new_lr_sizes: List[str] = ["LR-32", "LR-64"]):
    unstructured_path = "./quiscampi-lr-hr-real"

    for dir in new_lr_sizes:
        os.makedirs(f"{unstructured_path}/{dir}/")

    images_path = f"{unstructured_path}/LR/"
    images = os.listdir(f"{images_path}")
    for image_name in images:
        image = Image.open(f"{images_path}/{image_name}")
        for dir in new_lr_sizes:
            subject_dir = f"{unstructured_path}/{dir}"
            new_size = int(dir.split('-')[1])
            new_image = image.resize((new_size,new_size), Image.BICUBIC)
            new_image.save(f"{subject_dir}/{image_name}")
            print(f"{images_path}/{image_name}/{image_name} -> {subject_dir}/{image_name}")