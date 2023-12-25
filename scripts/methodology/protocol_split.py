import os
import sys
import shutil
import random
from typing import List

from PIL import Image

def read_paths_from_file(file_path: str):
    """
    Reads into an array the path to the reconstruction results
    produced by each network in an unstructed way.
    """
    paths = []
    with open(file_path, 'r') as file:
        for line in file:
            path = line.strip()  # Remove leading/trailing whitespace and newline characters
            if len(path) > 0:
                paths.append(path)
    return paths

def get_subject(name: str, index: int):
    """
    Gets the subject based on the image name and provided index.
    """
    return name.split('.')[0].split('_')[index]

def get_common_images(dirs: List[str]):
    """
    Gets the common images successfully reconstructed by
    all the networks to be compared.
    """
    set_list = [set(os.listdir(dir_path)) for dir_path in dirs]
    return list(set.intersection(*set_list))

def create_new_dir(dir_path: str):
    """
    Removes dir if exists to create a new one.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def create_reference(parent_dir: str, hr_dir: str, reference_images: List[str]):
    """
    Creates the reference directories containing the HR
    images to be used as gallery.
    """
    # Creates reference
    sizes = [256, 512]
    for size in sizes:
        reference_dir = f"{parent_dir}/reference-{size}/"
        os.mkdir(reference_dir)
        for image_name in reference_images:
            image = Image.open(f"{hr_dir}/{image_name}")
            new_image = image.resize((size,size))
            new_image.save(f"{reference_dir}/{image_name}")

def create_sr(parent_dir: str, sr_dirs: str, common_images: List[str]):
    """
    Creates the super-resolution directories containing the SR
    images to be used as probe.
    """
    # Creates sr
    for dir_name in sr_dirs:
        name = dir_name.replace('/', '-')[:-1]
        sr_dir = f"{parent_dir}/{name}"
        os.mkdir(sr_dir)
        for image in common_images:
            shutil.copy(f"{dir_name}/{image}", f"{sr_dir}/{image}")


def main(args: List[str]):
    path_to_hr = args[0] # Path to the HR images to form the gallery
    file_name = args[1]  # Path to the file containing the path to each network's reconstruction results
    db_name = args[2]    # Name of the database to be evaluated
    subject_index = int(args[3]) # Index that identifies the subject in the image name when split by '_'

    sr_dirs = read_paths_from_file(file_name)

    # Gets common reconstructed images for fair comparison
    common_images = get_common_images(sr_dirs)

    # Gets images names
    hr_images = os.listdir(path_to_hr)
    lr_images = common_images

    # Gets subject
    subjects = [get_subject(name, subject_index) for name in lr_images]
    lr_subjects = sorted([*set(subjects)])

    # Gets information to build open and closed sets
    open_set = False
    open_set_hr_images = []     # HR images when present, will form an open set
    closed_set_hr_images = []   # HR images when present, will form a closed set
    for name in hr_images:
        subject = get_subject(name, subject_index)
        open_set_hr_images.append(name)
        if subject not in lr_subjects:
            open_set = True
        else:
            closed_set_hr_images.append(name)

    print(f"Was originally open set: {open_set}")

    # Creates a processed dir
    base_dir_name = f"{db_name}-processed"
    create_new_dir(base_dir_name)

    if open_set:
        # Creates open set
        open_dir = f"{base_dir_name}/open-set"
        os.mkdir(open_dir)
        # Creates open set reference
        create_reference(open_dir, path_to_hr, hr_images)
        # Creates open set SR images
        create_sr(open_dir, sr_dirs, common_images)

        # Creates closed set
        closed_dir = f"{base_dir_name}/closed-set"
        os.mkdir(closed_dir)
        # Creates closed set reference
        create_reference(closed_dir, path_to_hr, closed_set_hr_images)
        # Creates closed set SR images
        create_sr(closed_dir, sr_dirs, common_images)
    else:
        # Calculates open set if it was not originally open set
        subjects_to_keep = lr_subjects[:]
        num_elements_to_remove = int(len(subjects_to_keep) * 0.2)
        for _ in range(num_elements_to_remove):
            element_to_remove = random.choice(subjects_to_keep)
            subjects_to_keep.remove(element_to_remove)

        # Creates open set
        open_dir = f"{base_dir_name}/open-set"
        os.mkdir(open_dir)
        # Creates open set reference
        create_reference(open_dir, path_to_hr, hr_images)
        # Creates open set SR images
        lr_images = []
        for i in common_images:
            if get_subject(i, subject_index) in subjects_to_keep:
                lr_images.append(i)
        create_sr(open_dir, sr_dirs, lr_images)

        # Creates closed set
        closed_dir = f"{base_dir_name}/closed-set"
        os.mkdir(closed_dir)
        # Creates closed set reference
        create_reference(closed_dir, path_to_hr, hr_images)
        # Creates closed set SR images
        create_sr(closed_dir, sr_dirs, common_images)


if __name__ == "__main__":
    main(sys.argv[1:])