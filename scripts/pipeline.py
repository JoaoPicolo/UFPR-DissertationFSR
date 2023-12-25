import os
import math
import random
import shutil
import subprocess

def main():
    # Foler containing the SR results for each dataset, called scface-processed
    # and quiscampi-processed. Each results folders is then split into the open-set
    # and closed-set protocols.
    base_folder = "./super-resolved/"
    
    # Pipeline execution
    samples = 30
    for it in range(1, samples+1):
        created_folders = []
        datasets = ["scface-processed", "quiscampi-processed"]
        for subject_index, dataset in enumerate(datasets):
            options = ["open-set", "closed-set"]
            for option in options:
                print(f"Processing {dataset}-{option}")
                base = f"{base_folder}/{dataset}/{option}"
                base_info = base.split('/')
                base_info = [item for item in base_info if len(item) > 1]
                new_folder = f"{base_info[-2].split('-')[0]}-{base_info[-1]}"

                shutil.copytree(base, new_folder)
                created_folders.append(new_folder)

                test_path = new_folder

                total_folds = 2
                base_lr = f"no-interpolation-512/"

                lr_images = os.listdir(f"{test_path}/{base_lr}")
                lr_subjects = list(set([name.split('_')[subject_index] for name in lr_images]))

                sample_n = math.floor(len(lr_subjects)/total_folds)

                sampled_total = 0
                folds = []
                while lr_subjects:
                    sampled_items = random.sample(lr_subjects, min(sample_n, len(lr_subjects)))

                    if sampled_total < total_folds:
                        folds.append(sampled_items)
                    else:
                        for index, item in enumerate(sampled_items):
                            folds[index].append(item)

                    lr_subjects = [item for item in lr_subjects if item not in sampled_items]
                    sampled_total += 1

                projects = os.listdir(test_path)
                projects = [name for name in projects if os.path.isdir(f"{test_path}/{name}")]
                for index, fold in enumerate(folds):
                    print(f"Moving fold {index+1}")
                    for project in projects:
                        project_path = f"{test_path}/{project}"
                        project_images = os.listdir(project_path)

                        fold_path = f"{project_path}/fold_{index+1}"
                        os.makedirs(fold_path)

                        for item in project_images:
                            # Ignore previously created folds
                            if not os.path.isdir(f"{project_path}/{item}"):
                                subject = item.split('_')[subject_index]
                                if subject in fold:
                                    shutil.move(f"{project_path}/{item}", f"{fold_path}/{item}")

                subprocess.Popen([f"ls {new_folder}/bicubic-interpolation-32/fold_1/ > {new_folder}-probe-{it}.txt"], shell=True).wait()

                # Will only execute on the open-set since reference will have more images than LR
                references = ["reference-512"]
                for reference in references:
                    reference_path = f"{test_path}/{reference}"
                    items = os.listdir(reference_path)
                    images = [item for item in items if not os.path.isdir(f"{reference_path}/{item}")]

                    print(f"Rerence {reference} still has {len(images)} images")
                    for image in images:
                        for index in range(1, total_folds+1):
                            shutil.copy(f"{reference_path}/{image}", f"{reference_path}/fold_{index}/{image}")
                        os.remove(f"{reference_path}/{image}")

                subprocess.Popen([f"ls {new_folder}/reference-256/fold_1/ > {new_folder}-gallery-{it}.txt"], shell=True).wait()

        subprocess.Popen(["bash ./methodology/execution.sh"], shell=True).wait()
        subprocess.Popen(["bash ./recognition/execution.sh"], shell=True).wait()

        for folder in created_folders:
            shutil.rmtree(folder)

if __name__ == "__main__":
    main()