import sys
import subprocess
from typing import List

def main(args: List[str]):
    path_to_bin = args[0]
    path_to_images = args[1]
    subject_index = int(args[2])

    general_hits = 0
    general_total = 0

    command = f"insightfacepaddle --rec --rec_model ArcFace --index {path_to_bin} --input {path_to_images}"
    delimiter = "INFO:root:File:"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
    for line in process.stdout:
        line = line.strip()
        is_subject_line = delimiter in line
        if is_subject_line:
            information = line.split(delimiter)[1]
            image = information.split(',')[0].strip()
            predict = information.split("label(s):")[1].strip()[2:-2]
            predict = int(predict) if predict != '' else -1

            # Consider all images
            subject = image.split('_')[subject_index]
            if int(subject) == int(predict):
                general_hits += 1
            general_total += 1

    general_accuracy = round(((general_hits / general_total) * 100), 2)
    with open("results.csv", 'a') as f:
        f.write(f"{path_to_images},{general_accuracy}\n")
    print(f"Accuracy for {path_to_images} is {general_accuracy}% among {general_hits} hits over a total of {general_total} general images")


if __name__ == "__main__":
    main(sys.argv[1:])