import os
import sys
from typing import List

def main(args: List[str]):
    path_to_gallery = args[0]
    gallery_size = int(args[1])
    subject_index = int(args[2])

    gallery = os.listdir(path_to_gallery)

    gallery_parent = os.path.abspath(os.path.join(path_to_gallery, os.pardir))
    gallery_parent = os.path.abspath(os.path.join(gallery_parent, os.pardir))
    f = open(f"{gallery_parent}/gallery-{gallery_size}.txt", 'w')
    for image in gallery:
        subject = image.split('_')[subject_index]
        f.write(f"{image}\t{subject}\n")
    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])