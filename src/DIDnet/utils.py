import os
from PIL import Image

def bicubic_downsample(in_path, out_path, out_shape):
    for img in os.listdir(in_path):
        print(f"Resizing {img}")
        original = Image.open(in_path + img)
        bicubic = original.resize(out_shape, Image.BICUBIC)
        bicubic.save(out_path + img)