import os
from PIL import Image
from keras.utils import image_dataset_from_directory

def bicubic_downsample(in_path, out_path, out_shape):
    for img in os.listdir(in_path):
        print(f"Resizing {img}")
        original = Image.open(in_path + img)
        bicubic = original.resize(out_shape, Image.BICUBIC)
        bicubic.save(out_path + img)


def load_datasets(lr_path, lr_shape, sr_path, sr_shape, test_size):
    lr = image_dataset_from_directory(lr_path, batch_size=None, image_size=lr_shape)
    sr = image_dataset_from_directory(sr_path, batch_size=None, image_size=sr_shape)

    return lr, sr
