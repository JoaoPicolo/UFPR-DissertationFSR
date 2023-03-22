import tensorflow as tf
from keras.utils import image_dataset_from_directory

def bicubic_downsample(img, _):
    bicubic = tf.image.resize(
        images=img,
        size=(90,65),
        method=tf.image.ResizeMethod.BICUBIC,
        preserve_aspect_ratio=False,
        antialias=False,
        name=None
    )
    return bicubic


def load_datasets(lr_path, lr_shape, sr_path, sr_shape, test_size):
    lr = image_dataset_from_directory(lr_path, batch_size=None, image_size=lr_shape)
    sr = image_dataset_from_directory(sr_path, batch_size=None, image_size=sr_shape)

    return lr, sr
