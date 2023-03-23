import tensorflow as tf
from keras.utils import image_dataset_from_directory

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-6)))

def bicubic_downsample(img):
    bicubic = tf.image.resize(
        images=img,
        size=(90,65),
        method=tf.image.ResizeMethod.BICUBIC
    )
    return bicubic


def load_datasets(lr_path, lr_shape, sr_path, sr_shape, test_size):
    lr = image_dataset_from_directory(lr_path, batch_size=None, image_size=lr_shape)
    sr = image_dataset_from_directory(sr_path, batch_size=None, image_size=sr_shape)

    return lr, sr
