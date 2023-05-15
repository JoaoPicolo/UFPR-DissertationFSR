import math

import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.layers import Normalization

def resize_image(image, output_shape, resize_method):
    resized = tf.image.resize(
        image, size=output_shape, method=resize_method
    )
    return resized


def manipulate_dataset(
        train, validation, test, batch_size=1, resize=False,
        resize_shape=(32, 32), resize_method=tf.image.ResizeMethod.BICUBIC
    ):
    if not resize:
        train_new = (train.map(lambda x, _: x).batch(batch_size))
        validation_new = (validation.map(lambda x, _: x).batch(batch_size))
        test_new = (test.map(lambda x, _: x).batch(batch_size))
    else:
        train_new = (train.map(
            lambda x, _: resize_image(x, resize_shape, resize_method)
        ).batch(batch_size))
        validation_new = (validation.map(
            lambda x, _: resize_image(x, resize_shape, resize_method)
        ).batch(batch_size))
        test_new = (test.map(
            lambda x, _: resize_image(x, resize_shape, resize_method)
        ).batch(batch_size))

    return train_new, validation_new, test_new

def apply_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    return image


def get_dataset_split(path, image_shape, train_split=0.7, evaluation_split=0.2, test_split=0.1, augment_data=False, augment_factor=10, batch_size=None):
    dataset = image_dataset_from_directory(
        directory=path, seed=123, image_size=image_shape, batch_size=batch_size)
    dataset_size = len(dataset)

    train_size = math.floor(train_split*dataset_size)
    eval_size = math.floor(evaluation_split*dataset_size)
    test_size = math.floor(test_split*dataset_size)

    train = dataset.take(train_size)
    validation = dataset.skip(train_size).take(eval_size)
    test = dataset.skip(train_size+eval_size).take(test_size)

    if augment_data:
        augmented = train.map(lambda x, y: (apply_augmentation(x), y)).repeat(augment_factor)
        train = train.concatenate(augmented)

    return train, validation, test


def get_normalization_layer(dataset):
    normalizer = Normalization()
    dataset_x = dataset.map(lambda x, _: x)
    normalizer.adapt(dataset_x)
    return normalizer
