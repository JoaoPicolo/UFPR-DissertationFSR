import math

import tensorflow as tf
from keras.utils import image_dataset_from_directory


def bicubic_downsample(image, output_shape, resize_method):
    bicubic = tf.image.resize(
        image, size=output_shape, method=resize_method
    )
    return bicubic


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
            lambda x, _: bicubic_downsample(x, resize_shape, resize_method)
        ).batch(batch_size))
        validation_new = (validation.map(
            lambda x, _: bicubic_downsample(x, resize_shape, resize_method)
        ).batch(batch_size))
        test_new = (test.map(
            lambda x, _: bicubic_downsample(x, resize_shape, resize_method)
        ).batch(batch_size))

    return train_new, validation_new, test_new


def get_dataset_split(path, image_shape, train_split=0.7, evaluation_split=0.2, test_split=0.1):
    dataset = image_dataset_from_directory(
        directory=path, seed=123, image_size=image_shape, batch_size=None)
    dataset_size = len(dataset)

    train_size = math.floor(train_split*dataset_size)
    eval_size = math.floor(evaluation_split*dataset_size)
    test_size = math.floor(test_split*dataset_size)

    train = dataset.take(train_size)
    validation = dataset.skip(train_size).take(eval_size)
    test = dataset.skip(train_size+eval_size).take(test_size)

    return train, validation, test
