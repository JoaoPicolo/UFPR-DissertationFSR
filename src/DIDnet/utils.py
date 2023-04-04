import math

import tensorflow as tf
from keras.utils import image_dataset_from_directory

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + 1e-6))


def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mse_from_embedding(fst_embedding, snd_embedding):
    sum = 0
    for idx, _ in enumerate(fst_embedding):
        sum += (fst_embedding[idx] - snd_embedding[idx])**2

    sum = sum / len(fst_embedding)
    return sum


def bicubic_downsample(img, _):
    bicubic = tf.image.resize(
        img,
        size=(90,65),
        method=tf.image.ResizeMethod.BICUBIC
    )
    return bicubic


def manipulate_dataset(train, validation, test, apply_bicubic=False):
    if apply_bicubic:
        function = bicubic_downsample
    else:
        function = lambda x, _: x

    train_new = (train.map(function).batch(1))
    validation_new = (validation.map(function).batch(1))
    test_new = (test.map(function).batch(1))

    return train_new, validation_new, test_new


def get_dataset_split(path, train_split, evaluation_split, image_shape=(360,260)):
    dataset = image_dataset_from_directory(directory=path, seed=123, image_size=image_shape, batch_size=None)
    dataset_size = len(dataset)

    train_size = math.floor(train_split*dataset_size)
    eval_size = math.floor(evaluation_split*dataset_size)

    train = dataset.take(train_size)
    validation = dataset.skip(train_size).take(eval_size)
    test = dataset.skip(train_size+eval_size).take(eval_size)

    return train, validation, test

