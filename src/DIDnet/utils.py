import tensorflow as tf

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

