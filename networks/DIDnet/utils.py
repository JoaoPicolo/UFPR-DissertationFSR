import tensorflow as tf

# TODO: In all the methods, see if use reduce sum is correct or should use reduce mean

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_sum(tf.sqrt(tf.square(y_true - y_pred) + 1e-6))


def mse_from_embedding(fst_embedding, snd_embedding):
    sum = 0
    for idx, _ in enumerate(fst_embedding):
        sum += (fst_embedding[idx] - snd_embedding[idx])**2

    return sum


def mae_loss(y_true, y_pred):
    return tf.reduce_sum(tf.abs(y_true - y_pred))
