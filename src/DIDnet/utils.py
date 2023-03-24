import tensorflow as tf

def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-6)))

def bicubic_downsample(img, _):
    bicubic = tf.image.resize(
        img,
        size=(90,65),
        method=tf.image.ResizeMethod.BICUBIC
    )
    return bicubic

