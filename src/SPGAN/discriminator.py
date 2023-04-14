import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, ReLU, Add

# Reference: https://paperswithcode.com/paper/fully-convolutional-networks-for-semantic
def convolution_block(inputs, channels):
    x = Conv2D(channels, kernel_size=3, strides=1, padding="same")(inputs)
    x = ReLU()(x)
    x = Conv2D(channels, kernel_size=3, strides=1, padding="same")(x)

    if x.shape != inputs.shape:
        shortcut = Conv2D(channels, kernel_size=1, strides=1)(inputs)
    else:
        shortcut = inputs

    x = Add()([x, shortcut])
    outputs = ReLU()(x)

    return outputs

def get_discriminator(input_shape):
    inputs = Input(shape=input_shape)

    # Builds the network
    x = Conv2D(8, kernel_size=3, strides=1, padding="same")(inputs)
    x = convolution_block(x, 8)
    x = convolution_block(x, 16)
    x = convolution_block(x, 32)
    x = convolution_block(x, 64)

    # TODO: Test with one channel
    outputs = Conv2D(3, kernel_size=1, strides=1, padding="same")(x)

    # Defines the model
    model = Model(inputs, outputs, name="Discriminator")

    return model