from keras import Input, Model
from keras.layers import Conv2D, concatenate

def convolution_block(inputs, channels):
    x = Conv2D(channels, kernel_size=3, strides=1, padding="same")(inputs)
    x = Conv2D(channels, kernel_size=3, strides=1, padding="same")(x)

    return concatenate([x, inputs], axis=3)

def get_discriminator(input_shape):
    inputs = Input(shape=input_shape)

    # Builds the network
    x = Conv2D(8, kernel_size=3, strides=1, padding="same")(inputs)
    x = convolution_block(x, 8)
    x = convolution_block(x, 16)
    x = convolution_block(x, 32)
    outputs = convolution_block(x, 64)

    # Defines the model
    model = Model(inputs, outputs, name="Discriminator")

    return model