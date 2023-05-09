import tensorflow as tf
from keras import Input, Model
from keras.layers import GlobalAveragePooling2D, Conv2D, ReLU, Multiply, BatchNormalization, Add, LeakyReLU

# CAL and RCAB are defined in: https://paperswithcode.com/paper/image-super-resolution-using-very-deep
def channel_attention_layer(inputs, filters, reduction):
    # TODO: Verify if activation is right for this two specific layers
    x = GlobalAveragePooling2D(keepdims=True)(inputs)
    x = Conv2D(filters//reduction, 1, padding="same", activation="relu", use_bias=True)(x)
    #x = ReLU()(x)
    x = Conv2D(filters, 1, padding="same", activation="sigmoid", use_bias=True)(x)
    #x = Activation("sigmoid")(x)
    return Multiply()([x, inputs])


def residual_channel_attention_block(inputs, filters, kernel_size, reduction, bn=False):
    x = inputs
    for i in range(2):
        x = Conv2D(filters, kernel_size, padding="same", activation="relu", use_bias=True)(x)
        if bn:
            x = BatchNormalization()(x)
        if i == 0:
            x = ReLU()(x)
    x = channel_attention_layer(x, filters, reduction)
    # TODO: Make sure this is right
    # This is commented in the original implementation, but since the number of
    # filters is 64 it is necessary to rescale the dimensions for the computations forward
    x = Conv2D(3, kernel_size, padding="same", activation="relu", use_bias=True)(x)
    return Add()([x, inputs])


# Pixel shuffle is defined in: https://paperswithcode.com/paper/real-time-single-image-and-video-super
def pixel_shuffle(inputs, filters, kernel_size, factor=2):
    x = Conv2D(filters * (factor ** 2), kernel_size, padding="same", activation="relu")(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


def body_module(inputs, filters, kernel_size=3, reduction=16):
    x = inputs
    for _ in range(30):
        x = residual_channel_attention_block(
            x, filters, kernel_size, reduction)

    return x


def upsampling_module(inputs, filters, kernel_size=3):
    # Conv 3x3
    x = Conv2D(filters, kernel_size, padding="same", activation="relu")(inputs)
    # Pixel-shuffle
    x = pixel_shuffle(x, filters, kernel_size)

    return x


def upsampling_block(inputs):
    filters = 64
    x = body_module(inputs, filters)
    x = upsampling_module(x, filters)
    # TODO: Revisit filters
    x = Conv2D(3, 1, padding="same", activation="relu")(x)

    # TODO: In the graph of the original papers there is an additional convolution
    # that is not described in the network structured. Should be added?
    # Maybe it's 1x1 to keep the dimensions - Check with Valfride

    return x


def get_model_G(input_shape, norm_layer=None):
    inputs = Input(shape=input_shape)

    if norm_layer:
        x = norm_layer(inputs)
    else:
        x = inputs

    x = upsampling_block(x)
    outputs = upsampling_block(x)
    model = Model(inputs, outputs, name="Generator_G")

    return model


def downsampling_block(inputs):
    filters = 64
    # TODO: review activation function in the future
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = LeakyReLU()(x)
    # TODO: Revisit filters
    x = Conv2D(3, 3, strides=(2, 2), padding="same", activation="relu")(x)

    return x


def get_model_F(input_shape, norm_layer=None):
    inputs = Input(shape=input_shape)

    if norm_layer:
        x = norm_layer(inputs)
    else:
        x = inputs

    x = downsampling_block(x)
    outputs = downsampling_block(x)
    model = Model(inputs, outputs, name="Generator_F")

    return model
