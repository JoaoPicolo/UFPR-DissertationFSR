from keras import Input, Model
from keras.layers import Conv2D, ReLU, PReLU, Add, Dense

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


# Reference: https://paperswithcode.com/paper/sphereface-deep-hypersphere-embedding-for
def get_sphere_face_embedding_net(inputs):
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same")(inputs)
    x = PReLU()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=128, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=512, kernel_size=3, strides=2, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    # TODO: Eltwise operation

    x = Conv2D(filters=512, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding="same")(x)
    x = PReLU()(x)

    outputs = Dense(512)(x)

    return outputs

def get_face_recognition(input_shape):
    inputs = Input(shape=input_shape)

    # Builds the network
    x = Conv2D(8, kernel_size=3, strides=1, padding="same")(inputs)
    x = convolution_block(x, 8)
    x = convolution_block(x, 16)
    x = convolution_block(x, 32)
    x = convolution_block(x, 64)
    outputs = get_sphere_face_embedding_net(x)

    # Defines the model
    model = Model(inputs, outputs, name="Face_Recognition")

    return model