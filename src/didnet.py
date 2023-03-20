import tensorflow as tf
from keras import Input, Model
from keras.layers import GlobalAveragePooling2D, Conv2D, ReLU, Activation, Multiply, Add, Lambda, BatchNormalization

# CAL and RCAB are defined in: https://github.com/yulunzhang/RCAN
def channel_attention_layer(inputs, channel, reduction):
    x = GlobalAveragePooling2D(keepdims=True)(inputs)
    x = Conv2D(filters=channel//reduction, kernel_size=1, padding="same", use_bias=True)(x)
    x = ReLU()(x)
    x = Conv2D(filters=channel, kernel_size=1, padding="same", use_bias=True)(x)
    x = Activation("sigmoid")(x)
    return Multiply()([x, inputs])


def residual_channel_attention_block(inputs, n_feat=64, kernel_size=3, reduction=16, bias=True, bn=False):
    x = inputs
    for i in range(2):
        x = Conv2D(filters=n_feat, kernel_size=kernel_size, padding="same", use_bias=bias)(x)
        if bn:
            x = BatchNormalization()(x)
        if i == 0:
            x= ReLU()(x)
    x = channel_attention_layer(x, n_feat, reduction)
    # TODO: Verify this one, it isn't in the original implementation
    x = Conv2D(filters=3, kernel_size=kernel_size, padding="same", use_bias=bias)(x)
    return Add()([x, inputs])


def body_module(inputs):
    x = inputs
    for _ in range(30):
        x = residual_channel_attention_block(x)
    
    return x

# Pixel shuffle is defined in: https://github.com/krasserm/super-resolution
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsampling_module(inputs):
    x = inputs
    # Pixel-shuffle
    x = Conv2D(256, 3, padding='same')(x)
    x = Lambda(pixel_shuffle(scale=2))(x)
    # Conv 3x3
    x = Conv2D(filters=64, kernel_size=3, activation='relu', padding="same")(x)

    return x


def upsampling_block(inputs):
    x = body_module(inputs)
    x = upsampling_module(x)
    # TODO: Verify if the number of filters is right
    x = Conv2D(filters=3, kernel_size=1, activation='relu')(x)

    return x

def get_model_G():
    inputs = Input(shape=(32,32,3))
    outputs = upsampling_block(inputs)
    outputs = upsampling_block(outputs)
    model = Model(inputs, outputs)
    return model

def main():
    g = get_model_G()
    print(g.summary())


if __name__ == "__main__":
    main()
