import tensorflow as tf
from keras import Input, Model
from keras.initializers import RandomUniform
from keras.layers import Add, Conv2D, UpSampling2D, concatenate

# SPGAN uses the generator from RDN, reference: https://paperswithcode.com/paper/residual-dense-network-for-image-super
def residual_dense_blocks(inputs, C, D, G, G0, kernel_size, initializer):
    rdb_concat = []
    rdb_in = inputs
    for _ in range(D):
        x = rdb_in
        for _ in range(C):
            out = Conv2D(G, kernel_size, padding="same",
                         activation="relu", kernel_initializer=initializer)(x)
            x = concatenate([x, out], axis=3)

        x = Conv2D(G0, kernel_size=1, kernel_initializer=initializer)(x)
        rdb_in = Add()([x, rdb_in])
        rdb_concat.append(rdb_in)

    return concatenate(rdb_concat, axis=3)


def upscale_block(inputs, output_dim, scale, initializer, method="ups"):
    x = Conv2D(64, kernel_size=5, strides=1, padding="same",
               activation="relu", kernel_initializer=initializer)(inputs)
    x = Conv2D(32, kernel_size=3, padding="same",
               activation="relu", kernel_initializer=initializer)(x)

    if method == "shuffle":
        x = Conv2D(output_dim * (scale ** 2), kernel_size=3,
                   padding="same", kernel_initializer=initializer)(x)
        x = tf.nn.depth_to_space(x, block_size=scale)
    elif method == "ups":
        x = Conv2D(output_dim * (scale ** 2), kernel_size=3,
                   padding="same", kernel_initializer=initializer)(x)
        x = UpSampling2D(size=scale)(x)

    return x


def get_generator(input_shape):
    inputs = Input(shape=input_shape)

    # Variables
    output_dim = 3
    kernel_size = 3
    C_layers_in_rdb = 6
    D_num_rdb = 20
    G_conv_output_dim = 64
    G0_rdb_output_dim = 64
    scale_factor = 4
    init_extreme_val = 0.05
    initializer = RandomUniform(minval=-init_extreme_val, maxval=init_extreme_val, seed=None)

    # Defines the network
    f_m1 = Conv2D(G0_rdb_output_dim, kernel_size, padding="same",
                  kernel_initializer=initializer)(inputs)
    f_0 = Conv2D(G0_rdb_output_dim, kernel_size, padding="same",
                 kernel_initializer=initializer)(f_m1)
    fd = residual_dense_blocks(f_0, C_layers_in_rdb, D_num_rdb,
                               G_conv_output_dim, G0_rdb_output_dim, kernel_size, initializer)
    gff_1 = Conv2D(G0_rdb_output_dim, kernel_size=1,
                   padding="same", kernel_initializer=initializer)(fd)
    gff_2 = Conv2D(G0_rdb_output_dim, kernel_size, padding="same",
                   kernel_initializer=initializer)(gff_1)
    fdf = Add()([gff_2, f_m1])
    fu = upscale_block(fdf, output_dim, scale_factor, initializer)
    outputs = Conv2D(output_dim, kernel_size, padding="same",
                     kernel_initializer=initializer)(fu)

    # Defines the model
    model = Model(inputs, outputs, name="Generator")

    return model


def main():
    print("Hello World")


if __name__ == "__main__":
    main()
