import tensorflow as tf
from keras import Input, Model
from keras.layers import GlobalAveragePooling2D, Conv2D, ReLU, Activation, Multiply, Add, LeakyReLU, BatchNormalization
from keras.losses import MeanAbsoluteError
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing import image
import matplotlib.pyplot as plt

# TODO: Verify activation function of each layers

# CAL and RCAB are defined in: https://paperswithcode.com/paper/image-super-resolution-using-very-deep
def channel_attention_layer(inputs, filters, reduction):
    x = GlobalAveragePooling2D(keepdims=True)(inputs)
    x = Conv2D(filters//reduction, 1, padding="same", use_bias=True)(x)
    x = ReLU()(x)
    x = Conv2D(filters, 1, padding="same", use_bias=True)(x)
    x = Activation("sigmoid")(x)
    return Multiply()([x, inputs])


def residual_channel_attention_block(inputs, filters, kernel_size, reduction, bn=False):
    x = inputs
    for i in range(2):
        x = Conv2D(filters, kernel_size, padding="same", use_bias=True)(x)
        if bn:
            x = BatchNormalization()(x)
        if i == 0:
            x = ReLU()(x)
    x = channel_attention_layer(x, filters, reduction)
    # TODO: Make sure this is right
    # This is commented in the original implementation, but since the number of
    # filters is 64 it is necessary to rescale the dimensions for the computations forward
    x = Conv2D(3, kernel_size, padding="same", use_bias=True)(x)
    return Add()([x, inputs])


# Pixel shuffle is defined in: https://paperswithcode.com/paper/real-time-single-image-and-video-super
def pixel_shuffle(inputs, filters, kernel_size, factor=2):
    x = Conv2D(filters * (factor ** 2), kernel_size, padding="same")(inputs)
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
    x = Conv2D(filters, kernel_size, padding="same")(inputs)
    # Pixel-shuffle
    x = pixel_shuffle(x, filters, kernel_size)

    return x


def upsampling_block(inputs):
    filters = 64
    x = body_module(inputs, filters)
    x = upsampling_module(x, filters)
    # TODO: Revisit filters
    x = Conv2D(3, 1, padding="same")(x)

    # TODO: In the graph of the original papers there is an additional convolution
    # that is not described in the network structured. Should be added?

    return x


def get_model_G():
    inputs = Input(shape=(32, 32, 3))
    outputs = upsampling_block(inputs)
    outputs = upsampling_block(outputs)
    model = Model(inputs, outputs, name="Generator_G")

    return model


def downsampling_block(inputs):
    filters = 64
    x = Conv2D(filters, 3, padding="same")(inputs)
    x = LeakyReLU()(x)
    # TODO: Revisit filters
    x = Conv2D(3, 3, strides=(2, 2), padding="same")(x)

    return x


def get_model_F():
    inputs = Input(shape=(128, 128, 3))
    outputs = downsampling_block(inputs)
    outputs = downsampling_block(outputs)
    model = Model(inputs, outputs, name="Generator_F")

    return model

# Reference: https://keras.io/examples/generative/cyclegan/#build-the-cyclegan-model
class GANMonitor(Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_horses.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = image.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()

# TODO: Remove lambda cycle and identity?
class DIDnet(Model):
    def __init__(
        self,
        generator_G,
        generator_F,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_G_optimizer,
        gen_F_optimizer,
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.cycle_loss_fn = MeanAbsoluteError()
        self.identity_loss_fn = MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is LR and y is SR
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # LR to SR
            fake_y = self.gen_G(real_x, training=True)
            # SR to LR
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (LR to SR to LR): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (SR to LR to SR) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            # same_x = self.gen_F(real_x, training=True)
            # same_y = self.gen_G(real_y, training=True)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            # id_loss_G = (
            #     self.identity_loss_fn(real_y, same_y)
            #     * self.lambda_cycle
            #     * self.lambda_identity
            # )
            # id_loss_F = (
            #     self.identity_loss_fn(real_x, same_x)
            #     * self.lambda_cycle
            #     * self.lambda_identity
            # )

            # Total generator loss
            total_loss_G = cycle_loss_G #+ id_loss_G
            total_loss_F = cycle_loss_F #+ id_loss_F

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
        }


def main():
    g = get_model_G()
    f = get_model_F()

    # Create cycle gan model
    didnet = DIDnet(g, f)

    # Compile the model
    didnet.compile(
        gen_G_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
    )

    # Callbacks
    plotter = GANMonitor()
    checkpoint_filepath = ".didnet_checkpoints.{epoch:03d}"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath
    )

    didnet.fit(
        tf.data.Dataset.zip((train_horses, train_zebras)),
        epochs=1,
        callbacks=[plotter, model_checkpoint_callback],
    )


if __name__ == "__main__":
    main()
