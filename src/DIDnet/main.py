import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.losses import MeanAbsoluteError
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import image_dataset_from_directory, array_to_img

from generators import get_model_G, get_model_F
from utils import charbonnier_loss, bicubic_downsample

# Reference: https://keras.io/examples/generative/cyclegan/#build-the-cyclegan-model
class GANMonitor(Callback):
    def __init__(self, test, num_img=4):
        self.test = test
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.test.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = array_to_img(prediction)
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
        cycle_loss_fn,
        identity_loss_fn
    ):
        super().compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

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
            # TODO: Verify if it is inverted
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
    g = get_model_G(input_shape=(90, 65, 3))
    f = get_model_F(input_shape=(360, 260, 3))

    # Create cycle gan model
    didnet = DIDnet(g, f)

    # Compile the model
    didnet.compile(
        gen_G_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        cycle_loss_fn=charbonnier_loss,
        identity_loss_fn=MeanAbsoluteError()
    )

    train_sr = image_dataset_from_directory(
        directory="../../datasets/FEI/", validation_split=0.025, subset="training", seed=123, image_size=(360, 260), batch_size=None)
    test_sr = image_dataset_from_directory(
        directory="../../datasets/FEI/", validation_split=0.025, subset="validation", seed=123, image_size=(360, 260), batch_size=None)
    
    train_sr = (train_sr.map(lambda x, _: x).batch(1))
    test_sr = (test_sr.map(lambda x, _: x).batch(1))
    train_lr = (train_sr.map(bicubic_downsample))
    test_lr = (test_sr.map(bicubic_downsample))

    # Callbacks
    plotter = GANMonitor(test=test_lr)
    checkpoint_filepath = ".didnet_checkpoints.{epoch:03d}"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath
    )

    didnet.fit(
        tf.data.Dataset.zip((train_lr, train_sr)),
        epochs=1,
        callbacks=[plotter, model_checkpoint_callback],
    )


if __name__ == "__main__":
    main()
