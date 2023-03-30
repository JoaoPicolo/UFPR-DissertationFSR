import sys

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils import image_dataset_from_directory, array_to_img

from generators import get_model_G, get_model_F
from utils import charbonnier_loss, mae_loss, mse_from_embedding, bicubic_downsample

sys.path.append("..")
from Facenet.facenet import get_embeddings

# Reference: https://keras.io/examples/generative/cyclegan/#build-the-cyclegan-model
class GANMonitor(Callback):
    def __init__(self, test, num_img=4):
        self.test = test
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if (epoch % 10 == 0):  # Saves every 10 epochs
            _, ax = plt.subplots(4, 2, figsize=(12, 12))
            for i, img in enumerate(self.test.take(self.num_img)):
                prediction = self.model.gen_G(img)

                img = np.array(img.numpy())
                img = img[0, :, :, :]
                img = array_to_img(img)

                prediction = np.array(prediction.numpy())
                prediction = prediction[0, :, :, :]
                prediction = array_to_img(prediction)

                ax[i, 0].imshow(img)
                ax[i, 1].imshow(prediction)
                ax[i, 0].set_title("Input image")
                ax[i, 1].set_title("Translated image")
                ax[i, 0].axis("off")
                ax[i, 1].axis("off")

                plt.savefig(f"./results/results_{epoch}.png")


class DIDnet(Model):
    def __init__(
        self,
        generator_G,
        generator_F,
    ):
        super().__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F

    def compile(
        self,
        optimizer,
        gen_G_optimizer,
        gen_F_optimizer,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super().compile()
        self.optimizer = optimizer
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

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_x, cycled_x)
            cycle_loss_F = self.cycle_loss_fn(real_y, cycled_y)

            # TODO: Verify if this is not inverted
            # Get face embeddings from Facenet
            id_embeddings_G = get_embeddings(fake_x, cycled_x)
            id_embeddings_F = get_embeddings(fake_y, cycled_y)

            # Generator identity loss
            id_loss_F = self.identity_loss_fn(
                id_embeddings_F[0], id_embeddings_F[1])
            id_loss_G = self.identity_loss_fn(
                id_embeddings_G[0], id_embeddings_G[1])

            # Total generator loss
            total_loss_G = cycle_loss_G + id_loss_G
            total_loss_F = cycle_loss_F + id_loss_F

            # Network Loss
            loop_loss = total_loss_G + total_loss_F
            identity_loss = id_loss_G + id_loss_F
            rec_loss = mae_loss(real_y, fake_y)
            channel_loss = charbonnier_loss(
                tf.image.rgb_to_yuv(fake_y), tf.image.rgb_to_yuv(real_y))
            network_loss = loop_loss + rec_loss + 0.5*identity_loss + 0.5*channel_loss

        # Gets the gradients for the generators and optimizes them
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )

        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Gets the gradients for the whole network and optimizes it
        grads_network = tape.gradient(network_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads_network, self.trainable_variables))

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "DIDnet_loss": network_loss,
        }


def main():
    g = get_model_G(input_shape=(90, 65, 3))
    f = get_model_F(input_shape=(360, 260, 3))

    # Create cycle gan model
    didnet = DIDnet(g, f)

    # Compile the model
    didnet.compile(
        optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        gen_G_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        gen_F_optimizer=Adam(learning_rate=2e-4, beta_1=0.5),
        cycle_loss_fn=charbonnier_loss,
        identity_loss_fn=mse_from_embedding
    )

    # TODO: Change to the right dataset
    train = image_dataset_from_directory(
        directory="../../datasets/FEI/", validation_split=0.1, subset="training", seed=123, image_size=(360, 260), batch_size=None)
    test = image_dataset_from_directory(
        directory="../../datasets/FEI/", validation_split=0.1, subset="validation", seed=123, image_size=(360, 260), batch_size=None)

    train_sr = (train.map(lambda x, _: x).batch(1))
    test_sr = (test.map(lambda x, _: x).batch(1))
    train_lr = (train.map(bicubic_downsample).batch(1))
    test_lr = (test.map(bicubic_downsample).batch(1))

    # Callbacks
    plotter = GANMonitor(test=test_lr)
    checkpoint_filepath = "./checkpoints/didnet_checkpoints.{epoch:03d}"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=True
    )

    didnet.fit(
        tf.data.Dataset.zip((train_lr, train_sr)),
        epochs=100,
        callbacks=[plotter, model_checkpoint_callback],
    )


if __name__ == "__main__":
    main()
