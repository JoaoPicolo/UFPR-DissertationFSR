import sys

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import array_to_img

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from generator import get_generator
from discriminator import get_discriminator

sys.path.append("..")
from shared.metrics import MetricsPlotCallback
from shared.utils import get_parser
from shared.plots import plot_metric_by_epoch, plot_test_dataset
from shared.data import resize_image, get_dataset_split, manipulate_dataset, get_normalization_layer


class EpochPlotCallback(Callback):
    def __init__(self, test_set):
        super().__init__()
        self.test_set = test_set 

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(self.test_set.take(4)):
            prediction = self.model.generator(img)
            mean = self.model.generator.layers[1].get_weights()[0]
            stddev = self.model.generator.layers[1].get_weights()[1]
            prediction = prediction * stddev + mean

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


class NetworkMetricsCallback(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.d_loss_train, self.d_loss_val = [], []
        self.g_loss_train, self.g_loss_val = [], []

    # Store the metric values in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        self.d_loss_train_aux, self.d_loss_val_aux = [], []
        self.g_loss_train_aux, self.g_loss_val_aux = [], []

    def on_train_batch_end(self, batch, logs=None):
        self.d_loss_train_aux.append(logs["d_loss"])
        self.g_loss_train_aux.append(logs["g_loss"])


    def on_test_batch_end(self, batch, logs=None):
        self.d_loss_val_aux.append(logs["d_loss"])
        self.g_loss_val_aux.append(logs["g_loss"])

    def on_epoch_end(self, epoch, logs=None):
        self.d_loss_train.append(np.mean(self.d_loss_train_aux))
        self.d_loss_val.append(np.mean(self.d_loss_val_aux))
        self.g_loss_train.append(np.mean(self.g_loss_train_aux))
        self.g_loss_val.append(np.mean(self.g_loss_val_aux))


    def on_train_end(self, logs=None):
        plot_metric_by_epoch(self.path, "D Loss", self.d_loss_train, self.d_loss_val)
        plot_metric_by_epoch(self.path, "G Loss", self.g_loss_train, self.g_loss_val)


class SPGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer, loss):
        super().compile()
        self.loss = loss
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def get_scalar(self, x_elem):
        # Uses -1 as the string supervised version
        return tf.where(tf.greater_equal(x_elem, 0), tf.ones_like(x_elem), -1*tf.ones_like(x_elem))
    
    def get_losses(self, lr_image, hr_image, isTraining=False):
        # Upsamples LR
        height, width = self.discriminator.input_shape[1], self.discriminator.input_shape[2]
        upsampled_lr = resize_image(lr_image, (height, width), tf.image.ResizeMethod.BICUBIC)

        # LR to SR
        sr_image = self.generator(lr_image, training=True)

        # Gets discriminative matrix
        concat_real = tf.concat([upsampled_lr, hr_image], axis=-1)
        concat_fake = tf.concat([upsampled_lr, sr_image], axis=-1)
        disc_real = self.discriminator(hr_image, training=isTraining)
        disc_fake = self.discriminator(sr_image, training=isTraining)
        
        d_sp = tf.subtract(disc_real, disc_fake)

        # Calculate losses
        pw_loss = tf.reduce_mean(tf.abs(hr_image - sr_image)) # Pixel wise loss between HR and SR
        disc_loss = -tf.reduce_sum(tf.math.minimum(0.0, (d_sp - pw_loss)))
        gen_loss = tf.reduce_sum(tf.multiply(d_sp, self.get_scalar(d_sp)))

        # Computes other metrics
        psnr = tf.image.psnr(hr_image, sr_image, max_val=255)
        ssim = tf.image.ssim(hr_image, sr_image, max_val=255)
        cs = tf.keras.losses.cosine_similarity(hr_image, sr_image)

        return { "g_loss": gen_loss, "d_loss": disc_loss, "psnr": psnr, "ssim": ssim, "cs": cs }

    def train_step(self, batch_data):
        # x is LR and y is HR
        lr_image, hr_image = batch_data

        with tf.GradientTape(persistent=True) as tape:
            losses = self.get_losses(lr_image, hr_image, isTraining=True)

        # Gets gradients and updates the models
        gen_grads = tape.gradient(losses["g_loss"], self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        disc_grads = tape.gradient(losses["d_loss"], self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grads, self.discriminator.trainable_variables)
        )

        return losses
    
    def test_step(self, batch_data):
        # x is LR and y is SR
        lr_image, hr_image = batch_data

        # Compute losses
        losses = self.get_losses(lr_image, hr_image)

        return losses

    def evaluate_test_datasets(self, path, lr_dataset):
        plot_test_dataset(path, "LR", self.generator, lr_dataset)

def main():
    args = get_parser()

    # Loads the dataset and splits
    lr_shape = (54, 44, 3)
    hr_shape = (216, 176, 3)
    train, validation, test = get_dataset_split(args.path, (hr_shape[0], hr_shape[1]), 0.8, 0.1, 0.1)
    train_hr, validation_hr, test_hr = manipulate_dataset(train, validation, test)
    train_lr, validation_lr, test_lr = manipulate_dataset(train, validation, test, resize=True, resize_shape=(lr_shape[0], lr_shape[1]))

    # Get models
    norm_layer_train = get_normalization_layer(train)
    generator = get_generator(input_shape=lr_shape, norm_layer=norm_layer_train)
    discriminator = get_discriminator(input_shape=hr_shape, norm_layer=norm_layer_train)
   
    # Create gan model
    spgan = SPGAN(generator, discriminator)

    # Compile the model
    spgan.compile(
        generator_optimizer=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999),
        discriminator_optimizer=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999),
        loss="g_loss"
    )

    # Callbacks
    early_stop = EarlyStopping(monitor="g_loss", patience=10, mode="min")
    model_checkpoint_callback = ModelCheckpoint(
        filepath= "./checkpoints/spgan_checkpoints.{epoch:03d}", save_weights_only=True
    )
    metrics = MetricsPlotCallback(path="./results")
    net_metrics = NetworkMetricsCallback(path="./results")
    epoch_save = EpochPlotCallback(test_set=validation_lr)

    # Trains
    spgan.fit(
        x=tf.data.Dataset.zip((train_lr, train_hr)),
        epochs=100,
        callbacks=[model_checkpoint_callback, metrics, net_metrics, epoch_save, early_stop],
        validation_data=tf.data.Dataset.zip((validation_lr, validation_hr))
    )

    # Plot the test datasets
    spgan.evaluate_test_datasets("./results", test_lr)


if __name__ == "__main__":
    main()
