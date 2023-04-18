import sys
import argparse

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from generator import get_generator
from discriminator import get_discriminator

sys.path.append("..")
from shared.utils import get_parser
from shared.plots import plot_loss_curve, plot_test_dataset
from shared.data import resize_image, get_dataset_split, manipulate_dataset

class SPGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer):
        super().compile()
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
        disc_real = self.discriminator(concat_real, training=isTraining)
        disc_fake = self.discriminator(concat_fake, training=isTraining)
        
        d_sp = tf.subtract(disc_real, disc_fake)

        # Calculate losses
        pw_loss = tf.reduce_mean(tf.abs(hr_image - sr_image)) # Pixel wise loss between HR and SR
        disc_loss = -tf.reduce_sum(tf.math.minimum(0.0, (d_sp - pw_loss)))
        gen_loss = tf.reduce_sum(tf.multiply(d_sp, self.get_scalar(d_sp)))

        return { "g_loss": gen_loss, "d_loss": disc_loss }

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
    hr_shape = (216, 176, 6)
    train, validation, test = get_dataset_split(args.path, (hr_shape[0], hr_shape[1]), 0.6, 0.2, 0.2)
    train_hr, validation_hr, test_hr = manipulate_dataset(train, validation, test)
    train_lr, validation_lr, test_lr = manipulate_dataset(train, validation, test, resize=True, resize_shape=(lr_shape[0], lr_shape[1]))

    # Get models
    generator = get_generator(input_shape=lr_shape)
    discriminator = get_discriminator(input_shape=hr_shape)
   
    # Create gan model
    spgan = SPGAN(generator, discriminator)

    # Compile the model
    spgan.compile(
        generator_optimizer=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999),
        discriminator_optimizer=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999),
    )

    # Callbacks
    model_checkpoint_callback = ModelCheckpoint(
        filepath= "./checkpoints/spgan_checkpoints.{epoch:03d}", save_weights_only=True
    )

    # Trains
    history = spgan.fit(
        x=tf.data.Dataset.zip((train_lr, train_hr)),
        epochs=10,
        callbacks=[model_checkpoint_callback],
        validation_data=tf.data.Dataset.zip((validation_lr, validation_hr))
    )

    # Plot networks curves
    plot_loss_curve("./results", history, "g_loss")
    plot_loss_curve("./results", history, "d_loss")

    # Plot the test datasets
    spgan.evaluate_test_datasets("./results", test_lr)


if __name__ == "__main__":
    main()
