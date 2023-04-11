import sys

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from generator import get_generator
from discriminator import get_discriminator

sys.path.append("..")
from shared.data import  get_dataset_split, manipulate_dataset

class SPGAN(Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

    def train_step(self, batch_data):
        # x is LR and y is HR
        lr_image, hr_image = batch_data

        with tf.GradientTape(persistent=True) as tape:
            # LR to SR
            sr_image = self.generator(lr_image, training=True)

            # TODO: Verify if it should be separeted or channel wise
            concatenated_image = tf.concat([sr_image, hr_image], axis=-1)
            sr_discriminated = self.discriminator(concatenated_image, training=True)
            print(concatenated_image.shape)
            print(sr_discriminated.shape)
            exit(0)

def main():
    lr_shape = (54, 44, 3)
    hr_shape = (216, 176, 6)

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

    # Loads the dataset and splits
    train, validation, test = get_dataset_split("../../datasets/CelebA_test/", (hr_shape[0], hr_shape[1]), 0.6, 0.2, 0.2)
    train_hr, validation_hr, test_hr = manipulate_dataset(train, validation, test)
    train_lr, validation_lr, test_lr = manipulate_dataset(train, validation, test, resize=True, resize_shape=(lr_shape[0], lr_shape[1]))

    # Callbacks
    model_checkpoint_callback = ModelCheckpoint(
        filepath= "./checkpoints/spgan_checkpoints.{epoch:03d}", save_weights_only=True
    )

    # Trains
    history = spgan.fit(
        x=tf.data.Dataset.zip((train_lr, train_hr)),
        epochs=1,
        callbacks=[model_checkpoint_callback],
        validation_data=tf.data.Dataset.zip((validation_lr, validation_hr))
    )


if __name__ == "__main__":
    main()
