import tensorflow as tf
from keras import Model
from keras.layers import Subtract
from keras.optimizers import Adam

from generator import get_generator
from discriminator import get_discriminator

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
            sr_discriminated = self.discriminator(sr_image, training=True)
            hr_discriminated = self.discriminator(hr_image, training=True)
            diff = Subtract()[sr_discriminated, hr_discriminated]
            print(diff.shape)
            exit(0)

def main():
    # Get models
    generator = get_generator(input_shape=(96, 96, 3))
    discriminator = get_discriminator(input_shape=(96, 96, 3))
   
    # Create gan model
    spgan = SPGAN(generator, discriminator)

    # Compile the model
    spgan.compile(
        generator_optimizer=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999),
        discriminator_optimizer=Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.999),
    )

    # Loads the dataset and splits
    train, validation, test = get_dataset_split("../../datasets/FEI/", 0.6, 0.2, (360,260))
    train_sr, validation_sr, test_sr = manipulate_dataset(train, validation, test)
    train_lr, validation_lr, test_lr = manipulate_dataset(train, validation, test, apply_bicubic=True)


if __name__ == "__main__":
    main()
