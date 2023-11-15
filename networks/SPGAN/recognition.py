import sys

import tensorflow as tf
from keras import Model
from keras.optimizers import Adam

sys.path.append("..")
from face_recognition import get_network
from common.utils import get_parser
from common.data import get_dataset_split

class FaceRecNet(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def compile(self, optimizer):
        super().compile(optimizer=optimizer)
        self.optimizer: Adam = optimizer

    def train_step(self, batch_data):
        image = batch_data[0]

        with tf.GradientTape(persistent=True) as tape:
            # Finish implementation
            loss = 0

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


# Trains the face recognition model
def main():
    args = get_parser()

    # Loads the dataset and splits
    hr_shape = (216, 176, 3)
    train, _, _ = get_dataset_split(args.path, (hr_shape[0], hr_shape[1]), 0.6, 0.2, 0.2, batch_size=1)

    # Get models
    model = get_network(input_shape=hr_shape)
    net = FaceRecNet(model=model)
    
    net.compile(
        optimizer=Adam(learning_rate=1e-4),
    )

    net.fit(
        x=train,
        epochs=100,
    )


if __name__ == "__main__":
    main()