import sys

import numpy as np
import tensorflow as tf
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from facenet import get_embeddings
from generators import get_model_G, get_model_F
from utils import charbonnier_loss, mae_loss, mse_from_embedding

sys.path.append("..")
from shared.metrics import NetworkMetricsPlotCallback
from shared.utils import get_parser
from shared.plots import plot_test_dataset
from shared.data import get_dataset_split, manipulate_dataset, get_normalization_layer


# Reference: https://keras.io/examples/generative/cyclegan/#build-the-cyclegan-model
class DIDnet(Model):
    def __init__(self, generator_G, generator_F):
        super().__init__()
        self.gen_G: Model = generator_G
        self.gen_F: Model = generator_F

    def compile(self, optimizer, gen_G_optimizer, gen_F_optimizer,
        cycle_loss_fn, identity_loss_fn, train_size,
    ):
        super().compile()
        self.optimizer: Adam = optimizer
        self.gen_G_optimizer: callable = gen_G_optimizer
        self.gen_F_optimizer: callable = gen_F_optimizer
        self.cycle_loss_fn: callable = cycle_loss_fn
        self.identity_loss_fn: callable = identity_loss_fn
        self.train_size: int = train_size

    def get_metrics(self, real_x, real_y, isTraining=False):
        # LR to SR
        fake_y = self.gen_G(real_x, training=isTraining)
        # SR to LR
        fake_x = self.gen_F(real_y, training=isTraining)

        # Cycle (LR to SR to LR): x -> y -> x
        cycled_x = self.gen_F(fake_y, training=isTraining)
        # Cycle (SR to LR to SR) y -> x -> y
        cycled_y = self.gen_G(fake_x, training=isTraining)

        # Generator cycle loss
        cycle_loss_G = self.cycle_loss_fn(real_x, cycled_x) / self.train_size
        cycle_loss_F = self.cycle_loss_fn(real_y, cycled_y) / self.train_size

        # Get face embeddings from Facenet
        id_embeddings_G = get_embeddings(fake_x, cycled_x)
        id_embeddings_F = get_embeddings(fake_y, cycled_y)

        id_loss_G = self.identity_loss_fn(
            id_embeddings_G[0], id_embeddings_G[1]) / self.train_size
        id_loss_F = self.identity_loss_fn(
            id_embeddings_F[0], id_embeddings_F[1]) / self.train_size

        # Total generator loss
        total_loss_G = cycle_loss_G + id_loss_G
        total_loss_F = cycle_loss_F + id_loss_F

        # Network Loss
        loop_loss = cycle_loss_G + cycle_loss_F
        identity_loss = id_loss_G + id_loss_F
        rec_loss = mae_loss(real_y, fake_y) / self.train_size
        channel_loss = charbonnier_loss(tf.image.rgb_to_yuv(fake_y), tf.image.rgb_to_yuv(real_y)) / self.train_size
        network_loss = loop_loss + rec_loss + 0.5*identity_loss + 0.5*channel_loss

        # Computes other metrics
        psnr = tf.image.psnr(real_y, fake_y, max_val=255)
        ssim = tf.image.ssim(real_y, fake_y, max_val=255)
        cs = tf.keras.losses.cosine_similarity(real_y, fake_y)

        return {
            "g_cycle_loss": cycle_loss_G, "f_cycle_loss": cycle_loss_F, "g_id_loss": id_loss_G,
            "f_id_loss": id_loss_F, "loop_loss": loop_loss, "id_loss": identity_loss, "rec_loss": rec_loss,
            "channel_loss": channel_loss, "g_loss": total_loss_G, "f_loss": total_loss_F,
            "network_loss": network_loss, "psnr": psnr, "ssim": ssim, "cs": cs
        }


    def train_step(self, batch_data):
        # x is LR and y is HR
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            metrics = self.get_metrics(real_x, real_y, isTraining=True)

        # Gets the gradients for the generators and optimizes them
        grads_G = tape.gradient(metrics["g_loss"], self.gen_G.trainable_variables)
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )

        grads_F = tape.gradient(metrics["f_loss"], self.gen_F.trainable_variables)
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Gets the gradients for the whole network and optimizes it
        grads_network = tape.gradient(metrics["network_loss"], self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads_network, self.trainable_variables))

        return { "network_loss": metrics["network_loss"] }
    
    def test_step(self, batch_data):
        # x is LR and y is SR
        real_x, real_y = batch_data

        # Compute metrics
        metrics = self.get_metrics(real_x, real_y)

        return { "network_loss": metrics["network_loss"] }
    
    def evaluate_test_datasets(self, path, lr_dataset, sr_dataset, normalized = False):
        plot_test_dataset(path, "LR", self.gen_G, lr_dataset, normalized=normalized)
        plot_test_dataset(path, "HR", self.gen_F, sr_dataset, normalized=normalized)


def main():
    args = get_parser()

    # Loads the dataset and splits
    lr_shape = (90, 65, 3)
    hr_shape = (360, 260, 3)
    train, validation, test = get_dataset_split(args.path, (hr_shape[0], hr_shape[1]), 0.88, 0.02, 0.1)
    train_hr, validation_hr, test_hr = manipulate_dataset(train, validation, test)
    train_lr, validation_lr, test_lr = manipulate_dataset(train, validation, test, resize=True, resize_shape=(lr_shape[0], lr_shape[1]))

    # Get models
    norm_layer = None
    normalize = False
    if normalize:
        norm_layer = get_normalization_layer(train)

    generator_g = get_model_G(input_shape=lr_shape, norm_layer=norm_layer)
    generator_f = get_model_F(input_shape=hr_shape, norm_layer=norm_layer)

    # Create cycle gan model
    didnet = DIDnet(generator_g, generator_f)

    # Compile the model
    didnet.compile(
        optimizer=Adam(learning_rate=1e-4),
        gen_G_optimizer=Adam(learning_rate=1e-4),
        gen_F_optimizer=Adam(learning_rate=1e-4),
        cycle_loss_fn=charbonnier_loss,
        identity_loss_fn=mse_from_embedding,
        train_size=len(train)
    )

    # Callbacks
    early_stop = EarlyStopping(monitor="network_loss", patience=10, mode="min")
    model_checkpoint_callback = ModelCheckpoint(
        filepath= "./checkpoints/didnet_checkpoints.{epoch:03d}", save_weights_only=True
    )
    net_metrics = NetworkMetricsPlotCallback(path="./results", metrics=["network_loss"])

    # Trains
    didnet.fit(
        x=tf.data.Dataset.zip((train_lr, train_hr)),
        epochs=100,
        callbacks=[model_checkpoint_callback, net_metrics],
        validation_data=tf.data.Dataset.zip((validation_lr, validation_hr))
    )

    # Plot the test datasets
    didnet.evaluate_test_datasets("./results", test_lr, test_hr, normalized=normalize)

if __name__ == "__main__":
    main()
