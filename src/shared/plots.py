import numpy as np
import matplotlib.pyplot as plt

from keras.utils import array_to_img

def plot_metric_by_epoch(path, metric_name, test_data, eval_data):
    plt.clf()
    plt.plot(test_data)
    plt.plot(eval_data)
    plt.title(f"{metric_name}")
    plt.ylabel("Metric")
    plt.xlabel("Epoch")
    plt.xticks(range(len(test_data)), range(1, len(test_data)+1))
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{path}/{metric_name}.png")

def plot_loss_curve(path, history, metric_name):
    plt.clf()
    plt.plot(history.history[metric_name])
    plt.plot(history.history[f"val_{metric_name}"])
    plt.title(f"{metric_name}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{path}/loss_curve_{metric_name}.png")


def plot_test_dataset(path, type, generator, dataset):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    for idx, img in enumerate(dataset):
        prediction = generator(img)
        img = np.array(img.numpy())
        img = img[0, :, :, :]
        img = array_to_img(img)

        prediction = np.array(prediction.numpy())
        prediction = prediction[0, :, :, :]
        prediction = array_to_img(prediction)

        ax1.imshow(img)
        ax2.imshow(prediction)
        ax1.set_title(f"Input {type} image")
        ax2.set_title("Translated image")
        ax1.axis("off")
        ax2.axis("off")

        plt.savefig(f"{path}/test_{type}_{idx}.png")
