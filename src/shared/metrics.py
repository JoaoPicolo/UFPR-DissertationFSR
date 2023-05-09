import numpy as np
from keras.callbacks import Callback

from .plots import plot_metric_by_epoch

class NetworkMetricsPlotCallback(Callback):
    def __init__(self, path, metrics):
        super().__init__()
        self.path = path
        self.metrics = metrics

        self.data = {}
        self.data_aux = {}
        for key in self.metrics:
            self.data[key + "_train"] = []
            self.data[key + "_val"] = []


    # Store the metric values in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        for key in self.metrics:
            self.data_aux[key + "_train"] = []
            self.data_aux[key + "_val"] = []


    def on_train_batch_end(self, batch, logs=None):
        for key in self.metrics:
            self.data_aux[key + "_train"].append(logs[key])


    def on_test_batch_end(self, batch, logs=None):
        for key in self.metrics:
            self.data_aux[key + "_val"].append(logs[key])


    def on_epoch_end(self, epoch, logs=None):
        for key in self.metrics:
            self.data[key + "_train"].append(np.mean(self.data_aux[key + "_train"]))
            self.data[key + "_val"].append(np.mean(self.data_aux[key + "_val"]))


    def on_train_end(self, logs=None):
        for key in self.metrics:
            plot_metric_by_epoch(self.path, "Loss " + key, self.data[key + "_train"], self.data[key + "_val"])