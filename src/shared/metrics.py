import numpy as np
from keras.callbacks import Callback

from .plots import plot_metric_by_epoch

class NetworkMetricsPlotCallback(Callback):
    def __init__(self, path, keys):
        super().__init__()
        self.path = path
        self.keys = keys

        self.vals = {}
        self.vals_aux = {}
        for key in self.keys:
            self.vals[key + "_train"] = []
            self.vals[key + "_val"] = []


    # Store the metric values in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        for key in self.keys:
            self.vals_aux[key + "_train"] = []
            self.vals_aux[key + "_val"] = []


    def on_train_batch_end(self, batch, logs=None):
        for key in self.keys:
            self.vals_aux[key + "_train"].append(logs[key])


    def on_test_batch_end(self, batch, logs=None):
        for key in self.keys:
            self.vals_aux[key + "_val"].append(logs[key])


    def on_epoch_end(self, epoch, logs=None):
        for key in self.keys:
            self.vals[key + "_train"].append(np.mean(self.vals_aux[key + "_train"]))
            self.vals[key + "_val"].append(np.mean(self.vals_aux[key + "_val"]))


    def on_train_end(self, logs=None):
        for key in self.keys:
            plot_metric_by_epoch(self.path, "Loss " + key, self.vals[key + "_train"], self.vals[key + "_val"])