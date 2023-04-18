import numpy as np
from keras.callbacks import Callback

from shared.plots import plot_metric_by_epoch

class MetricsCallback(Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.cs_train, self.cs_eval = [], []
        self.psnr_train, self.psnr_eval = [], []
        self.ssim_train, self.ssim_eval = [], []

    # Store the metric values in each epoch
    def on_epoch_begin(self, epoch, logs=None):
        self.cs_train_aux, self.cs_eval_aux = [], []
        self.psnr_train_aux, self.psnr_eval_aux = [], []
        self.ssim_train_aux, self.ssim_eval_aux = [], []


    def on_train_batch_end(self, batch, logs=None):
        self.cs_train_aux.append(logs["cs"])
        self.psnr_train_aux.append(logs["psnr"])
        self.ssim_train_aux.append(logs["ssim"])


    def on_test_batch_end(self, batch, logs=None):
        self.cs_eval_aux.append(logs["cs"])
        self.psnr_eval_aux.append(logs["psnr"])
        self.ssim_eval_aux.append(logs["ssim"])


    def on_epoch_end(self, epoch, logs=None):
        self.cs_train.append(np.mean(self.cs_train_aux))
        self.cs_eval.append(np.mean(self.cs_eval_aux))
        self.psnr_train.append(np.mean(self.psnr_train_aux))
        self.psnr_eval.append(np.mean(self.psnr_eval_aux))
        self.ssim_train.append(np.mean(self.ssim_train_aux))
        self.ssim_eval.append(np.mean(self.ssim_eval_aux))


    def on_train_end(self, logs=None):
        plot_metric_by_epoch(self.path, "Cosine Similarity", self.cs_train, self.cs_eval)
        plot_metric_by_epoch(self.path, "PSNR", self.psnr_train, self.psnr_eval)
        plot_metric_by_epoch(self.path, "SSIM", self.ssim_train, self.ssim_eval)