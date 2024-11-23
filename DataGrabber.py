import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from pathlib import Path

class ExtractIntermediateOutputs(Callback):
    def __init__(self, model, epochs_to_extract, layer_names, x_train, save_location, sweep_run):
        super(ExtractIntermediateOutputs, self).__init__()
        self.model = model
        self.epochs_to_extract = epochs_to_extract
        self.layer_names = layer_names
        self.x_train = x_train
        self.save_location = save_location
        self.sweep_run = sweep_run
        wandb.init()
        wandb.define_metric("tc", summary='min')
        wandb.define_metric("Metric2", summary='min')

    def on_epoch_end(self, epoch, logs=None):
        if epoch in self.epochs_to_extract:
            Path(f"data/{self.sweep_run}/epoch{epoch}/").mkdir(parents=True, exist_ok=True)
            for layer in range(len(self.layer_names)):
                intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.layer_names[layer]).output)
                intermediate_output= intermediate_model.predict(self.x_train)
                print(type(intermediate_output))
                with open(f"data/{self.sweep_run}/epoch{epoch}/layer{layer}.npy", "wb") as datafile:
                    np.save(datafile, intermediate_output)
