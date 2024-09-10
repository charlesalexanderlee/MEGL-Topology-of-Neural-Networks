# imports
import wandb
from constants import wandb_key
wandb.login(key=wandb_key)

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from enclosed_pipline import *

from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np

# Weights and Biases stuff
sweep_config = {
    'method': "random",
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize',
    },
    'parameters': {
        "optimizer": {
            "values": ['adam']
        },
        "l1": {
            "values": [64, 128, 256, 512]
        },
        "l2": {
            "values": [64, 128, 256, 512]
        },
        "l3": {
            "values": [64, 128, 256, 512]
        },
        "l4": {
            "values": [64, 128, 256, 512]
        },
        "l5": {
            "values": [64, 128, 256, 512]
        },
        "l6": {
            "values": [64, 128, 256, 512]
        },
    },
}
sweep_id = wandb.sweep(sweep_config, project="Cifar10_test")

def train():
    # Start a run, tracking hyperparameters
    wandb.init(
    # track hyperparameters and run metadata with wandb.config
        config={
            "l1": 128,
            "l2": 128, 
            "l3": 128, 
            "l4": 128,
            "l5": 128, 
            "l6": 128, 
            "optimizer": "adam", 
            "metric": "accuracy",
        }
    )

    # Use wandb.config as your config
    config = wandb.config

    activation_funct = 'relu'
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #variables needed for data extraction
    layer_names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    epochs_to_extract = [0, 9, 19, 29, 39, 49]
    save_location = "cifar10_test"

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model_input =  Input(shape=(32, 32, 3))


    x = Conv2D(8, (2,2), strides=2, activation=activation_funct, kernel_regularizer=l1_l2(l1=0.01, l2=0.1), padding='same') (model_input)
    x = BatchNormalization() (x)
    x = MaxPooling2D((3,3), padding='same') (x)
    x = Dropout(0.25) (x)

    x = Conv2D(16, (3,3), strides=2, activation=activation_funct, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), padding='same') (model_input)
    x = BatchNormalization() (x)
    x = MaxPooling2D((3,3), padding='same') (x)
    x = Dropout(0.25) (x)

    x = Flatten(name = "flatten")(x)

    x = Dense(config.l1, activation=activation_funct, name="f0") (x)
    x = Dense(config.l2, activation=activation_funct, name="f1") (x)
    x = Dense(config.l3, activation=activation_funct, name="f2") (x)
    x = Dense(config.l4, activation=activation_funct, name="f3") (x)
    x = Dense(config.l5, activation=activation_funct, name="f4") (x)
    x = Dense(config.l6, activation=activation_funct, name="f5") (x)

    x = Dropout(0.1) (x)

    output_layer = Dense(10, activation='softmax', name="out") (x)
    cifar10_model = Model(model_input, output_layer)

    # Optimization
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cifar10_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cb = [early_stopping, ExtractIntermediateOutputs(cifar10_model, epochs_to_extract, layer_names, x_train, save_location), WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")]
    results = cifar10_model.fit(x_train, y_train, epochs=50, callbacks=cb)
    cifar10_model.evaluate(x_test, y_test)

wandb.agent(sweep_id, train, count=1)
