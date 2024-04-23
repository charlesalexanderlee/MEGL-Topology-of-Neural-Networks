import wandb
wandb.login(key="d4050231a048ba552f448d9a5521b8456826d511")
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from enclosed_pipline import *

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
        "length" : {
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
    },
}
sweep_id = wandb.sweep(sweep_config, project="experiment2_cifar10_run3")

def train():
    # Start a run, tracking hyperparameters
    wandb.init(
    # track hyperparameters and run metadata with wandb.config
        config={
            "optimizer": "adam", 
            "metric": "accuracy",
            "length": 1
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
    save_location = "experiment2_cifar10_run3_length" + str(config.length)

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model_input =  Input(shape=(32, 32, 3))

    x = Conv2D(8, (2,2), strides=2, activation=activation_funct, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.1), padding='same') (model_input)
    x = BatchNormalization() (x)
    x = MaxPooling2D((3,3), padding='same') (x)
    x = Dropout(0.25) (x)

    x = Conv2D(16, (3,3), strides=2, activation=activation_funct, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01), padding='same') (model_input)
    x = BatchNormalization() (x)
    x = MaxPooling2D((3,3), padding='same') (x)
    x = Dropout(0.25) (x)

    x = Flatten(name = "flatten")(model_input)
    x = Dense(512, activation=activation_funct, name="f0") (x)
    x = Dense(512, activation=activation_funct, name="f1") (x)
    x = Dense(256, activation=activation_funct, name="f2") (x)
    x = Dense(128, activation=activation_funct, name="f3") (x)
    x = Dense(64, activation=activation_funct, name="f4") (x)
    x = Dense(32, activation=activation_funct, name="f5") (x)

    for layer in range(config.length):
        layer_names.append("f" +  str(layer + 6))
        x = Dense(128, activation=activation_funct, name="f" +  str(layer + 6)) (x)

    x = Dropout(0.25) (x)

    output_layer = Dense(10, activation='softmax', name="out") (x)
    cifar10_model = Model(model_input, output_layer)

    # Optimization
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cifar10_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cb = [early_stopping, ExtractIntermediateOutputs(cifar10_model, epochs_to_extract, layer_names, x_train, save_location), WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")]
    results = cifar10_model.fit(x_train, y_train, epochs=50, callbacks=cb)
    cifar10_model.evaluate(x_test, y_test)

wandb.agent(sweep_id, train, count=50)
