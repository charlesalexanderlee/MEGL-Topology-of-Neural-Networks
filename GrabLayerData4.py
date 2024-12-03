# imports
import wandb
#from constants import wandb_key
wandb.login(key="a1418080c9100b24c9431beda05cc3730cf801d8")

from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from DataGrabber import *

from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np

# AlexNet:


def train():
    # Start a run, tracking hyperparameters
    wandb.init(
    # track hyperparameters and run metadata with wandb.config
        config={
            # ========== LAYER HYPERPARAMETERS ==========
            "L1": 512,
            "L2": 256, 
            "L3": 128, 
            "L4": 64,
            "L5": 64,
            "L6": 64,
            "batch_size": 128,
            "patience": 5,
            "optimizer": "adam", 
            # ============================================
            "metric": "val_acc",
        }
    )

    # Use wandb.config as your config
    config = wandb.config

    activation_funct = 'relu'

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    #variables needed for data extraction
    layer_names = ["f0", "f1", "f2", "f3", "f4", "f5"]
    epochs_to_extract = [0, 9, 19, 29, 39, 49]
    save_location = "Cifar10_Test"

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # model_input =  Input(shape=(32, 32, 3))



    # x = Conv2D(8, (2,2), strides=2, activation=activation_funct, kernel_regularizer=l1_l2(l1=config.l1, l2=config.l2), padding='same') (model_input)
    # x = BatchNormalization() (x)
    # x = MaxPooling2D((3,3), padding='same') (x)
    # x = Dropout(0.25) (x)

    # x = Conv2D(16, (3,3), strides=2, activation=activation_funct, kernel_regularizer=l1_l2(l1=config.l1, l2=config.l2), padding='same') (model_input)
    # x = BatchNormalization() (x)
    # x = MaxPooling2D((3,3), padding='same') (x)
    # x = Dropout(0.25) (x)

    # x = Flatten(name = "flatten")(x)

    # x = Dense(config.L1, activation=activation_funct, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), name="f0") (x)
    # x = Dense(config.L2, activation=activation_funct, name="f1") (x)
    # x = Dense(config.L3, activation=activation_funct, kernel_regularizer=l1_l2(l1=config.l12, l2=config.l22) ,name="f2") (x)
    # x = Dense(config.L4, activation=activation_funct, name="f3") (x)
    # x = Dense(config.L5, activation=activation_funct, kernel_regularizer=l1_l2(l1=config.l11,l2=config.l21), name="f4") (x)
    # x = Dense(config.L6, activation=activation_funct, name="f5") (x)

    # x = Dropout(0.1) (x)

    # output_layer = Dense(10, activation='softmax', name="out") (x)
    # cifar10_model = Model(model_input, output_layer)


    cifar10_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding='same', input_shape = x_train.shape[1:], activation = 'relu'),
        tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(config.L1, activation='relu', name='f0'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(config.L2, name='f1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(config.L3, name='f2', kernel_regularizer=l1_l2(l1=0.0001, l2=0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(config.L4, activation='relu', name='f3', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(config.L5, activation='relu', name='f4', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(config.L6, activation='relu', name='f5', kernel_regularizer=l1_l2(l1=0.001, l2=0.0001)),


        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    # Optimization
    # early_stopping = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)

    cb = [ExtractIntermediateOutputs(cifar10_model, epochs_to_extract, layer_names, x_train, save_location, directory), WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")]
    cifar10_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # =============== Start Model Training ===============
    results = cifar10_model.fit(
        x_train, y_train, 
        batch_size=config.batch_size,
        epochs=50, 
        validation_split=0.1, # 10% of training data used for validation
        callbacks=cb
        )

    # ============ After model has been trained ============

    # Evaluate on test data

    test_loss, test_accuracy = cifar10_model.evaluate(x_test, y_test, verbose=0)


    # Evaluate on test data and log
    test_loss, test_accuracy = cifar10_model.evaluate(x_test, y_test)
    final_train_accuracy = results.history["accuracy"][-1]
    final_val_accuracy = results.history["val_acc"][-1]

    # Log all metrics to W&B
    wandb.log({
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss
    })
    wandb.finish()

for sweep_num in range(10):
    directory = f"sweep{sweep_num}"
    sweep_config = {
    "name": directory,
    'method': "bayes",
    'metric': {
        'name': 'accuracy',
        'goal': 'maximize',
    },
    'parameters': {
        "optimizer": {
            "values": ['adam']
        },
        "L1": {
            "values": [256, 512]
        },
        "L2": {
            "values": [128, 256, 512]
        },
        "L3": {
            "values": [128, 256, 512]
        },
        "L4": {
            "values": [64, 128, 256]
        },
        "L5": {
            "values": [64, 128]
        },
        "L6": {
            "values": [64, 128]
        },
    },
}
    sweep_id = wandb.sweep(sweep_config, project="DataSavingTestFINALprobably")
    wandb.agent(sweep_id, train, count=1)
