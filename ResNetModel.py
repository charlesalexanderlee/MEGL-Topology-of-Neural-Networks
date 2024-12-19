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

# ResNet Architecture:
def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x

    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def convolutional_block(x, filter):
    #copy tensor to variable called x_skip
    x_skip = x

    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)

    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def ResNet34(shape = (32, 32, 3), classes = 10):
    # Step 1 (Setup Input Layer)
    x_input = tf.keras.layers.Input(shape)
    x = tf.keras.layers.ZeroPadding2D((3,3))(x_input)

    # Step 2 (Initial Conv layer along with maxPool)
    x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64

    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            #For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)

        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] -1):
                x = identity_block(x, filter_size)

    # Step 4 End Dense Network
    x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation = 'relu', name='f0')(x)
    x = tf.keras.layers.Dense(256, activation = 'relu', name='f1')(x)
    x = tf.keras.layers.Dense(128, activation = 'relu', name='f2')(x)
    x = tf.keras.layers.Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model


def train():
    # Start a run, tracking hyperparameters
    wandb.init(
    # track hyperparameters and run metadata with wandb.config
        config={
            # ========== LAYER HYPERPARAMETERS ==========
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
    epochs_to_extract = [0, 2, 4, 6, 8, 10]
    save_location = "cifar10_test"

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    ''' ========= INSTANTIATE CIFAR10 MODEL ========= '''
    cifar10_model = ResNet34(shape = (32, 32, 3), classes = 10)
    ''' ============================================= '''

    # Optimization
    early_stopping = EarlyStopping(monitor='val_accuracy', restore_best_weights=True)
    cifar10_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    cb = [ExtractIntermediateOutputs(cifar10_model, epochs_to_extract, layer_names, x_train, save_location, directory), WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")]
    # =============== Start Model Training ===============
    results = cifar10_model.fit(
        x_train, y_train, 
        epochs=11, 
        validation_split=0.1, # 20% of training data used for validation
        callbacks=cb)

    # ============ After model has been trained ============

    # Evaluate on test data

    test_loss, test_accuracy = cifar10_model.evaluate(x_test, y_test, verbose=0)


    # Evaluate on test data and log
    test_loss, test_accuracy = cifar10_model.evaluate(x_test, y_test)
    final_train_accuracy = results.history["acc"][-1]
    final_val_accuracy = results.history["val_acc"][-1]

    # Log all metrics to W&B
    wandb.log({
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss
    })
    wandb.finish()


directory = f"sweep{6}"
sweep_config = {
    "name": f"ResNetArchitecture{6}",
    'method': "random",
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize',
    },
    'parameters': {
        "optimizer": {
            "values": ['adam']
        },
    },
}
sweep_id = wandb.sweep(sweep_config, project="Jo-500-Models-ResNet-ArchitectureFORACTUALDATA")
wandb.agent(sweep_id, train, count=1)
