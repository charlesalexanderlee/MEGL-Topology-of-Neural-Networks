import tensorflow as tf
import numpy as np
import wandb

from constants import wandb_key
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from enclosed_pipline import *

from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

wandb.login(key=wandb_key)

# Define the sweep configuration
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
    # Initialize WandB
    wandb.init(
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
    config = wandb.config

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Convert data to tf.data.Dataset and apply caching and prefetching
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Optimize with caching and prefetching
    batch_size = 256
    train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    # Use `tf.distribute.MirroredStrategy` for multi-GPU support
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Define the model architecture inside the strategy's scope
        model_input = Input(shape=(32, 32, 3))
        x = Conv2D(8, (2, 2), strides=2, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.1), padding='same')(model_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), padding='same')(x)
        x = Dropout(0.25)(x)

        x = Conv2D(16, (3, 3), strides=2, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), padding='same')(x)
        x = Dropout(0.25)(x)

        x = Flatten(name="flatten")(x)
        x = Dense(config.l1, activation='relu', name="f0")(x)
        x = Dense(config.l2, activation='relu', name="f1")(x)
        x = Dense(config.l3, activation='relu', name="f2")(x)
        x = Dense(config.l4, activation='relu', name="f3")(x)
        x = Dense(config.l5, activation='relu', name="f4")(x)
        x = Dense(config.l6, activation='relu', name="f5")(x)
        x = Dropout(0.1)(x)

        output_layer = Dense(10, activation='softmax', name="out")(x)
        cifar10_model = Model(model_input, output_layer)

        # Compile the model
        cifar10_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks and Training
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cb = [
        early_stopping,
        ExtractIntermediateOutputs(cifar10_model, [0, 9, 19, 29, 39, 49], ["f0", "f1", "f2", "f3", "f4", "f5"], x_train, "cifar10_test"),
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint("models")
    ]

    # Train the model using the optimized datasets
    results = cifar10_model.fit(train_dataset, epochs=50, callbacks=cb, validation_data=test_dataset)
    cifar10_model.evaluate(test_dataset)

# Run the WandB agent
wandb.agent(sweep_id, train, count=1)
