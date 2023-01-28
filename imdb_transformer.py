import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import argparse
from tensorflow_addons.optimizers import AdamW
from transformer import (
    TokenAndPositionEmbedding,
    TransformerBlock,
)

vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review


class LambdaSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_value, step_fn):
        self.initial_value = initial_value
        self.step_fn = step_fn

    def __call__(self, step):
        return self.initial_value * self.step_fn(step)


def train_transformer_imdb(config, verbose=False):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(1234)

    embedding_dim = config["dim"] * config["num_heads"]

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(
        maxlen,
        vocab_size,
        embedding_dim,
        config["embedding_norm"],
        config["embedding_dr"],
    )
    x = embedding_layer(inputs)
    for i in range(config["num_layers"]):
        x = TransformerBlock(
            embedding_dim,
            config["num_heads"],
            config["ff_dim"],
            dr=config["dr"],
        )(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    num_epochs = 20
    steps_per_epoch = 391
    decay_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        [
            config["warumup_steps"],
            config["warumup_steps"] + config["decay_epochs"] * steps_per_epoch,
            config["warumup_steps"] + 2 * config["decay_epochs"] * steps_per_epoch,
            config["warumup_steps"] + 3 * config["decay_epochs"] * steps_per_epoch,
        ],
        [
            0.1,
            1.0,
            config["decay_rate"],
            config["decay_rate"] ** 2,
            config["decay_rate"] ** 3,
        ],
    )
    learning_rate_fn = LambdaSchedule(config["lr"], decay_fn)
    weight_decay_fn = LambdaSchedule(config["wd"], decay_fn)

    optimizer = AdamW(learning_rate=learning_rate_fn, weight_decay=weight_decay_fn)
    model.compile(optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    hist = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=num_epochs,
        verbose=1 if verbose else 0,
        validation_data=(x_val, y_val) if verbose else None,
    )
    _, val_acc = model.evaluate(x_val, y_val)
    return val_acc


if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(
        num_words=vocab_size
    )
    # base_config = {
    #     "ff_dim": 256,
    #     "num_heads": 4,
    #     "dim": 32,
    #     "warumup_steps": 1000,
    #     "decay_epochs": 10,
    #     "decay_rate": 0.5,
    #     "lr": 1e-4,
    #     "wd": 1e-6,
    #     "dr": 0.1,
    #     "num_layers": 2,
    #     "embedding_norm": False,
    #     "embedding_dr": 0.1,
    # }
    # train_transformer_imdb(base_config, verbose=True)