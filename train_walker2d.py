# Adapted from # https://arxiv.org/abs/2006.04418
import os
import tensorflow as tf
from walker2d_data import Walker2dImitationData
from tensorflow_addons.optimizers import AdamW


class LambdaSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_value, step_fn):
        self.initial_value = initial_value
        self.step_fn = step_fn

    def __call__(self, step):
        return self.initial_value * self.step_fn(step)


def train_walker2d(config, verbose=False):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(1234)
    data = Walker2dImitationData(seq_len=64)

    signal_input = tf.keras.Input(shape=(data.seq_len, data.input_size))
    rnn = tf.keras.layers.LSTM(units=config["units"], return_sequences=True)
    output_states = rnn(signal_input)
    y = tf.keras.layers.Dense(data.output_size)(output_states)
    model = tf.keras.Model(inputs=signal_input, outputs=y)

    steps_per_epoch = 76
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

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
    )
    # model.summary()

    model.fit(
        x=data.train_x,
        y=data.train_y,
        batch_size=128,
        epochs=200,
        verbose=1 if verbose else 0,
        validation_data=(data.valid_x, data.valid_y) if verbose else None,
    )
    val_mse = model.evaluate(data.valid_x, data.valid_y, verbose=0)
    return val_mse


if __name__ == "__main__":
    train_walker2d(None, verbose=True)