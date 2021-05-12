import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from tools import plot_loss, plot_model


def get_optimizer(name, alpha):
    if name == 'momentum':
        return tf.optimizers.SGD(learning_rate=alpha)
    elif name == 'adagrad':
        return tf.optimizers.Adagrad(learning_rate=alpha)
    elif name == 'adadelta':
        return tf.optimizers.Adadelta(learning_rate=alpha)
    elif name == 'adam':
        return tf.optimizers.Adam(learning_rate=alpha)


def apply_models(x, y, size, test_results, itr, err, alpha, reg, opt, dir):
    # Split data
    train_x = x[:size]
    test_x = x[size:]
    train_y = y[:size]
    test_y = y[size:]

    model = np.array(train_x['date'])

    model_normalizer = preprocessing.Normalization(input_shape=[1, ])
    model_normalizer.adapt(model)

    model_model = tf.keras.Sequential([
        model_normalizer,
        layers.Dense(units=1, kernel_regularizer=reg)
    ])

    model_model.compile(
        optimizer=get_optimizer(opt, alpha),
        loss=err)

    history = model_model.fit(
        train_x['date'], train_y,
        epochs=itr,
        verbose=0,
        validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv(dir + '/loss_history.csv')

    plot_loss(history, dir + '/loss_history.png')

    test_results[dir] = model_model.evaluate(
        test_x['date'],
        test_y, verbose=0)

    covid_y_p = model_model.predict(x)

    plot_model(x, y, covid_y_p, dir + '/prediction.png')
