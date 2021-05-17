import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.layers.experimental import preprocessing

from tools import plot_loss, plot_model


def get_optimizer(name, alpha):
    if name == 'momentum':
        # return tf.optimizers.SGD(learning_rate=alpha)
        return tf.optimizers.SGD()
    elif name == 'adagrad':
        # return tf.optimizers.Adagrad(learning_rate=alpha)
        return tf.optimizers.Adagrad()
    elif name == 'adadelta':
        # return tf.optimizers.Adadelta(learning_rate=alpha)
        return tf.optimizers.Adadelta()
    elif name == 'adam':
        # return tf.optimizers.Adam(learning_rate=alpha)
        return tf.optimizers.Adam()


def apply_models(x, x_label, y, size, test_results, itr, err, alpha, reg, opt, dir):
    # Split data
    train_x = x[:size]
    test_x = x[size:]
    train_y = y[:size]
    test_y = y[size:]

    model = np.array(train_x)

    model_normalizer = preprocessing.Normalization()
    model_normalizer.adapt(model)

    # Non Linear Polynomial
    poly = PolynomialFeatures(degree=3)
    train_x_n = poly.fit_transform(model)

    model_model = tf.keras.Sequential([
        # model_normalizer,

        # Linear Regression
        # layers.Dense(units=1, kernel_regularizer=reg)

        # Non Linear Polynomial
        layers.Dense(units=1, kernel_regularizer=reg, input_shape=[20])
    ])

    model_model.compile(
        optimizer=get_optimizer(opt, alpha),
        loss=err)

    history = model_model.fit(
        train_x_n, train_y,
        epochs=itr,
        verbose=0,
        validation_split=0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv(dir + '/loss_history.csv')

    plot_loss(history, dir + '/loss_history.png')

    test_results[dir] = model_model.evaluate(
        # test_x,
        poly.fit_transform(np.array(test_x)),
        test_y, verbose=0)

    # covid_y_p = model_model.predict(x)
    # plot_model([x_label[x_i] for x_i in x['date_num']], y, covid_y_p, dir + '/prediction.png')
