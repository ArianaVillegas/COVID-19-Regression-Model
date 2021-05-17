import os

from matplotlib import pyplot as plt


def plot_loss(history, filename):
    f = plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close(f)


def plot_model(x, y, y_p, filename):
    f = plt.figure()
    plt.scatter(x, y, label='Data')
    plt.plot(x, y_p, color='k', label='Predictions')
    plt.xlabel('Fecha')
    plt.ylabel('Número de personas')
    plt.legend()
    plt.savefig(filename)
    plt.close(f)


def split_data(df, column):
    unique_values = list(set(df[column]))
    list_df = [df[df[column] == value] for value in unique_values]
    return list_df, unique_values


