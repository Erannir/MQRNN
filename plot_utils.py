import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch


# Taken from here: https://gist.github.com/salotz/8b4542d7fe9ea3e2eacc1a2eef2532c5
def move_axes(ax, fig, subplot_spec=111):
    """
    Move an Axes object from a figure to a new pyplot managed Figure in
    the specified subplot.
    :param ax: axes object
    :param fig: new figure to add given aces
    :param subplot_spec: subplot index
    """

    # get a reference to the old figure context so we can release it
    old_fig = ax.figure

    # remove the Axes from it's original Figure context
    ax.remove()

    # set the pointer from the Axes to the new figure
    ax.figure = fig

    # add the Axes to the registry of axes for the figure
    fig.axes.append(ax)
    # twice, I don't know why...
    fig.add_axes(ax)

    # then to actually show the Axes in the new figure we have to make
    # a subplot with the positions etc for the Axes to go, so make a
    # subplot which will have a dummy Axes
    dummy_ax = fig.add_subplot(subplot_spec)

    # then copy the relevant data from the dummy to the ax
    ax.set_position(dummy_ax.get_position())

    # then remove the dummy
    dummy_ax.remove()

    # close the figure the original axis was bound to
    plt.close(old_fig)


def plot_forecast(y_h, y_f, predictions, save=False, path=None):
    """
    Plots a history and forcast for data-series
    :param y_h: historical data
    :param y_f: actual future data
    :param predictions: quantiles prediction of future. includes q values for every y_f value
    :param save: boolean, indicating if to save plot to disk
    :param path: if save=True, path to save the plot
    :return:
    """
    len_hist = len(y_h)
    len_fct = len(y_f)
    forecast_range = np.arange(len_hist, len_hist + len_fct, 1)

    plt.plot(np.arange(0, len_hist + 1, 1), np.append(y_h, y_f[0]), label="history")
    plt.plot(forecast_range, y_f, label="actual", c="black")

    # defining color map
    num_quantiles = predictions.shape[-1]
    c = np.arange(num_quantiles + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    # filling the gaps between every two quantiles in different shade
    for j in range(num_quantiles - 1):
        plt.fill_between(forecast_range, predictions[:, j], predictions[:, j + 1], color=cmap.to_rgba(j + 1))

    plt.legend(loc=2)
    if save:
        plt.savefig(path)
    ax = plt.gca()
    plt.close()
    return ax