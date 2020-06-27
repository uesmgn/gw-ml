import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import matplotlib.colors as mcolors

from ..helper import *

__all__ = [
    'bar',
    'plot',
    'scatter',
    'plot_confusion_matrix'
]


def bar(input, out, reverse=True, **kwargs):
    input = check_array(input)
    counter = np.array([[label, np.count_nonzero(
        np.array(input) == label)] for label in list(set(input))])
    x_labels, yy = counter[:, 0].astype(str), counter[:, 1].astype(np.int32)
    idx = np.argsort(yy)
    if reverse:
        idx = idx[::-1]
    x_labels, yy = x_labels[idx], yy[idx]
    x_position = np.arange(len(x_labels))
    _init_plot(**kwargs)
    plt.bar(x_position, yy, tick_label=x_labels)
    _setup_plot(**kwargs)
    xticklabels = plt.gca().get_xticklabels()
    plt.setp(xticklabels, rotation=45, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot(data, out, **kwargs):
    data = check_array(data)
    if data.ndim is 1:
        yy = data.astype(np.float)
        xx = np.array(range(yy.size))
    elif data.ndim is 2:
        xx = data[:, 0]
        yy = data[:, 1].astype(np.float)
    else:
        raise ValueError('Invalid data format')
    if len(xx) > 1:
        _init_plot(**kwargs)
        plt.plot(xx, yy)
        _setup_plot(**kwargs)
        if kwargs.get('fit_x'):
            plt.xlim(min(xx), max(xx))
        if kwargs.get('fit_y'):
            plt.ylim(min(yy), max(yy))
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def scatter(xx, yy, labels, out, **kwargs):
    xx, yy, labels = check_array(
        xx, yy, labels, check_size=True, dtype=[np.float, np.float, np.str])
    labels_unique = check_array(labels, unique=True, sort=True, dtype=np.str)
    cmap = plt.get_cmap("tab20")
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(20)]
    colors = np.tile(colors, 3)
    _init_plot(**kwargs)
    for i, label in enumerate(labels_unique):
        idx = np.where(labels==label)
        x = xx[idx]
        y = yy[idx]
        label = acronym(label)
        color = colors[i]
        plt.scatter(x, y, c=color, s=8.0, label=label)
    _setup_plot(**kwargs)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
               borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_confusion_matrix(cm, xlabels, ylabels, out, **kwargs):
    cmap = plt.get_cmap('Blues')
    _init_plot(**kwargs)
    plt.imshow(cm.T, interpolation='nearest',
                         cmap=cmap,
                         origin='lower')
    ax = plt.gca()
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(len(xlabels)), range(len(ylabels))):
        num = "{:0.2f}".format(cm[i, j])
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(i, j, num, fontsize=10, color=color, ha='center', va='center')
    _setup_plot(**kwargs)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _init_plot(**kwargs):
    for k, v in kwargs.items():
        if k is 'figsize':
            plt.figure(figsize=v)


def _setup_plot(**kwargs):
    for k, v in kwargs.items():
        if k is 'xlabel':
            plt.xlabel(v)
        elif k is 'ylabel':
            plt.ylabel(v)
        elif k is 'title':
            plt.title(v)
        elif k is 'xlim':
            plt.xlim(v)
        elif k is 'ylim':
            plt.ylim(v)
        elif k is 'xmin':
            plt.gca().set_xlim(left=v)
        elif k is 'xmax':
            plt.gca().set_xlim(right=v)
        elif k is 'ymin':
            plt.gca().set_ylim(bottom=v)
        elif k is 'ymax':
            plt.gca().set_ylim(top=v)
