import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import itertools


def plot(xy, out, xlabel=None, ylabel=None,
         xlim=None, ylim=None):
    xy = np.array(xy)
    if xy.shape[0] > 2:
        xx = xy[:,0]
        yy = xy[:,1]
        plt.figure(figsize=[8, 4])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if type(xlim) in (tuple, list):
            plt.xlim([xlim[0], xlim[1]])
        else:
            plt.xlim([min(xx), max(xx)])
        if type(ylim) in (tuple, list):
            plt.ylim([ylim[0], ylim[1]])
        else:
            margin = (max(yy) - min(yy) + 0.1) / 20.
            plt.ylim([min(yy)-margin, max(yy)+margin])
        plt.plot(xx, yy)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def scatter(xx, yy, labels, out):
    df = pd.DataFrame((xx, yy, labels), index=['x', 'y', 'label']).T
    x_mean = np.mean(xx)
    x_sigma = 4. * np.std(xx)
    y_mean = np.mean(yy)
    y_sigma = 4. * np.std(yy)
    labels = sorted(list(set(labels)))
    cmap = plt.get_cmap("tab20")
    for i, label in enumerate(labels):
        idf = df[df['label'] == label]
        x = idf['x'].to_numpy()
        y = idf['y'].to_numpy()
        color = mcolors.rgb2hex(cmap(i))
        plt.scatter(x, y,
                    c=color,
                    s=8.0,
                    label=label)
    plt.xlim(x_mean - x_sigma, x_mean + x_sigma)
    plt.ylim(y_mean - y_sigma, y_mean + y_sigma)
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right',
               borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def cmshow(cm, index, columns, out, title=None):

    cmap = plt.get_cmap('Blues')

    width = 10
    height = width * len(index) / len(columns)
    margin = (max([len(str(i)) for i in index]) * 0.2,
              max([len(str(c)) for c in columns]) * 0.2)
    figsize = (width + margin[0], height + margin[1])

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest',
                   cmap=cmap,
                   aspect=0.7,
                   origin='lower')
    ax.set_xlim(None, None)
    ax.set_ylim(None, None)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(index)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(columns)
    ax.set_yticklabels(index)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 1.5
    # Loop over data dimensions and create text annotations.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = "{:0.2f}".format(cm[i, j])
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(j, i, num,
                fontsize=10, color=color,
                ha='center', va='center')

    if title:
        ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(out)
    plt.close()

def bar(xx, yy, out, reverse=False):
    d = np.array(list(zip(xx, yy)))
    d = np.array(sorted(d, key=lambda x: int(x[1]), reverse=reverse))
    xx = d[:,0]
    yy = d[:,1].astype(int)
    x_position = np.arange(len(xx))
    fig, ax = plt.subplots()
    ax.bar(x_position, yy, tick_label=xx)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(out)
    plt.close()
