import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors


def plot_loss(losses, out):
    if len(losses) > 2:
        plt.figure(figsize=[8, 4])
        xx = np.array(range(len(losses))) + 1
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xlim([min(xx), max(xx)])
        plt.ylim([min(losses), max(losses)])
        plt.plot(xx, losses)
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()


def plot_latent(xx, yy, labels, out):
    df = pd.DataFrame((xx, yy, labels), columns=['x', 'y', 'label'])
    x_mean = np.mean(xx)
    x_sigma = 3. * np.std(xx)
    y_mean = np.mean(yy)
    y_sigma = 3. * np.std(yy)
    labels = list(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, label in enumerate(labels):
        idf = df[df['label'] == label]
        x = idf['x'].to_numpy()
        y = idf['y'].to_numpy()
        color = mcolors.rgb2hex(cmap(i))
        plt.scatter(x, y,
                    c=color,
                    s=5.0,
                    label=label)
    plt.xlim(x_mean - x_sigma, x_mean + x_sigma)
    plt.ylim(y_mean - y_sigma, y_mean + y_sigma)
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper right',
               borderaxespad=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
