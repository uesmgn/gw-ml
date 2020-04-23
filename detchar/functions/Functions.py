import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Functions:

    @classmethod
    def plot_loss(cls, xx, losses, out, type=None):
        plt.figure(figsize=[8, 4])
        keys = losses.keys()
        y_median = 0
        for key in keys:
            yy = losses.get(key)
            plt.plot(xx, yy, label=key)
            if y_median < np.median(yy):
                y_median = np.median(yy)
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.xlim([min(xx), max(xx)])
        if type == 1:
            plt.ylim([0, y_median])
        plt.ylabel('loss')
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    @classmethod
    def plot_latent(cls, data, c, out):
        xx = data[:, 0]
        x_mean = np.mean(xx)
        x_sigma = 3. * np.std(xx)
        yy = data[:, 1]
        y_mean = np.mean(yy)
        y_sigma = 3. * np.std(yy)
        plt.scatter(xx, yy, c=c, cmap='tab20')
        plt.xlim(x_mean-x_sigma, x_mean+x_sigma)
        plt.ylim(y_mean-y_sigma, y_mean+y_sigma)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    @classmethod
    def plot_latent3d(cls, data, c, out):
        fig = plt.figure(figsize=[16,12])
        ax = fig.add_subplot(111, projection='3d')
        xx = data[:, 0]
        x_mean = np.mean(xx)
        x_sigma = 3. * np.std(xx)
        yy = data[:, 1]
        y_mean = np.mean(yy)
        y_sigma = 3. * np.std(yy)
        zz = data[:, 2]
        z_mean = np.mean(zz)
        z_sigma = 3. * np.std(zz)
        ax.scatter(xx, yy, zz, c=c, cmap='tab20')
        ax.set_xlim(x_mean-x_sigma, x_mean+x_sigma)
        ax.set_ylim(y_mean-y_sigma, y_mean+y_sigma)
        ax.set_zlim(z_mean-z_sigma, z_mean+z_sigma)
        fig.savefig(out)
        plt.close()

    @classmethod
    def plot_confusion_matrix(cls, cm, index, columns, out,
                              title='Confusion matrix', normalize=False):
        """
        Make plot of confusion matrix ``cm``.
        Parameters
        ----------
        cm: array-like
            confusion matrix
        index: array-like
            labels
        columns: array-like
            columns
        out: str
            output file path of plot
        title: str, default 'Confusion matrix'
            plotting title
        normalize: bool, default False
            normalize output matrix or not
        """

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
            num = "{:0.2f}".format(cm[i, j]) if normalize else int(cm[i, j])
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, num,
                    fontsize=10, color=color,
                    ha='center', va='center')

        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.tight_layout()
        fig.savefig(out)
        plt.close()
