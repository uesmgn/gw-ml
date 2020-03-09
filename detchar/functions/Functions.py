import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

class Functions:

    @classmethod
    def plot_losslog(cls, log, out):
        xx = [x for x in log.keys()]
        yy_keys = list(log.values())[0].keys()

        plt.figure(figsize=[8, 4])
        yy_stack = []
        for key in yy_keys:
            yy = [loss.get(key) for loss in log.values()]
            yy_stack.append(yy)
        plt.stackplot(xx,
                      np.vstack(yy_stack),
                      labels=yy_keys)
        plt.legend(loc='upper right')
        plt.xlabel('epoch')
        plt.xlim([min(xx), max(xx)])
        plt.ylabel('loss')
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    @classmethod
    def plot_latent(cls, data, labels, out):
        true_labels = labels
        n = len(labels)
        xx = np.array([d['x'] for d in data.values()])
        yy = np.array([d['y'] for d in data.values()])
        pred_labels = np.array([d['label'] for d in data.values()])
        colors = np.array([true_labels.index(l) / n for l in pred_labels])
        plt.scatter(xx, yy, c=colors, cmap='tab20')
        plt.tight_layout()
        plt.savefig(out)
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
        figsize = (width+margin[0], height+margin[1])

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
