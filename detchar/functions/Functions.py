from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import  numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import ticker


class Functions:

    def confution_matrix(self, xx, yy,
                         xlabels, ylabels, normalize=True):
        cm = np.zeros([len(xlabels), len(ylabels)])
        xlabels = list(xlabels)
        ylabels = list(ylabels)
        for (x, y) in zip(xx, yy):
            cm[xlabels.index(x), ylabels.index(y)] += 1.
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

    def fit_kmeans(self, n_clusters, data):
        arr = KMeans(n_clusters=n_clusters).fit_predict(data).astype(str)
        return arr

    def fit_tsne(self, n_components, data):
        arr = TSNE(n_components=n_components).fit_transform(data)
        return arr

    def plot_loss(self, losses, out):
        plt.figure(figsize=[8, 4])
        xx = list(range(len(losses)))
        median = np.median(losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xlim([min(xx), max(xx)])
        plt.ylim([min(yy), median])
        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    def plot_latent(self, xx, yy, preds, out):
        labels = list(set(preds))
        df = pd.DataFrame([xx, yy, preds], columns=['x', 'y', 'label'])
        x_mean = np.mean(xx)
        x_sigma = 3. * np.std(xx)
        y_mean = np.mean(yy)
        y_sigma = 3. * np.std(yy)
        for i, label in enumerate(labels):
            idf = df[df['label'] == label]
            x = idf['x'].to_numpy()
            y = idf['y'].to_numpy()
            plt.scatter(x, y, c=i,
                        label=label, cmap='tab20')
        plt.xlim(x_mean - x_sigma, x_mean + x_sigma)
        plt.ylim(y_mean - y_sigma, y_mean + y_sigma)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()

    def plot_cm(self, cm, index, columns,
                out, title=None):

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

        if title:
            ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.tight_layout()
        fig.savefig(out)
        plt.close()
