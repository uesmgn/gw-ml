from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import  numpy as np

class Functions:
    @classmethod
    def plot_latent2d(cls, data, labels, out):
        n = len(labels)
        zz = np.array([d['z'] for d in data.values()])
        zz_labels = np.array([d['label'] for d in data.values()])
        colors = np.array([labels.index(l) / n for l in zz_labels])

        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot(111)

        ax.scatter(zz[:, 0], zz[:, 1], c=colors, s=5.0, cmap='tab10')
        fig.tight_layout()
        fig.savefig(out)
        plt.close()

    @classmethod
    def plot_latent3d(cls, data, labels, out):
        n = len(labels)
        zz = np.array([d['z'] for d in data.values()])
        zz_labels = np.array([d['label'] for d in data.values()])
        colors = np.array([labels.index(l) / n for l in zz_labels])

        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(zz[:, 0], zz[:, 1], zz[:, 2], c=colors, s=5.0, cmap='tab10')
        fig.tight_layout()
        fig.savefig(out)
        plt.close()
