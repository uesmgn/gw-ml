import matplotlib.pyplot as plt
import  numpy as np

class Functions:
    @classmethod
    def plot_latent(cls, data, labels, out):
        n = len(labels)
        zz = np.array([d['z'] for d in data.values()])
        zz_labels = np.array([d['label'] for d in data.values()])
        colors = np.array([labels.index(l) / n for l in zz_labels])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        print(zz)

        ax.scatter(zz[:, 0], zz[:, 1], zz[:, 2], c=colors, cmap='tab20')
        fig.tight_layout()
        fig.savefig(out)
        fig.close()
