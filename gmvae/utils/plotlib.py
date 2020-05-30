import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

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
