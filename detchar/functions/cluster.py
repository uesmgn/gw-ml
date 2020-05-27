from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd


class Cluster:

    def __init__(self, labels_true, labels_pred):
        self.labels_true = labels_true
        self.labels_pred = labels_pred

    def get_cm(self, trues, preds):
        cm = np.zeros([len(self.labels_true), len(self.labels_pred)])
        for (true, pred) in zip(trues, preds):
            cm[self.labels_true.index(true), self.labels_pred.index(pred)] += 1.
        df = pd.DataFrame(cm, index=self.labels_true, columns=self.labels_pred)
        return df

    def fit_kmeans(n_clusters, data):
        arr = KMeans(n_clusters=K).fit_predict(data)
        return arr

    def fit_tsne(n_components, data):
        arr = TSNE(n_components=n_components).fit_transform(data)
        return arr
