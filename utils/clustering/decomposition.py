from sklearn.manifold import TSNE as NormalTSNE
# from MulticoreTSNE import MulticoreTSNE
import faiss
import torch


__all__ = [
    'TSNE', 'PCA'
]


class TSNE:
    def __init__(self, n_components=2, multi_core=True, n_jobs=4):
        # if multi_core:
        #     self.tsne = MulticoreTSNE(
        #         n_jobs=n_jobs, n_components=n_components)
        # else:
        #     self.tsne = NormalTSNE(n_components=n_components)
        self.tsne = NormalTSNE(n_components=n_components)

    def fit_transform(self, data):
        data = self.tsne.fit_transform(data)
        return data

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit_transform(self, data):
        mat = faiss.PCAMatrix(data.shape[-1], self.n_components)
        mat.train(data)
        return mat.apply_py(data)
