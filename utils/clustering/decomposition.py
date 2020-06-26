from sklearn.manifold import TSNE as NormalTSNE
# from MulticoreTSNE import MulticoreTSNE
import torch


__all__ = [
    'TSNE'
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
