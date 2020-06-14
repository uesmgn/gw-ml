from sklearn.manifold import TSNE as NormalTSNE
from MulticoreTSNE import MulticoreTSNE
from tsnecuda import TSNE as CudaTSNE
import torch


__all__ = [
    'TSNE'
]


class TSNE:
    def __init__(self, n_components=2, cuda=False, multi_core=False, n_jobs=4):
        if cuda:
            assert torch.cuda.is_available()
            # cudatsne only support n_components=2
            self.tsne = CudaTSNE(n_components=2)
        else:
            if multi_core:
                self.tsne = MulticoreTSNE(
                    n_jobs=n_jobs, n_components=n_components)
            else:
                self.tsne = NormalTSNE(n_components=n_components)

    def fit_transform(self, data):
        data = self.tsne.fit_transform(data)
        return data
