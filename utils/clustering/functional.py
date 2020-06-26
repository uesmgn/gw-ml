import numpy as np
import faiss


def run_kmeans(x, k):
    n_data, d = x.shape

    clus = faiss.Clustering(d, k)

    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 100
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    clus.train(x, index)
    _, I = index.search(x, 1)

    return np.array([int(n[0]) for n in I])
