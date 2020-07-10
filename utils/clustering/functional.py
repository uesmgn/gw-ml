import numpy as np
import faiss


def run_kmeans(x, k):
    n_data, d = x.shape

    kmeans = faiss.Kmeans(d, k)
    kmeans.seed = np.random.randint(1234)
    kmeans.train(x)
    _, I = kmeans.index.search(x, 1)
    labels = np.ravel(I)

    index = faiss.IndexFlatL2(d)
    index.add(x)
    _, I = index.search(kmeans.centroids, 1)
    centroid_ids = np.ravel(I)

    return labels, centroid_ids
