import numpy as np


ef random_split(N, k):
    devides = np.random.choice(list(range(1, N-1)), k-1, replace=False).astype(np.int)
    devides = np.sort(devides)
    devides = np.append(devides, N)
    ret = []
    x = 0
    for i in range(k):
        x_ = devides[i] - x
        ret.append(x_)
        x = devides[i]
    return ret

def make_spiral_moons(N=10000, k=5, noise=0.1, imbalance=True):
    x_stack = np.array([])
    y_stack = np.array([])
    labels = np.array([])
    if imbalance:
        sizes = random_split(N, k)
    else:
        eps = np.append(np.ones(N % k), np.zeros(k - N % k))
        eps = np.random.permutation(eps).astype(np.int)
        sizes = [N // k + eps[i] for i in range(k-1)]
        sizes.append(N - np.sum(sizes))
    for i in range(k):
        size = sizes[i]
        x = np.random.normal(loc=np.pi/2, scale=0.4, size=size)
        sin_gauss = np.sin(np.linspace(0, np.pi, size)) * (np.random.normal(loc=0, scale=noise, size=size))
        y = np.sin(x) + sin_gauss
        theta = 2*np.pi * i / k
        x_ = np.cos(theta)*x - np.sin(theta)*y
        y_ = np.sin(theta)*x + np.cos(theta)*y
        label = (np.ones(len(x_)) * i).astype(np.int)
        x_stack = np.append(x_stack, x_)
        y_stack = np.append(y_stack, y_)
        labels = np.append(labels, label)
    x_stack = np.ravel(x_stack)
    y_stack = np.ravel(y_stack)
    labels = np.ravel(labels)
    return x_stack, y_stack, labels
