import torch
import numpy as np


def reparameterize(mean, var):
    std = torch.pow(var, 0.5)
    eps = torch.randn_like(std)
    x = mean + eps * std
    return x


def activation(type='ReLU'):
    if type in ('Tanh', 'tanh'):
        return torch.nn.Tanh()
    elif type in ('Sigmoid', 'sigmoid'):
        return torch.nn.Sigmoid()
    elif type in ('Softmax', 'softmax'):
        return torch.nn.Softmax(dim=-1)
    elif type in ('ELU', 'elu'):
        return torch.nn.ELU()
    elif type in ('LeakyReLU', 'lrelu'):
        return torch.nn.LeakyReLU(0.1)
    elif type in ('PReLU', 'prelu'):
        return torch.nn.PReLU()
    else:
        return torch.nn.ReLU()


def confution_matrix(xx, yy, xlabels=None, ylabels=None, normalize=True):
    if xlabels is None:
        xlabels = sorted(list(set(xx)))
    if ylabels is None:
        ylabels = sorted(list(set(yy)))
    cm = np.zeros([len(xlabels), len(ylabels)])
    xlabels = list(xlabels)
    ylabels = list(ylabels)
    for (x, y) in zip(xx, yy):
        cm[xlabels.index(x), ylabels.index(y)] += 1.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm
