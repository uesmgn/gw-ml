import torch
import numpy as np
import sklearn.preprocessing as sp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def reparameterize(mean, var):
    if torch.is_tensor(var):
        std = torch.pow(var, 0.5)
    else:
        std = np.sqrt(var)
    eps = torch.randn_like(mean)
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
        return torch.nn.ELU(inplace=True)
    elif type in ('LeakyReLU', 'lrelu'):
        return torch.nn.LeakyReLU(0.1)
    elif type in ('PReLU', 'prelu'):
        return torch.nn.PReLU()
    else:
        return torch.nn.ReLU(inplace=True)


def confution_matrix(xx, yy, xlabels=None, ylabels=None, normalize=True):
    xlabels = sorted(list(set(xx))) if xlabels is None else sorted(list(xlabels))
    ylabels = sorted(list(set(yy))) if ylabels is None else sorted(list(ylabels))
    cm = np.zeros([len(xlabels), len(ylabels)])
    for (x, y) in zip(xx, yy):
        cm[xlabels.index(x), ylabels.index(y)] += 1.
    if normalize:
        cm = sp.normalize(cm, axis=1, norm='l1')
    return cm, xlabels, ylabels


def nmi(labels_true, labels_pred):
    trues, preds = np.array(labels_true), np.array(labels_pred)
    assert trues.size == preds.size
    return adjusted_mutual_info_score(trues, preds)


def ari(labels_true, labels_pred):
    trues, preds = np.array(labels_true), np.array(labels_pred)
    assert trues.size == preds.size
    return adjusted_rand_score(trues, preds)
