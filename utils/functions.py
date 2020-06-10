import torch
import numpy as np
import sklearn.preprocessing as sp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


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
