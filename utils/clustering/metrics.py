import itertools
import numpy as np
from sklearn import metrics
from sklearn import preprocessing

from ..helper import *

__all__ = [
    'confusion_matrix',
    'nmi',
    'ari'
]

# Confution matrix between 2 results have different labels
def confusion_matrix(true, pred, labels_true=None, labels_pred=None,
                     normalize=True, return_labels=False):
    true, pred = _check_array(true, pred, check_size=True)
    if labels_true is None:
        labels_true = _check_array(true, sort=True, unique=True)
    if labels_pred is None:
        labels_pred = _check_array(pred, sort=True, unique=True)
    matrix = np.zeros([len(labels_true), len(labels_pred)])
    for (t, p) in itertools.product(labels_true, labels_pred):
        idx = np.where(pred[np.where(true==t)]==p)[0]
        matrix[list(labels_true).index(t),
               list(labels_pred).index(p)] += len(idx)
    if normalize:
        matrix = preprocessing.normalize(matrix, axis=0, norm='l1')
    if return_labels:
        return matrix, labels_true, labels_pred
    return matrix


# Normalized Mutual Information Score
def nmi(true, pred):
    true, pred = _check_array(true, pred, check_size=True)
    return metrics.adjusted_mutual_info_score(true, pred)


# Adjusted Rand Index
def ari(true, pred):
    true, pred = _check_array(true, pred, check_size=True)
    return metrics.adjusted_rand_score(true, pred)
