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
    labels_true = _check_array(labels_true or true, sort=True, unique=True)
    labels_pred = _check_array(labels_pred or pred, sort=True, unique=True)

    matrix = np.zeros([len(labels_true), len(labels_pred)])
    for (t, p) in itertools.product(true, pred):
        matrix[list(labels_true).index(t),
               list(labels_pred).index(p)] += 1.
    if normalize:
        matrix = preprocessing.normalize(matrix, axis=1, norm='l1')
    if return_labels:
        return matrix, xlabels, ylabels
    return matrix


# Normalized Mutual Information Score
def nmi(true, pred):
    true, pred = _check_array(true, pred, check_size=True)
    return metrics.adjusted_mutual_info_score(true, pred)


# Adjusted Rand Index
def ari(true, pred):
    true, pred = _check_array(true, pred, check_size=True)
    return metrics.adjusted_rand_score(true, pred)