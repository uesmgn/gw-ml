import torch
import torch.nn.functional as F
import  numpy as np

eps = 1e-10

def bce_loss(inputs, targets):
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none').view(inputs.shape[0], -1)
    return loss

def log_norm_kl(mean, var, mean_=None, var_=None):
    if mean_ is None:
        mean_ = torch.zeros_like(mean)
    if var_ is None:
        var_ = torch.ones_like(mean)
    loss = 0.5 * ( torch.log(var_ / var) + (var + torch.pow(mean - mean_, 2)) / var_ - 1)
    return loss

def softmax_cross_entropy(input, target, clustering_weight=None):
    loss = F.cross_entropy(input, target, weight=clustering_weight)
    return loss
#
# def mse_loss(inputs, targets, reduction='mean'):
#     loss = F.mse_loss(inputs, targets).sum(-1)
#     return reduce(loss, reduction)
#
# def bce_loss(inputs, targets, reduction='mean'):
#     loss = F.binary_cross_entropy(inputs, targets, reduction='none').sum(-1)
#     return reduce(loss, reduction)
#
# def _log_norm(x, mean, var):
#     return -0.5 * (torch.log(2.0 * np.pi * var) + torch.pow(x - mean, 2) / var )
#
# def log_norm_kl(x, mean, var, mean_, var_, reduction='mean'):
#     log_p = _log_norm(x, mean, var).sum(-1)
#     log_q = _log_norm(x, mean_, var_).sum(-1)
#     loss = log_p - log_q
#     return reduce(loss, reduction)
#
# def entropy(logits, reduction='mean'):
#     p = logits.softmax(-1)
#     log_p = logits.log_softmax(-1)
#     entropy = -(p * log_p).sum(-1)
#     return reduce(entropy, reduction)
#
# def reduce(target, reduction):
#     if reduction is 'mean':
#         return target.mean()
#     if reduction is 'sum':
#         return target.sum()
#     return target
