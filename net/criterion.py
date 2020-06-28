import torch
import torch.nn.functional as F
import  numpy as np

__all__ = [
    'cvae'
]
eps = 1e-10

def cvae_loss(params, beta=(1.0, 1.0, 1.0)):
    rec_loss = beta[0] * bce_loss(params['x_reconst'], params['x']).view(-1)
    z_kl = beta[1] * log_norm_kl(
        params['z'], params['z_mean'], params['z_var'],
        params['z_prior_mean'], params['z_prior_var']).view(-1)
    y_entropy = beta[2] * entropy(params['y_logits']).view(-1)
    loss = (rec_loss + z_kl + y_entropy).sum()
    return loss, rec_loss, z_kl, y_entropy

def cross_entropy(input, target, clustering_weight=None, beta=1.0):
    loss = beta * F.cross_entropy(input,
                                  target,
                                  weight=clustering_weight).sum()
    return loss

def mse_loss(inputs, targets, reduction='mean'):
    loss = F.mse_loss(inputs, targets).sum(-1)
    return reduce(loss, reduction)

def bce_loss(inputs, targets, reduction='mean'):
    loss = F.binary_cross_entropy(inputs, targets, reduction='none').sum(-1)
    return reduce(loss, reduction)

def log_norm(x, mean, var):
    return -0.5 * (torch.log(2.0 * np.pi * var) * torch.pow(x - mean, 2) / var )

def log_norm_kl(x, mean, var, mean_, var_, reduction='mean'):
    log_p = log_norm(x, mean, var).sum(-1)
    log_q = log_norm(x, mean_, var_).sum(-1)
    loss = log_p - log_q
    return reduce(loss, reduction)

def entropy(logits, reduction='mean'):
    p = logits.softmax(-1)
    log_p = logits.log_softmax(-1)
    entropy = (p * log_p).sum(-1)
    return reduce(entropy, reduction)

def reduce(target, reduction):
    if reduction is 'mean':
        return target.mean()
    if reduction is 'sum':
        return target.sum()
    return target
