import torch
import torch.nn.functional as F
import  numpy as np

__all__ = [
    'cvae'
]
eps = 1e-10

def cvae(params, beta=(1.0, 1.0, 1.0), clustering_weight=None):
    rec_loss = beta[0] * bce_loss(params['x_reconst'], params['x']).view(-1)
    z_kl = beta[1] * log_norm_kl(
        params['z'], params['z_mean'], params['z_var'],
        params['z_prior_mean'], params['z_prior_var']).view(-1)
    y_kl = beta[2] * uniform_categorical_kl(params['y']).view(-1)
    features_loss = (rec_loss + z_kl + y_kl).sum()
    cluster_loss = F.cross_entropy(params['logits'].squeeze(1), params['pseudos'], weight=clustering_weight).sum()
    return features_loss, cluster_loss

def mse_loss(inputs, targets, reduction='mean'):
    loss = F.mse_loss(inputs, targets).sum(-1)
    return reduce(loss, reduction)

def bce_loss(inputs, targets, reduction='mean'):
    loss = F.binary_cross_entropy(inputs, targets, reduction='none').mean(-1)
    return reduce(loss, reduction)

def log_norm_kl(x, mean, var, mean_, var_, reduction='mean'):
    log_p = -0.5 * (torch.log(2.0 * np.pi * var) +
                    torch.pow(x - mean, 2) / var).mean(-1)
    log_q = -0.5 * (torch.log(2.0 * np.pi * var_) +
                    torch.pow(x - mean_, 2) / var_).mean(-1)
    loss = log_p - log_q
    return reduce(loss, reduction)

def uniform_categorical_kl(y, reduction='mean'):
    k = y.shape[-1]
    u = torch.ones_like(y) / k
    kl = (u * torch.log(u / y + eps)).mean(-1)
    return reduce(kl, reduction)

def reduce(target, reduction):
    if reduction is 'mean':
        return target.mean()
    if reduction is 'sum':
        return target.sum()
    return target
