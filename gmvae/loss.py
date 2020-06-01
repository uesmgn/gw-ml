import torch
import torch.nn.functional as F
import numpy as np


def reconstruction_loss(x, x_, sigma=1.):
    # 1/2σ * (x - x_)**2
    # loss = F.binary_cross_entropy(x_, x, reduction='none')
    loss = 0.5 / sigma * F.mse_loss(x_, x, reduction='none')
    loss = loss.sum(-1).sum(-1)
    return loss


def conditional_kl_loss(z_x, z_x_mean, z_x_var,
                        z_wy_means, z_wy_vars, y_wz):
    eps = 1e-6
    logq = -0.5 * (torch.log(z_x_var + eps)
                   + torch.pow(z_x - z_x_mean, 2) / z_x_var).sum(1)
    K = z_wy_means.shape[-1]
    z_wy = z_x.repeat(1, K).view(z_x.shape[0], K, -1).transpose(1, 2)
    aux = torch.pow(z_wy - z_wy_means, 2) / z_wy_vars)
    logp = -0.5 * (y_wz * torch.log(z_wy_vars + eps).sum(1) + y_wz * aux.sum(1)).sum(1)
    kl = logq - logp
    return kl


def w_prior_kl_loss(w_mean, w_var):
    eps = 1e-6
    kl = 0.5 * (w_var - 1 - torch.log(w_var + eps) + torch.pow(w_mean, 2)).sum(-1)
    return kl


def y_prior_kl_loss(y_wz):
    # y_wz_k: (batch_size, y_dim)
    eps = 1e-6
    k = y_wz.shape[1]
    # kl = -torch.mean(y_wz, -1) - np.log(k)
    kl = (y_wz * (torch.log(y_wz + eps) + np.log(k))).sum(1)
    return kl
