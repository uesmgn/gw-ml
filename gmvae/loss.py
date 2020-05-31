import torch
import torch.nn.functional as F
import numpy as np


def reconstruction_loss(x, x_, sigma=0.001):
    # 1/2σ * (x - x_)**2
    loss = 0.5 / sigma * F.mse_loss(x_, x, reduction='none')
    loss = loss.sum(-1).sum(-1)
    return loss.mean()


def conditional_kl_loss(z_x, z_x_mean, z_x_logvar,
                        z_wy_means, z_wy_logvars, y_wz):
    logq = -0.5 * (z_x_logvar
                   + torch.pow(z_x - z_x_mean, 2) / torch.exp(z_x_logvar)).sum(1)
    K = z_wy_means.shape[-1]
    z_wy = z_x.repeat(1, K).view(z_x.shape[0], K, -1).transpose(1, 2)
    aux = torch.pow(z_wy - z_wy_means, 2) / torch.exp(z_wy_logvars)
    logp = -0.5 * (y_wz * z_wy_logvars.sum(1) + y_wz * aux.sum(1)).sum(1)
    kl = logq - logp
    return kl.mean()


def w_prior_kl_loss(w_mean, w_logvar):
    kl = 0.5 * (torch.exp(w_logvar) - 1 - w_logvar + torch.pow(w_mean, 2)).sum(-1)
    return kl.mean()


def y_prior_kl_loss(y_wz):
    # y_wz_k: (batch_size, y_dim)
    k = y_wz.shape[1]
    # kl = -torch.mean(y_wz, -1) - np.log(k)
    kl = (y_wz * torch.log(10e-4 + k * y_wz)).sum(1)
    return kl.mean()
