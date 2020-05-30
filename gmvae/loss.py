import torch
import torch.nn.functional as F
import numpy as np


def reconstruction_loss(x, x_, sigma=0.001):
    loss = -0.5 / sigma * F.mse_loss(x_, x, reduction='sum')
    return loss


def log_normal(self, x, mean, log_var):
    return -0.5 * torch.sum(
        log_var + torch.pow(x - mean, 2) / torch.exp(log_var))


def conditional_kl(z_x, z_x_mean, z_x_logvar,
                   z_wys, z_wy_means, z_wy_logvars, z_wy_pi):
    # z_x: (batch_size, z_dim)
    # z_x_mean: (batch_size, z_dim)
    # z_x_logvar: (batch_size, z_dim)
    # z_wys: (batch_size, z_dim, K)
    # z_wy_means: (batch_size, z_dim, K)
    # z_wy_logvars: (batch_size, z_dim, K)
    # z_wy_pi: (batch_size, K)
    logp = log_normal(z_x, z_x_mean, z_x_logvar)
    log_det_sigma = z_wy_pi * torch.sum(z_wy_logvars, 2)  # (batch_size, K)
    aux = z_wy_pi * \
        torch.sum(torch.pow(z_wys - z_wy_means, 2) /
                  torch.exp(z_wy_logvars), 2) # (batch_size, K)
    logq = -0.5 * torch.sum(log_det_sigma + aux)
    return logq - logp


def w_prior_kl(w_mean, w_logvar):
    kl = 0.5 * torch.sum(torch.exp(w_logvar) - 1 - w_logvar + torch.pow(w_mean, 2))
    return kl


def y_prior_kl(y_wz, k):
    kl = -np.log(k) - 1 / k * torch.sum(p_y_wz)
    return kl
