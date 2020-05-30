import torch
import torch.nn.functional as F
import  numpy as np


def reconstruction_loss(x, x_, sigma=0.001):
    loss = -0.5 / sigma * F.mse_loss(x_, x, reduction='none')
    loss = loss.sum(-1).sum(-1)
    return loss.mean()


def __log_normal(x, mean, logvar):
    return -0.5 * torch.sum(
        logvar + torch.pow(x - mean, 2) / torch.exp(logvar), -1)


def conditional_kl(z_x, z_x_mean, z_x_logvar,
                   z_wy, z_wy_mean, z_wy_logvar):
    logp = __log_normal(z_x, z_x_mean, z_x_logvar)
    logq = __log_normal(z_wy, z_wy_mean, z_wy_logvar)
    return (logp - logq).mean()


def w_prior_kl(w_mean, w_logvar):
    kl = 0.5 * (torch.exp(w_logvar) - 1 - w_logvar + torch.pow(w_mean, 2)).sum(-1)
    return kl.mean()


def y_prior_kl(y_wz):
    k = y_wz.shape[1]
    kl = -np.log(k) - 1 / k * torch.sum(y_wz, -1)
    return kl.mean()
