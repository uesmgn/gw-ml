import torch
import torch.nn.functional as F
import numpy as np


def reconstruction_loss(x, x_, sigma=1.):
    # Reconstruction loss
    # 1/2σ * Σ(x - x_)**2
    loss = 0.5 / sigma * F.mse_loss(x_, x, reduction='none')
    loss = loss.sum(-1).sum(-1)
    return loss


def conditional_kl_loss(z_x, z_x_mean, z_x_var,
                        z_wy_means, z_wy_vars, y_wz):
    # Conditional loss
    # q(z|x)=N(μ_x,σ_x)
    # logp = −0.5 * { log(det(σ_x^2)) + (z − μ_x)^2 / σ_x^2 }
    # q(z|w,y=1)=N(μ_w,σ_w)
    # logq = −0.5 * { Σπlog(det(σ_w^2)) + Σπ(z − μ_w)^2 / σ_w^2 }
    eps = 1e-6
    logq = -0.5 * (torch.log(z_x_var + eps)
           + torch.pow(z_x - z_x_mean, 2) / z_x_var).sum(1)
    K = z_wy_means.shape[-1]
    z_wy = z_x.repeat(1, K).view(z_x.shape[0], K, -1).transpose(1, 2) # (batch_size, z_dim, K)
    logp = -0.5 * (y_wz * torch.log(z_wy_vars.sum(1) + eps)
           + y_wz * (torch.pow(z_wy - z_wy_means, 2) / z_wy_vars).sum(1)).sum(1)
    kl = logq - logp
    return kl


def w_prior_kl_loss(w_mean, w_var):
    eps = 1e-6
    kl = 0.5 * (w_var - 1 - torch.log(w_var + eps) + torch.pow(w_mean, 2)).sum(-1)
    return kl


def y_prior_kl_loss(y_wz):
    eps = 1e-6
    k = y_wz.shape[1]
    kl = (y_wz * (torch.log(y_wz + eps) + np.log(k))).sum(1)
    return kl
