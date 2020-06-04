import torch
import torch.nn.functional as F
import numpy as np


def reconstruction_loss(x, x_, sigma=1.):
    # E_q(z|x)[p(x|z)] = -(1/2σ^2*(x-x')^2)
    # Reconstruction loss
    # 1/2σ * Σ(x - x_)**2
    loss = 0.5 / sigma * F.mse_loss(x_, x, reduction='mean')
    return loss


def conditional_kl(z_x, z_x_mean, z_x_var,
                   z_wy_means, z_wy_vars, y_wz):
    # Conditional loss
    # q(z|x)=N(μ_x,σ_x)
    # logp = −0.5 * { log(det(σ_x^2)) + (z − μ_x)^2 / σ_x^2 }
    # q(z|w,y=1)=N(μ_w,σ_w)
    # logq = −0.5 * { Σπlog(det(σ_w^2)) + Σπ(z − μ_w)^2 / σ_w^2 }
    eps = 1e-10
    logq = -0.5 * (torch.log(z_x_var + eps)
                   + torch.pow(z_x - z_x_mean, 2) / z_x_var).sum(1)
    K = y_wz.shape[-1]
    z_wy = z_x.repeat(1, K).view(
        z_x.shape[0], K, -1).transpose(1, 2)  # (batch_size, z_dim, K)
    logp = -0.5 * (y_wz * torch.log(z_wy_vars + eps).sum(1)
                   + y_wz * (torch.pow(z_wy - z_wy_means, 2) / z_wy_vars).sum(1)).sum(1)
    kl = (logq - logp).mean()
    return kl


def w_prior_kl(w_mean, w_var):
    # input: μ_θ(w), (batch_size, w_dim)
    # input: σ_θ(w), (batch_size, w_dim)
    eps = 1e-10
    kl = 0.5 * (w_var - 1 - torch.log(w_var + eps) +
                torch.pow(w_mean, 2)).sum(-1)
    kl = kl.mean()
    return kl.mean()


def y_prior_negative_kl(y_wz, thres=1.5):
    # input: p_θ(y=1|w,z), (batch_size, K)
    # KL(p_θ(y=1|w,z)||p(y)) = Σ{p(y=1)log[p_θ(y=1|w,z)/p(y=1)]} < 0
    k = y_wz.shape[-1]
    eps = 1e-10
    pi = 1 / k
    kl = (pi * torch.log(y_wz / pi + eps)).sum(-1)
    # output is E_q[KL]
    kl = kl.mean() # negative value minimize(kl)
    # kl = torch.max(kl, torch.ones_like(kl) * thres)
    return kl
