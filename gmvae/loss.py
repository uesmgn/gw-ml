import torch
import torch.nn.functional as F
import numpy as np


def reconstruction_loss(x, x_):
    # Reconstruction loss
    # https://arxiv.org/pdf/1312.6114.pdf -> C.1
    # E_q[log p(x^(i)|z^(i))]=1/LΣ(log p(x_m^(i)|z_m^(i,l)))
    # x, p ~ β(p, x)=p^x+(1-p)^(1-x)
    # log p(x_m^(i)|z_m^(i,l) = log(p_i^x_i+(1-p_i)^(1-x_i))
    #                         = x_i log(p_i)+(1-x_i) log(1-p_i)
    weight = torch.ones_like(x)
    loss = F.binary_cross_entropy(x_, x, weight=weight, reduction='sum')
    loss /= x.shape[0]
    return loss


def conditional_negative_kl(z_x, z_x_mean, z_x_var,
                   z_wy_means, z_wy_vars, y_wz):
    # Conditional loss
    # q(z|x)=N(μ_x,σ_x)
    # logp = −0.5 * { log(det(σ_x^2)) + (z − μ_x)^2 / σ_x^2 }
    # q(z|w,y=1)=N(μ_w,σ_w)
    # logq = −0.5 * { Σπlog(det(σ_w^2)) + Σπ(z − μ_w)^2 / σ_w^2 }
    eps = 1e-10
    logq = -0.5 * (torch.log(z_x_var + eps).sum(1)
                  + (torch.pow(z_x - z_x_mean, 2) / z_x_var).sum(1))
    K = y_wz.shape[-1]
    z_wy = z_x.repeat(1, K).view(z_x.shape[0], K, -1).transpose(1,2)  # (batch_size, z_dim, K)
    logp = -0.5 * (y_wz * torch.log(z_wy_vars + eps).sum(1)
                   + y_wz * (torch.pow(z_wy - z_wy_means, 2) / z_wy_vars).sum(1)).sum(1)
    kl = (logq - logp).mean()
    return -kl


def gaussian_negative_kl(mean, var):
    # input: μ_θ(w), (batch_size, w_dim)
    # input: σ_θ(w), (batch_size, w_dim)
    eps = 1e-10
    kl = 0.5 * (1 + torch.log(var) -
                torch.pow(mean, 2) - var).sum(-1)
    kl = kl.mean()
    return kl


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
