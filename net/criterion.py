import torch
import torch.nn.functional as F

__all__ = [
    'gmvae_loss'
]

def gmvae_loss(params, beta=1.0):
    x, x_z = params['x'], params['x_z']
    z_x = params['z_x']
    z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var']
    z_wys = params['z_wys']
    z_wy_means, z_wy_vars = params['z_wy_means'], params['z_wy_vars']
    w_x = params['w_x']
    w_x_mean, w_x_var = params['w_x_mean'], params['w_x_var']
    y_wz, y_pred = params['y_wz'], params['y_pred']

    rec_loss = binary_cross_entropy(x_z, x)
    cond_kl = gaussian_gmm_kl(z_x, z_x_mean, z_x_var,
                              z_wy_means, z_wy_vars, y_wz)
    w_kl = standard_gaussian_kl(w_x_mean, w_x_var)
    y_kl = uniform_categorical_kl(y_wz)

    total = rec_loss + beta * (cond_kl + w_kl + y_kl)

    each = torch.stack([total,
                        rec_loss,
                        cond_kl,
                        w_kl,
                        y_kl]).detach()

    return total, each

def binary_cross_entropy(input, target):
    # x: (batch_size, x_size, x_size)
    # x_: (batch_size, x_size, x_size)
    # loss: (batch_size, )
    assert input.shape == target.shape
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)
    loss = F.binary_cross_entropy(input, target, reduction='none').sum(1)
    return loss.mean()

def gaussian_gmm_kl(z, mean, variance, means, variances, pi):
    # mean: (batch_size, dim)
    # var: (batch_size, dim) > 0
    # means: (batch_size, dim, K)
    # vars: (batch_size, dim, K) > 0
    # pi: (batch_size, K)
    # kl: (batch_size, )
    k = pi.shape[-1]
    z_repeat = z.unsqueeze(-1).repeat(1, 1, k)
    log_q = -0.5 * (torch.log(variance) + torch.pow(z - mean, 2) / variance ).sum(-1)
    log_p = -0.5 * (
        (pi * torch.log(variances).sum(1)).sum(-1) + \
        (pi * (torch.pow(z_repeat - means, 2) / variances).sum(1)).sum(-1)
    )
    kl = log_q - log_p
    return kl.mean()

def standard_gaussian_kl(mean, var):
    # mean: (batch_size, dim, ..)
    # var: (batch_size, dim, ..)
    # kl: (batch_size, ..)
    kl = 0.5 * (var - 1 - torch.log(var) + torch.pow(mean, 2)).sum(1)
    # sum over dimenntion of w
    return kl.mean()

def uniform_categorical_kl(y):
    # y: (batch_size, K)
    # kl: (batch_size, )
    k = y.shape[-1]
    u = torch.ones_like(y) / k
    # (y * torch.log(y/u)).sum() = F.kl_div(torch.log(u), y, reduction='none')
    kl = F.kl_div(torch.log(u), y, reduction='none').sum(1)
    return kl.mean()
