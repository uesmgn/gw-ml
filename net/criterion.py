import torch
import torch.nn.functional as F

__all__ = [
    'gmvae_loss'
]

def gmvae_loss(x, x_z, z_x, z_x_mean, z_x_var,
               z_wys, z_wy_means, z_wy_vars,
               w_x, w_x_mean, w_x_var,
               y_wz, pi, y_pred, beta=1.0):

    rec_loss = binary_cross_entropy(x_z, x)
    cond_kl = gaussian_gmm_kl(z_x_mean, z_x_var,
                              z_wy_means, z_wy_vars, pi)
    w_kl = standard_gaussian_kl(w_x_mean, w_x_var)
    y_kl = uniform_categorical_kl(pi)

    total = rec_loss + beta * (cond_kl + w_kl + y_kl)

    each = torch.stack([total,
                        rec_loss,
                        cond_kl,
                        w_kl,
                        y_kl], 1).detach()

    return total.sum(0), each.sum(0)

def binary_cross_entropy(input, target):
    # x: (batch_size, x_size, x_size)
    # x_: (batch_size, x_size, x_size)
    # loss: (batch_size, )
    assert input.shape == target.shape
    input = input.view(input.shape[0], -1)
    target = target.view(target.shape[0], -1)
    loss = F.binary_cross_entropy(input, target, reduction='none')
    return loss.sum(1)

def gaussian_gmm_kl(mean, var, means, variances, pi):
    # mean: (batch_size, dim)
    # var: (batch_size, dim) > 0
    # means: (batch_size, dim, K)
    # vars: (batch_size, dim, K) > 0
    # pi: (batch_size, K)
    # kl: (batch_size, )
    K = pi.shape[-1]
    mean_repeat = mean.unsqueeze(-1).repeat(1, 1, K)
    var_repeat = var.unsqueeze(-1).repeat(1, 1, K)
    kl = (pi * gaussian_kl(mean_repeat, var_repeat, means, variances))
    return kl.mean(1)

def gaussian_kl(mean1, var1, mean2, var2):
    # mean1: (batch_size, dim, .. )
    # mean2: (batch_size, dim, .. )
    # var1: (batch_size, dim, .. ) > 0
    # var2: (batch_size, dim, .. ) > 0
    # kl: (batch_size, .. )
    assert (torch.cat([var1, var2]) > 0).all()
    kl = 0.5 * (torch.log(var2 / var1) + var1 / var2 + torch.pow(mean1 - mean2, 2) / var2 - 1)
    return kl.sum(1)

def standard_gaussian_kl(mean, var):
    # mean: (batch_size, dim, ..)
    # var: (batch_size, dim, ..)
    # kl: (batch_size, ..)
    kl = 0.5 * (var - 1 - torch.log(var) + torch.pow(mean, 2))
    return kl.sum(1)

def uniform_categorical_kl(y):
    # y: (batch_size, K)
    # kl: (batch_size, )
    k = y.shape[-1]
    u = torch.ones_like(y) / k
    kl = F.kl_div(torch.log(u), y, reduction='none')
    return kl.sum(1)
