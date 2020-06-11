import torch
import torch.nn.functional as F
import numpy as np


eps = 1e-10

class Criterion:

    def gmvae_loss(self, params, beta=1.0, reduction='sum'):
        # get parameters from model
        x = params['x']
        x_z = params['x_z']
        w_x_mean, w_x_var = params['w_x_mean'], params['w_x_var']
        y_wz = params['y_wz']
        z_x = params['z_x']
        z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var'],
        z_wy_means, z_wy_vars = params['z_wy_means'], params['z_wy_vars']

        rec_loss = self.binary_cross_entropy(x, x_z)
        cond_kl = self.gaussian_gmm_kl(z_x_mean, z_x_var,
                                       z_wy_means, z_wy_vars, y_wz)
        w_kl = self.standard_gaussian_kl(w_x_mean, w_x_var)
        y_kl = self.uniform_categorical_kl(y_wz)

        total = rec_loss + beta * (cond_kl + w_kl + y_kl)

        # (batch_size, 5)
        each = torch.stack([total,
                            rec_loss,
                            cond_kl,
                            w_kl,
                            y_kl], 1).detach()

        if reduction is 'mean':
            total = total.mean()
            each = each.mean(0)
        elif reduction is 'sum':
            total = total.sum()
            each = each.sum(0)
        return total, each

    def binary_cross_entropy(self, x, x_):
        # x: (batch_size, x_size, x_size)
        # x_: (batch_size, x_size, x_size)
        # loss: (batch_size, )
        assert x.shape == x_.shape
        x = x.view(x.shape[0], -1)
        x_ = x_.view(x_.shape[0], -1)
        loss = F.binary_cross_entropy(x_, x, reduction='none')
        loss = loss.sum(1)
        return loss

    def gaussian_gmm_kl(self, mean, var, means, variances, pi):
        # mean: (batch_size, dim)
        # var: (batch_size, dim) > 0
        # means: (batch_size, dim, K)
        # vars: (batch_size, dim, K) > 0
        # pi: (batch_size, K)
        # kl: (batch_size, )
        K = pi.shape[-1]
        mean_repeat = mean.unsqueeze(-1).repeat(1, 1, K)
        var_repeat = var.unsqueeze(-1).repeat(1, 1, K)
        kl = (pi * self.gaussian_kl(mean_repeat, var_repeat, means, variances)).mean(1)
        return kl

    def gaussian_kl(self, mean1, var1, mean2, var2):
        # mean1: (batch_size, dim, .. )
        # mean2: (batch_size, dim, .. )
        # var1: (batch_size, dim, .. ) > 0
        # var2: (batch_size, dim, .. ) > 0
        # kl: (batch_size, .. )
        assert (torch.cat([var1, var2]) > 0).all()
        kl = 0.5 * (torch.log(var2 / var1) + var1 / var2 + torch.pow(mean1 - mean2, 2) / var2 - 1)
        return kl.sum(1)
        return kl

    def standard_gaussian_kl(self, mean, var):
        # mean: (batch_size, dim, ..)
        # var: (batch_size, dim, ..)
        # kl: (batch_size, ..)
        kl = 0.5 * (var - 1 - torch.log(var) + torch.pow(mean, 2)).sum(1)
        return kl

    def uniform_categorical_kl(self, y):
        # y: (batch_size, K)
        # kl: (batch_size, )
        k = y.shape[-1]
        u = torch.ones_like(y) / k
        kl = (u * torch.log(u / y + eps)).mean(1)
        return kl
