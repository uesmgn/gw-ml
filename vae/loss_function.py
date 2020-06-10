import torch
import torch.nn.functional as F
import numpy as np

class Criterion:
    def __init__(self):
        pass

    def vae_loss(self, params, reduction='none'):
        # get parameters from model
        x = params['x']
        x_z = params['x_z']
        z_x = params['z_x']
        z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var'],

        rec_loss = self.binary_cross_entropy(x, x_z)
        gaussian_negative_kl = self.gaussian_negative_kl(z_x_mean, z_x_var)

        total = torch.cat([rec_loss.view(-1),
                           gaussian_negative_kl.view(-1)])

        if reduction is 'sum':
            return total.sum()
        return total

    def binary_cross_entropy(self, x, x_):
        # Reconstruction loss
        # https://arxiv.org/pdf/1312.6114.pdf -> C.1
        # E_q[log p(x^(i)|z^(i))]=1/LΣ(log p(x_m^(i)|z_m^(i,l)))
        # x, p ~ β(p, x)=p^x+(1-p)^(1-x)
        # log p(x_m^(i)|z_m^(i,l) = log(p_i^x_i+(1-p_i)^(1-x_i))
        #                         = x_i log(p_i)+(1-x_i) log(1-p_i)
        loss = F.binary_cross_entropy(x_, x, reduction='sum')
        loss /= x.shape[0]
        return loss

    def gaussian_negative_kl(self, mean, var):
        # input: μ_θ(w), (batch_size, w_dim)
        # input: σ_θ(w), (batch_size, w_dim)
        eps = 1e-10
        kl = 0.5 * (var - 1 - torch.log(var) + torch.pow(mean, 2)).sum(-1)
        kl = kl.mean()
        return -kl
