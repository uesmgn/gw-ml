import torch
import torch.nn.functional as F
import numpy as np

class Criterion:
    def __init__(self):
        pass

    def vae_loss(self, params, beta=1.0):
        # get parameters from model
        x = params['x']
        x_z = params['x_z']
        z_x = params['z_x']
        z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var'],

        rec_loss = self.binary_cross_entropy(x, x_z)
        gaussian_kl = self.gaussian_kl(z_x_mean, z_x_var)

        total = rec_loss + beta * gaussian_kl
        losses = torch.cat([rec_loss.view(-1), gaussian_kl.view(-1)])

        return total, losses

    def binary_cross_entropy(self, x, x_):
        loss = F.binary_cross_entropy(x_, x, reduction='sum')
        loss = loss.sum()
        return loss

    def gaussian_kl(self, mean, var):
        eps = 1e-10
        kl = -0.5 * (1 + torch.log(var) - torch.pow(mean, 2) - var).sum()
        return kl
