import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, x, x_reconst, type='bce'):
        if type == 'mse':
            loss = F.mse_loss(x_reconst, x, reduction='none')
        elif type == 'bce':
            loss = F.binary_cross_entropy(x_reconst, x, reduction='none')
        else:
            raise "invalid loss function... try bce or mse..."
        return loss.sum(-1).sum(-1).mean()

    def log_normal(self, x, mu, var):
        var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - \
            self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.mean()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))

    def gaussian_kl_loss(self, z, z_mu, z_logvar):
        lossi = -1 - z_logvar + z_mu.pow(2) + z_logvar.exp()
        return 0.5 * lossi.sum()

    def categorical_kl_loss(self, pi):
        k = pi.size(-1)
        lossi = pi * torch.log(k * pi + self.eps)
        return lossi.sum()
