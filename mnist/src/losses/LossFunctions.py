import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, x, x_):
        loss = F.binary_cross_entropy(x_, x, reduction='sum')
        return loss.mean()

    def gaussian_kl_divergence(self, z, z_mu, z_var):
        kl = 0.5 * torch.sum(1 + z_var.pow(2).log()
                             - z_mu.pow(2) - z_var.pow(2))
        return kl
