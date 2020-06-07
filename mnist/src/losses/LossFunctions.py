import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, x, x_):
        loss = F.binary_cross_entropy(x_, x, reduction='none')
        return loss.sum(-1).mean()

    def gaussian_kl_loss(self, z, z_mu, z_logvar):
        loss = -0.5 * torch.sum(1 + z_logvar
                                - z_mu.pow(2) - z_logvar.exp())
        return loss
