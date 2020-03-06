import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class LossFunctions:
    eps = 1e-8

    def reconstruction_loss(self, x, x_, type='mse'):
      if type == 'mse':
        loss = F.mse_loss(x, x_.detach(), reduction='none')
      elif type == 'bce':
        loss = F.binary_cross_entropy(x, x_.detach(), reduction='none')
      else:
        raise "invalid loss function... try bce or mse..."
      return loss.sum(-1).mean()

    def __log_normal(self, x, mu, var):
      if self.eps > 0.0:
        var = var + self.eps
      return -0.5 * torch.sum(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)

    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
      loss = self.__log_normal(z, z_mu, z_var) - self.__log_normal(z, z_mu_prior, z_var_prior)
      return loss.mean()

    def entropy(self, logits, targets):
      log_q = F.log_softmax(logits, dim=-1)
      return -torch.mean(torch.sum(targets * log_q, dim=-1))
