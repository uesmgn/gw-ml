import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.logvar = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, logvar):
        # reparameterize trick
        # std = torch.sqrt(var + 1e-10)
        # noise = torch.randn_like(std)
        # z = mu + noise * std
        sigma = torch.exp(0.5*logvar)
        noise = torch.randn_like(sigma)
        return mu + noise * sigma

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
