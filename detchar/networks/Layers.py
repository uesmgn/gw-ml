import torch
from torch import nn
from torch.nn import functional as F


class ConvModule(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel=3, activation='ReLu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kernel,
                              stride=1,
                              padding=(kernel-1)//2)
        self.bn = nn.BatchNorm2d(out_channel)
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x

# Flatten layer
class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)

# Reshape layer
class Reshape(nn.Module):
  def __init__(self, outer_shape):
    super(Reshape, self).__init__()
    self.outer_shape = outer_shape
  def forward(self, x):
    return x.view(x.size(0), *self.outer_shape)

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        # p(c|f)
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim

    def _sample_gumbel(self, shape, eps=1e-20, is_cuda=False):
        u = torch.rand(shape)
        if is_cuda:
            u = u.cuda()
        return -torch.log(-torch.log(u + eps) + eps)

    def _gumbel_softmax_sample(self, logits, temperature):
        y = logits + self._sample_gumbel(logits.size(), is_cuda=logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        y = self._gumbel_softmax_sample(logits, temperature)
        return y

    def forward(self, x, temperature=1.0):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature)
        return logits, prob, y


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        # make ``z_dim``-D means and vars of ``z_dim``-D gaussian distribution
        # from ``in_dim``-D features
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        # reparameterize trick:
        # sample ``z`` from N(``mu``, ``var``) differentiable
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        # var is non-zero
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z
