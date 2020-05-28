import torch
from torch import nn
import torch.nn.functional as F


def get_activation(activation='Relu'):
    if activation in ('Sigmoid', 'sigmoid'):
        return nn.Sigmoid()
    elif activation in ('Tanh', 'tanh'):
        return nn.Tanh()
    else:
        return nn.ReLU(inplace=True)


class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel=None,
                 kernel_size=3, stride=1, activation='ReLu'):
        super().__init__()
        out_channel = out_channel or in_channel
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size - stride) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


class ConvtModule(nn.Module):
    def __init__(self, in_channel, out_channel=None,
                 kernel_size=3, stride=1, activation='ReLu'):
        super().__init__()
        out_channel = out_channel or in_channel
        self.convt = nn.ConvTranspose2d(
            in_channel, out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.activation(self.bn(self.convt(x)))
        return x


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=3, activation='ReLu'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - stride) // 2)
        self.conv = ConvModule(in_channel,
                               out_channel,
                               kernel_size=1,
                               stride=1,
                               activation=activation)

    def forward(self, x):
        x = self.conv(self.maxpool(x))
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=3, activation='ReLu'):
        super().__init__()
        self.convt = ConvtModule(in_channel,
                                 out_channel,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 activation=activation)

    def forward(self, x):
        x = self.convt(x)
        return x


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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        return x

    def forward(self, x):
        z_mu = self.mu(x)
        z_logvar = self.logvar(x)
        z = self.reparameterize(z_mu, z_logvar)
        return z, z_mu, z_logvar


class GumbelSoftmax(nn.Module):

    eps = 1e-8

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Flatten()
        )

    def gumbel_softmax(self, logits, temp):
        u = torch.rand(logits.size())
        if logits.is_cuda:
            u = u.to(logits.device)
        g = -torch.log(-torch.log(u + self.eps) + self.eps)
        y = logits + g
        return F.softmax(y / temp, dim=-1)

    def forward(self, x, temp=1.0):
        logits = self.logits(x)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temp)
        return logits, prob, y
