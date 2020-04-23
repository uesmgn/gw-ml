import numpy as np
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


class ConvInceptionModule(nn.Module):
    def __init__(self, channel, bottle_channel=1, stride=1,
                 activation='ReLu', hard=None):
        super().__init__()
        self.hard = hard

        self.bottle = ConvModule(channel,
                                 bottle_channel,
                                 kernel_size=1,
                                 stride=1,
                                 activation=activation)
        self.conv_15 = ConvModule(bottle_channel,
                                  channel,
                                  kernel_size=15,
                                  stride=1,
                                  activation=activation)
        self.conv_7 = ConvModule(bottle_channel,
                                 channel,
                                 kernel_size=7,
                                 stride=1,
                                 activation=activation)
        self.conv_5 = ConvModule(bottle_channel,
                                 channel,
                                 kernel_size=5,
                                 stride=1,
                                 activation=activation)
        self.conv_3 = ConvModule(bottle_channel,
                                 channel,
                                 kernel_size=3,
                                 stride=1,
                                 activation=activation)

        self.pool_3 = nn.MaxPool2d(kernel_size=3,
                                   stride=stride,
                                   padding=(3 - stride) // 2)
        self.pool_5 = nn.MaxPool2d(kernel_size=5,
                                   stride=stride,
                                   padding=(5 - stride) // 2)
        self.pool_7 = nn.MaxPool2d(kernel_size=7,
                                   stride=stride,
                                   padding=(7 - stride) // 2)
        self.activation = get_activation(activation)

    def forward(self, x):

        bottle = self.bottle(x)
        if self.hard == 1:
            x = self.conv_3(bottle) + self.conv_7(bottle) + \
                self.conv_15(bottle) + self.pool_3(x) + self.pool_7(x)
        elif self.hard == 2:
            x = self.conv_3(bottle) + self.conv_5(bottle) + \
                self.conv_7(bottle) + self.pool_3(x) + self.pool_5(x)
        else:
            x = self.conv_3(bottle) + self.pool_3(x)
        x = self.activation(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=3, stride=3, activation='ReLu',
                 type='max', return_indices=False):
        super().__init__()
        self.type = type
        self.return_indices = return_indices
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - stride) // 2,
                                    return_indices=return_indices)
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - stride) // 2)
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=1,
                              stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = get_activation(activation)
        self.indices = None

    def forward(self, x):
        if self.type is 'max':
            if self.return_indices:
                x, self.indices = self.maxpool(x)
            else:
                x = self.maxpool(x)
        elif self.type is 'avg':
            x = self.avgpool(x)
        else:
            raise 'type must be max or avg...'
        x = self.activation(self.bn(self.conv(x)))
        return (x, self.indices) if self.return_indices else x


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


class ConvtInceptionModule(nn.Module):
    def __init__(self, channel, bottle_channel=1, stride=1,
                 activation='ReLu', hard=None):
        super().__init__()
        self.hard = hard

        self.bottle = ConvtModule(channel,
                                  bottle_channel,
                                  kernel_size=1,
                                  stride=1,
                                  activation=activation)
        self.convt_15 = ConvtModule(bottle_channel,
                                    channel,
                                    kernel_size=15,
                                    stride=stride,
                                    activation=activation)
        self.convt_7 = ConvtModule(bottle_channel,
                                   channel,
                                   kernel_size=7,
                                   stride=stride,
                                   activation=activation)
        self.convt_5 = ConvtModule(bottle_channel,
                                   channel,
                                   kernel_size=5,
                                   stride=stride,
                                   activation=activation)
        self.convt_3 = ConvtModule(bottle_channel,
                                   channel,
                                   kernel_size=3,
                                   stride=stride,
                                   activation=activation)

        self.pool_3 = nn.MaxPool2d(kernel_size=3,
                                   stride=stride,
                                   padding=(3 - stride) // 2)
        self.pool_5 = nn.MaxPool2d(kernel_size=5,
                                   stride=stride,
                                   padding=(5 - stride) // 2)
        self.pool_7 = nn.MaxPool2d(kernel_size=7,
                                   stride=stride,
                                   padding=(7 - stride) // 2)

        self.activation = get_activation(activation)

    def forward(self, x):

        bottle = self.bottle(x)
        if self.hard == 1:
            x = self.convt_3(bottle) + self.convt_7(bottle) + \
                self.convt_15(bottle) + self.pool_3(x) + self.pool_7(x)
        elif self.hard == 2:
            x = self.convt_3(bottle) + self.convt_5(bottle) + \
                self.convt_7(bottle) + self.pool_3(x) + self.pool_5(x)
        else:
            x = self.convt_3(bottle) + self.pool_3(x)
        x = self.activation(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel=None,
                 kernel_size=3, stride=3, activation='ReLu',
                 accept_indices=True):
        super().__init__()
        self.accept_indices = accept_indices
        out_channel = out_channel or in_channel
        self.unpool = nn.MaxUnpool2d(kernel_size=kernel_size,
                                     stride=stride,
                                     padding=(kernel_size - stride) // 2)
        self.bottle = nn.ConvTranspose2d(in_channel,
                                         out_channel,
                                         kernel_size=1,
                                         stride=1)
        self.convt = nn.ConvTranspose2d(in_channel,
                                        out_channel,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size - stride) // 2)

        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = get_activation(activation)

    def forward(self, x, indices=None):
        if self.accept_indices:
            x = self.bottle(x)
            x = self.unpool(x, indices)
        else:
            x = self.convt(x)
        x = self.activation(self.bn(x))
        return x


class ConvtModule(nn.Module):
    def __init__(self, in_channel, out_channel=None,
                 kernel_size=3, stride=1, activation='ReLu'):
        super().__init__()
        out_channel = out_channel or in_channel
        self.convt = nn.ConvTranspose2d(in_channel,
                                        out_channel,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=(kernel_size - stride) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.activation(self.bn(self.convt(x)))
        return x


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim, middle_dim=None):
        super().__init__()
        middle_dim = middle_dim or z_dim
        self.h = nn.Sequential(
            nn.Linear(in_dim, middle_dim),
            nn.Tanh()
        )
        self.mu = nn.Linear(middle_dim, z_dim)
        self.logvar = nn.Linear(middle_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        return x

    def forward(self, x, reparameterize=True):
        h = self.h(x)
        z_mu = self.mu(h)
        z_logvar = self.logvar(h)
        z = self.reparameterize(z_mu, z_logvar) if reparameterize else z_mu
        return z, z_mu, z_logvar

class GumbelSoftmax(nn.Module):

    eps=1e-8

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Flatten()
            )

    def gumbel_softmax(self, logits, temp):
        u = torch.rand(logits.size())
        if prob.is_cuda:
            u = u.to(logits.device)
        g  = -torch.log(-torch.log(u + self.eps) + self.eps)
        y = logits + g
        return F.softmax(y / temp, dim=-1)

    def forward(self, x, temp=1.0):
        logits = self.logits(x)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temp)
        return logits, prob, y
