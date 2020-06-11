import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import *

eps = 1e-10

class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class Conv2dModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=1,
                 stride=1,
                 activation='ReLU'):
        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module('Conv2d',
                                 nn.Conv2d(in_ch, out_ch,
                                           kernel_size=kernel,
                                           stride=stride,
                                           padding=(kernel - stride) // 2))
        self.features.add_module('BatchNorm2d',
                                 nn.BatchNorm2d(out_ch))
        if activation in ACTIVATIONS:
            self.features.add_module(f'{activation}',
                                     get_activation(activation))

    def forward(self, x):
        x = self.features(x)
        return x


class ConvTranspose2dModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=1,
                 stride=1,
                 activation='ReLU'):
        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module('ConvTranspose2d',
                                 nn.ConvTranspose2d(in_ch, out_ch,
                                                    kernel_size=kernel,
                                                    stride=stride,
                                                    padding=(kernel - stride) // 2))
        self.features.add_module('BatchNorm2d',
                                 nn.BatchNorm2d(out_ch))
        if activation in ACTIVATIONS:
            self.features.add_module('activation',
                                     get_activation(activation))

    def forward(self, x):
        x = self.features(x)
        return x

class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.tanh()
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        if self.training:
            x = reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var

class GumbelSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, tau=0.7):
        # x: (batch_size, K)
        pi = F.softmax(logits, dim=-1)
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + eps) + eps)
        y = F.softmax(logits + g / tau, dim=-1)
        return pi, y

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel=3,
                 pool_kernel=3,
                 pooling='max',
                 activation='ReLu'):
        super().__init__()
        self.features = nn.Sequential()
        assert kernel >= pool_kernel
        if pooling in ('avg', 'AvgPool'):
            self.features.add_module(f'{kernel}x{kernel}AvgPool',
                                     nn.AvgPool2d(kernel_size=kernel,
                                                  stride=pool_kernel,
                                                  padding=(kernel-pool_kernel) // 2))
        elif pooling in ('max', 'MaxPool'):
            self.features.add_module(f'{kernel}x{kernel}MaxPool',
                                     nn.MaxPool2d(kernel_size=kernel,
                                                  stride=pool_kernel,
                                                  padding=(kernel-pool_kernel) // 2))
        else:
            self.features.add_module(f'{kernel}x{kernel}conv',
                                     Conv2dModule(in_ch, in_ch,
                                                  kernel=kernel,
                                                  stride=pool_kernel,
                                                  activation=activation))
        self.features.add_module(f'1x1conv',
                                 Conv2dModule(in_ch, out_ch,
                                              activation=activation))

    def forward(self, x):
        x = self.features(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch,
                 unpool_kernel=3,
                 activation='ReLu'):
        super().__init__()
        self.features = nn.Sequential(
            ConvTranspose2dModule(in_ch,
                                  in_ch,
                                  kernel=unpool_kernel,
                                  stride=unpool_kernel,
                                  activation=activation),
            ConvTranspose2dModule(in_ch, out_ch,
                                  activation=activation))

    def forward(self, x):
        x = self.features(x)
        return x


class DenseModule(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_layers=0,
                 hidden_dim=64,
                 act_trans='ReLU',
                 act_out=None):
        super().__init__()
        layers = []
        if n_layers > 0:
            for i in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                if act_trans in ACTIVATIONS:
                    layers.append(get_activation(act_trans))
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        if act_out in ACTIVATIONS:
            layers.append(get_activation(act_out))

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x
