import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import nn as cn


ACTIVATION_NAMES = ['ReLU', 'ELU', 'Tanh']


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
        if activation in ACTIVATION_NAMES:
            self.features.add_module(f'{activation}',
                                     cn.activation(activation))

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
        if activation in ACTIVATION_NAMES:
            self.features.add_module('activation',
                                     cn.activation(activation))

    def forward(self, x):
        x = self.features(x)
        return x


class GlobalPool2d(nn.Module):
    def __init__(self, pooling='avg'):
        super().__init__()
        self.pooling = pooling

    def forward(self, x):
        if self.pooling is 'avg':
            x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        else:
            x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.shape[0], -1)
        return x


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, act_regur='Tanh'):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2)
        )
        if act_regur in ACTIVATION_NAMES:
            self.features.add_module('act_regur',
                                     cn.activation(act_regur))

    def forward(self, x):
        eps = 1e-10
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        if self.training:
            x = cn.reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var

class VAE(nn.Module):
    def __init__(self, nargs=None):
        super().__init__()

        nargs = nargs or {}
        x_shape = nargs.get('x_shape')
        in_ch = x_shape[0]
        x_dim = x_shape[1] * x_shape[2]
        z_dim = nargs.get('z_dim') or 20

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(x_dim, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, z_dim*2),
            Gaussian(z_dim*2, z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, x_dim),
            nn.Sigmoid(),
            cn.Reshape(x_shape)
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, return_params=False):
        z_x, z_x_mean, z_x_var = self.encoder(x)
        x_z = self.decoder(z_x)
        if return_params:
            return { 'x': x, 'x_z': x_z,
                     'z_x': z_x,
                     'z_x_mean': z_x_mean,
                     'z_x_var': z_x_var }
        else:
            return x_z
