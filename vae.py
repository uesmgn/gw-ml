import torch.nn as nn
import numpy as np

from ..layers import *
from ..helper import *

__all__ = [
    'Encoder',
    'Decoder'
]

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        x_shape = kwargs['x_shape']
        in_channel = x_shape[0]
        bottle_channel = kwargs['bottle_channel']
        channels = kwargs['channels']
        kernels = kwargs['kernels']
        poolings = kwargs['poolings']
        pool_func = kwargs['pool_func']
        act_func = kwargs['act_func']

        self.features = nn.Sequential(
            Conv2dModule(in_channel, bottle_channel,
                         act_func=act_func),
            DownSample(bottle_channel,
                       channels[0],
                       kernel=kernels[0],
                       pooling=poolings[0],
                       pool_func=pool_func,
                       act_func=act_func),
            *[DownSample(channels[i-1], channels[i],
                         kernel=kernels[i],
                         pooling=poolings[i],
                         pool_func=pool_func,
                         act_func=act_func) for i in range(1, len(channels))],
            Conv2dModule(channels[-1], bottle_channel,
                         act_func=act_func),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        z_dim = kwargs['z_dim']
        x_shape = kwargs['x_shape']
        in_channel = x_shape[0]
        bottle_channel = kwargs['bottle_channel']
        channels = kwargs['channels']
        kernels = kwargs['kernels']
        poolings = kwargs['poolings']
        pool_func = kwargs['pool_func']
        act_func = kwargs['act_func']
        middle_dim = get_middle_dim(x_shape, poolings)
        f_dim = bottle_channel * middle_dim**2

        self.features = nn.Sequential(
            nn.Linear(z_dim, f_dim),
            Reshape((bottle_channel, middle_dim, middle_dim)),
            ConvTranspose2dModule(bottle_channel, channels[-1],
                                  act_func=act_func),
            *[Upsample(channels[-i], channels[-i-1],
                       unpooling=poolings[-i],
                       act_func=act_func) for i in range(1, len(channels))],
            Upsample(channels[0], bottle_channel,
                     unpooling=poolings[0],
                     act_func=act_func),
            ConvTranspose2dModule(bottle_channel, in_channel,
                                  act_func='Sigmoid')
        )

    def forward(self, x):
        return self.features(x)
