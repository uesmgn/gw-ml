import torch
import torch.nn as nn

from ..layers import *
from ..helper import *

class GMVAE(nn.Module):
    def __init__(self, x_shape, y_dim, w_dim, z_dim,
                 **kwargs):
        super().__init__()

        self.x_shape = x_shape
        self.y_dim = y_dim
        self.w_dim = w_dim
        self.z_dim = z_dim

        in_ch = x_shape[0]
        x_dim = x_shape[-1]
        poolings = kwargs.get('poolings') or (3, 3, 3, 3)
        middle_dim = get_middle_dim(x_dim, poolings)
        channels = kwargs.get('channels') or (32, 48, 64, 32)
        bottle = kwargs.get('bottle') or 16
        kernels = kwargs.get('kernels') or (3, 3, 3, 3)
        hidden = kwargs.get('hidden') or 256
        activation = kwargs.get('activation') or 'ReLU'
        pool = kwargs.get('pool') or 'max'

        middle_flat = bottle * middle_dim * middle_dim

        assert len(channels) == len(poolings) == len(kernels)

        self.zw_x_graph = nn.Sequential(
            Conv2dModule(in_ch, bottle,
                         activation=activation),
            DownSample(bottle, channels[0],
                         kernel=kernels[0],
                         pool_kernel=poolings[0],
                         pooling=pool,
                         activation=activation),
            *[DownSample(channels[i-1], channels[i],
                         kernel=kernels[i],
                         pool_kernel=poolings[i],
                         pooling=pool,
                         activation=activation) for i in range(1, len(channels))],
            Conv2dModule(channels[-1], bottle,
                         activation=activation),
            nn.Flatten()
        )

        self.z_x_graph = nn.Sequential(
            Gaussian(middle_flat, z_dim)
        )

        self.w_x_graph = nn.Sequential(
            Gaussian(middle_flat, w_dim)
        )

        self.y_wz_graph = DenseModule(w_dim + z_dim,
                                      y_dim,
                                      n_layers=1,
                                      hidden_dims=(hidden,),
                                      act_trans=activation,
                                      act_out='Softmax')

        self.z_wy_graph = GaussianMixture(w_dim, z_dim, y_dim)

        self.x_z_graph = nn.Sequential(
            DenseModule(z_dim, middle_flat,
                        n_layers=0,
                        act_trans=activation,
                        act_out=activation),
            Reshape((bottle, middle_dim, middle_dim)),
            ConvTranspose2dModule(bottle, channels[-1],
                                  activation=activation),
            *[Upsample(channels[-i], channels[-i-1],
                       unpool_kernel=poolings[-i],
                       activation=activation) for i in range(1, len(channels))],
            Upsample(channels[0], bottle,
                     unpool_kernel=poolings[0],
                     activation=activation),
            ConvTranspose2dModule(bottle, in_ch,
                                  activation='Sigmoid')
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, return_params=False, **kwargs):
        tau = kwargs.get('tau') or .5

        h = self.zw_x_graph(x)
        z_x, z_x_mean, z_x_var = self.z_x_graph(h)
        w_x, w_x_mean, w_x_var = self.w_x_graph(h)
        y_wz = self.y_wz_graph(torch.cat((w_x, z_x), -1))
        z_wys, z_wy_means, z_wy_vars = self.z_wy_graph(w_x, y_wz)
        _, y_pred = torch.max(y_wz, dim=-1)
        z_wy = z_wys[torch.arange(z_wys.shape[0]), :, y_pred]
        x_z = self.x_z_graph(z_x)

        if return_params:
            return {'x': x, 'x_z': x_z,
                    'z_x': z_x,
                    'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                    'z_wys': z_wys,
                    'z_wy_means': z_wy_means, 'z_wy_vars': z_wy_vars,
                    'w_x': w_x,
                    'w_x_mean': w_x_mean, 'w_x_var': w_x_var,
                    'y_wz': y_wz, 'y_pred': y_pred }

        else:
            return x_z
