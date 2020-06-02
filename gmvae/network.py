import torch
import torch.nn as nn
from .utils import nn as cn
from . import utils as ut
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=3,
                 activation='ReLU'):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=kernel,
                      stride=1,
                      padding=(kernel - 1) // 2),
            nn.BatchNorm2d(out_ch),
            ut.activation(activation)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class ConvTransposeModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=3,
                 activation='ReLU'):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=kernel,
                               stride=kernel,
                               padding=0),
            nn.BatchNorm2d(out_ch),
            ut.activation(activation)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Gaussian(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit)
        if self.training:
            x = ut.reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch,
                 pool_kernel=3,
                 activation='ReLu'):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=pool_kernel,
                         stride=pool_kernel),
            ConvModule(in_ch, out_ch,
                       kernel=1,
                       activation=activation)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch,
                 pool_kernel=3,
                 activation='ReLu'):
        super().__init__()
        self.features = ConvTransposeModule(in_ch,
                                            out_ch,
                                            kernel=pool_kernel,
                                            activation=activation)

    def forward(self, x):
        x = self.features(x)
        return x


class DenseModule(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 middle_dim=1024,
                 n_middle_layers=0,
                 drop=0,
                 drop_rate=0.5,
                 act_trans='ReLU',
                 act_out=None):
        super().__init__()
        self.features = nn.Sequential()
        if n_middle_layers > 0:
            self.features.add_module(f'h0',
                                     nn.Linear(in_dim, middle_dim))
        else:
            self.features.add_module(f'h0',
                                     nn.Linear(in_dim, out_dim))
        for i in range(n_middle_layers):
            if drop:
                self.features.add_module(f'drop{i+1}',
                                         nn.Dropout(p=drop_rate,
                                                    inplace=True))
            else:
                self.features.add_module(f'bn{i+1}',
                                         nn.BatchNorm1d(middle_dim))
            if act_trans is not None:
                self.features.add_module(f'act_trans{i+1}',
                                         ut.activation(act_trans))
            # last layer
            if i == n_middle_layers - 1:
                self.features.add_module(f'h{i+1}',
                                         nn.Linear(middle_dim, out_dim))
            else:
                self.features.add_module(f'h{i+1}',
                                         nn.Linear(middle_dim, middle_dim))

        if act_out is not None:
            self.features.add_module('act_out',
                                     ut.activation(act_out))

    def forward(self, x):
        x = self.features(x)
        return x


class GMVAE_graph(nn.Module):
    def __init__(self,
                 x_shape,
                 y_dim,
                 z_dim,
                 w_dim,
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim

        nargs = nargs or dict()
        bottle_ch = nargs.get('bottle_channel') or 32
        conv_ch = nargs.get('conv_channels') or [64, 128, 256]
        kernels = nargs.get('conv_kernels') or [3, 3, 3]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or 'ReLU'
        drop_rate = nargs.get('drop_rate') or 0.5

        self.sigma = nargs.get('sigma') or 0.01

        self.z_x_graph = nn.Sequential(
            ConvModule(in_ch, bottle_ch,
                       kernel=1,
                       activation=activation),
            DownSample(bottle_ch, conv_ch[0],
                       pool_kernel=pool_kernels[0],
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       pool_kernel=pool_kernels[1],
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       pool_kernel=pool_kernels[2],
                       activation=activation),
            nn.Flatten(),
            DenseModule(middle_dim, z_dim * 2,
                        n_middle_layers=1,
                        drop_rate=drop_rate,
                        act_trans=activation,
                        act_out='Tanh'), # (batch_size, z_dim * 2)
            Gaussian()
        )

        self.w_x_graph = nn.Sequential(
            ConvModule(in_ch, bottle_ch,
                       kernel=1,
                       activation=activation),
            DownSample(bottle_ch, conv_ch[0],
                       pool_kernel=pool_kernels[0],
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       pool_kernel=pool_kernels[1],
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       pool_kernel=pool_kernels[2],
                       activation=activation),
            nn.Flatten(),
            DenseModule(middle_dim, w_dim * 2,
                        n_middle_layers=1,
                        drop_rate=drop_rate,
                        act_trans=activation,
                        act_out='Tanh'), # (batch_size, z_dim * 2)
            Gaussian()
        )

        self.y_wz_graph = DenseModule(w_dim + z_dim,
                                      y_dim,
                                      n_middle_layers=1,
                                      drop_rate=drop_rate,
                                      act_out='Softmax')

        self.z_wy_graph = nn.Sequential(
            DenseModule(w_dim, z_dim * 2 * y_dim,
                        n_middle_layers=1,
                        drop_rate=drop_rate,
                        act_trans=activation,
                        act_out='Tanh'), # (batch_size, z_dim * 2)
            cn.Reshape((z_dim * 2, y_dim)),
            Gaussian()
        )

        self.x_z_graph = nn.Sequential(
            nn.Linear(z_dim, middle_dim),
            cn.Reshape((conv_ch[-1], middle_size, middle_size)),
            Upsample(conv_ch[-1], conv_ch[-2],
                     pool_kernel=pool_kernels[-1],
                     activation=activation),
            Upsample(conv_ch[-2], conv_ch[-3],
                     pool_kernel=pool_kernels[-2],
                     activation=activation),
            Upsample(conv_ch[-3], bottle_ch,
                     pool_kernel=pool_kernels[-3],
                     activation=activation),
            ConvTransposeModule(bottle_ch, in_ch,
                                kernel=1,
                                activation='Sigmoid'),
        )

    def forward(self, x):
        # Encoder
        z_x, z_x_mean, z_x_var = self.z_x_graph(x)
        w_x, w_x_mean, w_x_var = self.w_x_graph(x)
        y_wz = self.y_wz_graph(torch.cat((w_x, z_x), 1))
        # Decoder
        z_wys, z_wy_means, z_wy_vars = self.z_wy_graph(w_x) # (batch_size, z_dim, K)
        _, p = torch.max(y_wz, dim=1) # (batch_size, )
        z_wy = z_wys[torch.arange(z_wys.shape[0]),:,p] # (batch_size, z_dim)
        x_z_mean = self.x_z_graph(z_x) # EDIT
        x_z = ut.reparameterize(x_z_mean, self.sigma)
        return {'x': x,
                'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                'w_x': w_x, 'w_x_mean': w_x_mean, 'w_x_var': w_x_var,
                'y_wz': y_wz,
                'y_pred': p,
                'z_wy': z_wy, # (batch_size, z_dim, K)
                'z_wys': z_wys,
                'z_wy_means': z_wy_means,
                'z_wy_vars': z_wy_vars,
                'x_z': x_z }


class GMVAE(nn.Module):
    def __init__(self,
                 x_shape,
                 y_dim,
                 z_dim,
                 w_dim,
                 nargs=None):
        super().__init__()

        self.net = GMVAE_graph(x_shape, y_dim, z_dim, w_dim, nargs)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        args = self.net(x)
        return args
