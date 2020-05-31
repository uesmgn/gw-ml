import torch
import torch.nn as nn
from .utils import nn as cn
from . import utils as ut

class ConvModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 conv_kernel=3,
                 pool_kernel=3,
                 activation='ReLU'):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=conv_kernel,
                      stride=pool_kernel,
                      padding=(conv_kernel - pool_kernel) // 2),
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
                 convt_kernel=3,
                 activation='ReLU'):
        super().__init__()
        self.features = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=convt_kernel,
                               stride=convt_kernel,
                               padding=0),
            nn.BatchNorm2d(out_ch),
            ut.activation(activation)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class Encoder(nn.Module):
    def __init__(self,
                 x_shape,
                 y_dim,
                 z_dim,
                 w_dim,
                 activation='ReLU',
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim

        nargs = nargs or dict()
        conv_ch = nargs.get('conv_channels') or [16, 32, 64]
        kernels = nargs.get('conv_kernels') or [3, 3, 3]
        pools = nargs.get('pool_kernels') or [3, 3, 3]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or activation

        self.z_x_graph = nn.Sequential(
            ConvModule(in_ch, conv_ch[0],
                       conv_kernel=kernels[0],
                       pool_kernel=pools[0],
                       activation=activation),
            ConvModule(conv_ch[0], conv_ch[1],
                       conv_kernel=kernels[1],
                       pool_kernel=pools[1],
                       activation=activation),
            ConvModule(conv_ch[1], conv_ch[2],
                       conv_kernel=kernels[2],
                       pool_kernel=pools[2],
                       activation=activation),
            nn.Flatten(),
            nn.Linear(middle_dim, z_dim * 2)  # (batch_size, z_dim * 2)
        )

        self.w_x_graph = nn.Sequential(
            ConvModule(in_ch, conv_ch[0],
                       conv_kernel=kernels[0],
                       pool_kernel=pools[0],
                       activation=activation),
            ConvModule(conv_ch[0], conv_ch[1],
                       conv_kernel=kernels[1],
                       pool_kernel=pools[1],
                       activation=activation),
            ConvModule(conv_ch[1], conv_ch[2],
                       conv_kernel=kernels[2],
                       pool_kernel=pools[2],
                       activation=activation),
            nn.Flatten(),
            nn.Linear(middle_dim, w_dim * 2)  # (batch_size, w_dim * 2)
        )

        self.y_wz_graph = nn.Sequential(
            nn.Linear(w_dim + z_dim, dense_dim),
            nn.Linear(dense_dim, y_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        _z_x = self.z_x_graph(x)
        z_x_mean, z_x_logvar = torch.split(_z_x, self.z_dim, 1)
        z_x = ut.reparameterize(z_x_mean, z_x_logvar)
        _w_x = self.w_x_graph(x)
        w_x_mean, w_x_logvar = torch.split(_w_x, self.w_dim, 1)
        w_x = ut.reparameterize(w_x_mean, w_x_logvar)
        y_wz = self.y_wz_graph(torch.cat((w_x, z_x), 1))
        return {'x': x,
                'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_logvar': z_x_logvar,
                'w_x': w_x, 'w_x_mean': w_x_mean, 'w_x_logvar': w_x_logvar,
                'y_wz': y_wz }


class Decoder(nn.Module):
    def __init__(self,
                 x_shape,
                 y_dim,
                 z_dim,
                 w_dim,
                 activation='ReLU',
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim

        nargs = nargs or dict()
        conv_ch = nargs.get('conv_channels') or [16, 32, 64]
        pools = nargs.get('pool_kernels') or [3, 3, 3]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or activation

        self.z_wy_graph = nn.Sequential(
            nn.Linear(w_dim, dense_dim),
            nn.Linear(dense_dim, z_dim * 2 * self.y_dim),
            cn.Reshape((z_dim * 2, self.y_dim))
        )

        self.x_z_graph = nn.Sequential(
            nn.Linear(z_dim, middle_dim),
            cn.Reshape((conv_ch[-1], middle_size, middle_size)),
            ConvTransposeModule(conv_ch[-1], conv_ch[-2],
                                convt_kernel=pools[-1],
                                activation=activation),
            ConvTransposeModule(conv_ch[-2], conv_ch[-3],
                                convt_kernel=pools[-2],
                                activation=activation),
            ConvTransposeModule(conv_ch[-3], in_ch,
                                convt_kernel=pools[-3],
                                activation=activation),
            nn.Sigmoid()
        )

    def forward(self, z, w):
        _z_wy = self.z_wy_graph(w)
        z_wy_means, z_wy_logvars = torch.split(_z_wy, self.z_dim, 1)
        x_z = self.x_z_graph(z)
        return {'z_wy_means': z_wy_means, # (batch_size, z_dim, K)
                'z_wy_logvars': z_wy_logvars, # (batch_size, z_dim, K)
                'x_z': x_z }

class GMVAE(nn.Module):
    def __init__(self,
                 x_shape,
                 y_dim,
                 z_dim,
                 w_dim,
                 nargs=None):
        super().__init__()

        self.encoder = Encoder(x_shape, y_dim, z_dim, w_dim, nargs)
        self.decoder = Decoder(x_shape, y_dim, z_dim, w_dim, nargs)

    def forward(self, x):
        encoder_out = self.encoder(x)
        z_x, w_x = encoder_out['z_x'], encoder_out['w_x']
        decoder_out = self.decoder(z_x, w_x)
        encoder_out.update(decoder_out)
        return encoder_out
