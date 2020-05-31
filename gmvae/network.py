import torch
import torch.nn as nn
from .utils import nn as cn
from . import utils as ut


class Encoder(nn.Module):
    def __init__(self,
                 x_shape,
                 y_dim,
                 z_dim,
                 w_dim,
                 activation=nn.Relu,
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.activation = activation

        nargs = nargs or dict()
        conv_ch = nargs.get('conv_channels') or [2, 4, 6]
        kernels = nargs.get('conv_kernels') or [3, 3, 3]
        strides = nargs.get('conv_strides') or [3, 3, 3]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024

        self.z_x_graph = nn.Sequential(
            nn.Conv2d(in_ch, conv_ch[0],
                      kernel_size=kernels[0],
                      stride=strides[0],
                      padding=(kernels[0] - strides[0]) // 2),
            nn.BatchNorm2d(conv_ch[0]),
            self.activation(),
            nn.Conv2d(conv_ch[0], conv_ch[1],
                      kernel_size=kernels[1],
                      stride=strides[1],
                      padding=(kernels[1] - strides[1]) // 2),
            nn.BatchNorm2d(conv_ch[1]),
            self.activation(),
            nn.Conv2d(conv_ch[1], conv_ch[2],
                      kernel_size=kernels[2],
                      stride=strides[2],
                      padding=(kernels[2] - strides[2]) // 2),
            nn.BatchNorm2d(conv_ch[2]),
            self.activation(),
            nn.Flatten(),
            nn.Linear(middle_dim, z_dim * 2)  # (batch_size, z_dim * 2)
        )

        self.w_x_graph = nn.Sequential(
            nn.Conv2d(in_ch, conv_ch[0],
                      kernel_size=kernels[0],
                      stride=strides[0],
                      padding=(kernels[0] - strides[0]) // 2),
            nn.BatchNorm2d(conv_ch[0]),
            self.activation(),
            nn.Conv2d(conv_ch[0], conv_ch[1],
                      kernel_size=kernels[1],
                      stride=strides[1],
                      padding=(kernels[1] - strides[1]) // 2),
            nn.BatchNorm2d(conv_ch[1]),
            self.activation(),
            nn.Conv2d(conv_ch[1], conv_ch[2],
                      kernel_size=kernels[2],
                      stride=strides[2],
                      padding=(kernels[2] - strides[2]) // 2),
            nn.BatchNorm2d(conv_ch[2]),
            self.activation(),
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
                 activation=nn.Relu
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.activation = activation

        nargs = nargs or dict()
        conv_ch = nargs.get('conv_channels') or (2, 4, 6)
        kernels = nargs.get('conv_kernels') or (3, 3, 3)
        strides = nargs.get('conv_strides') or (3, 3, 3)
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024

        self.z_wy_graph = nn.Sequential(
            nn.Linear(w_dim, dense_dim),
            nn.Linear(dense_dim, z_dim * 2 * self.y_dim),
            cn.Reshape((z_dim * 2, self.y_dim))
        )

        self.x_z_graph = nn.Sequential(
            nn.Linear(z_dim, middle_dim),
            cn.Reshape((conv_ch[-1], middle_size, middle_size)),
            nn.ConvTranspose2d(conv_ch[-1], conv_ch[-2],
                               kernel_size=kernels[-1],
                               stride=strides[-1],
                               padding=(kernels[-1] - strides[-1]) // 2),
            nn.BatchNorm2d(conv_ch[-2]),
            self.activation(),
            nn.ConvTranspose2d(conv_ch[-2], conv_ch[-3],
                               kernel_size=kernels[-2],
                               stride=strides[-2],
                               padding=(kernels[-2] - strides[-2]) // 2),
            nn.BatchNorm2d(conv_ch[-3]),
            self.activation(),
            nn.ConvTranspose2d(conv_ch[-3], in_ch,
                               kernel_size=kernels[-3],
                               stride=strides[-3],
                               padding=(kernels[-3] - strides[-3]) // 2),
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
