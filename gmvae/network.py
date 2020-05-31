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
        means, logits = torch.split(x, x.shape[0] // 2, 1)
        vars = F.softplus(logits)
        x = ut.reparameterize(means, vars)
        return x, means, vars


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
            self.features.add_module(f'dropout',
                                     nn.Dropout())
            if act_trans is not None:
                self.features.add_module(act_trans,
                                         ut.activation(act_trans))
            # last layer
            if i == n_middle_layers - 1:
                self.features.add_module(f'h{i+1}',
                                         nn.Linear(middle_dim, out_dim))
            else:
                self.features.add_module(f'h{i+1}',
                                         nn.Linear(middle_dim, middle_dim))

        if act_out:
            self.features.add_module(act_out,
                                     ut.activation(act_out))

    def forward(self, x):
        x = self.features(x)
        return x


class Encoder(nn.Module):
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
        conv_ch = nargs.get('conv_channels') or [16, 32, 64]
        kernels = nargs.get('conv_kernels') or [3, 3, 3]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or 'ReLU'

        self.z_x_graph = nn.Sequential(
            DownSample(in_ch, conv_ch[0],
                       pool_kernel=pool_kernels[0],
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       pool_kernel=pool_kernels[1],
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       pool_kernel=pool_kernels[2],
                       activation=activation),
            nn.Flatten(),
            nn.Linear(middle_dim, z_dim * 2),  # (batch_size, z_dim * 2)
            Gaussian()
        )

        self.w_x_graph = nn.Sequential(
            DownSample(in_ch, conv_ch[0],
                       pool_kernel=pool_kernels[0],
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       pool_kernel=pool_kernels[1],
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       pool_kernel=pool_kernels[2],
                       activation=activation),
            nn.Flatten(),
            nn.Linear(middle_dim, w_dim * 2),  # (batch_size, w_dim * 2)
            Gaussian()
        )

        self.y_wz_graph = DenseModule(w_dim + z_dim,
                                      y_dim,
                                      n_middle_layers=1,
                                      act_out='Softmax')

    def forward(self, x):
        z_x, z_x_mean, z_x_var = self.z_x_graph(x)
        w_x, w_x_mean, w_x_var = self.w_x_graph(x)
        y_wz = self.y_wz_graph(torch.cat((w_x, z_x), 1))
        return {'x': x,
                'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                'w_x': w_x, 'w_x_mean': w_x_mean, 'w_x_var': w_x_var,
                'y_wz': y_wz}


class Decoder(nn.Module):
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
        conv_ch = nargs.get('conv_channels') or [16, 32, 64]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or 'ReLU'

        self.z_wy_graph = nn.Sequential(
            DenseModule(w_dim,
                        z_dim * 2 * self.y_dim,
                        n_middle_layers=1),
            cn.Reshape((z_dim * 2, self.y_dim),
            Gaussian())
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
            Upsample(conv_ch[-3], in_ch,
                     pool_kernel=pool_kernels[-3],
                     activation='Sigmoid')
        )

    def forward(self, z, w):
        z_wy, z_wy_means, z_wy_vars = self.z_wy_graph(w)
        x_z = self.x_z_graph(z)
        return {'z_wy_means': z_wy_means,  # (batch_size, z_dim, K)
                'z_wy_vars': z_wy_vars,  # (batch_size, z_dim, K)
                'x_z': x_z}


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

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        encoder_out = self.encoder(x)
        z_x, w_x = encoder_out['z_x'], encoder_out['w_x']
        decoder_out = self.decoder(z_x, w_x)
        encoder_out.update(decoder_out)
        return encoder_out
