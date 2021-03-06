import os
import torch
import torch.nn as nn
from .utils import nn as cn
from . import utils as ut
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=1,
                 stride=1,
                 activation='ReLU',
                 dim=2):
        super().__init__()

        self.features = nn.Sequential()

        if dim == 2:
            self.features.add_module('Conv2d',
                                     nn.Conv2d(in_ch, out_ch,
                                               kernel_size=kernel,
                                               stride=stride,
                                               padding=(kernel - stride) // 2))
            self.features.add_module('BatchNorm2d',
                                     nn.BatchNorm2d(out_ch))
        else:
            self.features.add_module('Conv1d',
                                     nn.Conv1d(in_ch, out_ch,
                                               kernel_size=kernel,
                                               stride=stride,
                                               padding=(kernel - stride) // 2))
            self.features.add_module('BatchNorm1d',
                                     nn.BatchNorm1d(out_ch))
        self.features.add_module('activation',
                                 ut.activation(activation))

    def forward(self, x):
        x = self.features(x)
        return x


class ConvTransposeModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=1,
                 stride=1,
                 activation=None,
                 dim=2):
        super().__init__()

        self.features = nn.Sequential()

        if dim == 2:
            self.features.add_module('ConvTranspose2d',
                                     nn.ConvTranspose2d(in_ch, out_ch,
                                                        kernel_size=kernel,
                                                        stride=stride,
                                                        padding=(kernel - stride) // 2))
            self.features.add_module('BatchNorm2d',
                                     nn.BatchNorm2d(out_ch))
        else:
            self.features.add_module('ConvTranspose1d',
                                     nn.ConvTranspose1d(in_ch, out_ch,
                                                        kernel_size=kernel,
                                                        stride=stride,
                                                        padding=(kernel - stride) // 2))
            self.features.add_module('BatchNorm1d',
                                     nn.BatchNorm1d(out_ch))
        if activation is not None:
            self.features.add_module('activation',
                                     ut.activation(activation))

    def forward(self, x):
        x = self.features(x)
        return x


class GlobalPool(nn.Module):
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
    def __init__(self,
                 in_dim,
                 out_dim,
                 act_regur='Tanh'):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2)
        )
        if act_regur is not None:
            self.features.add_module('act_regur',
                                     ut.activation(act_regur))

    def forward(self, x):
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + 1e-8
        # if self.training:
        #     x = ut.reparameterize(mean, var)
        # else:
        #     x = mean
        x = ut.reparameterize(mean, var)
        return x, mean, var


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
                                     ConvModule(in_ch, in_ch,
                                                kernel=kernel,
                                                stride=pool_kernel,
                                                activation=activation))
        self.features.add_module(f'1x1conv',
                                 ConvModule(in_ch, out_ch,
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
            ConvTransposeModule(in_ch,
                                in_ch,
                                kernel=unpool_kernel,
                                stride=unpool_kernel,
                                activation=activation),
            ConvTransposeModule(in_ch, out_ch,
                                activation=activation))

    def forward(self, x):
        x = self.features(x)
        return x


class DenseModule(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 middle_dim=1024,
                 n_middle_layers=0,
                 norm_trans='bn',
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
            if norm_trans in ('Dropout', 'dropout', 'drop'):
                self.features.add_module(f'drop{i+1}',
                                         nn.Dropout(p=drop_rate,
                                                    inplace=True))
            elif norm_trans in ('BatchNorm', 'batchnorm', 'bn'):
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


class VAE_F(nn.Module):
    def __init__(self,
                 x_shape,
                 z_dim,
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.z_dim = z_dim

        nargs = nargs or dict()
        bottle_ch = nargs.get('bottle_channel') or 32
        conv_ch = nargs.get('conv_channels') or [64, 128, 256]
        kernels = nargs.get('conv_kernels') or [3, 3, 3]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3]
        unpool_kernels = nargs.get('unpool_kernels') or [5, 5, 5]
        middle_size = nargs.get('middle_size') or 18
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or 'ReLU'
        drop_rate = nargs.get('drop_rate') or 0.5
        pooling = nargs.get('pooling') or 'max'

        self.z_x_graph = nn.Sequential(
            ConvModule(in_ch, bottle_ch,
                       activation=activation),
            DownSample(bottle_ch, conv_ch[0],
                       pool_kernel=pool_kernels[0],
                       pooling=pooling,
                       conv_kernel=kernels[0],
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       pool_kernel=pool_kernels[1],
                       pooling=pooling,
                       conv_kernel=kernels[1],
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       pool_kernel=pool_kernels[2],
                       pooling=pooling,
                       conv_kernel=kernels[2],
                       activation=activation),
            nn.Flatten(),
            DenseModule(middle_dim, z_dim * 2,
                        n_middle_layers=0),
            Gaussian(in_dim=z_dim * 2,
                     out_dim=z_dim)
        )

        self.x_z_graph = nn.Sequential(
            DenseModule(z_dim, middle_dim,
                        n_middle_layers=0,
                        act_out=activation),
            cn.Reshape((conv_ch[-1], middle_size, middle_size)),
            Upsample(conv_ch[-1], conv_ch[-2],
                     pool_kernel=unpool_kernels[-1],
                     stride=pool_kernels[-1],
                     activation=activation),
            Upsample(conv_ch[-2], conv_ch[-3],
                     pool_kernel=unpool_kernels[-2],
                     stride=pool_kernels[-2],
                     activation=activation),
            Upsample(conv_ch[-3], bottle_ch,
                     pool_kernel=unpool_kernels[-3],
                     stride=pool_kernels[-3],
                     activation=activation),
            ConvTransposeModule(bottle_ch, in_ch,
                                kernel=1,
                                activation='Sigmoid'),
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_params=False):
        # Encoder
        z_x, z_x_mean, z_x_var = self.z_x_graph(x)
        # Decoder
        x_z = self.x_z_graph(z_x) # EDIT
        if return_params:
            return {'x': x,
                    'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                    'x_z': x_z}
        else:
            return x_z


class VAE(nn.Module):
    def __init__(self,
                 x_shape,
                 z_dim,
                 nargs=None):
        super().__init__()
        in_ch = x_shape[0]
        self.in_width = x_shape[1]
        self.in_height = x_shape[2]
        self.z_dim = z_dim

        nargs = nargs or dict()
        bottle_ch = nargs.get('bottle_channel') or 20
        conv_ch = nargs.get('conv_channels') or [40, 60, 80, 100]
        kernels = nargs.get('conv_kernels') or [11, 5, 3, 3]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3, 3]
        unpool_kernels = nargs.get('unpool_kernels') or [3, 3, 3, 3]
        middle_size = nargs.get('middle_size') or 6
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or 'ReLU'
        drop_rate = nargs.get('drop_rate') or 0.5
        pooling = nargs.get('pooling') or 'max'

        self.z_x_graph = nn.Sequential(
            ConvModule(in_ch, bottle_ch,
                       activation=activation),
            DownSample(bottle_ch, conv_ch[0],
                       kernel=kernels[0],
                       pool_kernel=pool_kernels[0],
                       pooling=pooling,
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       pool_kernel=pool_kernels[1],
                       pooling=pooling,
                       conv_kernel=kernels[1],
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       pool_kernel=pool_kernels[2],
                       pooling=pooling,
                       conv_kernel=kernels[2],
                       activation=activation),
            DownSample(conv_ch[2], conv_ch[3],
                       pool_kernel=pool_kernels[3],
                       pooling=pooling,
                       conv_kernel=kernels[3],
                       activation=activation),
            nn.Flatten(),
            DenseModule(middle_dim, z_dim * 2,
                        n_middle_layers=0),  # (batch_size, z_dim * 2)
            Gaussian(in_dim=z_dim * 2,
                     out_dim=z_dim)
        )

        self.x_z_graph = nn.Sequential(
            DenseModule(z_dim, middle_dim,
                        n_middle_layers=0,
                        act_out=activation),
            cn.Reshape((conv_ch[-1], middle_size, middle_size)),
            Upsample(conv_ch[-1], conv_ch[-2],
                     pool_kernel=pool_kernels[-1],
                     stride=unpool_kernels[-1],
                     activation=activation),
            Upsample(conv_ch[-2], conv_ch[-3],
                     pool_kernel=pool_kernels[-2],
                     stride=unpool_kernels[-2],
                     activation=activation),
            Upsample(conv_ch[-3], conv_ch[-4],
                     pool_kernel=pool_kernels[-3],
                     stride=unpool_kernels[-3],
                     activation=activation),
            Upsample(conv_ch[-4], bottle_ch,
                     pool_kernel=pool_kernels[-4],
                     stride=unpool_kernels[-4],
                     activation=activation),
            ConvTransposeModule(bottle_ch, in_ch,
                                kernel=1,
                                activation='Sigmoid')
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_params=False):
        # Encoder
        z_x, z_x_mean, z_x_var = self.z_x_graph(x)
        # Decoder
        x_z = self.x_z_graph(z_x) # EDIT
        if return_params:
            return {'x': x,
                    'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                    'x_z': x_z}
        else:
            return x_z




class GMVAE(nn.Module):
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
        bottle_ch = nargs.get('bottle_channel') or 16
        conv_ch = nargs.get('conv_channels') or [32, 48, 64, 80]
        kernels = nargs.get('kernels') or [3, 3, 3, 3]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3, 3]
        unpool_kernels = nargs.get('unpool_kernels') or [3, 3, 3, 3]
        middle_size = nargs.get('middle_size') or 6
        middle_dim = conv_ch[-1] * middle_size * middle_size
        dense_dim = nargs.get('dense_dim') or 1024
        activation = nargs.get('activation') or 'ReLU'
        drop_rate = nargs.get('drop_rate') or 0.5
        pooling = nargs.get('pooling') or 'max'

        self.zw_x_graph = nn.Sequential(
            ConvModule(in_ch, bottle_ch,
                       activation=activation),
            DownSample(bottle_ch, conv_ch[0],
                       kernel=kernels[0],
                       pool_kernel=pool_kernels[0],
                       pooling=pooling,
                       activation=activation),
            DownSample(conv_ch[0], conv_ch[1],
                       kernel=kernels[1],
                       pool_kernel=pool_kernels[1],
                       pooling=pooling,
                       activation=activation),
            DownSample(conv_ch[1], conv_ch[2],
                       kernel=kernels[2],
                       pool_kernel=pool_kernels[2],
                       pooling=pooling,
                       activation=activation),
            DownSample(conv_ch[2], conv_ch[3],
                       kernel=kernels[3],
                       pool_kernel=pool_kernels[3],
                       pooling=pooling,
                       activation=activation),
            nn.Flatten()
        )

        self.z_x_graph = nn.Sequential(
            DenseModule(middle_dim, z_dim * 2,
                        n_middle_layers=0),  # (batch_size, z_dim * 2)
            Gaussian(in_dim=z_dim * 2,
                     out_dim=z_dim)
        )

        self.w_x_graph = nn.Sequential(
            DenseModule(middle_dim, w_dim * 2,
                        n_middle_layers=0),  # (batch_size, z_dim * 2)
            Gaussian(in_dim=w_dim * 2,
                     out_dim=w_dim)
        )

        self.y_wz_graph = DenseModule(w_dim + z_dim,
                                      y_dim,
                                      n_middle_layers=1,
                                      act_trans=activation,
                                      act_out='Softmax')


        self.z_wy_graphs = nn.ModuleList([
            nn.Sequential(
                DenseModule(w_dim, z_dim * 2,
                            n_middle_layers=0), # (batch_size, z_dim * 2)
                Gaussian(in_dim=z_dim * 2,
                         out_dim=z_dim)
            ) for _ in range(self.y_dim)
        ])


        self.x_z_graph = nn.Sequential(
            DenseModule(z_dim, middle_dim,
                        n_middle_layers=0,
                        act_out=activation),
            cn.Reshape((conv_ch[-1], middle_size, middle_size)),
            Upsample(conv_ch[-1], conv_ch[-2],
                     unpool_kernel=unpool_kernels[-1],
                     activation=activation),
            Upsample(conv_ch[-2], conv_ch[-3],
                     unpool_kernel=unpool_kernels[-2],
                     activation=activation),
            Upsample(conv_ch[-3], conv_ch[-4],
                     unpool_kernel=unpool_kernels[-3],
                     activation=activation),
            Upsample(conv_ch[-4], bottle_ch,
                     unpool_kernel=unpool_kernels[-4],
                     activation=activation),
            ConvTransposeModule(bottle_ch, in_ch,
                                kernel=1,
                                activation='Sigmoid')
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_params=False):
        # Encoder
        h = self.zw_x_graph(x) # (batch_size, 1, 486, 486) -> 100*6*6
        z_x, z_x_mean, z_x_var = self.z_x_graph(h) # (batch_size, 100*6*6) -> z_dim
        w_x, w_x_mean, w_x_var = self.w_x_graph(h) # (batch_size, 100*6*6) -> w_dim
        y_wz = self.y_wz_graph(torch.cat((w_x, z_x), 1)) # z_dim+w_dim -> y_dim
        # Decoder
        z_wys_stack = []
        z_wy_means_stack = []
        z_wy_vars_stack = []
        for graph in self.z_wy_graphs:
            # (batch_size, z_dim)
            z_wy, z_wy_mean, z_wy_var = graph(w_x)
            z_wys_stack.append(z_wy)
            z_wy_means_stack.append(z_wy_mean)
            z_wy_vars_stack.append(z_wy_var)
        z_wys = torch.stack(z_wys_stack, 2)
        z_wy_means = torch.stack(z_wy_means_stack, 2)
        z_wy_vars = torch.stack(z_wy_vars_stack, 2)
        _, p = torch.max(y_wz, dim=1)  # (batch_size, )
        z_wy = z_wys[torch.arange(z_wys.shape[0]), :, p]  # (batch_size, z_dim)
        x_z = self.x_z_graph(z_x) 
        if return_params:
            return {'x': x,
                    'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                    'w_x': w_x, 'w_x_mean': w_x_mean, 'w_x_var': w_x_var,
                    'y_wz': y_wz,
                    'y_pred': p,
                    'z_wy': z_wy,  # (batch_size, z_dim, K)
                    'z_wys': z_wys,
                    'z_wy_means': z_wy_means,
                    'z_wy_vars': z_wy_vars,
                    'x_z': x_z}
        else:
            return x_z
