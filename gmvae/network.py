import os
import torch
import torch.nn as nn
from . import nn as cn
import torch.nn.functional as F


class GMVAE(nn.Module):
    def __init__(self, nargs=None):
        super().__init__()

        nargs = nargs or dict()
        x_shape = nargs.get('x_shape') or (1, 486, 486)
        in_ch = x_shape[0]
        y_dim = nargs.get('y_dim') or 10
        z_dim = nargs.get('z_dim') or 20
        w_dim = nargs.get('w_dim') or 20
        bottle_ch = nargs.get('bottle_channel') or 16
        conv_ch = nargs.get('conv_channels') or [32, 48, 64, 32]
        kernels = nargs.get('kernels') or [3, 3, 3, 3]
        pool_kernels = nargs.get('pool_kernels') or [3, 3, 3, 3]
        middle_size = nargs.get('middle_size') or 6
        middle_dim = bottle_ch * middle_size * middle_size
        hidden_dim = nargs.get('hidden_dim') or 256
        activation = nargs.get('activation') or 'ReLU'
        pooling = nargs.get('pooling') or 'max'

        self.zw_x_graph = nn.Sequential(
            cn.Conv2dModule(in_ch, bottle_ch,
                            activation=activation),
            cn.DownSample(bottle_ch, conv_ch[0],
                          kernel=kernels[0],
                          pool_kernel=pool_kernels[0],
                          pooling=pooling,
                          activation=activation),
            cn.DownSample(conv_ch[0], conv_ch[1],
                          kernel=kernels[1],
                          pool_kernel=pool_kernels[1],
                          pooling=pooling,
                          activation=activation),
            cn.DownSample(conv_ch[1], conv_ch[2],
                          kernel=kernels[2],
                          pool_kernel=pool_kernels[2],
                          pooling=pooling,
                          activation=activation),
            cn.DownSample(conv_ch[2], conv_ch[3],
                          kernel=kernels[3],
                          pool_kernel=pool_kernels[3],
                          pooling=pooling,
                          activation=activation),
            cn.Conv2dModule(conv_ch[3], bottle_ch,
                            activation=activation),
            nn.Flatten()
        )

        self.z_x_graph = nn.Sequential(
            nn.Linear(middle_dim, z_dim * 2),
            cn.Gaussian()
        )

        self.w_x_graph = nn.Sequential(
            nn.Linear(middle_dim, w_dim * 2),
            cn.Gaussian()
        )

        self.y_wz_graph = nn.Sequential(
            cn.DenseModule(w_dim + z_dim,
                           y_dim,
                           n_layers=1,
                           hidden_dim=hidden_dim,
                           act_trans=activation),
            cn.GumbelSoftmax()
        )

        self.z_wy_graphs = nn.ModuleList([
            nn.Sequential(
                cn.DenseModule(w_dim, z_dim * 2),
                cn.Gaussian()
            ) for _ in range(y_dim)
        ])


        self.x_z_graph = nn.Sequential(
            cn.DenseModule(z_dim, middle_dim,
                           act_out=activation),
            cn.Reshape((bottle_ch, middle_size, middle_size)),
            cn.ConvTranspose2dModule(bottle_ch, conv_ch[-1],
                                     activation=activation),
            cn.Upsample(conv_ch[-1], conv_ch[-2],
                        unpool_kernel=pool_kernels[-1],
                        activation=activation),
            cn.Upsample(conv_ch[-2], conv_ch[-3],
                        unpool_kernel=pool_kernels[-2],
                        activation=activation),
            cn.Upsample(conv_ch[-3], conv_ch[-4],
                        unpool_kernel=pool_kernels[-3],
                        activation=activation),
            cn.Upsample(conv_ch[-4], bottle_ch,
                        unpool_kernel=pool_kernels[-4],
                        activation=activation),
            cn.ConvTranspose2dModule(bottle_ch, in_ch,
                                     kernel=1,
                                     activation='Sigmoid')
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x, clustering=False, return_params=False):
        if clustering:
            return self.clustering(x)
        else:
            return self.fit_train(x, return_params)

    def fit_train(self, x, return_params=False):
        # Encoder
        h = self.zw_x_graph(x)
        z_x, z_x_mean, z_x_var = self.z_x_graph(h)
        w_x, w_x_mean, w_x_var = self.w_x_graph(h)
        # pi: softmax(MLP(torch.cat((w_x, z_x)))
        # y_wz: gumbel-sample
        pi, y_wz = self.y_wz_graph(torch.cat((w_x, z_x), 1))
        x_z = self.x_z_graph(z_x)

        # Decoder
        z_wys_stack = []
        z_wy_means_stack = []
        z_wy_vars_stack = []
        for graph in self.z_wy_graphs:
            z_wy, z_wy_mean, z_wy_var = graph(w_x)
            z_wys_stack.append(z_wy)
            z_wy_means_stack.append(z_wy_mean)
            z_wy_vars_stack.append(z_wy_var)
        z_wys = torch.stack(z_wys_stack, 2)
        z_wy_means = torch.stack(z_wy_means_stack, 2)
        z_wy_vars = torch.stack(z_wy_vars_stack, 2)

        _, y_pred = torch.max(pi, dim=1)
        z_wy = z_wys[torch.arange(z_wys.shape[0]), :, y_pred]

        if return_params:
            return  {'x': x, 'x_z': x_z,
                     'z_x': z_x, 'z_x_mean': z_x_mean, 'z_x_var': z_x_var,
                     'w_x': w_x, 'w_x_mean': w_x_mean, 'w_x_var': w_x_var,
                     'pi': pi,
                     'y_wz': y_wz,
                     'y_pred': y_pred,
                     'z_wy': z_wy,
                     'z_wys': z_wys,
                     'z_wy_means': z_wy_means, 'z_wy_vars': z_wy_vars }
        else:
            return x_z

    def clustering(self, x):
        # Encoder
        h = self.zw_x_graph(x)
        z_x, _, _ = self.z_x_graph(h)
        w_x, _, _ = self.w_x_graph(h)
        # pi: softmax(MLP(torch.cat((w_x, z_x)))
        # y_wz: gumbel-sample
        pi, y_wz = self.y_wz_graph(torch.cat((w_x, z_x), 1))
        _, y_pred = torch.max(pi, dim=1)

        return  {'pi': pi,
                 'y_wz': y_wz,
                 'y_pred': y_pred }
