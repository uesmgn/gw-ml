import torch
from torch import nn
from torch.nn import functional as F
from .Layers import *

class VAENet(nn.Module):
    def __init__(self, x_size, z_dim, y_dim):
        super().__init__()

        middle_channel = 192
        middle_size = 18
        middle_dim = middle_channel * middle_size ** 2
        n_bottle = 16

        self.encoder = nn.Sequential(
            ConvModule(1, 32, 1, 1),
            DownSample(32, 64, kernel_size=3, stride=3,
                       return_indices=False, type='max'),
            DownSample(64, 128, kernel_size=3, stride=3,
                       return_indices=False, type='max'),
            DownSample(128, 192, kernel_size=3, stride=3,
                       return_indices=False, type='max'),
            nn.Flatten()
        )
        self.gumbel = GumbelSoftmax(middle_dim, y_dim)
        self.gaussian = Gaussian(middle_dim + y_dim, z_dim)

        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_logvar = nn.Linear(y_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, middle_dim),
            Reshape((middle_channel, middle_size, middle_size)),
            Upsample(192, 128, kernel_size=3, stride=3,
                     accept_indices=False),
            Upsample(128, 64, kernel_size=3, stride=3,
                     accept_indices=False),
            Upsample(64, 32, kernel_size=3, stride=3,
                     accept_indices=False),
            ConvtModule(32, 1, 1, 1, activation='Sigmoid')
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, temp=1.0, reparameterize=True):
        indices = []

        for layer in self.encoder:
            if hasattr(layer, 'return_indices') and layer.return_indices:
                x, indice = layer(x)
                indices.append(indice)
            else:
                x = layer(x)

        y_logits, y_prob, y = self.gumbel(x, temp=temp)
        y_mu = self.y_mu(y)
        y_logvar = self.y_logvar(y)
        z, z_mu, z_logvar = self.gaussian(torch.cat((x, y), 1), reparameterize)

        x_ = z

        for layer in self.decoder:
            if hasattr(layer, 'accept_indices') and layer.accept_indices:
                x_ = layer(x_, indices.pop(-1))
            else:
                x_ = layer(x_)

        return {'x_reconst': x_,
                'z': z,
                'z_mu': z_mu,
                'z_logvar': z_logvar,
                'y': y,
                'y_logits': y_logits,
                'y_prob': y_prob,
                'y_mu': y_mu,
                'y_logvar': y_logvar }

#
# class VAENet_Inception(nn.Module):
#     def __init__(self, x_size, z_dim, y_dim):
#         super().__init__()
#
#         middle_channel = 384
#         middle_size = 18
#         middle_dim = middle_channel * middle_size ** 2
#         n_bottle = 16
#
#         self.encoder = nn.Sequential(
#             ConvModule(1, 64, 1, 1, activation='tanh'),
#             ConvInceptionModule(
#                 64, bottle_channel=n_bottle, activation='tanh'),
#             DownSample(64, 128, kernel_size=3, stride=3,
#                        activation='tanh', return_indices=False, type='avg'),
#             ConvInceptionModule(
#                 128, bottle_channel=n_bottle, activation='tanh', hard=False),
#             DownSample(128, 192, kernel_size=3, stride=3,
#                        activation='tanh', return_indices=False, type='avg'),
#             ConvInceptionModule(
#                 192, bottle_channel=n_bottle, activation='tanh', hard=False),
#             DownSample(192, 384, kernel_size=3, stride=3,
#                        activation='tanh', return_indices=False, type='avg'),
#             ConvInceptionModule(
#                 384, bottle_channel=n_bottle, activation='tanh', hard=False),
#             nn.Flatten()
#         )
#         self.gumbel = GumbelSoftmax(middle_dim, y_dim)
#         self.gaussian = Gaussian(middle_dim + y_dim, z_dim)
#
#         self.y_mu = nn.Linear(y_dim, z_dim)
#         self.y_logvar = nn.Linear(y_dim, z_dim)
#
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, middle_dim),
#             Reshape((middle_channel, middle_size, middle_size)),
#             ConvtInceptionModule(
#                 384, bottle_channel=n_bottle, activation='tanh', hard=False),
#             Upsample(384, 192, kernel_size=3, stride=3,
#                      activation='tanh', accept_indices=False),
#             ConvtInceptionModule(
#                 192, bottle_channel=n_bottle, activation='tanh', hard=False),
#             Upsample(192, 128, kernel_size=3, stride=3,
#                      activation='tanh', accept_indices=False),
#             ConvtInceptionModule(
#                 128, bottle_channel=n_bottle, activation='tanh', hard=False),
#             Upsample(128, 64, kernel_size=3, stride=3,
#                      activation='tanh', accept_indices=False),
#             ConvtInceptionModule(
#                 64, bottle_channel=n_bottle, activation='tanh'),
#             ConvtModule(64, 1, 1, 1, activation='Sigmoid')
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias.data is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, reparameterize=True):
#         indices = []
#
#         for layer in self.encoder:
#             if hasattr(layer, 'return_indices') and layer.return_indices:
#                 x, indice = layer(x)
#                 indices.append(indice)
#             else:
#                 x = layer(x)
#
#         y_pi, y = self.gumbel(x, temp=1.0)
#         y_mu = self.y_mu(y)
#         y_logvar = self.y_logvar(y)
#         z, z_mu, z_logvar = self.gaussian(torch.cat((x, y), 1), reparameterize)
#
#         x_ = z
#
#         for layer in self.decoder:
#             if hasattr(layer, 'accept_indices') and layer.accept_indices:
#                 x_ = layer(x_, indices.pop(-1))
#             else:
#                 x_ = layer(x_)
#
#         return {'x_reconst': x_,
#                 'z': z,
#                 'z_mu': z_mu,
#                 'z_logvar': z_logvar,
#                 'y': y,
#                 'y_pi': y_pi,
#                 'y_mu': y_mu,
#                 'y_logvar': y_var }
#
#
# class VAENet_M2(nn.Module):
#     def __init__(self, x_size, z_dim, y_dim):
#         super().__init__()
#
#         middle_channel = 192
#         middle_size = 18
#         middle_dim = middle_channel * middle_size ** 2
#         n_bottle = 16
#
#         self.encoder = nn.Sequential(
#             ConvModule(1, 32, 1, 1, activation='tanh'),
#             ConvInceptionModule(
#                 32, bottle_channel=n_bottle, activation='tanh', hard=2),
#             DownSample(32, 64, kernel_size=3, stride=3,
#                        activation='tanh', return_indices=False, type='max'),
#             ConvInceptionModule(
#                 64, bottle_channel=n_bottle, activation='tanh', hard=2),
#             DownSample(64, 128, kernel_size=3, stride=3,
#                        activation='tanh', return_indices=False, type='max'),
#             ConvInceptionModule(
#                 128, bottle_channel=n_bottle, activation='tanh', hard=2),
#             DownSample(128, 192, kernel_size=3, stride=3,
#                        activation='tanh', return_indices=False, type='max'),
#             ConvInceptionModule(
#                 192, bottle_channel=n_bottle, activation='tanh', hard=2),
#             nn.Flatten()
#         )
#         self.gaussian = Gaussian(middle_dim, z_dim)
#
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, middle_dim),
#             Reshape((middle_channel, middle_size, middle_size)),
#             ConvtInceptionModule(
#                 192, bottle_channel=n_bottle, activation='tanh', hard=2),
#             Upsample(192, 128, kernel_size=3, stride=3,
#                      activation='tanh', accept_indices=False),
#             ConvtInceptionModule(
#                 128, bottle_channel=n_bottle, activation='tanh', hard=2),
#             Upsample(128, 64, kernel_size=3, stride=3,
#                      activation='tanh', accept_indices=False),
#             ConvtInceptionModule(
#                 64, bottle_channel=n_bottle, activation='tanh', hard=2),
#             Upsample(64, 32, kernel_size=3, stride=3,
#                      activation='tanh', accept_indices=False),
#             ConvtInceptionModule(
#                 32, bottle_channel=n_bottle, activation='tanh', hard=2),
#             ConvtModule(32, 1, 1, 1, activation='Sigmoid')
#         )
#
#         # weight initialization
#         for m in self.modules():
#             if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias.data is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, reparameterize=True):
#         indices = []
#
#         for layer in self.encoder:
#             if hasattr(layer, 'return_indices') and layer.return_indices:
#                 x, indice = layer(x)
#                 indices.append(indice)
#             else:
#                 x = layer(x)
#
#         z, z_mu, z_logvar = self.gaussian(x, reparameterize)
#
#         x_ = z
#
#         for layer in self.decoder:
#             if hasattr(layer, 'accept_indices') and layer.accept_indices:
#                 x_ = layer(x_, indices.pop(-1))
#             else:
#                 x_ = layer(x_)
#
#         return {'x_reconst': x_,
#                 'z': z,
#                 'z_mu': z_mu,
#                 'z_logvar': z_logvar}
