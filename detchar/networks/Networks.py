import torch
from torch import nn
from torch.nn import functional as F
from .Layers import *


class VAENet(nn.Module):
    def __init__(self, x_size, z_dim, y_dim,
                 layers=(32, 64, 128, 192), middle_size=18, activation='Tanh'):
        super().__init__()

        middle_channel = layers[3]
        middle_dim = middle_channel * middle_size ** 2

        self.encoder = nn.Sequential(
            ConvModule(1, layers[0], 1, 1, activation=activation),
            DownSample(layers[0], layers[1], kernel_size=3, stride=3,
                       activation=activation),
            DownSample(layers[1], layers[2], kernel_size=3, stride=3,
                       activation=activation),
            DownSample(layers[2], layers[3], kernel_size=3, stride=3,
                       activation=activation),
            nn.Flatten()
        )
        self.gumbel = GumbelSoftmax(middle_dim, y_dim)
        self.gaussian = Gaussian(middle_dim + y_dim, z_dim)

        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_logvar = nn.Linear(y_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, middle_dim),
            Reshape((middle_channel, middle_size, middle_size)),
            Upsample(layers[3], layers[2], kernel_size=3, stride=3,
                     activation=activation),
            Upsample(layers[2], layers[1], kernel_size=3, stride=3,
                     activation=activation),
            Upsample(layers[1], layers[0], kernel_size=3, stride=3,
                     activation=activation),
            ConvtModule(layers[0], 1, 1, 1, activation='Sigmoid')
        )

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, temp=1.0):
        x = self.encoder(x)
        y_logits, y_prob, y = self.gumbel(x, temp=temp)
        y_mu = self.y_mu(y)
        y_logvar = self.y_logvar(y)
        z, z_mu, z_logvar = self.gaussian(torch.cat((x, y), 1))
        x = self.decoder(z)

        return {'x_reconst': x,
                'z': z,
                'z_mu': z_mu,
                'z_logvar': z_logvar,
                'y': y,
                'y_logits': y_logits,
                'y_prob': y_prob,
                'y_mu': y_mu,
                'y_logvar': y_logvar}
