import torch
import torch.nn as nn

from ..layers import *
from .. import utils as ut
from .. import criterion as crit


class CVAE(nn.Module):
    def __init__(self, attr):
        super().__init__()

        x_channel = attr.x_channel
        x_dim = attr.x_dim
        x_shape = (x_channel, *list(x_dim))
        y_dim = attr.y_dim
        z_dim = attr.z_dim
        bottle_channel = attr.bottle_channel
        channels = attr.channels
        kernels = attr.kernels
        poolings = attr.poolings
        pool_func = attr.pool_func
        act_func = attr.act_func

        middle_dim = ut.get_middle_dim(x_shape, poolings)
        f_dim = bottle_channel * middle_dim ** 2

        # encoder: (M, C, W, H) -> (M, features_dim)
        self.encoder = nn.Sequential(
            Conv2dModule(x_channel, bottle_channel,
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
        # classifier: (M, features_dim) -> (M, y_dim)
        self.classifier = GumbelSoftmax(f_dim, y_dim)
        # z: (M, features_dim + y_dim) -> (M, z_dim)
        self.z = Gaussian(f_dim + y_dim, z_dim)
        # z_prior: (M, y_dim) -> (M, z_dim)
        self.z_prior = Gaussian(y_dim, z_dim)
        # decoder: (M, z_dim) -> (M, C, W, H)
        self.decoder = nn.Sequential(
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
            ConvTranspose2dModule(bottle_channel, x_channel,
                                  act_func='Sigmoid')
        )

        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, target, return_params=False):
        f = self.encoder(x)
        y_logits, y = self.classifier(f)
        xy = torch.cat((f, y), -1)
        z, z_mean, z_var = self.z(xy.detach())
        _, z_prior_mean, z_prior_var = self.z_prior(y)
        x_reconst = self.decoder(z)

        if return_params:
            return dict(
                x=x,
                x_reconst=x_reconst,
                y_logits=y_logits, y=y,
                z=z, z_mean=z_mean, z_var=z_var,
                z_prior_mean=z_prior_mean, z_prior_var=z_prior_var
            )
        return x_reconst

    def criterion(self, **kwargs):
        rec_loss = crit.bce_loss(kwargs['x_reconst'], kwargs['x'])
        z_kl = crit.log_norm_kl(
            kwargs['z'], kwargs['z_mean'], kwargs['z_var'],
            kwargs['z_prior_mean'], kwargs['z_prior_var'])
        y_entropy = crit.entropy(kwargs['y_logits'])
        loss = torch.stack((rec_loss, z_kl, y_entropy))
        weights = kwargs.get('weights') or (1., 1., 1.)
        weights = torch.Tensor(list(weights)).to(loss.device)
        return weights * loss
