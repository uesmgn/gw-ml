import torch
import torch.nn as nn

from .vae import *
from ..layers import *
from ..helper import *
from ..criterionn import *

__all__ = [
    'CVAE'
]

class CVAE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.x_shape = kwargs['x_shape']
        self.y_dim = kwargs['y_dim']
        self.z_dim = kwargs['z_dim']
        self.bottle_channel = kwargs['bottle_channel']
        self.poolings = kwargs['poolings']
        self.middle_dim = get_middle_dim(self.x_shape, self.poolings)
        self.f_dim = self.bottle_channel * self.middle_dim**2

        # encoder: (M, C, W, H) -> (M, features_dim)
        self.encoder = Encoder(**kwargs)
        # classifier: (M, features_dim) -> (M, y_dim)
        self.classifier = GumbelSoftmax(self.f_dim, self.y_dim)
        # z: (M, features_dim + y_dim) -> (M, z_dim)
        self.z = Gaussian(self.f_dim + self.y_dim, self.z_dim)
        # z_prior: (M, y_dim) -> (M, z_dim)
        self.z_prior = Gaussian(self.y_dim, self.z_dim)
        # decoder: (M, z_dim) -> (M, C, W, H)
        self.decoder = Decoder(**kwargs)

        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_loss=True, beta=(1., 1., 1.)):
        f = self.encoder(x)
        y_logits, y = self.classifier(f)
        xy = torch.cat((f, y), -1)
        z, z_mean, z_var = self.z(xy)
        _, z_prior_mean, z_prior_var = self.z_prior(y)
        x_reconst = self.decoder(z)

        params = {'x': x, 'f': f, 'x_reconst': x_reconst, 'y': y,
                  'z': z, 'z_mean': z_mean, 'z_var': z_var,
                  'z_prior_mean': z_prior_mean, 'z_prior_var': z_prior_var }

        if return_loss:
            return criterion.cvae(params, beta)

        return {'x': x, 'f': f, 'x_reconst': x_reconst, 'y': y,
                'z': z, 'z_mean': z_mean, 'z_var': z_var,
                'z_prior_mean': z_prior_mean, 'z_prior_var': z_prior_var }

    def features(self, x):
        f = self.encoder(x)
        y_logits, y = self.classifier(f)
        _, z, _ = self.z(torch.cat((f, y), -1))
        return z

    def clustering_logits(self, x):
        f = self.encoder(x)
        y_logits, y = self.classifier(f)
        return y_logits
