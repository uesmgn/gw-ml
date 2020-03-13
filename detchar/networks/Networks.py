import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .Layers import *


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()

        features_dim = 256 * (x_dim // 3 // 3 // 3)**2

        self.features = torch.nn.ModuleList([
            ConvModule(1, 64, 11),
            nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True),
            ConvModule(64, 192, 3),
            nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True),
            ConvModule(192, 256, 3),
            nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True),
            nn.Flatten()
        ])

        # Gumbel-softmax(x_features)
        self.gumbel = GumbelSoftmax(features_dim, y_dim)
        # Gaussian(x_features)
        self.gaussian = Gaussian(features_dim, z_dim)

    # q(y,z|x)
    def x_features(self, x):
        indices = []
        indices.append(x)
        for i, layer in enumerate(self.features):
            if i % 2 == 1:
                # get indice from max-pooling
                x, indice = layer(x)
                indices.append(indice)
            else:
                x = layer(x)
        return x, indices

    def forward(self, x, temp=1.0):
        x_features, indices = self.x_features(x)
        # p(y|x) y ~ Cat(y)
        logits, prob, y = self.gumbel(x_features, temp)
        # q(z|x) z ~ N(z_mu, z_var)
        z_mu, z_var, z = self.gaussian(x_features)

        return {'z_mu': z_mu, 'z_var': z_var, 'z': z,
                'indices': indices, 'logits': logits,
                'prob_cat': prob, 'categorical': y}


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(Decoder, self).__init__()

        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)
        middle_size = x_dim // 3 // 3 // 3
        features_dim = 256 * middle_size**2

        self.fc = nn.Sequential(
            nn.Linear(z_dim, features_dim),
            Reshape((256, middle_size, middle_size))
        )

        self.reconst = torch.nn.ModuleList([
            nn.MaxUnpool2d(kernel_size=3, stride=3),
            ConvModule(256, 192, 3),
            nn.MaxUnpool2d(kernel_size=3, stride=3),
            ConvModule(192, 64, 3),
            nn.MaxUnpool2d(kernel_size=3, stride=3),
            ConvModule(64, 1, 11, activation='Sigmoid')
        ])

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z, indices):
        indices.reverse()
        idx = 0
        for i, layer in enumerate(self.reconst):
            if i % 2 == 0:
                # MaxUnpool2d layer
                z = layer(z, indices[idx])
                idx += 1
            else:
                z = layer(z)
        return z

    def forward(self, z, y, indices):
        y_mu, y_var = self.pzy(y)
        x = self.fc(z)
        x_reconst = self.pxz(x, indices)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_reconst': x_reconst}
        return output


class VAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(VAENet, self).__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.encoder = Encoder(x_dim, z_dim, y_dim)
        self.decoder = Decoder(x_dim, z_dim, y_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    # weight initialization
    # for m in self.modules():
    #   if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
    #     torch.nn.init.xavier_normal_(m.weight)
    #     if m.bias.data is not None:
    #       init.constant_(m.bias, 0)

    def forward(self, x, temperature=1):
        from_encoder = self.encoder(x, temperature)
        z = from_encoder['gaussian']
        y = from_encoder['categorical']
        indices = from_encoder['indices']
        from_decoder = self.decoder(z, y, indices)

        # merge output
        output = from_encoder
        for key, value in from_decoder.items():
            output[key] = value
        return output
