import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from .Layers import *

# Inference Network
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(Encoder, self).__init__()

        # q(y|x)
        self.__features = torch.nn.ModuleList([
            ConvModule(1, 32, 11),
            nn.MaxPool2d(kernel_size=5, stride=5, return_indices=True),
            ConvModule(32, 64, 5),
            nn.MaxPool2d(kernel_size=5, stride=5, return_indices=True),
            nn.Flatten()
        ])

        _features_dim = 64*(x_dim//5//5)**2
        self.pyx = GumbelSoftmax(_features_dim, y_dim)
        self.pzxy = Gaussian(_features_dim+y_dim, z_dim)

    # q(y|x)
    def conv(self, x, temperature):
        indices = []
        indices.append(x)
        for i, layer in enumerate(self.__features):
            if i % 2 == 1:
                # get indices from max-pooling
                x, indice = layer(x)
                indices.append(indice)
            else:
                x = layer(x)
        return x, indices

    def forward(self, x, temperature=1.0):
        features, indices = self.conv(x, temperature)
        logits, prob, y = self.pyx(features, temperature)
        concat = torch.cat((features, y), dim=1)
        mu, var, z = self.pzxy(concat)
        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'indices': indices, 'logits': logits,
                  'prob_cat': prob, 'categorical': y}
        return output

class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super(Decoder, self).__init__()

        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)
        self._middle_size = x_dim//5//5
        self._features_dim = 64*self._middle_size**2

        self.fc = nn.Sequential(
            nn.Linear(z_dim, self._features_dim),
            Reshape((64, self._middle_size, self._middle_size))
        )

        self.reconst = torch.nn.ModuleList([
            nn.MaxUnpool2d(kernel_size=5, stride=5),
            ConvModule(64, 32, 5),
            nn.MaxUnpool2d(kernel_size=5, stride=5),
            ConvModule(32, 1, 11, activation='Sigmoid')
        ])

    # p(z|y)
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
                z = layer(z, indices[idx],
                          output_size=indices[idx+1].size())
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

        self.encoder = Encoder(x_dim, z_dim, y_dim)
        self.decoder = Decoder(x_dim, z_dim, y_dim)

    # weight initialization
    # for m in self.modules():
    #   if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
    #     torch.nn.init.xavier_normal_(m.weight)
    #     if m.bias.data is not None:
    #       init.constant_(m.bias, 0)

    def forward(self, x, temperature=1.0):
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
