import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel=3, activation='ReLu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kernel,
                              stride=1,
                              padding=(kernel - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channel)
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def reparameterize(self, mu, var):
        # reparameterize trick
        # std = torch.sqrt(var + 1e-10)
        # noise = torch.randn_like(std)
        # z = mu + noise * std
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std
        return z

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)
        # var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var)
        return mu, var, z


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()

        features_dim = 64 * (x_dim // 2 // 2)**2

        self.features = torch.nn.ModuleList([
            ConvModule(1, 32, 5),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            ConvModule(32, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
            nn.Flatten()
        ])
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

    def forward(self, x):
        x_features, indices = self.x_features(x)
        z_mu, z_var, z = self.gaussian(x_features)

        return {'z_mu': z_mu, 'z_var': z_var,
                'z': z, 'indices': indices}


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()

        middle_size = x_dim // 2 // 2
        features_dim = 64 * middle_size**2

        self.fc = nn.Sequential(
            nn.Linear(z_dim, features_dim),
            Reshape((64, middle_size, middle_size))
        )

        self.reconst = torch.nn.ModuleList([
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvModule(64, 32, 3),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            ConvModule(32, 1, 3, activation='Sigmoid')
        ])

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

    def forward(self, z, indices):
        x = self.fc(z)
        x_reconst = self.pxz(x, indices)

        output = {'x_reconst': x_reconst}

        return output
