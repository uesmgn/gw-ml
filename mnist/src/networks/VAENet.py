from torch import nn
from .Layers import Reshape, Gaussian

class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, stride=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.gaussian = Gaussian(features_dim, z_dim)

    def forward(self, x):
        features = self.features(x)
        z_mu, z_logvar, z = self.gaussian(features)

        return {'z_mu': z_mu,
                'z_logvar': z_logvar,
                'z': z}


class Decoder(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()


        self.fc = nn.Sequential(
            nn.Linear(z_dim, features_dim),
            Reshape((32, middle_size, middle_size))
        )

        self.reconst = torch.nn.ModuleList([
            ConvModule(32, 32, 3),
            ConvModule(32, 1, 5, activation='Sigmoid')
        ])

    def pxz(self, z, indices):
        indices.reverse()
        idx = 0
        for i, layer in enumerate(self.reconst):
            # if i % 2 == 0:
            #     # MaxUnpool2d layer
            #     z = layer(z, indices[idx])
            #     idx += 1
            # else:
            z = layer(z)
        return z

    def forward(self, z, indices):
        x = self.fc(z)
        x_reconst = self.pxz(x, indices)

        output = {'x_reconst': x_reconst}

        return output

class VAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.encoder = Encoder(x_dim, z_dim, y_dim)
        # self.decoder = Decoder(x_dim, z_dim, y_dim)

    def forward(self, x):
        from_encoder = self.encoder(x)
        print(from_encoder)
        z = from_encoder['z']
        indices = from_encoder['indices']
        from_decoder = self.decoder(z, indices)

        # merge output
        output = from_encoder
        for key, value in from_decoder.items():
            output[key] = value

        return output
