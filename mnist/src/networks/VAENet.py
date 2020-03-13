import torch.nn as nn
from .Layers import Encoder, Decoder

class VAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.encoder = Encoder(x_dim, z_dim, y_dim)
        self.decoder = Decoder(x_dim, z_dim, y_dim)

    def forward(self, x):
        from_encoder = self.encoder(x)
        z = from_encoder['z']
        indices = from_encoder['indices']
        from_decoder = self.decoder(z, indices)

        # merge output
        output = from_encoder
        for key, value in from_decoder.items():
            output[key] = value

        return output
