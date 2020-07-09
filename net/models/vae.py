import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class BasicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

def bce_loss(inputs, targets, reduction='mean'):
    loss = F.binary_cross_entropy(inputs, targets, reduction='sum')
    return loss

class VAE_439(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                BasicConv2d(1, 32, kernel_size=3, stride=2),
                BasicConv2d(32, 32, kernel_size=3),
                BasicConv2d(32, 64, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2),
                BasicConv2d(64, 80, kernel_size=1),
                BasicConv2d(80, 128, kernel_size=3, stride=2),
                BasicConv2d(128, 128, kernel_size=3),
                nn.MaxPool2d(kernel_size=3, stride=2),
                BasicConv2d(128, 64, kernel_size=1),
                BasicConv2d(64, 64, kernel_size=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Flatten(),
                nn.Linear(1600, 256)
        )

        self.decoder = nn.Sequential(
                nn.Linear(256, 64*6*6),
                Reshape((64, 6, 6)),
                BasicConvTranspose2d(64, 64, kernel_size=3, stride=3),
                BasicConvTranspose2d(64, 128, kernel_size=1),
                BasicConvTranspose2d(128, 128, kernel_size=3, stride=3),
                BasicConvTranspose2d(128, 80, kernel_size=1),
                BasicConvTranspose2d(80, 80, kernel_size=3, stride=2),
                BasicConvTranspose2d(80, 64, kernel_size=1),
                BasicConvTranspose2d(64, 64, kernel_size=3, stride=2),
                BasicConvTranspose2d(64, 32, kernel_size=1),
                BasicConvTranspose2d(32, 1, kernel_size=3, stride=2),
                nn.Sigmoid()
        )


    def forward(self, x, target=None, return_params=False):
        x_reconst = self.decoder(self.encoder(x))
        return {'x': x, 'x_reconst': x_reconst} if return_params else x_reconst

    def criterion(self, **kwargs):
        loss = F.binary_cross_entropy(kwargs['x_reconst'], kwargs['x'], reduction='sum')
        return loss
