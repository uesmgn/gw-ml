from torch import nn
from torchvision import transforms
import numpy as np

class ConvModule(nn.Module):

    def __init__(self, input_channels, output_channels,
                 kernel_conv=3, kernel_pool=3, return_indices=False):
        super().__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=kernel_conv,
                              stride=1,
                              padding=(kernel_conv-1)//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_pool,
                                    stride=kernel_pool,
                                    return_indices=return_indices)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        (x, indices) = self.maxpool(self.bn(self.conv(x)))
        x = self.relu(x)
        return (x, indices)


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = ConvModule(1, 64, 11, 3, return_indices=True)
        self.conv2 = ConvModule(64, 192, 5, 3, return_indices=True)
        self.conv3 = ConvModule(192, 256, 3, 3, return_indices=True)

    def forward(self, x):
        indices = [[], [], []]
        (x, indices[0]) = self.conv1(x)
        (x, indices[1]) = self.conv2(x)
        (x, indices[2]) = self.conv3(x)
        return x, indices


class DeconvModule(nn.Module):

    def __init__(self, input_channels, output_channels,
                 kernel_conv=3, kernel_pool=3, indice=None):
        super().__init__()
        self.indice = indice
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=kernel_conv,
                              stride=1,
                              padding=(kernel_conv-1)//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=kernel_pool,
                                        stride=kernel_pool)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.maxunpool(self.bn(self.conv(x)), self.indice)
        x = self.relu(x)
        return x


class Decoder(nn.Module):

    def __init__(self, indices):
        super().__init__()

        self.features = nn.Sequential(
            DeconvModule(256, 192, 3, 3, indice=indices[3]),
            DeconvModule(192, 64, 5, 3, indice=indices[2]),
            DeconvModule(64, 1, 11, 3, indice=indices[1]),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        return x
