from torch import nn


class ConvModule(nn.Module):

    def __init__(self, input_channels, output_channels,
                 kernel_conv=3, kernel_pool=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=kernel_conv,
                              stride=1,
                              padding=(kernel_conv-1)//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_pool,
                                    stride=kernel_pool)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.maxpool(self.bn(self.conv(x))))
        return x


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvModule(1, 64, 11, 3),
            ConvModule(64, 192, 5, 3),
            ConvModule(192, 384, 3, 3),
            ConvModule(384, 256, 3, 3)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = nn.Flatten(x)
        print(x.size())
        return x
