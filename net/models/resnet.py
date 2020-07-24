import torch
import torch.nn as nn

__all__ = ['BasicResBlock', 'BottleNeckResBlock', 'ResNet']


def get_activation(activation, inplace=True):
    if activation in ('sigmoid', 'Sigmoid'):
        return nn.Sigmoid()
    return nn.ReLU(inplace=inplace)


def get_global_pool(gp, output_size=(1, 1)):
    if gp in ('max', 'Max'):
        return nn.AdaptiveMaxPool2d(output_size)
    return nn.AdaptiveAvgPool2d(output_size)


def get_global_pool(gp, output_size=(1, 1)):
    if gp in ('max', 'Max'):
        return nn.AdaptiveMaxPool2d(output_size)
    return nn.AdaptiveAvgPool2d(output_size)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def convt3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def convt1x1(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, connection=None,
                 groups=1, base_width=64, dilation=1, activation='relu'):
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        transpose = in_planes > out_planes

        if transpose:
            self.features = nn.Sequential(
                convt3x3(in_planes, out_planes, stride=stride),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                convt3x3(out_planes, out_planes),
                nn.BatchNorm2d(out_planes)
            )
            self.connection = None
            if stride != 1 or in_planes != out_planes * self.expansion:
                self.connection = nn.Sequential(
                    convt1x1(in_planes, out_planes *
                             self.expansion, stride=stride),
                    nn.BatchNorm2d(out_planes * self.expansion)
                )
        else:
            self.features = nn.Sequential(
                conv3x3(in_planes, out_planes, stride=stride),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                conv3x3(out_planes, out_planes),
                nn.BatchNorm2d(out_planes)
            )
            self.connection = None
            if stride != 1 or in_planes != out_planes * self.expansion:
                self.connection = nn.Sequential(
                    conv1x1(in_planes, out_planes *
                            self.expansion, stride=stride),
                    nn.BatchNorm2d(out_planes * self.expansion)
                )

        self.activation = get_activation(activation)

    def forward(self, x):
        identity = x
        features = self.features(x)
        if self.connection is not None:
            identity = self.connection(x)
        features += identity
        return self.activation(features)


class BottleNeckResBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, connection=None,
                 groups=1, base_width=64, dilation=1, activation='relu'):
        super().__init__()

        width = int(out_planes * (base_width / 64.)) * groups

        self.features = nn.Sequential(
            conv1x1(in_planes, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            conv3x3(width, width, stride, groups, dilation),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            conv1x1(width, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion)
        )

        self.connection = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.connection = nn.Sequential(
                conv1x1(in_planes, out_planes * self.expansion, stride=stride),
                nn.BatchNorm2d(out_planes * self.expansion)
            )

        self.activation = get_activation(activation)

    def forward(self, x):
        identity = x
        features = self.features(x)
        if self.connection is not None:
            identity = self.connection(x)
        features += identity
        return self.activation(features)


class ResNet(nn.Module):

    def __init__(self, in_planes=1, block='basic', num_blocks=(2, 2, 2, 2),
                 num_classes=1000, groups=1, width_per_group=64, dilate=None, gp='avg'):
        super().__init__()

        assert len(num_blocks) == 4, 'num_blocks must be array have length of 4'

        self.next_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.block = self._get_block(block)

        if dilate is None:
            dilate = (False, False, False)
        elif isinstance(dilate, (int, bool)):
            dilate = (dilate, dilate, dilate)
        elif isinstance(dilate, (list, tuple)):
            if len(dilate) != 3:
                raise ValueError('dilates must have length of 3')
        else:
            raise ValueError('dilate must be int(s) or bool(s)')

        self.conv1_x = nn.Sequential(
            nn.Conv2d(in_planes, self.next_planes, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.next_planes),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, self.block, num_blocks[0])
        )

        self.conv3_x = nn.Sequential(
            self._make_layer(
                128, self.block, num_blocks[1], stride=2, dilate=dilate[0])
        )

        self.conv4_x = nn.Sequential(
            self._make_layer(
                256, self.block, num_blocks[2], stride=2, dilate=dilate[1])
        )

        self.conv5_x = nn.Sequential(
            self._make_layer(
                512, self.block, num_blocks[3], stride=2, dilate=dilate[2])
        )

        self.pool = nn.Sequential(
            get_global_pool(gp),
            nn.Flatten()
        )

        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_block(self, block):
        assert block in (
            'basic', 'bottle_neck'), 'block only supports ("basic", "bottle_neck")'
        if block is 'bottle_neck':
            return BottleNeckResBlock
        return BasicResBlock

    def _make_layer(self, out_planes, block, num_block, stride=1, dilate=False):
        assert num_block > 0
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(block(self.next_planes, out_planes, stride=stride, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation))
        self.next_planes = out_planes * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.next_planes, out_planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
