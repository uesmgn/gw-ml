import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

__all__ = ['Block', 'BasicResBlock', 'BNResBlock',
           'BasicResTransposeBlock', 'ResNet']


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def _conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(nn.Module):

    expansion = 1

    def __init__(self):
        super().__init__()
        self.block = None
        self.connection = None

    def compile(self, in_planes, out_planes, **kwargs):
        stride = kwargs.get('stride') or 1
        self.block = nn.Sequential(
            _conv3x3(in_planes, out_planes * self.expansion, stride=stride),
            nn.BatchNorm2d(out_planes * self.expansion),
        )
        self.connection = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.connection = nn.Sequential(
                _conv1x1(in_planes, out_planes *
                         self.expansion, stride=stride),
                nn.BatchNorm2d(out_planes * self.expansion)
            )
        return self

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return F.relu(x, inplace=True)


class BasicResBlock(Block):

    expansion = 1

    def compile(self, in_planes, out_planes, **kwargs):
        stride = kwargs.get('stride') or 1
        connection = kwargs.get('connection') or None
        activation = kwargs.get('activation') or nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            _conv3x3(in_planes, out_planes, stride=stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            _conv3x3(out_planes, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion)
        )
        self.connection = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.connection = nn.Sequential(
                _conv1x1(in_planes, out_planes *
                         self.expansion, stride=stride),
                nn.BatchNorm2d(out_planes * self.expansion)
            )
        self.activation = activation

        return self

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class BNResBlock(Block):

    expansion = 4

    def compile(self, in_planes, out_planes, **kwargs):
        stride = kwargs.get('stride') or 1
        connection = kwargs.get('connection') or None
        activation = kwargs.get('activation') or nn.ReLU(inplace=True)
        groups = kwargs.get('groups') or 1
        base_width = kwargs.get('base_width') or 64

        width = int(out_planes * (base_width / 64.)) * groups

        self.block = nn.Sequential(
            _conv1x1(in_planes, width),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _conv3x3(width, width, stride=stride, groups=groups),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            _conv1x1(width, out_planes * self.expansion),
            nn.BatchNorm2d(out_planes * self.expansion)
        )
        self.connection = None
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.connection = nn.Sequential(
                _conv1x1(in_planes, out_planes *
                         self.expansion, stride=stride),
                nn.BatchNorm2d(out_planes * self.expansion)
            )
        self.activation = activation

        return self

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class ResNet(nn.Module):

    def __init__(self, in_planes=1, block=BasicResBlock(), num_blocks=(2, 2, 2, 2),
                 num_classes=1000, gp='avg'):
        super().__init__()

        assert len(num_blocks) == 4, 'num_blocks must be array have length of 4'

        self.in_planes = in_planes
        self.next_planes = 64
        assert hasattr(block, 'compile')
        self.expansion = block.expansion

        self.conv1_x = nn.Sequential(
            nn.Conv2d(in_planes, self.next_planes, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.next_planes),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, block, num_blocks[0])
        )

        self.conv3_x = nn.Sequential(
            self._make_layer(128, block, num_blocks[1], stride=2)
        )

        self.conv4_x = nn.Sequential(
            self._make_layer(256, block, num_blocks[2], stride=2)
        )

        self.conv5_x = nn.Sequential(
            self._make_layer(512, block, num_blocks[3], stride=2)
        )

        self.pool = nn.Sequential(
            self._global_pool(gp),
            nn.Flatten()
        )

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_planes, block, num_block, stride=1):
        assert num_block > 0

        layers = []
        layers.append(self._block(block, in_planes=self.next_planes,
                                  out_planes=out_planes, stride=stride))
        self.next_planes = out_planes * block.expansion

        for _ in range(1, num_block):
            layers.append(self._block(
                block, in_planes=self.next_planes, out_planes=out_planes))

        return nn.Sequential(*layers)

    def _global_pool(self, gp='avg', output_size=(1, 1)):
        if gp is 'avg':
            return nn.AdaptiveAvgPool2d(output_size)
        elif gp is 'max':
            return nn.AdaptiveMaxPool2d(output_size)
        else:
            raise ValueError('Global Pooling gp only supports ("avg", "max")')

    def _block(self, block, **kwargs):
        block = copy.deepcopy(block)
        block.compile(**kwargs)
        return block

    def forward(self, x):
        x = self.conv1_x(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
