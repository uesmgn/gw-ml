import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super().__init__()

        self.features = nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            conv3x3(out_planes, out_planes),
            nn.BatchNorm2d(out_planes)
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        features = self.features(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        features += identity
        return F.relu(features, inplace=True)

class ResNet(nn.Module):

    def __init__(self, in_planes, num_classes, block, num_blocks):
        super().__init__()

        assert len(num_blocks) == 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 64, block, num_blocks[0])
        )

        self.conv3_x = nn.Sequential(
            self._make_layer(64, 128, block, num_blocks[1], stride=2)
        )

        self.conv4_x = nn.Sequential(
            self._make_layer(128, 256, block, num_blocks[2], stride=2)
        )

        self.conv5_x = nn.Sequential(
            self._make_layer(256, 512, block, num_blocks[3], stride=2)
        )

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, in_planes, out_planes, block, num_block, stride=1):
        assert num_block > 0
        layers = []
        downsample = None
        if stride > 1:
            downsample = nn.Sequential(
                            conv1x1(in_planes, out_planes, stride=stride),
                            nn.BatchNorm2d(out_planes)
                        )
        layers.append(block(in_planes, out_planes, stride=stride, downsample=downsample))

        for _ in range(1, num_block):
            layers.append(block(out_planes, out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, eps=1e-8):
        super().__init__()
        self.features = nn.Linear(in_dim, out_dim * 2)
        self.eps = eps

    def forward(self, x, reparameterize=True):
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, -1)
        var = F.softplus(logit) + self.eps
        if reparameterize:
            x = self._reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var

    def _reparameterize(self, mean, var):
        if torch.is_tensor(var):
            std = torch.pow(var, 0.5)
        else:
            std = np.sqrt(var)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x

class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

def bce_loss(inputs, targets):
    loss = F.binary_cross_entropy(inputs, targets, reduction='none').view(inputs.shape[0], -1).sum(-1)
    return loss

def _log_norm(x, mean=None, var=None):
    if mean is None:
        mean = torch.zeros_like(x)
    if var is None:
        var = torch.ones_like(x)
    return -0.5 * (torch.log(2.0 * np.pi * var) + torch.pow(x - mean, 2) / var )

def log_norm_kl(x, mean, var, mean_=None, var_=None):
    log_p = _log_norm(x, mean, var).sum(-1)
    log_q = _log_norm(x, mean_, var_).sum(-1)
    loss = log_p - log_q
    return loss

class ResNet_VAE(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = ResNet(1, 1000, BasicBlock, [3, 3, 3, 3])
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.gaussian = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=0.01),
            Gaussian(256, 64)
        )
        self.upsample = nn.Sequential(
            nn.Linear(64, 512),
            nn.BatchNorm1d(512, momentum=0.01),
            nn.Linear(512, 512 * 64),
            nn.BatchNorm1d(512 * 64, momentum=0.01),
            Reshape((512, 8, 8))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=7, stride=3, padding=2),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_params=False):
        z, z_mean, z_var = self.gaussian(self.encoder(x))
        x_reconst = self.decoder(self.upsample(z))
        if return_params:
            return {'x': x,'x_reconst': x_reconst, 'z': z, 'z_mean': z_mean, 'z_var': z_var}
        return x_reconst

    def criterion(self, x, params):
        loss = bce_loss(params['x_reconst'], x)
        kl = log_norm_kl(params['z'], params['z_mean'], params['z_var'])
        return (loss + kl).mean()
