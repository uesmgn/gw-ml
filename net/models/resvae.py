import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from .resnet import *
from .. import layers, criterion


class Encoder(nn.Module):
    def __init__(self, in_planes=1, out_dim=2, block=BasicResBlock(),
                 planes=(64, 128, 256, 512), num_blocks=(2, 2, 2, 2)):
        super().__init__()

        resnet = ResNet(in_planes=in_planes, block=block, planes=planes, num_blocks=num_blocks)
        self.num_final_fc = resnet.num_final_fc
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1],  # remove final layer
            nn.BatchNorm1d(self.num_final_fc),
            nn.ReLU(inplace=True)
        )
        self.gaussian = layers.Gaussian(self.num_final_fc, out_dim)

    def forward(self, x):
        x = self.encoder(x)
        z, z_mean, z_var = self.gaussian(x)
        return z, z_mean, z_var


class Decoder(nn.Module):
    def __init__(self, in_dim=2, out_planes=1, block=TransposeResBlock(),
                 planes=(512, 256, 128, 64), num_blocks=(2, 2, 2, 2),
                 filter_size=8, activation=nn.Sigmoid()):
        super().__init__()

        assert len(num_blocks) == 4, 'num_blocks must be array have length of 4'
        assert len(planes) == 4, 'planes must be array have length of 4'
        self.next_planes = planes[0]
        assert hasattr(block, 'compile')

        self.fc = nn.Sequential(
            nn.Linear(in_dim, planes[0] * filter_size * filter_size),
            nn.ReLU(inplace=True),
            layers.Reshape((planes[0], filter_size, filter_size))
        )
        self.convt1_x = nn.Sequential(
            self._make_layer(planes[1], block, num_blocks[0], stride=2),
        )
        self.convt2_x = nn.Sequential(
            self._make_layer(planes[2], block, num_blocks[1], stride=2)
        )
        self.convt3_x = nn.Sequential(
            self._make_layer(planes[3], block, num_blocks[2], stride=2)
        )
        self.convt4_x = nn.Sequential(
            self._make_layer(planes[3], block, num_blocks[3], stride=2),
        )
        self.convt5_x = nn.Sequential(
            self._block(block, in_planes=planes[3] * block.expansion,
                        out_planes=out_planes, stride=2, activation=activation)
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
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

    def _block(self, block, **kwargs):
        block = copy.deepcopy(block)
        block.compile(**kwargs)
        return block

    def forward(self, x):
        x = self.fc(x)
        x = self.convt1_x(x)
        x = self.convt2_x(x)
        x = self.convt3_x(x)
        x = self.convt4_x(x)
        x = self.convt5_x(x)
        return x


class M1(nn.Module):
    def __init__(self, resnet, z_dim=64, filter_size=6,
                 activation=nn.Sigmoid(), verbose=False):
        super().__init__()

        self.z_dim = z_dim
        self.in_planes = resnet.in_planes
        self.encoder = Encoder(resnet, z_dim)
        self.decoder = Decoder(z_dim, out_planes=self.in_planes,
                               filter_size=filter_size, activation=activation)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.verbose = verbose

    def forward(self, x, return_params=False):
        x, x_reconst, z, z_mean, z_var = self._forward_func(x)
        if return_params:
            return dict(x=x, x_reconst=x_reconst,
                        z=z, z_mean=z_mean, z_var=z_var)
        return x_reconst

    def criterion(self, x):
        x, x_reconst, z, z_mean, z_var = self._forward_func(x)

        log_p_x = criterion.bce_loss(x_reconst, x).sum(-1)
        log_p_z = criterion.log_norm_kl(z_mean, z_var).sum(-1)

        loss = (log_p_x + log_p_z).mean()  # batch mean

        return loss

    def _forward_func(self, x):
        z, z_mean, z_var = self.encoder(x)
        x_reconst = self.decoder(z)

        if x_reconst.shape != x.shape:
            if self.verbose:
                print(
                    f'output tensor shape {x_reconst.shape} is resized into tensor shape {x.shape}')
            x_reconst = F.interpolate(
                x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

        return x, x_reconst, z, z_mean, z_var


class ResVAE_M2(nn.Module):
    def __init__(self, b1=BasicResBlock(), b2=TransposeResBlock(),
                 x_dim=256, z_dim=64, y_dim=10,
                 num_blocks=(2, 2, 2, 2), planes=(64, 128, 256, 512),
                 filter_size=8, activation=nn.Sigmoid(), verbose=False):
        super().__init__()

        self.z_dim = z_dim
        self.y_dim = y_dim
        self.in_planes = 1
        encoder = Encoder(in_planes=self.in_planes, block=b1, planes=planes, num_blocks=num_blocks)
        num_final_fc= encoder.num_final_fc
        self.encoder = nn.Sequential(
            *list(encoder.children())[:-1],  # remove final gaussian layer
            nn.Linear(num_final_fc, x_dim),
            nn.BatchNorm1d(x_dim),
            nn.ReLU(inplace=True)
        )
        self.y_inference = nn.Sequential(
            nn.Linear(x_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            layers.Clustering(512, y_dim)
        )
        self.z_inference = nn.Sequential(
            nn.Linear(x_dim + y_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            layers.Gaussian(512, z_dim)
        )
        self.decoder = Decoder(in_dim=z_dim + y_dim, out_planes=self.in_planes,
                               block=b2, planes=planes[::-1], num_blocks=num_blocks[::-1],
                               filter_size=filter_size, activation=activation)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.verbose = verbose

    def forward(self, ux, lx=None, target=None, alpha=1., return_params=False):
        if lx is not None and target is not None:
            return self.criterion(ux, lx, target, alpha)
        else:
            params = self._forward_func(ux)
            if return_params:
                return params
            return params['x_reconst']

    def criterion(self, ux, lx, target, alpha=1.):
        labeled_loss = self._labeled_loss(lx, target, alpha=alpha)
        unlabeled_loss = self._unlabeled_loss(ux)
        return labeled_loss, unlabeled_loss

    def _forward_func(self, x, alpha=1.):
        x_ = self.encoder(x)
        _, qy = self.y_inference(x_)
        _, y_pred = torch.max(qy, -1)
        y = F.one_hot(y_pred, num_classes=self.y_dim).to(x_.device, dtype=x_.dtype)
        z, z_mean, z_var = self.z_inference(torch.cat((x_, y), -1))
        x_reconst = self.decoder(torch.cat((z, y), -1))

        if x_reconst.shape != x.shape:
            if self.verbose:
                print(
                    f'output tensor shape {x_reconst.shape} is resized into tensor shape {x.shape}')
            x_reconst = F.interpolate(
                x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

        return dict(
            qy=qy, y_pred=y_pred, z=z, z_mean=z_mean, x_reconst=x_reconst
        )

    def _labeled_loss(self, x, y, alpha=1.):
        x_ = self.encoder(x)
        y_ = y.to(x_.dtype)
        z, z_mean, z_var = self.z_inference(torch.cat((x_, y_), -1))
        x_reconst = self.decoder(torch.cat((z, y_), -1))

        if x_reconst.shape != x.shape:
            if self.verbose:
                print(
                    f'output tensor shape {x_reconst.shape} is resized into tensor shape {x.shape}')
            x_reconst = F.interpolate(
                x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

        log_p_x = criterion.bce_loss(x_reconst, x).sum(-1)
        log_p_y = -np.log(1 / self.y_dim)
        log_p_z = criterion.log_norm_kl(z_mean, z_var).sum(-1)

        y_logits, _ = self.y_inference(x_)

        target = torch.argmax(y, dim=1)
        sup_loss = alpha * \
            criterion.softmax_cross_entropy(y_logits, target).sum(-1)

        labeled_loss = log_p_x + log_p_y + log_p_z + sup_loss
        return labeled_loss

    def _unlabeled_loss(self, x):
        unlabeled_loss = 0
        x_ = self.encoder(x)  # (batch_size, 512 * block.expansion)
        _, qy = self.y_inference(x_)
        y = F.one_hot(torch.arange(self.y_dim), num_classes=self.y_dim).to(x_.device, dtype=x_.dtype)
        for i in range(self.y_dim):
            qy_i = qy[:, i]
            y_ = y[:, i].repeat(x_.shape[0], 1)
            z, z_mean, z_var = self.z_inference(torch.cat((x_, y_), -1))
            x_reconst = self.decoder(torch.cat((z, y_), -1))

            if x_reconst.shape != x.shape:
                x_reconst = F.interpolate(
                    x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

            log_p_x = criterion.bce_loss(x_reconst, x).sum(-1)
            log_p_y = -np.log(1 / self.y_dim)
            log_p_z = criterion.log_norm_kl(z_mean, z_var).sum(-1)
            log_q_y = torch.log(qy_i + 1e-8)

            unlabeled_loss += (log_p_x + log_p_y + log_p_z + log_q_y) * qy_i

        return unlabeled_loss
