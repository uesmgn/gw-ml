import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .. import layers, criterion


class Encoder(nn.Module):
    def __init__(self, resnet, z_dim=64):
        super().__init__()

        self.z_dim = z_dim
        exp = resnet.expansion
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1],  # remove final layer
            nn.BatchNorm1d(512 * exp),
            nn.ReLU(inplace=True)
        )
        self.gaussian = layers.Gaussian(512 * exp, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        z, z_mean, z_var = self.gaussian(x)
        return z, z_mean, z_var

class TransposeBlock(nn.Module):

    def __init__(self, in_planes, out_planes, **kwargs):
        super().__init__()
        stride = kwargs.get('stride') or 2
        activation = kwargs.get('activation') or nn.ReLU(inplace=True)
        kernel_size = stride + 2

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                               stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_planes, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        self.connection = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               bias=False, output_padding=stride-1),
            nn.BatchNorm2d(out_planes)
        )
        self.activation = activation

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class Decoder(nn.Module):
    def __init__(self, in_dim, out_planes=1, planes=(512, 256, 128, 64, 64),
                 scale_factor=8, activation=nn.Sigmoid()):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            layers.Reshape((planes[0], 1, 1)),
            nn.Upsample(scale_factor=scale_factor),
            *[TransposeBlock(planes[i], planes[i+1]) for i in range(len(planes)-1)],
            TransposeBlock(planes[-1], out_planes, activation=activation),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class ResVAE_M1(nn.Module):
    def __init__(self, resnet, device='cpu', z_dim=64, planes=(512, 256, 128, 64, 64),
                 scale_factor=8, activation=nn.Sigmoid(), verbose=False):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.in_planes = resnet.in_planes
        self.encoder = Encoder(resnet, z_dim)
        self.decoder = Decoder(z_dim, out_planes=self.in_planes, planes=planes,
                               scale_factor=scale_factor, activation=activation)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.verbose = verbose
        self = self.to(device)

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
        x = x.to(self.device)
        z, z_mean, z_var = self.encoder(x)
        x_reconst = self.decoder(z)

        if x_reconst.shape != x.shape:
            if self.verbose:
                print(f'output tensor shape {x_reconst.shape} is resized into tensor shape {x.shape}')
            x_reconst = F.interpolate(
                x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

        return x, x_reconst, z, z_mean, z_var


class ResVAE_M2(nn.Module):
    def __init__(self, resnet, device='cpu', x_dim=256, z_dim=64, y_dim=10,
                 planes=(512, 256, 128, 64, 64), scale_factor=8,
                 activation=nn.Sigmoid(), verbose=False):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.in_planes = resnet.in_planes
        encoder = Encoder(resnet, z_dim)
        exp = resnet.expansion
        self.encoder = nn.Sequential(
            *list(encoder.children())[:-1],  # remove final layer
            nn.Linear(512 * exp, x_dim),
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
        self.decoder = Decoder(z_dim + y_dim, out_planes=self.in_planes, planes=planes,
                               scale_factor=scale_factor, activation=activation)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.verbose = verbose
        self = self.to(device)

    def forward(self, x, return_params=False):
        params = self._forward_func(x)
        if return_params:
            return params
        return params['x_reconst']

    def criterion(self, ux, lx, target, alpha=1.):
        ux = ux.to(self.device)
        lx = lx.to(self.device)
        target = target.to(self.device)
        labeled_loss = self._labeled_loss(lx, target, alpha=alpha)
        unlabeled_loss = self._unlabeled_loss(ux)
        return labeled_loss + unlabeled_loss

    def _forward_func(self, x, alpha=1.):
        x = x.to(self.device)
        x_ = self.encoder(x)
        _, qy = self.y_inference(x_)
        _, y_pred = torch.max(qy, -1)
        y = F.one_hot(y_pred, num_classes=self.y_dim).to(
            self.device, dtype=torch.float32)
        z, z_mean, z_var = self.z_inference(torch.cat((x_, y), -1))
        x_reconst = self.decoder(torch.cat((z, y), -1))

        if x_reconst.shape != x.shape:
            if self.verbose:
                print(f'output tensor shape {x_reconst.shape} is resized into tensor shape {x.shape}')
            x_reconst = F.interpolate(
                x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

        return dict(
            qy=qy, y_pred=y_pred, z=z, z_mean=z_mean, x_reconst=x_reconst
        )

    def _labeled_loss(self, x, target, alpha=1.):
        x_ = self.encoder(x)
        y = F.one_hot(target, num_classes=self.y_dim).to(
            self.device, dtype=torch.float32)
        z, z_mean, z_var = self.z_inference(torch.cat((x_, y), -1))
        x_reconst = self.decoder(torch.cat((z, y), -1))

        if x_reconst.shape != x.shape:
            if self.verbose:
                print(f'output tensor shape {x_reconst.shape} is resized into tensor shape {x.shape}')
            x_reconst = F.interpolate(
                x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

        log_p_x = criterion.bce_loss(x_reconst, x).sum(-1)
        log_p_y = -np.log(1 / self.y_dim)
        log_p_z = criterion.log_norm_kl(z_mean, z_var).sum(-1)

        y_logits, _ = self.y_inference(x_)

        sup_loss = alpha * \
            criterion.softmax_cross_entropy(y_logits, target).sum(-1)

        return (log_p_x + log_p_y + log_p_z + sup_loss).mean()

    def _unlabeled_loss(self, x):
        unlabeled_loss = 0
        x_ = self.encoder(x)  # (batch_size, 512 * block.expansion)
        _, qy = self.y_inference(x_)

        for i in range(self.y_dim):
            qy_i = qy[:, i]
            y = F.one_hot(torch.tensor(i), num_classes=self.y_dim).repeat(
                x.shape[0], 1).to(self.device, dtype=torch.float32)
            z, z_mean, z_var = self.z_inference(torch.cat((x_, y), -1))
            x_reconst = self.decoder(torch.cat((z, y), -1))

            if x_reconst.shape != x.shape:
                x_reconst = F.interpolate(
                    x_reconst, size=x.shape[2:], mode='bilinear', align_corners=True)

            log_p_x = criterion.bce_loss(x_reconst, x).sum(-1)
            log_p_y = -np.log(1 / self.y_dim)
            log_p_z = criterion.log_norm_kl(z_mean, z_var).sum(-1)
            log_q_y = torch.log(qy_i + 1e-10)

            unlabeled_loss += (log_p_x + log_p_y + log_p_z + log_q_y) * qy_i

        return unlabeled_loss.mean()  # batch mean
