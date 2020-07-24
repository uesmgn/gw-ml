import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .. import criterion


def _reparameterize(mean, var):
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


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.features = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x, eps=1e-10):
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        if self.training:
            x = _reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var


class Clustering(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.logits = nn.Linear(dim, dim)

    def forward(self, x):
        logits = self.logits(x)
        y = F.softmax(logits, -1)
        return logits, y


class ResVAE_M1(nn.Module):
    def __init__(self, resnet, device, z_dim=64, hidden_dim=6, verbose=False):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1],  # remove final layer
            nn.BatchNorm1d(512 * resnet.block.expansion),
            nn.ReLU(inplace=True),
            nn.Linear(512 * resnet.block.expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            Gaussian(256, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 8 * hidden_dim * hidden_dim),
            nn.BatchNorm1d(8 * hidden_dim * hidden_dim),
            nn.ReLU(inplace=True),
            Reshape((8, hidden_dim, hidden_dim)),
            resnet.block(8, 512, stride=2),
            resnet.block(512, 256, stride=2),
            resnet.block(256, 128, stride=2),
            resnet.block(128, 64, stride=2),
            resnet.block(64, 1, stride=2, activation='sigmoid')
        )

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
    def __init__(self, resnet, device, z_dim=64, y_dim=10, hidden_dim=6, verbose=False):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1],  # remove final layer
            nn.BatchNorm1d(512 * resnet.block.expansion),
            nn.ReLU(inplace=True),
            nn.Linear(512 * resnet.block.expansion, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.y_inference = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, y_dim),
            nn.BatchNorm1d(y_dim),
            nn.ReLU(inplace=True),
            Clustering(y_dim)
        )
        self.z_inference = nn.Sequential(
            nn.Linear(256 + y_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            Gaussian(64, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + y_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            Reshape((512, 1, 1)),
            nn.Upsample(scale_factor=hidden_dim),
            resnet.block(512, 256, stride=2),
            resnet.block(256, 128, stride=2),
            resnet.block(128, 64, stride=2),
            resnet.block(64, 1, stride=2, activation='sigmoid')
        )

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
        loss = self._forward_func(ux, lx, target, alpha)
        return loss

    def _forward_func(self, ux, lx=None, target=None, alpha=1.):
        ux = ux.to(self.device)
        if lx is not None and target is not None:
            lx = lx.to(self.device)
            target = target.to(self.device)

            labeled_loss = self._labeled_loss(lx, target, alpha=alpha)
            unlabeled_loss = self._unlabeled_loss(ux)
            return labeled_loss + unlabeled_loss
        else:
            params = self._predict(ux)
            return params

    def _predict(self, x):
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
