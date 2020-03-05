import datetime
import time

from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
import numpy as np

class ConvModule(nn.Module):

    def __init__(self, input_channels, output_channels,
                 kernel=3, pooling_kernel=3, return_indices=False):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=kernel,
                              stride=1,
                              padding=(kernel-1)//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=pooling_kernel,
                                    stride=pooling_kernel,
                                    return_indices=return_indices)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x, indices = self.maxpool(x)
        return x, indices

class Latent(nn.Module):
    def __init__(self, dim_input, dim_latent):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.mu = nn.Linear(dim_input, dim_latent)
        self.var = nn.Linear(dim_input, dim_latent)
        self.fc = nn.Linear(dim_latent, dim_input)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def forward(self, x):
        mu, var = self.mu(x), self.var(x)
        z = self._reparameterize(mu, var)
        return z, mu, var

class Cluster(nn.Module):
    def __init__(self, dim_input, n_cluster):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.fc = nn.Linear(dim_input, n_cluster)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        y = self.softmax(x)
        return y

class Middle(nn.Module):
    def __init__(self, dim_input, dim_latent, n_cluster):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.latent = Latent(dim_input, dim_latent)
        self.cluster = Cluster(dim_input, n_cluster)
        self.fc1 = nn.Linear(n_cluster, dim_latent)
        self.fc2 = nn.Linear(dim_latent, dim_input)

    def forward(self, x):
        z, mu, var = self.latent(x)
        y = self.cluster(x)
        x = z + self.fc1(y)
        x = self.fc2(x)
        return x, y, z, mu, var


class DeconvModule(nn.Module):

    def __init__(self, input_channels, output_channels,
                 kernel=3, pooling_kernel=3, activation='ReLu'):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.maxunpool = nn.MaxUnpool2d(kernel_size=pooling_kernel,
                                        stride=pooling_kernel)
        self.conv = nn.Conv2d(input_channels,
                              output_channels,
                              kernel_size=kernel,
                              stride=1,
                              padding=(kernel-1)//2)
        self.bn = nn.BatchNorm2d(output_channels)
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x, indices, output_size):
        x = self.maxunpool(x, indices, output_size)
        x = self.activation(self.bn(self.conv(x)))
        return x


class VAE(nn.Module):
    def __init__(self, input_size, dim_latent, n_cluster):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()
        self.conv1 = ConvModule(1, 64, 11, 3, return_indices=True)
        self.conv2 = ConvModule(64, 192, 5, 3, return_indices=True)
        self.conv3 = ConvModule(192, 256, 3, 3, return_indices=True)
        self.fc = nn.Flatten()

        self.middle_size = input_size//3//3//3
        self.middle_dim = 256*self.middle_size**2

        self.middle = Middle(self.middle_dim, dim_latent, n_cluster)

        self.deconv1 = DeconvModule(256, 192, 3, 3)
        self.deconv2 = DeconvModule(192, 64, 5, 3)
        self.deconv3 = DeconvModule(64, 1, 11, 3, activation='Sigmoid')

    def init_model(self, train_loader, optimizer):
        self.train_loader = train_loader
        self.optimizer = optimizer

    def forward(self, x):
        input_size = x.size()

        x, indice1 = self.conv1(x)
        x, indice2 = self.conv2(x)
        x, indice3 = self.conv3(x)

        x = self.fc(x)

        x, y, z, mu, var = self.middle(x)

        x = x.view(-1, 256, self.middle_size, self.middle_size)

        x = self.deconv1(x, indice3, indice2.size())
        x = self.deconv2(x, indice2, indice1.size())
        x = self.deconv3(x, indice1, input_size)

        return x, y, z, mu, var

    def loss_function(self, recon_x, x, mu, var):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + var - mu.pow(2) - var.exp())
        return BCE + KLD

    def fit_train(self, epoch):
        print(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")

        self.train()
        train_loss = 0
        samples_cnt = 0
        start_t = time.time()
        for batch_idx, (x, label) in enumerate(self.train_loader):
            print(f'\rBatch: {batch_idx+1}', end='')
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_, y, z, mu, var = self(x)
            loss = self.loss_function(x_, x, mu, var)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            samples_cnt += x.size(0)
        elapsed_t = time.time() - start_t
        print(f"Loss: {train_loss / samples_cnt:f}")
        print(f"Calc time: {elapsed_t} sec/epoch")
