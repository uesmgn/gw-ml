import configparser
import os
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vae import loss_function
from vae.network import *
import utils as ut


LOSS_LABELS = ['total_loss', 'reconstruction_loss', 'kl_divergence']

if __name__ == '__main__':

    config_ini = 'config_mnist.ini'
    ini.read(config_ini, 'utf-8')

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    is_cuda = torch.cuda.is_available()

    x_size = ini.getint('conf', 'x_size')
    x_shape = (1, x_size, x_size)
    z_dim = ini.getint('conf', 'z_dim')
    num_workers = ini.getint('conf', 'num_workers')
    batch_size =  ini.getint('conf', 'batch_size')
    n_epoch =  ini.getint('conf', 'n_epoch')

    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if is_cuda else {}
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=data_transform),
                    batch_size=args.batch_size, shuffle=True,
                    **kwargs )
    test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, transform=data_transform)),
                    batch_size=args.batch_size, shuffle=True,
                    **kwargs )

    nargs = {}
    nargs['x_shape'] = x_shape
    nargs['z_dim'] = z_dim

    model = VAE(nargs)
    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    loss_stats = None

    for epoch in range(n_epoch):
        # training
        model.train()
        print(f'----- training at epoch {epoch + 1} -----')
        n_samples = 0
        vae_loss_epoch = torch.zeros(len(LOSS_LABELS)).to(device)

        for batch, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            params = model(x, return_params=True)
            vae_loss = criterion.vae_loss(params, reduction='none')
            vae_loss_total = vae_loss.sum()
            vae_loss_total.backward()
            optimizer.step()
            vae_loss_epoch += torch.cat([vae_loss_total.view(-1),
                                         vae_loss.view(-1)])
            n_samples += 1
        vae_loss_epoch /= n_samples
        vae_loss_epoch = vae_loss_epoch.detach().cpu().numpy()
        if loss_stats is None:
            loss_stats = vae_loss_epoch
        else:
            loss_stats = np.vstack([loss_stats, vae_loss_epoch])
        print(f'train loss = {vae_loss_epoch[0]:.3f} at epoch {epoch + 1}')
