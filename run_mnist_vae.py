import argparse
import configparser
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

from vae import loss_function
from vae.network import *
import utils as ut


LOSS_LABELS = ['total_loss', 'reconstruction_loss', 'kl_divergence']

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of VAE')
parser.add_argument('--eval_itvl', type=int,
                    help='evaluationn interval')
args = parser.parse_args()

if __name__ == '__main__':

    outdir = 'result_vae'

    ini = configparser.ConfigParser()
    config_ini = 'config_mnist.ini'
    ini.read(config_ini, 'utf-8')

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    is_cuda = torch.cuda.is_available()

    x_size = ini.getint('conf', 'x_size')
    x_shape = (1, x_size, x_size)
    z_dim = ini.getint('conf', 'z_dim')
    num_workers = ini.getint('conf', 'num_workers')
    batch_size = ini.getint('conf', 'batch_size')
    n_epoch = ini.getint('conf', 'n_epoch')
    lr = ini.getfloat('conf', 'lr')

    eval_itvl = args.eval_itvl or 10

    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if is_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=data_transform),
        batch_size=batch_size, shuffle=True,
        **kwargs )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=data_transform),
        batch_size = batch_size,
        shuffle=True,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_function.Criterion()

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
            n_samples += x.shape[0]
        vae_loss_epoch /= n_samples
        vae_loss_epoch = vae_loss_epoch.detach().cpu().numpy()
        if loss_stats is None:
            loss_stats = vae_loss_epoch
        else:
            loss_stats = np.vstack([loss_stats, vae_loss_epoch])
        print(f'train loss = {vae_loss_epoch[0]:.3f} at epoch {epoch + 1}, n_samples {n_samples}')

        if (epoch + 1) % eval_itvl == 0:
            with torch.no_grad():
                model.eval()
                print(f'----- evaluating at epoch {epoch} -----')
                z_x = torch.Tensor().to(device)
                labels_true = []
                n_samples = 0

                for batch, (x, l) in enumerate(test_loader):
                    x = x.to(device)
                    params = model(x, return_params=True)
                    z_x = torch.cat((z_x, params['z_x']), 0)
                    labels_true += l
                    n_samples += x.shape[0]
                print(f'test epoch {epoch + 1}, n_samples {n_samples}')

                # decompose...
                print(f'----- decomposing and plotting -----')
                tsne = TSNE(n_jobs=4)
                z_x = z_x.cpu().numpy()

                z_x_tsne = tsne.fit_transform(z_x)

                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                ut.scatter(z_x_tsne[:, 0], z_x_tsne[:, 1],
                           labels_true, f'{outdir}/zx_tsne_{epoch}.png')
                for i in range(loss_stats.shape[1]):
                    loss_label = LOSS_LABELS[i]
                    yy = loss_stats[:,i]
                    ut.plot(yy, f'{outdir}/{loss_label}_{epoch}.png', 'epoch', loss_label)
