import argparse
import configparser
import json
import time
import os
import multiprocessing as mp
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gmvae.dataset import Dataset
from gmvae.network import GMVAE
import gmvae.utils as ut
from gmvae import loss

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of GMVAE Clustering')

# NN Architecture
parser.add_argument('-y', '--y_dim', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('-z', '--z_dim', default=512, type=int,
                    help='gaussian size (default: 512)')
parser.add_argument('-w', '--w_dim', default=20, type=int,
                    help='w dim (default: 20)')
parser.add_argument('-e', '--n_epoch', default=1000, type=int,
                    help='number of epochs (default: 1000)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='batch size (default: 32)')
parser.add_argument('-n', '--num_workers', default=4, type=int,
                    help='num_workers of DataLoader (default: 4)')
parser.add_argument('-s', '--sigma', default=0.01, type=float,
                    help='sigma to use reconstruction loss (default: 0.01)')
parser.add_argument('-i', '--eval_itvl', default=5, type=int,
                    help='eval interval (default: 5)')
args = parser.parse_args()

def get_loss(params):
    x = params['x']
    x_z = params['x_z']
    w_x_mean, w_x_logvar = params['w_x_mean'], params['w_x_logvar']
    y_wz = params['y_wz']
    z_x = params['z_x'] # (batch_size, z_dim)
    z_x_mean, z_x_logvar = params['z_x_mean'], params['z_x_logvar'],
    z_wy_means, z_wy_logvars = params['z_wy_means'], params['z_wy_logvars']
    rec_loss = loss.reconstruction_loss(x, x_z)
    conditional_kl_loss = loss.conditional_kl_loss(z_x, z_x_mean, z_x_logvar,
                                                   z_wy_means, z_wy_logvars, y_wz)
    w_prior_kl_loss = loss.w_prior_kl_loss(w_x_mean, w_x_logvar)
    y_prior_kl_loss = loss.y_prior_kl_loss(y_wz)
    total = rec_loss + conditional_kl_loss + w_prior_kl_loss + y_prior_kl_loss
    return total, {
        'reconstruction': rec_loss,
        'conditional_kl_loss': conditional_kl_loss,
        'w_prior_kl_loss': w_prior_kl_loss,
        'y_prior_kl_loss': y_prior_kl_loss
    }

def update_loss(loss_dict_total, loss_dict):
    for k, v in loss_dict.items():
        loss_dict_total[k] += v.item()

if __name__ == '__main__':
    # network params
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'utf-8')
    nargs = dict()
    nargs['conv_channels'] = json.loads(ini['net']['conv_channels'])
    nargs['conv_kernels'] = json.loads(ini['net']['conv_kernels'])
    nargs['conv_strides'] = json.loads(ini['net']['conv_strides'])
    nargs['middle_size'] = ini['net']['middle_size']
    nargs['dense_dim'] = ini['net']['dense_dim']
    nargs['activation'] = ini['net']['activation']

    # test params
    x_shape = (1, 486, 486)
    y_dim = args.y_dim
    z_dim = args.z_dim
    w_dim = args.w_dim

    sigma = args.sigma

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers

    eval_itvl = args.eval_itvl
    outdir = 'result_gmvae'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    df = pd.read_json('dataset.json')
    data_transform = transforms.Compose([
        transforms.CenterCrop((x_shape[1], x_shape[2])),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = Dataset(df, data_transform)
    train_set, test_set = dataset.split_dataset(0.7)
    labels = np.array(dataset.get_labels()).astype(str)
    labels_pred = np.array(range(y_dim)).astype(str)
    model = GMVAE(x_shape, y_dim, z_dim, w_dim,
                  nargs)
    model.to(device)
    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True)
    losses = []
    for epoch_idx in range(n_epoch):
        epoch = epoch_idx + 1
        # ---------- train ----------
        model.train()
        print(f'---------- train: {epoch} ... ----------')
        time_start = time.time()
        loss_total = 0
        loss_dict_total = defaultdict(lambda: 0)
        for batch_idx, (x, l) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            total, loss_dict = get_loss(output)
            total.backward()
            optimizer.step()
            loss_total += total.item()
            update_loss(loss_dict_total, loss_dict)
        losses.append(loss_total)
        time_elapse = time.time() - time_start
        print(f'train loss = {loss_total:.3f} at epoch {epoch_idx+1}')
        loss_info = ", ".join([f'{k}: {v:.3f}' for k, v in loss_dict.items()])
        print(loss_info)
        print(f"calc time = {time_elapse:.3f} sec")

        # ---------- eval ----------
        if epoch % eval_itvl == 0:
            with torch.no_grad():
                model.eval()
                print(f'---------- eval: {epoch} ... ----------')
                time_start = time.time()
                loss_total = 0
                loss_dict_total = defaultdict(lambda: 0)
                z_x = torch.Tensor().to(device)
                w_x = torch.Tensor().to(device)
                labels = []
                labels_pred = []
                for batch_idx, (x, l) in enumerate(train_loader):
                    x = x.to(device)
                    output = model(x)
                    z_x = torch.cat((z_x, output['z_x']), 0)
                    w_x = torch.cat((w_x, output['w_x']), 0)
                    _, p = torch.max(output['y_wz'], dim=1)
                    labels += l
                    labels_pred += list(p.cpu().numpy())
                    total, loss_dict = get_loss(output)
                    loss_total += total.item()
                    update_loss(loss_dict_total, loss_dict)
                time_elapse = time.time() - time_start
                print(f'test loss = {loss_total:.3f} at epoch {epoch_idx+1}')
                loss_info = ", ".join([f'{k}: {v:.3f}' for k, v in loss_dict.items()])
                print(loss_info)
                print(f"calc time = {time_elapse:.3f} sec")

                print(f'---------- decomposition and plot ... ----------')
                time_start = time.time()
                pca = PCA(n_components=2)
                tsne = TSNE(n_components=2)
                z_x = z_x.cpu().numpy()
                w_x = w_x.cpu().numpy()

                with mp.Pool(4) as pool:
                    z_x_pca = pool.apply_async(pca.fit_transform, (z_x, )).get()
                    w_x_pca = pool.apply_async(pca.fit_transform, (w_x, )).get()
                    z_x_tsne = pool.apply_async(tsne.fit_transform, (z_x, )).get()
                    w_x_tsne = pool.apply_async(tsne.fit_transform, (w_x, )).get()

                ut.plot_latent(z_x_pca[:,0], z_x_pca[:,1], labels, f'{outdir}/z_pca_{epoch}_t.png')
                ut.plot_latent(z_x_pca[:,0], z_x_pca[:,1], labels_pred, f'{outdir}/z_pca_{epoch}_p.png')
                ut.plot_latent(w_x_pca[:,0], w_x_pca[:,1], labels, f'{outdir}/w_pca_{epoch}_t.png')
                ut.plot_latent(w_x_pca[:,0], w_x_pca[:,1], labels_pred, f'{outdir}/w_pca_{epoch}_p.png')

                ut.plot_latent(z_x_tsne[:,0], z_x_tsne[:,1], labels, f'{outdir}/z_tsne_{epoch}_t.png')
                ut.plot_latent(z_x_tsne[:,0], z_x_tsne[:,1], labels_pred, f'{outdir}/z_tsne_{epoch}_p.png')
                ut.plot_latent(w_x_tsne[:,0], w_x_tsne[:,1], labels, f'{outdir}/w_tsne_{epoch}_t.png')
                ut.plot_latent(w_x_tsne[:,0], w_x_tsne[:,1], labels_pred, f'{outdir}/w_tsne_{epoch}_p.png')

                ut.plot_loss(losses, f'{outdir}/loss_{epoch}.png')

                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")
