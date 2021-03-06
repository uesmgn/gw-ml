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
from torchsummary import summary
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gmvae.dataset import Dataset
from gmvae.network import VAE
import gmvae.utils as ut
from gmvae import loss

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of GMVAE Clustering')

# NN Architecture
parser.add_argument('-z', '--z_dim', type=int,
                    help='gaussian size')
parser.add_argument('-e', '--n_epoch', type=int,
                    help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int,
                    help='batch size')
parser.add_argument('-n', '--num_workers', type=int,
                    help='num_workers of DataLoader')
parser.add_argument('-i', '--eval_itvl', type=int,
                    help='eval interval')
parser.add_argument('-lr', '--lr', type=float,
                    help='learning rate')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose')
parser.add_argument('-s', '--sample', action='store_true',
                    help='sample dataset into minimum N of each class')
parser.add_argument('-l', '--load_model', action='store_true',
                    help='load saved model')
args = parser.parse_args()


def get_loss(params):
    x = params['x']
    x_z = params['x_z']
    z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var']
    z_x = params['z_x']  # (batch_size, z_dim)

    # minimize reconstruction loss
    rec_loss = loss.reconstruction_loss(x, x_z, 'bce')
    gaussian_negative_kl = loss.gaussian_negative_kl(z_x_mean, z_x_var)
    total = rec_loss - gaussian_negative_kl
    return total, {'rec_loss': rec_loss,
                   'gaussian_negative_kl': gaussian_negative_kl }

def update_loss(loss_dict, loss_latest):
    for k, v in loss_latest.items():
        loss_dict[k] += v.item()

if __name__ == '__main__':
    # network params
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'utf-8')

    model_path = ini.get('net', 'model_path')
    load_model = args.load_model or False
    verbose = args.verbose or False

    nargs = dict()
    nargs['bottle_channel'] = ini.getint('net', 'bottle_channel')
    nargs['conv_channels'] = json.loads(ini['net']['conv_channels'])
    nargs['conv_kernels'] = json.loads(ini['net']['conv_kernels'])
    nargs['pool_kernels'] = json.loads(ini['net']['pool_kernels'])
    nargs['unpool_kernels'] = json.loads(ini['net']['unpool_kernels'])
    nargs['middle_size'] = ini.getint('net', 'middle_size')
    nargs['dense_dim'] = ini.getint('net', 'dense_dim')
    nargs['activation'] = ini.get('net', 'activation')
    nargs['drop_rate'] = ini.getfloat('net', 'drop_rate')
    nargs['pooling'] = ini.get('net', 'pooling')
    print(nargs)

    # test params
    x_shape = (1, 486, 486)
    z_dim = args.z_dim or ini.getint('net', 'z_dim')

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    n_epoch = args.n_epoch or ini.getint('conf', 'n_epoch')
    batch_size = args.batch_size or ini.getint('conf', 'batch_size')
    num_workers = args.num_workers or ini.getint('conf', 'num_workers')
    eval_itvl = args.eval_itvl or ini.getint('conf', 'eval_itvl')
    save_itvl = ini.getint('conf', 'save_itvl')
    lr = args.lr or ini.getfloat('conf', 'lr')
    sample = args.sample or False

    outdir = 'result_vae'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    df = pd.read_json('dataset.json')
    data_transform = transforms.Compose([
        transforms.CenterCrop((x_shape[1], x_shape[2])),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = Dataset(df, data_transform)
    train_set, test_set = dataset.split_by_labels(['Helix', 'Scratchy'],
                                                  sample=sample,
                                                  min_cat=200)
    xlabels = np.array(train_set.get_labels()).astype(str)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             drop_last=True)

    model = VAE(x_shape,
                z_dim,
                nargs)

    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    model.eval()
    print(model)
    # summary(model, x_shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    init_epoch = 0
    loss_cum = defaultdict(list)
    times = []

    if load_model and type(model_path) is str:
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss_cum = checkpoint['loss_cum']
        times = checkpoint['times']
        print(f'load model from epoch {init_epoch}')

    for epoch_idx in range(init_epoch, n_epoch):
        epoch = epoch_idx + 1
        # train...
        model.train()
        print(f'----- training at epoch {epoch}... -----')
        time_start = time.time()
        loss_dict = defaultdict(lambda: 0)
        n_samples = 0
        loss_total = 0
        for batch_idx, (x, l) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            params = model(x, return_params=True)
            total, loss_latest = get_loss(params)
            if verbose:
                loss_info = ', '.join([f'{k}: {v.item():.3f}' for k, v in loss_latest.items()])
                print(f'{total.item():.3f},', loss_info)
            total.backward()
            optimizer.step()
            loss_total += total.item()
            update_loss(loss_dict, loss_latest)
            n_samples += 1
        loss_total /= n_samples
        loss_cum['total_loss'].append([epoch, loss_total])
        for k, v in loss_dict.items():
            loss_dict[k] /= n_samples
            loss_cum[k].append([epoch, loss_dict[k]])
        time_elapse = time.time() - time_start
        times.append(time_elapse)
        print(f'train loss = {loss_total:.3f} at epoch {epoch_idx+1}')
        loss_info = ', '.join([f'{k}: {v:.3f}' for k, v in loss_dict.items()])
        print(loss_info)
        print(f"calc time = {time_elapse:.3f} sec")
        print(f"average calc time = {np.array(times).mean():.3f} sec")

        # eval...
        if epoch % eval_itvl == 0:
            with torch.no_grad():
                model.eval()
                print(f'----- evaluating at epoch {epoch}... -----')
                time_start = time.time()

                z_x = torch.Tensor().to(device)
                labels_true = []
                n_samples = 0
                loss_total = 0
                for batch_idx, (x, l) in enumerate(train_loader):
                    x = x.to(device)
                    params = model(x, return_params=True)
                    total, loss_latest = get_loss(params)
                    z_x = torch.cat((z_x, params['z_x']), 0)
                    labels_true += l
                    loss_total += total.item()
                    n_samples += 1
                loss_total /= n_samples
                print(f'test loss = {loss_total:.3f} at epoch {epoch_idx+1}')

                z_x_new = torch.Tensor().to(device)
                labels_new = []
                n_samples = 0
                loss_total_new = 0
                for batch_idx, (x, l) in enumerate(test_loader):
                    x = x.to(device)
                    params = model(x, return_params=True)
                    total, loss_latest = get_loss(params)
                    z_x = torch.cat((z_x, params['z_x']), 0)
                    labels_new += l
                    loss_total_new += total.item()
                    n_samples += 1
                loss_total_new /= n_samples
                time_elapse = time.time() - time_start
                print(f'test-newclass loss = {loss_total:.3f} at epoch {epoch_idx+1}')

                print(f"calc time = {time_elapse:.3f} sec")

                # decompose...
                print(f'----- decomposing and plotting... -----')
                time_start = time.time()
                pca = PCA(n_components=2)
                tsne = TSNE(n_components=2)
                z_x = z_x.cpu().numpy()

                z_x_tsne = tsne.fit_transform(z_x)

                ut.scatter(z_x_tsne[:, 0], z_x_tsne[:, 1],
                           labels_true, f'{outdir}/zx_tsne_{epoch}_t.png')
                for k, v in loss_cum.items():
                    ut.plot(v, f'{outdir}/{k}_{epoch}.png', 'epoch', k)

                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")

        if epoch % save_itvl == 0:
            torch.save({
            'epoch': epoch,
            'loss_cum': loss_cum,
            'times': times,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path )
