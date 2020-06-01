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
parser.add_argument('-y', '--y_dim', type=int,
                    help='number of classes')
parser.add_argument('-z', '--z_dim', type=int,
                    help='gaussian size')
parser.add_argument('-w', '--w_dim', type=int,
                    help='w dim')
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
args = parser.parse_args()


def get_loss(params, args):
    x = params['x']
    x_z = params['x_z']
    w_x_mean, w_x_var = params['w_x_mean'], params['w_x_var']
    y_wz = params['y_wz']
    z_x = params['z_x']  # (batch_size, z_dim)
    z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var'],
    z_wys = params['z_wys']
    z_wy_means, z_wy_vars = params['z_wy_means'], params['z_wy_vars']

    sigma = args.get('sigma') or 1.
    rec_wei = args.get('rec_wei') or 1.
    cond_wei = args.get('cond_wei') or 1.
    w_wei = args.get('w_wei') or 1.
    y_wei = args.get('y_wei') or 1.

    rec_loss = loss.reconstruction_loss(x, x_z, sigma)
    conditional_kl_loss = loss.conditional_kl_loss(z_x, z_x_mean, z_x_var,
                                                   z_wys, z_wy_means, z_wy_vars,
                                                   y_wz)
    w_prior_kl_loss = loss.w_prior_kl_loss(w_x_mean, w_x_var)
    y_prior_kl_loss = loss.y_prior_kl_loss(y_wz)
    total = rec_loss * rec_wei + conditional_kl_loss * cond_wei \
        + w_prior_kl_loss * w_wei + y_prior_kl_loss * y_wei
    total_m = total.mean()
    return total_m, {
        'reconstruction': rec_loss.mean(),
        'conditional_kl_loss': conditional_kl_loss.mean(),
        'w_prior_kl_loss': w_prior_kl_loss.mean(),
        'y_prior_kl_loss': y_prior_kl_loss.mean()
    }


def update_loss(loss_dict_total, loss_dict):
    for k, v in loss_dict.items():
        loss_dict_total[k] += v.item()


if __name__ == '__main__':
    # network params
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'utf-8')
    nargs = dict()

    nargs['bottle_channel'] = ini.getint('net', 'bottle_channel')
    nargs['conv_channels'] = json.loads(ini['net']['conv_channels'])
    nargs['conv_kernels'] = json.loads(ini['net']['conv_kernels'])
    nargs['pool_kernels'] = json.loads(ini['net']['pool_kernels'])
    nargs['middle_size'] = ini.getint('net', 'middle_size')
    nargs['dense_dim'] = ini.getint('net', 'dense_dim')
    nargs['activation'] = ini.get('net', 'activation')
    nargs['drop_rate'] = ini.getfloat('net', 'drop_rate')
    print(nargs)

    # test params
    x_shape = (1, 486, 486)
    y_dim = args.y_dim or ini.getint('net', 'y_dim')
    z_dim = args.z_dim or ini.getint('net', 'z_dim')
    w_dim = args.w_dim or ini.getint('net', 'w_dim')

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    n_epoch = args.n_epoch or ini.getint('conf', 'n_epoch')
    batch_size = args.batch_size or ini.getint('conf', 'batch_size')
    num_workers = args.num_workers or ini.getint('conf', 'num_workers')
    eval_itvl = args.eval_itvl or ini.getint('conf', 'eval_itvl')
    lr = args.lr or ini.getfloat('conf', 'lr')

    largs = dict()
    largs['sigma'] = ini.getfloat('loss', 'sigma') or 1.
    largs['rec_wei'] = ini.getfloat('loss', 'rec_wei') or 1.
    largs['cond_wei'] = ini.getfloat('loss', 'cond_wei') or 1.
    largs['w_wei'] = ini.getfloat('loss', 'w_wei') or 1.
    largs['y_wei'] = ini.getfloat('loss', 'y_wei') or 1.
    print(largs)

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
    train_set, test_set = dataset.split_by_labels(['Helix', 'Scratchy'],
                                                  n_cat=200,
                                                  min_cat=200)
    xlabels = np.array(train_set.get_labels()).astype(str)
    ylabels = np.array(range(y_dim)).astype(str)
    model = GMVAE(x_shape,
                  y_dim,
                  z_dim,
                  w_dim,
                  nargs)
    model.to(device)
    print(model)
    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True)
    losses = []
    nmis = []
    times = []
    for epoch_idx in range(n_epoch):
        epoch = epoch_idx + 1
        # train...
        model.train()
        print(f'----- training at epoch {epoch}... -----')
        time_start = time.time()
        loss_total = 0
        loss_dict_total = defaultdict(lambda: 0)
        for batch_idx, (x, l) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            total, loss_dict = get_loss(output, largs)
            total.backward()
            optimizer.step()
            loss_total += total.item()
            update_loss(loss_dict_total, loss_dict)
        losses.append([epoch, loss_total])
        time_elapse = time.time() - time_start
        times.append(time_elapse)
        print(f'train loss = {loss_total:.3f} at epoch {epoch_idx+1}')
        loss_info = ", ".join(
            [f'{k}: {v:.3f}' for k, v in loss_dict_total.items()])
        print(loss_info)
        print(f"calc time = {time_elapse:.3f} sec")
        print(f"average calc time = {np.array(times).mean():.3f} sec")

        # eval...
        if epoch % eval_itvl == 0:
            with torch.no_grad():
                model.eval()
                print(f'----- evaluating at epoch {epoch}... -----')
                time_start = time.time()
                loss_total = 0
                loss_dict_total = defaultdict(lambda: 0)
                z_x = torch.Tensor().to(device)
                w_x = torch.Tensor().to(device)
                labels_true = []
                labels_pred = []

                for batch_idx, (x, l) in enumerate(train_loader):
                    x = x.to(device)
                    output = model(x)
                    z_x = torch.cat((z_x, output['z_x']), 0)
                    w_x = torch.cat((w_x, output['w_x']), 0)
                    p = output['y_pred']
                    labels_true += l
                    labels_pred += list(p.cpu().numpy().astype(str))
                    total, loss_dict = get_loss(output, largs)
                    loss_total += total.item()
                    update_loss(loss_dict_total, loss_dict)
                time_elapse = time.time() - time_start
                print(f'test loss = {loss_total:.3f} at epoch {epoch_idx+1}')
                loss_info = ", ".join(
                    [f'{k}: {v:.3f}' for k, v in loss_dict_total.items()])
                print(loss_info)
                print(f"calc time = {time_elapse:.3f} sec")

                nmi = ut.nmi(labels_true, labels_pred)
                nmis.append([epoch, nmi])

                # decompose...
                print(f'----- decomposing and plotting... -----')
                print(f'N classes predicted: {len(set(labels_pred))}')
                print(f'NMI: {nmi:.3f}')

                time_start = time.time()
                pca = PCA(n_components=2)
                tsne = TSNE(n_components=2)
                z_x = z_x.cpu().numpy()
                w_x = w_x.cpu().numpy()

                # multi processing
                with mp.Pool(6) as pool:
                    z_x_pca = pool.apply_async(
                        pca.fit_transform, (z_x, )).get()
                    w_x_pca = pool.apply_async(
                        pca.fit_transform, (w_x, )).get()
                    z_x_tsne = pool.apply_async(
                        tsne.fit_transform, (z_x, )).get()
                    w_x_tsne = pool.apply_async(
                        tsne.fit_transform, (w_x, )).get()
                    cm = pool.apply_async(ut.confution_matrix, (labels_true,
                                                                labels_pred,
                                                                xlabels,
                                                                ylabels)).get()

                # output plots
                ut.scatter(z_x_pca[:, 0], z_x_pca[:, 1],
                           labels_true, f'{outdir}/z_pca_{epoch}_t.png')
                ut.scatter(z_x_pca[:, 0], z_x_pca[:, 1],
                           labels_pred, f'{outdir}/z_pca_{epoch}_p.png')
                ut.scatter(w_x_pca[:, 0], w_x_pca[:, 1],
                           labels_true, f'{outdir}/w_pca_{epoch}_t.png')
                ut.scatter(w_x_pca[:, 0], w_x_pca[:, 1],
                           labels_pred, f'{outdir}/w_pca_{epoch}_p.png')

                ut.scatter(z_x_tsne[:, 0], z_x_tsne[:, 1],
                           labels_true, f'{outdir}/z_tsne_{epoch}_t.png')
                ut.scatter(z_x_tsne[:, 0], z_x_tsne[:, 1],
                           labels_pred, f'{outdir}/z_tsne_{epoch}_p.png')
                ut.scatter(w_x_tsne[:, 0], w_x_tsne[:, 1],
                           labels_true, f'{outdir}/w_tsne_{epoch}_t.png')
                ut.scatter(w_x_tsne[:, 0], w_x_tsne[:, 1],
                           labels_pred, f'{outdir}/w_tsne_{epoch}_p.png')

                ut.cmshow(cm, xlabels, ylabels, f'{outdir}/cm_{epoch}.png')

                ut.plot(losses, f'{outdir}/loss_{epoch}.png', 'epoch', 'loss')
                ut.plot(nmis, f'{outdir}/nmi_{epoch}.png', 'epoch', 'normalized mutual information')

                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")
