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
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose')
parser.add_argument('-s', '--sample', action='store_true',
                    help='sample dataset into minimum N of each class')
parser.add_argument('-l', '--load_model', action='store_true',
                    help='load saved model')
args = parser.parse_args()


def get_loss(params, args):
    x = params['x']
    x_z = params['x_z']
    w_x_mean, w_x_var = params['w_x_mean'], params['w_x_var']
    y_wz = params['y_wz']
    z_x = params['z_x']  # (batch_size, z_dim)
    z_x_mean, z_x_var = params['z_x_mean'], params['z_x_var'],
    z_wy_means, z_wy_vars = params['z_wy_means'], params['z_wy_vars']

    sigma = args.get('sigma') or 1.
    rec_wei = args.get('rec_wei') or 1.
    cond_wei = args.get('cond_wei') or 1.
    w_wei = args.get('w_wei') or 1.
    y_wei = args.get('y_wei') or 1.
    y_thres = args.get('y_thres') or 1000.

    rec_loss = loss.reconstruction_loss(x, x_z, sigma)
    conditional_kl = loss.conditional_kl(z_x, z_x_mean, z_x_var,
                                                   z_wy_means, z_wy_vars,
                                                   y_wz)
    w_prior_kl = loss.w_prior_kl(w_x_mean, w_x_var)
    y_prior_kl = loss.y_prior_kl(y_wz, y_thres)
    total = rec_loss * rec_wei - conditional_kl * cond_wei \
            - w_prior_kl * w_wei - y_prior_kl * y_wei
    total = total.sum()
    return total


if __name__ == '__main__':
    # network params
    ini = configparser.ConfigParser()
    ini.read('./config.ini', 'utf-8')

    model_path = ini.get('net', 'model_path')
    load_model = args.load_model or False
    sigma = ini.getfloat('net', 'sigma')
    verbose = args.verbose or False

    nargs = dict()
    nargs['bottle_channel'] = ini.getint('net', 'bottle_channel')
    nargs['conv_channels'] = json.loads(ini['net']['conv_channels'])
    nargs['conv_kernels'] = json.loads(ini['net']['conv_kernels'])
    nargs['pool_kernels'] = json.loads(ini['net']['pool_kernels'])
    nargs['middle_size'] = ini.getint('net', 'middle_size')
    nargs['dense_dim'] = ini.getint('net', 'dense_dim')
    nargs['activation'] = ini.get('net', 'activation')
    nargs['drop_rate'] = ini.getfloat('net', 'drop_rate')
    nargs['sigma'] = sigma
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
    save_itvl = ini.getint('conf', 'save_itvl')
    lr = args.lr or ini.getfloat('conf', 'lr')
    sample = args.sample or False

    largs = dict()
    largs['sigma'] = sigma
    largs['rec_wei'] = ini.getfloat('loss', 'rec_wei') or 1.
    largs['cond_wei'] = ini.getfloat('loss', 'cond_wei') or 1.
    largs['w_wei'] = ini.getfloat('loss', 'w_wei') or 1.
    largs['y_wei'] = ini.getfloat('loss', 'y_wei') or 1.
    largs['y_thres'] = ini.getfloat('loss', 'y_thres') or 1000.
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
                                                  sample=sample,
                                                  min_cat=200)
    xlabels = np.array(train_set.get_labels()).astype(str)
    ylabels = np.array(range(y_dim)).astype(str)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True)

    model = GMVAE(x_shape,
                  y_dim,
                  z_dim,
                  w_dim,
                  nargs)
    model.to(device)
    summary(model, x_shape)
    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    init_epoch = 0
    losses = []
    nmis = []
    times = []

    if load_model and type(model_path) is str:
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        losses = checkpoint['losses']
        nmis = checkpoint['nmis']
        times = checkpoint['times']
        print(f'load model from epoch {init_epoch}')

    for epoch_idx in range(init_epoch, n_epoch):
        epoch = epoch_idx + 1
        # train...
        model.train()
        print(f'----- training at epoch {epoch}... -----')
        time_start = time.time()
        loss_total = 0
        n_samples = 0
        for batch_idx, (x, l) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_z, params = model(x)
            total = get_loss(params, largs)
            total.backward()
            optimizer.step()
            loss_total += total.item()
            n_samples += x.shape[0]
        loss_total /= n_samples
        losses.append([epoch, loss_total])
        time_elapse = time.time() - time_start
        times.append(time_elapse)
        print(f'train loss = {loss_total:.3f} at epoch {epoch_idx+1}')
        print(f"calc time = {time_elapse:.3f} sec")
        print(f"average calc time = {np.array(times).mean():.3f} sec")

        # eval...
        if epoch % eval_itvl == 0:
            with torch.no_grad():
                model.eval()
                print(f'----- evaluating at epoch {epoch}... -----')
                time_start = time.time()
                z_x = torch.Tensor().to(device)
                z_wy = torch.Tensor().to(device)
                w_x = torch.Tensor().to(device)
                labels_true = []
                labels_pred = []

                for batch_idx, (x, l) in enumerate(train_loader):
                    x = x.to(device)
                    x_z, params = model(x)
                    z_x = torch.cat((z_x, params['z_x']), 0)
                    z_wy = torch.cat((z_wy, params['z_wy']), 0)
                    w_x = torch.cat((w_x, params['w_x']), 0)
                    p = params['y_pred']
                    labels_true += l
                    labels_pred += list(p.cpu().numpy().astype(str))
                nmi = ut.nmi(labels_true, labels_pred)
                nmis.append([epoch, nmi])
                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")
                print(f'# classes predicted: {len(set(labels_pred))}')
                print(f'NMI: {nmi:.3f}')

                # decompose...
                print(f'----- decomposing and plotting... -----')
                time_start = time.time()
                pca = PCA(n_components=2)
                tsne = TSNE(n_components=2)
                z_x = z_x.cpu().numpy()
                z_wy = z_wy.cpu().numpy()
                w_x = w_x.cpu().numpy()

                # multi processing
                with mp.Pool(6) as pool:
                    z_x_tsne = pool.apply_async(
                        tsne.fit_transform, (z_x, )).get()
                    z_wy_tsne = pool.apply_async(
                        tsne.fit_transform, (z_wy, )).get()
                    w_x_tsne = pool.apply_async(
                        tsne.fit_transform, (w_x, )).get()
                    cm, cm_index, cm_columnns = pool.apply_async(ut.confution_matrix, (labels_true,
                                                                 labels_pred,
                                                                 xlabels,
                                                                 ylabels)).get()

                # output plots
                # ut.scatter(z_x_pca[:, 0], z_x_pca[:, 1],
                #            labels_true, f'{outdir}/z_pca_{epoch}_t.png')
                # ut.scatter(z_x_pca[:, 0], z_x_pca[:, 1],
                #            labels_pred, f'{outdir}/z_pca_{epoch}_p.png')
                # ut.scatter(w_x_pca[:, 0], w_x_pca[:, 1],
                #            labels_true, f'{outdir}/w_pca_{epoch}_t.png')
                # ut.scatter(w_x_pca[:, 0], w_x_pca[:, 1],
                #            labels_pred, f'{outdir}/w_pca_{epoch}_p.png')

                ut.scatter(z_x_tsne[:, 0], z_x_tsne[:, 1],
                           labels_true, f'{outdir}/zx_tsne_{epoch}_t.png')
                ut.scatter(z_x_tsne[:, 0], z_x_tsne[:, 1],
                           labels_pred, f'{outdir}/zx_tsne_{epoch}_p.png')
                ut.scatter(z_wy_tsne[:, 0], z_wy_tsne[:, 1],
                           labels_true, f'{outdir}/zwy_tsne_{epoch}_t.png')
                ut.scatter(z_wy_tsne[:, 0], z_wy_tsne[:, 1],
                           labels_pred, f'{outdir}/zwy_tsne_{epoch}_p.png')
                ut.scatter(w_x_tsne[:, 0], w_x_tsne[:, 1],
                           labels_true, f'{outdir}/wx_tsne_{epoch}_t.png')
                ut.scatter(w_x_tsne[:, 0], w_x_tsne[:, 1],
                           labels_pred, f'{outdir}/wx_tsne_{epoch}_p.png')

                ut.cmshow(cm, cm_index, cm_columnns, f'{outdir}/cm_{epoch}.png')

                ut.plot(losses, f'{outdir}/loss_{epoch}.png', 'epoch', 'loss')
                ut.plot(nmis, f'{outdir}/nmi_{epoch}.png', 'epoch', 'normalized mutual information')

                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")

        if epoch % save_itvl == 0:
            torch.save({
            'epoch': epoch,
            'losses': losses,
            'nmis': nmis,
            'times': times,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path )
