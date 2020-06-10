import argparse
import configparser
import json
import time
import os
import multiprocessing as mp
from collections import defaultdict
from pprint import  pprint
from lal import gpstime

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gmvae.dataset import Dataset
from gmvae.network import GMVAE
import gmvae.utils as ut
from gmvae import loss_function

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of GMVAE Unsupervised Clustering')

parser.add_argument('-e', '--n_epoch', type=int,
                    help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int,
                    help='batch size')
parser.add_argument('-lr', '--lr', type=float,
                    help='learning rate')
parser.add_argument('--num_workers', type=int,
                    help='num_workers of DataLoader')
parser.add_argument('--eval_itvl', type=int,
                    help='eval interval')
parser.add_argument('--save_itvl', type=int,
                    help='save interval')

parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose')
parser.add_argument('-s', '--n_sample', type=int,
                    help='N of sampling data of each class')
parser.add_argument('-l', '--load_model', action='store_true',
                    help='load saved model')

args = parser.parse_args()

if __name__ == '__main__':

    config_ini = 'config.ini'

    # import parameters from config.ini files
    assert os.path.exists(config_ini)
    basedir = os.path.dirname(os.path.abspath(__file__))

    ini = configparser.ConfigParser()
    ini.read(f'{basedir}/{config_ini}', 'utf-8')

    load_model = args.load_model or False
    verbose = args.verbose or False
    n_epoch = args.n_epoch or ini.getint('conf', 'n_epoch')
    batch_size = args.batch_size or ini.getint('conf', 'batch_size')
    num_workers = args.num_workers or ini.getint('conf', 'num_workers')
    eval_itvl = args.eval_itvl or ini.getint('conf', 'eval_itvl')
    save_itvl = args.save_itvl or ini.getint('conf', 'save_itvl')
    lr = args.lr or ini.getfloat('conf', 'lr')
    n_sample = args.n_sample or 0

    # model params
    nargs = dict()
    x_size = ini.getint('net', 'x_size')
    y_dim = ini.getint('net', 'y_dim')
    nargs['x_shape'] = (1, x_size, x_size)
    nargs['y_dim'] = y_dim
    nargs['z_dim'] = ini.getint('net', 'z_dim')
    nargs['w_dim'] = ini.getint('net', 'w_dim')
    nargs['bottle_channel'] = ini.getint('net', 'bottle_channel')
    nargs['conv_channels'] = json.loads(ini.get('net', 'conv_channels'))
    nargs['kernels'] = json.loads(ini.get('net', 'kernels'))
    nargs['pool_kernels'] = json.loads(ini.get('net', 'pool_kernels'))
    nargs['middle_size'] = ini.getint('net', 'middle_size')
    nargs['dense_dim'] = ini.getint('net', 'dense_dim')
    nargs['activation'] = ini.get('net', 'activation')
    nargs['pooling'] = ini.get('net', 'pooling')
    if verbose:
        pprint(nargs)

    largs = dict()
    largs['rec_wei'] = ini.getfloat('loss', 'rec_wei') or 1.
    largs['cond_wei'] = ini.getfloat('loss', 'cond_wei') or 1.
    largs['w_wei'] = ini.getfloat('loss', 'w_wei') or 1.
    largs['y_wei'] = ini.getfloat('loss', 'y_wei') or 1.
    largs['y_thres'] = ini.getfloat('loss', 'y_thres') or 0.
    if verbose:
        pprint(largs)

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

    time_exec = gpstime.gps_time_now()
    outdir = ini.get('conf', 'outdir') + f'_{time_exec}'
    model_path = f'{outdir}/' + ini.get('conf', 'model_path')
    dataset_json = ini.get('conf', 'dataset_json')

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    data_transform = transforms.Compose([
        transforms.CenterCrop((x_size, x_size)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    df = pd.read_json(dataset_json)
    dataset = Dataset(df, data_transform)
    # dataset.sample('label', min_value_count=200, n_sample=200)
    # new_set = dataset.get_by_keys('label', new_labels)
    # dataset = dataset.get_by_keys('label', labels)
    train_set, test_set = dataset.random_split()
    if verbose:
        print(f'length of training set: ', len(train_set))
        print(f'length of test set: ', len(test_set))

    xlabels = dataset.unique_column('label')
    ylabels = np.array(range(y_dim))

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

    model = GMVAE_mult(nargs)

    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_function.Criterion()

    model.eval()
    if verbose:
        summary(model, x_shape)

    init_epoch = 0
    loss_labels = ['total',
                   'reconstruction_loss',
                   'conditional_loss',
                   'w-prior_loss',
                   'y-prior_loss']
    loss_stats = None
    nmi_stats = []
    ari_stats = []
    time_stats = []

    if load_model and type(model_path) is str:
        assert os.path.exists(model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss_stats = checkpoint['loss_stats']
        nmi_stats = checkpoint['nmi_stats']
        ari_stats = checkpoint['ari_stats']
        time_stats = checkpoint['time_stats']
        print(f'load model from epoch {init_epoch}')

    for epoch_idx in range(init_epoch, n_epoch):
        epoch = epoch_idx + 1
        # training
        model.train()
        print(f'----- training at epoch {epoch} -----')
        time_start = time.time()

        n_samples = 0
        total_loss_epoch = 0
        gmvae_loss_epoch = torch.zeros(len(loss_labels)).to(device)

        for batch_idx, (x, l) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()

            params = model(x, return_params=True)
            y_pred = params['y_pred'].to(int)
            gmvae_loss = criterion.gmvae_loss(params, largs, reduction='none')
            gmvae_loss_total = gmvae_loss.sum()
            gmvae_loss_total.backward()

            params = model(x, clustering=True)
            y_wz = params['y_wz']
            cross_entropy = criterion.cross_entropy(y_wz, y_pred.detach())
            cross_entropy.backward()

            optimizer.step()
            gmvae_loss_epoch += torch.cat([gmvae_loss_total.view(-1),
                                           gmvae_loss.view(-1)])
            total_loss_epoch += gmvae_loss_total.item()
            n_samples += 1
        gmvae_loss_epoch /= n_samples
        gmvae_loss_epoch = gmvae_loss_epoch.detach().cpu().numpy()
        # initialize or append loss
        if loss_stats is None:
            loss_stats = gmvae_loss_epoch
        else:
            loss_stats = np.vstack([loss_stats, gmvae_loss_epoch])
        time_elapse = time.time() - time_start
        time_stats.append(time_elapse)
        print(f'train loss = {gmvae_loss_epoch[0]:.3f} at epoch {epoch_idx+1}')
        print(f"calc time = {time_elapse:.3f} sec")
        print(f"average calc time = {np.array(time_stats).mean():.3f} sec")

        # eval...
        if epoch % eval_itvl == 0:
            with torch.no_grad():
                model.eval()
                print(f'----- evaluating at epoch {epoch} -----')
                time_start = time.time()
                z_x = torch.Tensor().to(device)
                z_wy = torch.Tensor().to(device)
                w_x = torch.Tensor().to(device)
                labels_true = []
                labels_pred = []

                for batch_idx, (x, l) in enumerate(train_loader):
                    x = x.to(device)
                    params = model(x, return_params=True)
                    z_x = torch.cat((z_x, params['z_x']), 0)
                    z_wy = torch.cat((z_wy, params['z_wy']), 0)
                    w_x = torch.cat((w_x, params['w_x']), 0)
                    y_pred = params['y_pred']
                    labels_true += l
                    labels_pred += list(y_pred.cpu().numpy().astype(int))
                nmi = ut.nmi(labels_true, labels_pred)
                nmi_stats.append(nmi)
                ari = ut.ari(labels_true, labels_pred)
                ari_stats.append(ari)
                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")
                print(f'# classes predicted: {len(set(labels_pred))}')
                print(f'NMI: {nmi:.3f}')
                print(f'ARI: {ari:.3f}')

                # decompose...
                print(f'----- decomposing and plotting -----')
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

                counter = np.array(
                    [[label, np.count_nonzero(labels_pred==label)] for label in ylabels])
                ut.bar(counter[:,0], counter[:,1], f'{outdir}/bar_{epoch}.png', reverse=True)

                ut.cmshow(cm, cm_index, cm_columnns, f'{outdir}/cm_{epoch}.png')

                for i in range(loss_stats.shape[1]):
                    loss_label = loss_labels[i]
                    yy = loss_stats[:,i]
                    ut.plot(yy, f'{outdir}/{loss_label}_{epoch}.png', 'epoch', loss_label)
                ut.plot(nmi_stats, f'{outdir}/nmi_{epoch}.png', 'epoch', 'adjusted mutual info score',
                        ylim=(-0.1,1))
                ut.plot(ari_stats, f'{outdir}/ari_{epoch}.png', 'epoch', 'adjusted rand score',
                        ylim=(-0.1,1))

                time_elapse = time.time() - time_start
                print(f"calc time = {time_elapse:.3f} sec")

        if epoch % save_itvl == 0:
            torch.save({
            'epoch': epoch,
            'loss_stats': loss_stats,
            'ari_stats': ari_stats,
            'nmi_stats': nmi_stats,
            'time_stats': time_stats,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, model_path )
