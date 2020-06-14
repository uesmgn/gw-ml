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

from gmvae.dataset import Dataset
from gmvae.network import *
from gmvae import loss_function

from utils.clustering import decomposition, metrics
from utils.plotlib import plot as plt

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of GMVAE Unsupervised Clustering')

parser.add_argument('-e', '--n_epoch', type=int,
                    help='number of epochs')
parser.add_argument('-o', '--outdir', type=str,
                    help='output directory')
parser.add_argument('-m', '--model_id', type=int,
                    help='model id')
parser.add_argument('--eval_itvl', type=int,
                    help='eval interval')
parser.add_argument('--save_itvl', type=int,
                    help='save interval')

parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose')
parser.add_argument('-l', '--load_model', action='store_true',
                    help='load saved model')

args = parser.parse_args()

if __name__ == '__main__':

    time_exec = gpstime.gps_time_now()

    config_ini = 'config_gmvae.ini'
    assert os.path.exists(config_ini)
    basedir = os.path.dirname(os.path.abspath(__file__))
    ini = configparser.ConfigParser()
    ini.read(f'{basedir}/{config_ini}', 'utf-8')

    load_model = args.load_model or False
    verbose = args.verbose or False
    outdir = args.outdir or ini.get('conf', 'outdir') + f'_{time_exec}'
    n_epoch = args.n_epoch or ini.getint('conf', 'n_epoch')
    eval_itvl = args.eval_itvl or ini.getint('conf', 'eval_itvl')
    save_itvl = args.save_itvl or ini.getint('conf', 'save_itvl')
    model_id = args.model_id or 0

    batch_size = ini.getint('conf', 'batch_size') or 32
    num_workers = ini.getint('conf', 'num_workers') or 1
    lr = ini.getfloat('conf', 'lr') or 1e-3
    model_path = ini.get('conf', 'model_path')
    dataset_json = ini.get('conf', 'dataset_json')
    x_size = ini.getint('net', 'x_size')
    y_dim = ini.getint('net', 'y_dim')

    nargs = dict()
    nargs['x_shape'] = (1, x_size, x_size)
    nargs['y_dim'] = y_dim
    nargs['z_dim'] = ini.getint('net', 'z_dim')
    nargs['w_dim'] = ini.getint('net', 'w_dim')
    nargs['bottle_channel'] = ini.getint('net', 'bottle_channel')
    nargs['conv_channels'] = json.loads(ini.get('net', 'conv_channels'))
    nargs['kernels'] = json.loads(ini.get('net', 'kernels'))
    nargs['pool_kernels'] = json.loads(ini.get('net', 'pool_kernels'))
    nargs['middle_size'] = ini.getint('net', 'middle_size')
    nargs['hidden_dim'] = ini.getint('net', 'hidden_dim')
    nargs['activation'] = ini.get('net', 'activation')
    nargs['pooling'] = ini.get('net', 'pooling')
    if verbose:
        pprint(nargs)

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

    data_transform = transforms.Compose([
        transforms.CenterCrop((x_size, x_size)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    df = pd.read_json(dataset_json)
    dataset = Dataset(df, data_transform)
    dataset.sample('label', min_value_count=200)
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
    if model_id == 0:
        model = GMVAE(nargs)
    elif model_id == 1:
        model = GMVAE_gumbel(nargs)

    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_function.Criterion()

    # model.eval()
    # if verbose:
    #     summary(model, x_shape)

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

    if load_model:
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

    beta_rate = 0.05

    for epoch in range(init_epoch, n_epoch):
        epoch = epoch + 1
        # training
        model.train()
        print(f'----- training at epoch {epoch} -----')
        time_start = time.time()

        n_samples = 0
        gmvae_loss_epoch = np.zeros(len(loss_labels))
        # beta(x) = exp(-10*exp(-0.1x))
        beta = max((1., np.exp(-20 * np.exp(-beta_rate * epoch))), 0)

        for batch_idx, (x, l) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            params = model(x, return_params=True)
            total_loss, each_loss = criterion.gmvae_loss(params, beta,
                                                         reduction='mean')
            if verbose:
                print(f'batch: {batch_idx}, loss: {total_loss:.3f}')
                print(', '.join([f'{loss_labels[i]}: {l:.3f}' for i, l in enumerate(each_loss)]))
            total_loss.backward()
            optimizer.step()
            gmvae_loss_epoch += each_loss.cpu().numpy()
            n_samples += x.shape[0]
        gmvae_loss_epoch /= n_samples
        # initialize or append loss
        if loss_stats is None:
            loss_stats = gmvae_loss_epoch
        else:
            loss_stats = np.vstack([loss_stats, gmvae_loss_epoch])
        time_elapse = time.time() - time_start
        time_stats.append(time_elapse)
        print(f'train loss = {gmvae_loss_epoch[0]:.3f} at epoch {epoch}, beta {beta:.3f}')
        print(', '.join([f'{loss_labels[i]}: {l:.3f}' for i, l in enumerate(gmvae_loss_epoch)]))
        print(f"calc time = {time_elapse:.3f} sec")
        print(f"average calc time = {np.array(time_stats).mean():.3f} sec")

        # ----------
        # evaluation
        # ----------
        if epoch % eval_itvl == 0:
            print(f'----- evaluating at epoch {epoch} -----')
            time_start = time.time()
            model.eval()
            with torch.no_grad():
                # latent features for visualizing
                z_x = torch.Tensor().to(device)
                w_x = torch.Tensor().to(device)
                labels_true = np.array([])
                labels_pred = np.array([])

                for batch_idx, (x, l) in enumerate(train_loader):
                    x = x.to(device)
                    params = model(x)
                    # stack features over all batches
                    z_x = torch.cat((z_x, params['z_x']), 0)
                    w_x = torch.cat((w_x, params['w_x']), 0)
                    # concatenate all labels
                    labels_pred = np.append(labels_pred, params['y_pred'].cpu().numpy())
                    labels_true = np.append(labels_true, l)
            # metrics
            nmi = metrics.nmi(labels_true, labels_pred)
            nmi_stats.append(nmi)
            ari = metrics.ari(labels_true, labels_pred)
            ari_stats.append(ari)
            cm, xlabels, ylabels = metrics.confution_matrix(
                labels_true, labels_pred, xlabels, ylabels, return_labels=True)
            time_elapse = time.time() - time_start

            print(f"calc time = {time_elapse:.3f} sec")
            print(f'# classes predicted: {len(set(labels_pred))}')
            print(f'nmi: {nmi:.3f}')
            print(f'ari: {ari:.3f}')

            # ----------
            # decomposing and plotting latent features
            # ----------
            print(f'----- decomposing and plotting -----')
            time_start = time.time()
            tsne = decomposition.TSNE(cuda=True)
            z_x_tsne = tsne.fit_transform(z_x)
            w_x_tsne = tsne.fit_transform(w_x)

            if not os.path.exists(outdir):
                os.mkdir(outdir)
            # output plots
            plt.scatter(z_x_tsne, labels_true,
                        f'{outdir}/zx_tsne_{epoch}_t.png')
            plt.scatter(z_x_tsne, labels_pred,
                        f'{outdir}/zx_tsne_{epoch}_p.png')
            plt.scatter(w_x_tsne, labels_true,
                        f'{outdir}/wx_tsne_{epoch}_t.png')
            plt.scatter(w_x_tsne, labels_pred,
                        f'{outdir}/wx_tsne_{epoch}_p.png')

            counter = np.array(
                [[label, np.count_nonzero(labels_pred==label)] for label in ylabels])
            plt.bar(counter[:,0], counter[:,1],
                    f'{outdir}/bar_{epoch}.png', reverse=True)

            plt.plot_confusion_matrix(cm, cm_index, cm_columnns,
                                      f'{outdir}/cm_{epoch}.png')

            for i in range(loss_stats.shape[1]):
                loss_label = loss_labels[i]
                yy = loss_stats[:,i]
                plt.plot(yy, f'{outdir}/{loss_label}_{epoch}.png', 'epoch', loss_label)
            plt.plot(nmi_stats, f'{outdir}/nmi_{epoch}.png', ymin=-0.1)
            plt.plot(ari_stats, f'{outdir}/ari_{epoch}.png', ymin=-0.1)

            time_elapse = time.time() - time_start
            print(f"calc time = {time_elapse:.3f} sec")

        # if epoch % save_itvl == 0:
        #     torch.save({
        #     'epoch': epoch,
        #     'loss_stats': loss_stats,
        #     'ari_stats': ari_stats,
        #     'nmi_stats': nmi_stats,
        #     'time_stats': time_stats,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     }, model_path )
