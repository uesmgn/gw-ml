import argparse
import configparser
import json
import datetime
import time
import os
import random
import multiprocessing as mp
from collections import defaultdict
from pprint import  pprint
from multiprocessing import Manager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import pandas as pd
import numpy as np

from gwspy.dataset import Dataset
from net.models import *
from net import criterion
from net.helper import get_middle_dim

from utils.clustering import decomposition, metrics, functional
from utils.plotlib import plot as plt

def main(args, **kwargs):

    seed = kwargs.get('seed') or 123

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    beta = kwargs.get('beta') or (1., 1., 1.)

    time_exec = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    config_ini = 'config/cvae.ini'
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

    batch_size = ini.getint('conf', 'batch_size') or 32
    num_workers = ini.getint('conf', 'num_workers') or 1
    lr = ini.getfloat('conf', 'lr') or 1e-3
    model_path = ini.get('conf', 'model_path')
    dataset_json = ini.get('conf', 'dataset_json')
    x_size = ini.getint('net', 'x_size')
    x_shape = (1, x_size, x_size)
    y_dim = ini.getint('net', 'y_dim')
    z_dim = ini.getint('net', 'z_dim')
    bottle_channel = ini.getint('net', 'bottle_channel')
    poolings = json.loads(ini.get('net', 'poolings'))
    middle_dim = get_middle_dim(x_shape, poolings)
    f_dim = bottle_channel * middle_dim**2

    nkwargs = dict()
    nkwargs['x_shape'] = x_shape
    nkwargs['y_dim'] = y_dim
    nkwargs['z_dim'] = z_dim
    nkwargs['bottle_channel'] = bottle_channel
    nkwargs['channels'] = json.loads(ini.get('net', 'channels'))
    nkwargs['kernels'] = json.loads(ini.get('net', 'kernels'))
    nkwargs['poolings'] = poolings
    nkwargs['hidden_dim'] = ini.getint('net', 'hidden_dim')
    nkwargs['act_func'] = ini.get('net', 'act_func')
    nkwargs['pool_func'] = ini.get('net', 'pool_func')

    if verbose:
        pprint(nkwargs)

    device_ids = list(range(torch.cuda.device_count()))
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print('device_main:', device)
        print('device_ids:', device_ids)

    data_transform = transforms.Compose([
        transforms.CenterCrop((x_size, x_size)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    df = pd.read_json(dataset_json)
    dataset = Dataset(df, transform=data_transform, seed=seed)
    dataset.sample('label', min_value_count=200, n_samples=200)
    # train_set, test_set = dataset.random_split()
    if verbose:
        print(f'length of dataset: ', len(dataset))

    true_labels = dataset.unique_column('label')
    pred_labels = np.array(range(y_dim)).astype(np.int32)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                        drop_last=True)

    pseudo_dict = Manager().dict()
    train_set = Dataset(df, transform=data_transform, pseudo_dict=pseudo_dict)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)

    model = CVAE(**nkwargs)

    if torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)
    # get encoder and classifier
    classifier = nn.Sequential(*list(model.children())[:2])

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    optim_c = torch.optim.Adam(classifier.parameters(), lr=lr)

    if verbose:
        # model.eval()
        # summary(model, x_shape)
        print(model)

    stats = defaultdict(list)

    for epoch in range(1, n_epoch+1):
        if verbose:
            print(f'---------- epoch: {epoch} ----------')

        losses = defaultdict(lambda: 0)

        model.train()
        for b, (x, t, idx) in enumerate(loader):
            x = x.to(device)
            params = model(x)
            loss, rec_loss, z_kl, y_entropy = criterion.cvae_loss(params, beta)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses['features_loss'] += loss.item()
            losses['reconstruntion_loss'] += rec_loss.item()
            losses['z_kl_divergence'] += z_kl.item()
            losses['y_entropy'] += y_entropy.item()
        if verbose:
            print(f'features_loss: {losses["features_loss"]:.3f}')
            print(f'reconstruntion_loss: {losses["reconstruntion_loss"]:.3f}')
            print(f'z_kl_divergence: {losses["z_kl_divergence"]:.3f}')
            print(f'y_entropy: {losses["y_entropy"]:.3f}')

        features, trues, idxs = compute_features(loader, model)

        # assign cluster labels to pseudo_loader
        pseudos = functional.run_kmeans(features, y_dim)
        for i, p in zip(idxs, pseudos):
            pseudo_dict[i] = int(p)

        clustering_weight = []
        for i in range(y_dim):
            w = np.count_nonzero(pseudos == i)
            clustering_weight.append(w)
        clustering_weight = 1. / torch.Tensor(clustering_weight).to(device)

        model.train()
        for b, (x, t, p, idx) in enumerate(train_loader):
            x = x.to(device)
            y_logits, y = model.clustering(x)
            loss = criterion.cross_entropy(y_logits, p.to(device), clustering_weight)
            optim_c.zero_grad()
            loss.backward()
            optim_c.step()
            losses['clustering_loss'] += loss.item()
        if verbose:
            print(f'clustering_loss: {losses["clustering_loss"]:.3f}')

        nmi = metrics.nmi(trues, pseudos)
        if verbose:
            print(f'nmi: {nmi:.3f}')

        # statistics
        stats['features_loss'].append(losses['features_loss'])
        stats['reconstruntion_loss'].append(losses['reconstruntion_loss'])
        stats['z_kl_divergence'].append(losses['z_kl_divergence'])
        stats['y_entropy'].append(losses['y_entropy'])
        stats['clustering_loss'].append(losses['clustering_loss'])
        stats['nmi'].append(nmi)

        if epoch % eval_itvl == 0:
            if verbose:
                print(f'plotting...')

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            if verbose:
                print(f'calculating confusion_matrix...')

            cm, pred_labels, true_labels = metrics.confusion_matrix(
                pseudos, trues, pred_labels, true_labels, return_labels=True)
            figsize_cm = (len(pred_labels) / 1.5, len(true_labels) / 2.0)

            if verbose:
                print(f'plotting confusion_matrix...')

            plt.plot_confusion_matrix(cm, pred_labels, true_labels,
                                      f'{outdir}/cm_{epoch}.png',
                                      figsize=figsize_cm)
            if verbose:
                print(f'plotting bar...')

            plt.bar(pseudos, f'{outdir}/bar_{epoch}.png')

            if verbose:
                print(f'calculating tsne...')

            tsne = decomposition.TSNE(n_components=2)

            features = tsne.fit_transform(features)

            if verbose:
                print(f'plotting tsne...')

            plt.scatter(features[:, 0], features[:, 1], trues,
                        f'{outdir}/features_t_{epoch}.png')
            plt.scatter(features[:, 0], features[:, 1], pseudos,
                        f'{outdir}/features_p_{epoch}.png')

            if verbose:
                print(f'plotting stats...')

            for k, v in stats.items():
                plt.plot(v, f'{outdir}/{k}_{epoch}.png',
                         fit_x=True)



def compute_features(loader, model):
    device = next(model.parameters()).device
    features = torch.Tensor([]).to(device)
    labels = np.array([])
    idxs = np.array([], dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for b, (x, l, idx) in enumerate(loader):
            x = x.to(device)
            z = model.features(x)
            features = torch.cat([features, z], 0)
            labels = np.append(labels, np.ravel(l))
            idxs = np.append(idxs, np.ravel(idx).astype(np.int32))
        features = features.squeeze(1).cpu().numpy()
    return features, labels, idxs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PyTorch Implementation of CVAE Self-supervised Clustering')
    parser.add_argument('-e', '--n_epoch', type=int,
                        help='number of epochs')
    parser.add_argument('-o', '--outdir', type=str,
                        help='output directory')
    parser.add_argument('--eval_itvl', type=int,
                        help='eval interval')
    parser.add_argument('--save_itvl', type=int,
                        help='save interval')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose')
    parser.add_argument('-l', '--load_model', action='store_true',
                        help='load saved model')
    args = parser.parse_args()
    main(args, beta=(1e-3, .5, 1.), seed=123)
