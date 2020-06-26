import argparse
import configparser
import json
import datetime
import time
import os
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

if __name__ == '__main__':

    beta = (1., .9, 0.01)

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
    dataset = Dataset(df, transform=data_transform)
    dataset.sample('label', min_value_count=200)
    # train_set, test_set = dataset.random_split()
    if verbose:
        print(f'length of dataset: ', len(dataset))

    xlabels = dataset.unique_column('label')
    ylabels = np.array(range(y_dim))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                        drop_last=True)

    pseudo_dict = Manager().dict()
    pseudo_dataset = Dataset(df, transform=data_transform, pseudo_dict=pseudo_dict)
    pseudo_loader = DataLoader(pseudo_dataset,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=True,
                               drop_last=True)

    model = CVAE(**nkwargs)

    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    model.to(device)

    classifier = nn.Sequential(
        nn.Linear(f_dim, 256),
        nn.Linear(256, y_dim)
    )
    classifier.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_pseudo = torch.optim.Adam(classifier.parameters(), lr=lr)

    if verbose:
        # model.eval()
        # summary(model, x_shape)
        print(model)

    init_epoch = 0
    loss_stats = []
    nmi_stats = []
    ari_stats = []
    time_stats = []

    for epoch in range(1, n_epoch+1):
        trues = np.array([])
        idxs = np.array([], dtype=np.int32)
        z = torch.Tensor([]).to(device)

        model.eval()
        with torch.no_grad():
            for b, (x, l, idx) in enumerate(loader):
                print(b)
                x = x.to(device)
                params = model(x)
                idxs = np.append(idxs, np.ravel(idx))
                z = torch.cat([z, params['z']], 0)
                trues = np.append(trues, np.ravel(l))

        features = z.squeeze(1).cpu().numpy()
        pseudos = functional.run_kmeans(features, y_dim)
        for i, p in zip(idxs, pseudos):
            pseudo_dict[i] = int(p)

        clustering_weight = []
        for i in range(y_dim):
            w = np.count_nonzero(pseudos == i)
            clustering_weight.append(w)
        clustering_weight = 1. / torch.Tensor(clustering_weight).to(device)

        model.train()
        losses = defaultdict(lambda: 0)
        for b, (x, p, idx) in enumerate(pseudo_loader):
            print(b)
            x = x.to(device)
            params = model(x)
            params['pseudos'] = p.to(device)
            params['logits'] = classifier(params['f'].detach())
            features_loss, clustering_loss = criterion.cvae(params, beta, clustering_weight)
            optimizer.zero_grad()
            optimizer_pseudo.zero_grad()
            features_loss.backward()
            clustering_loss.backward()
            optimizer.step()
            optimizer_pseudo.step()
            losses['features_loss'] += features_loss.item()
            losses['clustering_loss'] += clustering_loss.item()
        loss_stats.append(losses.values())
        nmi = metrics.nmi(trues, pseudos)
        nmi_stats.append(nmi)
        print(epoch, losses.values(), nmi)

        if epoch % eval_itvl == 0:

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            z = z.squeeze(1).detach().cpu().numpy()

            cm, labels_pred, labels_true = metrics.confusion_matrix(pseudos, trues, ylabels, xlabels, return_labels=True)

            counter = np.array([[label, np.count_nonzero(
                np.array(pseudos) == label)] for label in list(set(pseudos))])
            x_position = np.arange(len(counter[:, 0]))
            plt.bar(counter[:, 0], counter[:, 1], f'{outdir}/bar_{epoch}.png')

            plt.scatter(z[:, 0], z[:, 1], trues,
                        f'{outdir}/latent_t_{epoch}.png')
            plt.scatter(z[:, 0], z[:, 1], pseudos,
                        f'{outdir}/latent_p_{epoch}.png')

            plt.plot(nmi_stats, f'{outdir}/nmi_{epoch}.png')

            plt.plot_confusion_matrix(cm, labels_pred, labels_true,
                                      f'{outdir}/cm_{epoch}.png')
