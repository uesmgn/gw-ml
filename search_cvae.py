import argparse
import configparser
import json
import datetime
import time
import os
import random
import multiprocessing as mp
from collections import defaultdict
from multiprocessing import Manager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from gwspy.dataset import Dataset
from net import models
from net import criterion

from utils.clustering import decomposition, metrics, functional
from utils.parameter import suggestions as su

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', type=int, default=100,
                    help='num epoch')
parser.add_argument('-b', '--batch_size', type=int, default=100,
                    help='batch size')
parser.add_argument('-n', '--num_workers', type=int, default=1,
                    help='num workers')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='verbose')
args = parser.parse_args()

# random seed
seed = 123

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# setting
n_epoch = args.n_epoch
batch_size = args.batch_size
num_workers = args.num_workers
verbose = args.verbose

device_ids = list(range(torch.cuda.device_count()))
device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

config_ini = 'config/cvae.ini'
assert os.path.exists(config_ini)
basedir = os.path.dirname(os.path.abspath(__file__))
ini = configparser.ConfigParser()
ini.read(f'{basedir}/{config_ini}', 'utf-8')

# dataset init
dataset_json = ini.get('conf', 'dataset_json')
df = pd.read_json(dataset_json)
data_transform = transforms.Compose([
    transforms.CenterCrop((x_size, x_size)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
dataset = Dataset(df, transform=data_transform, seed=seed)
dataset.sample('label', min_value_count=200, n_samples=200)
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

# unique labels
true_unique = dataset.unique_column('label')
pred_unique = np.array(range(y_dim)).astype(np.int32)


def objective(trial):

    # suggest parameters
    features_beta = su.suggest_loguniform_list(trial, low=1e-3, high=1, size=3, 'features_beta')
    clustering_beta = su.suggest_loguniform(trial, low=1e-3, high=1, 'clustering_beta')

    nkwargs = {
        'x_shape': (1, 486, 486),
        'z_dim' = su.suggest_uniform(trial, low=64, high=512, q=64, 'z_dim'),
        'y_dim' = 20,
        'bottle_channel': 32,
        'channels': su.suggest_uniform_list(trial, low=64, high=192, q=32, size=4, 'channel'),
        'kernels': su.suggest_uniform_list(trial, low=3, high=11, q=2, size=4, 'kernel'),
        'poolings': [3, 3, 3, 3],
        'pool_func': 'max',
        'act_func': su.suggest_categorical(trial, ['ReLU', 'Tanh', 'ELU'], 'act_func')
    }

    model = models.CVAE(**nkwargs)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    model.to(device)

    # get encoder and classifier
    classifier = nn.Sequential(*list(model.children())[:2])

    optim = su.suggest_optimizer(trial, model, 'optim')
    optim_c = su.suggest_optimizer(trial, model, 'optim_c')

    for epoch in range(1, n_epoch+1):
        if verbose:
            print(f'epoch: {epoch}')

        model.train()
        for b, (x, _, _) in enumerate(loader):
            x = x.to(device)
            params = model(x)
            loss, _, _, _ = criterion.cvae_loss(params, features_beta)
            optim.zero_grad()
            loss.backward()
            optim.step()

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
            loss = criterion.cross_entropy(y_logits, p.to(device),
                                           clustering_weight, clustering_beta)
            optim_c.zero_grad()
            loss.backward()
            optim_c.step()

        nmi = metrics.nmi(trues, pseudos)

    return nmi


if __name__ == '__main__':
    trial_size = 10000
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trial_size)
    print(study.best_params)
