import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from detchar.dataset import Dataset
from detchar.models.VAE import VAE
from detchar.functions.Functions import Functions as F

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of VAE Clustering')

# Architecture
parser.add_argument('-y', '--y_dim', type=int, default=16,
                    help='number of classes (default: 16)')
parser.add_argument('-z', '--z_dim', default=64, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('-i', '--input_size', default=486, type=int,
                    help='input size (default: 486)')
parser.add_argument('-e', '--epochs', default=1000, type=int,
                    help='input size (default: 1000)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    help='batch size (default: 8)')
parser.add_argument('-o', '--outdir', default='result', type=str,
                    help='output directory name (default: result)')

# Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='bce', help='desired reconstruction loss function (default: bce)')

args = parser.parse_args()


if __name__ == '__main__':
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    df = pd.read_json('dataset.json')
    input_size = args.input_size
    data_transform = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Dataset(df, data_transform)
    old_set, new_set = dataset.split_by_labels(['Helix', 'Scratchy'])
    args.labels = old_set.get_labels()
    train_set, test_set = old_set.split_dataset(0.7)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False)

    vae = VAE(args)
    print(vae.net)
    optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-3)
    vae.init_model(train_loader, test_loader, optimizer)
    log = {}

    for e in range(args.epochs):
        epoch = e+1
        init_temp = 5.
        min_temp = 0.5
        decay_temp_rate = 0.16
        # init_temp -> min_temp
        gumbel_temp = np.maximum(init_temp*np.exp(-decay_temp_rate*e), min_temp)

        train_out = vae.fit_train(epoch, gumbel_temp)
        test_out = vae.fit_test(epoch, gumbel_temp, outdir=outdir)
        log[epoch] = {
            'train_loss': train_out['total'],
            'test_loss': test_out['total']
        }
        if epoch % 10 == 0:
            cm_out = f'{outdir}/cm_{epoch}.png'
            cm_title = f'Confusion matrix epoch-{epoch}, loss_cat: {test_out["categorical"]}'
            cm_index = args.labels
            cm_columns = list(range(args.y_dim))
            F.plot_confusion_matrix(test_out['cm'],
                                    cm_index,
                                    cm_columns,
                                    cm_out,
                                    normalize=True)
            log_out = f'{outdir}/loss_{epoch}.png'
            F.plot_losslog(log, log_out)
