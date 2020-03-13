import os
import re
import json
import datetime
import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import utils, transforms, datasets
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from src.models.VAE import VAE
from src.functions.Functions import Functions as F


parser = argparse.ArgumentParser(
    description='PyTorch Implementation of VAE Clustering with MNIST')

parser.add_argument('-cuda', '--n_cuda', type=int, choices=(0, 1),
                    default=0,
                    help='cuda index (default: 0)')
parser.add_argument('-v', '--verbose', type=int, choices=(0, 1),
                    default=1,
                    help='print verbose output (default: 1 => True)')
parser.add_argument('-s', '--sample', type=int, choices=(0, 1),
                    default=0,
                    help='sample data in each labels (default: 0 => False)')
# Architecture
parser.add_argument('-y', '--y_dim', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('-z', '--z_dim', default=3, type=int,
                    help='gaussian size (default: 3)')
parser.add_argument('-e', '--epochs', default=1000, type=int,
                    help='input size (default: 1000)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='batch size (default: 32)')
parser.add_argument('-o', '--outdir', default='result', type=str,
                    help='output directory name (default: result)')
parser.add_argument('-lr', '--init_lr', default=1e-3, type=float,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('-lr_step', '--step_lr', default=100, type=int,
                    help='step size of learning rate to decay \
                    (default: 100)')

# Loss function parameters
parser.add_argument('-it', '--init_temp', default=1., type=float,
                    help='initial value of gumble-temperature \
                    (default: 1.)')
parser.add_argument('-mt', '--min_temp', default=0.5, type=float,
                    help='minimum of gumble-temperature (default: 0.5)')
parser.add_argument('-rt', '--rate_temp', default=3e-3, type=float,
                    help='rate of gumble-temperature to decay \
                    (default: 3e-3)')
parser.add_argument('-wg', '--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('-wc', '--w_cat', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('-wr', '--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str,
                    choices=['bce', 'mse'],
                    default='bce',
                    help='desired reconstruction loss function \
                    (default: bce)')

args = parser.parse_args()


if __name__ == '__main__':
    # get parameters
    if torch.cuda.is_available():
        args.device = f'cuda:{args.n_cuda}'
    else:
        args.device = 'cpu'

    verbose = args.verbose
    outdir = args.outdir
    batch_size = args.batch_size
    epochs = args.epochs

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # Download or load downloaded MNIST dataset
    train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_dataset[0][0].size()
    args.input_size = input_size[1]
    args.labels = list(range(10))

    vae = VAE(args)
    optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-3)
    vae.init_model(train_loader, test_loader, optimizer)

    if verbose:
        print('----------')
        print('Model:')
        print(vae.net)
        print('----------')
        print('Parameters:')
        for k, v in vars(args).items():
            print(f'{k}: {v}')
        print('----------')

    for i in range(epochs):
        epoch = i + 1
        train_out = vae.train(epoch, verbose=verbose)
        test_out = vae.test(epoch, verbose=verbose)

        reconst = test_out['reconst']
        utils.save_image(
            reconst,
            f"{outdir}/reconst_epoch{epoch}.png",
            nrow=8
        )
        latent_features = test_out['latent_features']

        if args.z_dim == 2:
            F.plot_latent2d(latent_features,
                            args.labels,
                            f"{outdir}/latent_epoch{epoch}.png")
        elif args.z_dim == 3:
            F.plot_latent3d(latent_features,
                            args.labels,
                            f"{outdir}/latent_epoch{epoch}.png")
    #
    # log = {}
    #
    # for e in range(epochs):
    #     epoch = e + 1
    #     verbose_plot = (epoch % 2 == 0)
    #
    #     init_temp = args.init_temp
    #     min_temp = args.min_temp
    #     decay_temp_rate = args.rate_temp
    #     # init_temp -> min_temp
    #     gumbel_temp = np.maximum(
    #         init_temp * np.exp(-decay_temp_rate * e), min_temp)
    #
    #     print(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")
    #
    #     train_out = vae.train(epoch, gumbel_temp, verbose=verbose)
    #     test_out = vae.test(epoch, gumbel_temp, verbose=verbose,
    #                         plot=verbose_plot, outdir=outdir)
    #
    #     log[epoch] = {
    #         'reconst': train_out['reconstruction'] * args.w_rec,
    #         'gaussian': train_out['gaussian'] * args.w_gauss,
    #         'categorical': train_out['categorical'] * args.w_cat
    #     }
    #     if verbose_plot:
    #         latent_features = test_out['latent_features']
    #         F.plot_latent(latent_features,
    #                       vae.labels,
    #                       f"{outdir}/Latent_epoch{epoch}.png")
    #         cm_out = f'{outdir}/cm_{epoch}.png'
    #         cm_title = 'Confusion Matrix'
    #         cm_title += f' Epoch: {epoch}'
    #         cm_title += f', Loss-categorical: {test_out["categorical"]:.3f}'
    #         cm_index = vae.labels
    #         cm_columns = list(range(vae.y_dim))
    #         F.plot_confusion_matrix(test_out['cm'],
    #                                 cm_index,
    #                                 cm_columns,
    #                                 cm_out,
    #                                 title=cm_title,
    #                                 normalize=True)
    #         log_out = f'{outdir}/loss_{epoch}.png'
    #         F.plot_losslog(log, log_out)
