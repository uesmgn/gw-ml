import os
import torch
import time
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import utils, transforms
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from multiprocessing import Pool

from detchar.dataset import Dataset
from detchar.models.VAE import VAE
from detchar.functions.Functions import Functions as F
from detchar.networks.Networks import VAENet

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of VAE Clustering')

# Architecture
parser.add_argument('-y', '--y_dim', type=int, default=16,
                    help='number of classes (default: 16)')
parser.add_argument('-z', '--z_dim', default=512, type=int,
                    help='gaussian size (default: 512)')
parser.add_argument('-i', '--input_size', default=486, type=int,
                    help='input size (default: 486)')
parser.add_argument('-e', '--epochs', default=5000, type=int,
                    help='number of epochs (default: 5000)')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    help='batch size (default: 4)')
parser.add_argument('-o', '--outdir', default='result', type=str,
                    help='output directory name (default: result)')
parser.add_argument('-cuda', '--cuda', default=0, type=int,
                    help='cuda index')
# Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_cat', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='mse', help='desired reconstruction loss function (default: mse)')

parser.add_argument('--init_temp', type=float,
                    default=1.0, help='')
parser.add_argument('--decay_temp_rate', type=float,
                    default=0.013862944, help='')
parser.add_argument('--min_temp', type=float,
                    default=0.5, help='')
parser.add_argument('--plot_itvl', type=int,
                    default=1, help='')

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
    args.device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print(args.device)
    dataset = Dataset(df, data_transform)
    old_set, new_set = dataset.split_by_labels(['Helix', 'Scratchy'])
    args.labels = old_set.get_labels()
    args.labels_pred = list(range(args.y_dim))
    loader = DataLoader(old_set,
                        batch_size=args.batch_size,
                        shuffle=True)
    net = VAENet(args.input_size, args.z_dim, args.y_dim)
    vae = VAE(args, net)
    print(vae.net)
    optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=200, gamma=0.5)
    vae.init_model(loader, optimizer, scheduler=scheduler)

    losses = []
    epochs = []

    for e in range(args.epochs):

        epoch = e + 1
        epochs.append(epoch)

        temp = max(args.init_temp * np.exp(-args.decay_temp_rate * e), args.min_temp)
        print(f"gumbel temp: {temp:.3f}, epoch: {epoch}")

        start_t = time.time()
        vae_out = vae.fit(epoch, temp=temp)

        losses.append(vae_out['loss_total'])

        if epoch % args.plot_itvl == 0:
            F.plot_result(epoch,
                          args.labels,
                          args.labels_pred,
                          vae_out['latents'],
                          vae_out['true'],
                          vae_out['pred'],
                          args.outdir)
            F.plot_loss(losses,
                        f"{args.outdir}/loss_{epoch}.png")

        elapsed_t = time.time() - start_t
        print(f"Calc time: {elapsed_t:.3f} sec / epoch")
        print('----------')
