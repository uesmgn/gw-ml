import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import numpy as np
import multiprocessing as mp
import pandas as pd

from detchar.dataset import Dataset
from detchar.models.VAE import VAE
from detchar.functions.Functions import Functions
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
parser.add_argument('-n', '--num_workers', default=2, type=int,
                    help='num_workers of DataLoader (default: 2)')
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
                    default=5, help='')

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
    device_ids = range(torch.cuda.device_count())
    args.device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    print(args.device)
    dataset = Dataset(df, data_transform)
    old_set, new_set = dataset.split_by_labels(['Helix', 'Scratchy'], n_cat=200)
    args.labels = np.array(old_set.get_labels()).astype(str)
    args.labels_pred = np.array(range(args.y_dim)).astype(str)
    loader = DataLoader(old_set,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=False)
    model = VAENet(args.input_size, args.z_dim, args.y_dim, activation='ELU')
    vae = VAE(args, model)
    print(vae.net)
    optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=200, gamma=0.5)
    vae.init_model(loader, optimizer, scheduler=scheduler)

    losses = []
    epochs = []
    tlabels = args.labels
    plabels = args.labels_pred
    F = Functions()

    for e in range(args.epochs):

        epoch = e + 1
        epochs.append(epoch)

        temp = max(args.init_temp *
                   np.exp(-args.decay_temp_rate * e), args.min_temp)
        print(f"gumbel temp: {temp:.3f}, epoch: {epoch}")

        start_t = time.time()
        vae_out = vae.fit(epoch, temp=temp)

        losses.append(vae_out['loss_total'])

        if epoch % args.plot_itvl == 0:

            K = len(plabels)
            latents = vae_out['latents']
            latents_2d = []
            trues = vae_out['true']
            preds = vae_out['pred']
            preds_kmeans = []
            cm = pd.DataFrame()
            cm_kmeans = pd.DataFrame()

            with mp.Pool(4) as pool:
                latents_2d = pool.apply_async(F.fit_tsne, (2, latents)).get()
                preds_kmeans = pool.apply_async(
                    F.fit_kmeans, (K, latents)).get()

            with mp.Pool(4) as pool:
                cm = pool.apply_async(F.confution_matrix,
                                      (trues, preds, tlabels, plabels)).get()
                cm_kmeans = pool.apply_async(F.confution_matrix,
                                             (trues, preds_kmeans, tlabels, plabels)).get()

            F.plot_cm(cm, tlabels, plabels, f'{outdir}/cm_{epoch}_vae.png')
            F.plot_cm(cm_kmeans, tlabels, plabels,
                      f'{outdir}/cm_{epoch}_kmeans.png')
            F.plot_latent(latents_2d[:, 0], latents_2d[:, 1],
                          trues, f'{outdir}/latents_{epoch}_true.png')
            F.plot_latent(latents_2d[:, 0], latents_2d[:, 1],
                          preds, f'{outdir}/latents_{epoch}_pred.png')
            F.plot_latent(latents_2d[:, 0], latents_2d[:, 1],
                          preds_kmeans, f'{outdir}/latents_{epoch}_kmeans.png')
            F.plot_loss(losses, f"{outdir}/loss_{epoch}.png")

        elapsed_t = time.time() - start_t
        print(f"Calc time: {elapsed_t:.3f} sec / epoch")
        print('----------')
