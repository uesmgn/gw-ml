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
from sklearn.manifold import TSNE

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
    train_set, test_set = old_set.split_dataset(0.7)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False)
    net = VAENet(args.input_size, args.z_dim, args.y_dim)
    vae = VAE(args, net)
    print(vae.net)
    optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-3)
    vae.init_model(train_loader, test_loader, optimizer, enable_scheduler=False)

    losses = {'train': [],
              'test': []}
    epochs = []

    for e in range(args.epochs):

        epoch = e + 1
        epochs.append(epoch)

        temp = max(args.init_temp * np.exp(-args.decay_temp_rate * e), args.min_temp)
        print(f"gumbel temp: {temp:.3f}, epoch: {epoch}")

        start_t = time.time()

        train_out = vae.train(epoch, temp=temp)
        test_out = vae.test(epoch, temp=temp)

        losses['train'].append(train_out['loss_total'])
        losses['test'].append(test_out['loss_total'])

        if epoch % 5 == 0:
            latents = test_out['latents']
            labels = test_out['labels']
            comparison = test_out['comparison']
            cm = test_out['cm']

            latents_2d = TSNE(
                n_components=2, random_state=0).fit_transform(latents)
            labels = np.array([args.labels.index(l) / len(args.labels)
                               for l in labels])

            F.plot_latent(latents_2d, labels,
                            f'{outdir}/latents_{epoch}.png')

            utils.save_image(comparison,
                             f"{outdir}/VAE_epoch{epoch}.png",
                             nrow=12)

            cm_out = f'{outdir}/cm_{epoch}.png'
            cm_title = f'Confusion matrix epoch-{epoch}'
            cm_index = args.labels
            cm_columns = list(range(args.y_dim))
            F.plot_confusion_matrix(cm,
                                    cm_index,
                                    cm_columns,
                                    cm_out,
                                    normalize=True)

        if epoch % 10 == 0:
            if epoch % 100 == 0:
                F.plot_loss(epochs, losses, f"{outdir}/Loss_epoch{epoch}.png")
            else:
                F.plot_loss(epochs, losses,
                            f"{outdir}/Loss_epoch{epoch}.png", type=1)

        elapsed_t = time.time() - start_t
        print(f"Calc time: {elapsed_t:.3f} sec / epoch")
        print('----------')
        # test_out = vae.fit_test(epoch, gumbel_temp, outdir=outdir)
        # log[epoch] = {
        #     'train_loss': train_out['total'],
        #     'test_loss': test_out['total']
        # }
        # if epoch % 10 == 0:
        #     cm_out = f'{outdir}/cm_{epoch}.png'
        #     cm_title = f'Confusion matrix epoch-{epoch}, loss_cat: {test_out["categorical"]}'
        #     cm_index = args.labels
        #     cm_columns = list(range(args.y_dim))
        #     F.plot_confusion_matrix(test_out['cm'],
        #                             cm_index,
        #                             cm_columns,
        #                             cm_out,
        #                             normalize=True)
        #     log_out = f'{outdir}/loss_{epoch}.png'
        #     F.plot_losslog(log, log_out)
