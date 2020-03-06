import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from matplotlib import ticker

from detchar.dataset import Dataset
from detchar.models.VAE import VAE

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of VAE Clustering')

# Architecture
parser.add_argument('-y', '--y_dim', type=int, default=16,
                    help='number of classes (default: 16)')
parser.add_argument('-z', '--z_dim', default=64, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('-i', '--input_size', default=480, type=int,
                    help='input size (default: 480)')
parser.add_argument('-e', '--epochs', default=1000, type=int,
                    help='input size (default: 1000)')
parser.add_argument('-b', '--batch_size', default=8, type=int,
                    help='batch size (default: 8)')

# Loss function parameters
parser.add_argument('--w_gauss', default=1, type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1, type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='mse', help='desired reconstruction loss function (default: mse)')

args = parser.parse_args()


if __name__ == '__main__':

    df = pd.read_json('dataset.json')
    input_size = args.input_size
    data_transform = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    dataset = Dataset(df, data_transform)
    old_set, new_set = dataset.split_by_labels(['Helix', 'Scratchy'])
    train_set, test_set = old_set.split_dataset(0.7)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False)

    vae = VAE(args)
    optimizer = torch.optim.Adam(vae.net.parameters(), lr=1e-3)
    vae.init_model(train_loader, test_loader, optimizer)
    log = {}

    for epoch in range(args.epochs):
        train_loss = vae.fit_train(epoch + 1)
        test_loss = vae.fit_test(epoch + 1)
        log[epoch] = {
            'train_loss': train_loss,
            'test_loss': test_loss
        }
    xx = [epoch + 1 for epoch in log.keys()]
    yy_train = [loss.get('train_loss') for loss in log.values()]
    yy_test = [loss.get('test_loss') for loss in log.values()]

    plt.figure(figsize=[8, 4])
    plt.plot(xx, yy_train, color='red')
    plt.plot(xx, yy_test, color='blue')

    plt.xlabel('epoch')
    plt.xlim([min(xx), max(xx)])
    plt.ylabel('loss')
    plt.ylim([min(yy_train + yy_test), max(yy_train + yy_test)])
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    plt.tight_layout()
    plt.savefig(f'result/loss.png')
