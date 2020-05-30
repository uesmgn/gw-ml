import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import argparse

from gmvae.dataset import Dataset
from gmvae.network import GMVAE
from gmvae.loss import *

parser = argparse.ArgumentParser(
    description='PyTorch Implementation of GMVAE Clustering')

# NN Architecture
parser.add_argument('-y', '--y_dim', type=int, default=10,
                    help='number of classes (default: 10)')
parser.add_argument('-z', '--z_dim', default=512, type=int,
                    help='gaussian size (default: 512)')
parser.add_argument('-e', '--n_epoch', default=1000, type=int,
                    help='number of epochs (default: 1000)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    help='batch size (default: 32)')
parser.add_argument('-n', '--num_workers', default=4, type=int,
                    help='num_workers of DataLoader (default: 4)')
args = parser.parse_args()

if __name__ == '__main__':
    # test params
    x_shape = (1, 486, 486)
    y_dim = args.y_dim
    z_dim = args.z_dim
    w_dim = args.w_dim

    device_ids = range(torch.cuda.device_count())
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    num_workers = args.num_workers

    df = pd.read_json('dataset.json')
    data_transform = transforms.Compose([
        transforms.CenterCrop((x_shape[1], x_shape[2])),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    dataset = Dataset(df, data_transform)
    labels = np.array(dataset.get_labels()).astype(str)
    labels_pred = np.array(range(y_dim)).astype(str)
    model = GMVAE(x_shape, y_dim, z_dim, w_dim)
    model.to(self.device)
    # GPU Parallelize
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)
    n_samples = 0
    loss_total = 0
    for epoch_idx in range(n_epoch):
        print(epoch_idx)
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            x_z = output['x_z']
            loss = reconstruction_loss(x, x_z)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
            n_samples += x.size(0)
        loss_total /= n_samples
        print(f'loss = {loss_total:.3f} at epoch {epoch_idx+1}')
