import os
import sys
import datetime
import tarfile
import glob
import urllib.request as urlrequest
import warnings
import copy
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from matplotlib.ticker import MultipleLocator

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, utils

from skimage import io, color, transform
from PIL import Image

# torchvision, pandas
class GWSpyDataset(Dataset):
    '''
    >>> from test03 import GWSpyDataset, Encoder
    >>> from torch.utils.data import Dataset, DataLoader
    >>> from torchvision import transforms
    >>> data_transform = transforms.Compose([transforms.CenterCrop(512), transforms.Grayscale(), transforms.ToTensor()])
    >>> gwspy_set = GWSpyDataset(csv_file="test/dataset_ok.csv", root_dir="TrainingSet", transform=data_transform)
    >>> dumy_set = GWSpyDataset(csv_file="test/dataset_fail.csv", root_dir="TrainingSet", transform=data_transform)
    Traceback (most recent call last):
        ...
    AssertionError: expect file has columns: ['class', 'path'], but input file has columns: ['name', 'path']
    >>> encoder = Encoder()
    >>> train_loader = DataLoader(gwspy_set, batch_size=32, shuffle=True)
    >>> for img, category in train_loader:
    >>>     out = encoder(img)
    >>>     break
    '''

    def __init__(self, csv_file: str, root_dir: str, transform: torchvision.transforms.transforms.Compose=None):
        '''
        Parameters
        ----------
        csv_file  : str
            path to .csv file which has column ['class', 'path'], 'class' is name of category, 'path' is path to image
        root_dir  : str
            path to root directory of dataset, dataset directory has tree [root_dir]-[class]-[path]
        transform : torchvision.transforms.transforms.Compose
            transform function for image preprocessing
        '''
        self._columns = ['class', 'path']
        if os.path.exists(csv_file):
            try:
                self.df = pd.read_csv(csv_file)
                assert self._columns[0] in self.df.columns and self._columns[1] in self.df.columns, 'expect file has columns: {}, but input file has columns: {}'.format(self._columns, list(self.df.columns))
            except AssertionError as err:
                raise err
        else:
            raise FileNotFoundError(f'{csv_file} does not exists.')
        self.root_dir = root_dir
        self.transform = transform

    def get_all_categiries(self):
        return self.df[self._columns[0]].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx, :]
        img_name = f'{self.root_dir}/{item[self._columns[0]]}/{item[self._columns[1]]}'
        img = Image.open(img_name)
        if self.transform:
            img = self.transform(img)
        return (img, item[self._columns[0]])


class ConvModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    '''

    '''
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = nn.Flatten(x)
        print(x.size())
        return x


class split_gwspy(object):
    def __call__(self, dset: GWSpyDataset):

        min_cats_size = 200
        newly_cats = ['Helix', 'Scratchy']

        dset_ = copy.copy(dset)
        df = dset.df[dset.df['path'].str.contains('1.0', regex=False)]
        older_df = df[~df['class'].isin(newly_cats)]
        all_cats = older_df['class'].unique()
        sizeof_cats = [len(older_df[older_df['class'] == c]) for c in all_cats]
        cats_sizes = dict(zip(all_cats, sizeof_cats))
        repr_cats = [k for k, v in cats_sizes.items() if v >= min_cats_size]

        older_cdf = pd.DataFrame(columns=['class', 'path'])
        for c in repr_cats:
            older_cdf = older_cdf.append(
                                # older_df[older_df['class'] == c].sample(n=min_cats_size, random_state=1)
                                older_df[older_df['class'] == c]
                            )
        print(f"# of old classes: {len(repr_cats)}")
        print(f"{repr_cats}")
        print(f"# of old data: {len(older_cdf)}")

        newly_df = df[df['class'].isin(newly_cats)]
        newly_cdf = pd.DataFrame(columns=['class', 'path'])
        for c in newly_cats:
            newly_cdf = newly_cdf.append(
                                # newly_df[newly_df['class'] == c].sample(n=min_cats_size, random_state=1)
                                newly_df[newly_df['class'] == c]
                            )

        print(f"# of new classes: {len(newly_cats)}")
        print(f"{newly_cats}")
        print(f"# of new data: {len(newly_cdf)}")

        return dset.set_df(older_cdf), dset_.set_df(newly_cdf)

def split_dataset(dataset: GWSpyDataset, a=0.2):
    N_test = int(len(dataset)*a)
    N_train = len(dataset) - N_test
    return random_split(dataset, [N_train, N_test])

def save_plot(arr, out):
    img = arr.permute(2, 3, 0, 1)[:,:,0,0]
    fig = plt.figure(figsize=(1, 1), dpi=512)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.imshow(img, cmap=plt.cm.gray)
    fig.add_axes(ax)
    fig.savefig(out, dpi=512)
    plt.close()


class VAE(nn.Module):
    def __init__(self, name, input_size, color_channels, n_latent_features, pooling_kernels, conv_kernels, channels):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        super().__init__()

        # Encoder
        self.encoder = Encoder()

        if not os.path.exists(name):
            os.mkdir(name)

        self.model_name = name

    def init_model(self, optimizer, train_loader, test_loader):
        self.train_loader, self.test_loader = train_loader, test_loader
        self.optimizer = optimizer


        if self.device == "cuda":
            self = self.cuda()
            torch.backends.cudnn.benchmark=True

        self.to(self.device)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        # Encoder
        h = self.encoder(x)

        return h

    def loss_function(self, recon_x, x, mu, logvar):
        # https://arxiv.org/abs/1312.6114 (Appendix B)
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    # Train
    def fit_train(self, epoch, log):
        self.train()
        print(f"\nEpoch: {epoch + 1:d} {datetime.datetime.now()}")
        # sys.stdout.write(f"\nEpoch: {epoch + 1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        start_t = time.time()
        for batch_idx, (inputs, cat) in enumerate(self.train_loader):
            img_in = inputs.to(self.device)
            self.optimizer.zero_grad()
            img_out, z, mu, logvar = self(img_in)
            loss = self.loss_function(img_out, img_in, mu, logvar)
            loss.backward()
            self.optimizer.step()
            # print(f"Train: {epoch+1}-{batch_idx+1}")

            train_loss += loss.item()
            samples_cnt += img_in.size(0)
        elapsed_t = time.time() - start_t
        print(f"Loss: {train_loss / samples_cnt:f}")
        print(f"Calc time: {elapsed_t} sec/epoch")
        # sys.stdout.write(f"epoch: {epoch}, batch_idx: {batch_idx}, Loss: {train_loss / samples_cnt:f}")
        log.append([epoch, train_loss/samples_cnt])
        return log

    # Test
    def test(self, epoch, loader=None, out=None):
        test_loader = loader if loader is not None else self.test_loader
        test_out = out if out is not None else "reconst"
        self.eval()
        with torch.no_grad():
            features = torch.Tensor().to(self.device)
            cats = []
            losses = []
            for batch_idx, (inputs, cat) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                reconsts, z, mu, logvar = self(inputs)
                loss = self.loss_function(inputs, reconsts, mu, logvar)
                features = torch.cat([features, z], 0)
                cats += list(cat)
                utils.save_image(torch.cat([inputs[:4], reconsts[:4]]), f"{self.model_name}/{test_out}_epoch{epoch+1}_batch{batch_idx+1}.png", nrow=4)
            return features, np.array(cats)

TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 32

def main():
    test_name = "test_200210-Inception-GE200"
    input_size = 480
    log_file = test_name + '/config.txt'

    net = VAE(test_name,
              input_size,
              color_channels,
              n_latent_features,
              pooling_kernels,
              conv_kernels,
              channels)

    with open(log_file, 'w') as f:
        f.write(
            f"""
{test_name}, {datetime.datetime.now()}
----------------------------------------
[configure]
input: ({color_channels}, {input_size}, {input_size})
n_latent_features: {n_latent_features}
pooling_kernels: {pooling_kernels}
conv_kernels: {conv_kernels}
channels: {channels}
comment:
The numbers of data for each categories are greater than or equals to 200
----------------------------------------
[network]
{net}
            """
            )

    data_transform = transforms.Compose([
                         transforms.CenterCrop(input_size),
                         transforms.Grayscale(),
                         transforms.ToTensor()
                     ])

    gwspy_set = GWSpyDataset(csv_file="gwspy_set.csv", root_dir="TrainingSet", transform=data_transform)

    dataset_split = split_gwspy()

    older_set, newly_set = dataset_split(gwspy_set)

    train_set, test_set = split_dataset(older_set, a=0.2)

    old_classes = np.array(older_set.get_cats())
    new_classes = np.array(newly_set.get_cats())

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
    new_loader = DataLoader(newly_set, batch_size=TEST_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    net.init_model(optimizer, train_loader, test_loader)
    log = []
    for i in range(2000):
        log = net.fit_train(i, log)
        features, cats = net.test(i)

        if i > 0 and i % 10 == 9:

            cats_idx = np.array([list(old_classes).index(cat) for cat in cats])
            torch.save(features, f"{net.model_name}/features.tensor")

            features = features.cpu().numpy()
            c_df = pd.DataFrame({'class':cats_idx})
            f_df = pd.DataFrame(features, columns=[f"f{i+1}" for i in range(features.shape[1])])
            df = pd.concat([c_df, f_df], axis=1)
            df.to_csv(f"{net.model_name}/features_vecs_{i+1}.csv")

            new_features, new_cats = net.test(i, loader=new_loader, out="new_reconst")
            ncats_idx = np.array([list(new_classes).index(cat) for cat in new_cats])
            new_features = new_features.cpu().numpy()
            nc_df = pd.DataFrame({'class':ncats_idx})
            nf_df = pd.DataFrame(new_features, columns=[f"f{i+1}" for i in range(new_features.shape[1])])
            ndf = pd.concat([nc_df, nf_df], axis=1)
            ndf.to_csv(f"{net.model_name}/new_features_vecs_{i+1}.csv")

            try:
                nplog = np.array(log).T
                plt.figure(figsize=[8,4])
                plt.plot(nplog[0], nplog[1])
                plt.xlabel('epoch')
                plt.xlim([min(nplog[0]), max(nplog[0])])
                plt.ylabel('loss')
                plt.ylim([min(nplog[1]), np.median(nplog[1])])
                ax = plt.gca()
                ax.xaxis.set_major_locator(ticker.MaxNLocator(5, integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
                plt.tight_layout()
                plt.savefig(f'{net.model_name}/loss_epoch{i+1}.png')
                plt.close()
                torch.save(net, f"{test_name}/GWSPY.model")
            except:
                print(f'Save graph failed at epoch {i+1}')
                continue

if __name__ == '__main__':
    import doctest
    doctest.testmod()
