import datetime
import time
import os
from tqdm import tqdm
import torch
from torch import nn
from torchvision import utils, transforms
import numpy as np
from collections import defaultdict

from ..networks.Networks import VAENet
from ..losses.LossFunctions import LossFunctions


class VAE:
    def __init__(self, args):
        self.input_size = args.input_size
        self.z_dim = args.z_dim
        self.y_dim = args.y_dim
        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_cat
        self.rec_type = args.rec_type
        self.labels = args.labels
        self.device = args.device
        self.init_lr = args.init_lr
        self.step_lr = args.step_lr

        self.net = VAENet(self.input_size,
                          self.z_dim,
                          self.y_dim)

        if torch.cuda.is_available():
            self.net = self.net.cuda()
            torch.backends.cudnn.benchmark = True
        self.net.to(self.device)
        self.losses = LossFunctions()
        self.__initialized = False

    def init_model(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.init_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_lr, gamma=0.5)
        self.__initialized = True

    def unlabeled_loss(self, x, out):
        # obtain network variables
        z, x_ = out['gaussian'], out['x_reconst']
        logits, prob_cat = out['logits'], out['prob_cat']
        y, y_mu, y_var = out['categorical'], out['y_mean'], out['y_var']
        mu, var = out['mean'], out['var']

        loss_rec = self.losses.reconstruction_loss(x, x_, self.rec_type)
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)
        # loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)
        loss_cat = -self.losses.entropy(logits, prob_cat)
        # loss_cat = -self.losses.entropy(logits, y)
        loss_total = self.w_rec * loss_rec + \
            self.w_gauss * loss_gauss + self.w_cat * loss_cat

        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {'total': loss_total,
                    'predicted_labels': predicted_labels,
                    'reconstruction': loss_rec,
                    'gaussian': loss_gauss,
                    'categorical': loss_cat}

        return loss_dic

    def train(self, epoch, temp, verbose=1):
        assert self.__initialized

        self.net.train()

        loss = defaultdict(lambda: 0)
        n_samples = 0

        start_t = time.time()

        for b, (x, labels) in enumerate(self.train_loader):
            batch = b + 1

            x = x.to(self.device)
            self.optimizer.zero_grad()

            out = self.net(x, temperature=temp)

            loss_dic = self.unlabeled_loss(x, out)
            total = loss_dic['total']
            loss_rec = loss_dic['reconstruction']
            loss_gauss = loss_dic['gaussian']
            loss_cat = loss_dic['categorical']

            total.backward()
            self.optimizer.step()

            loss['total'] += total.item()
            loss['reconstruction'] += loss_rec.item()
            loss['gaussian'] += loss_gauss.item()
            loss['categorical'] += loss_cat.item()
            n_samples += x.size(0)

        elapsed_t = time.time() - start_t
        lr = self.scheduler.get_lr()[0]
        self.scheduler.step()

        for key in loss.keys():
            loss[key] /= n_samples

        if verbose:
            print(f"Calc time: {elapsed_t:.3f} sec/epoch")
            print(f"Learning rate: {lr:.6f}")
            loss_info = ", ".join(
                [f'Loss-{k}: {v:.3f}' for k, v in loss.items()])
            print(f'Train {loss_info}')

        out = loss
        # additional output key-value ↓

        return out

    # Test
    def test(self, epoch, temp, verbose=1, plot=0,
             outdir='result', plot_itvl=50):
        assert self.__initialized

        self.net.eval()

        loss = defaultdict(lambda: 0)
        n_samples = 0
        cm = np.zeros([len(self.labels), self.net.y_dim])

        latent_features = {}

        with torch.no_grad():
            for b, (x, labels) in enumerate(self.test_loader):
                batch = b + 1
                x = x.to(self.device)

                out = self.net(x, temperature=temp)
                x_reconst = out['x_reconst']

                loss_dic = self.unlabeled_loss(x, out)
                total = loss_dic['total']
                loss_rec = loss_dic['reconstruction']
                loss_gauss = loss_dic['gaussian']
                loss_cat = loss_dic['categorical']

                for i, _ in enumerate(out['gaussian'][:, 0]):

                    latent_features[f'{epoch}-{batch}'] = {
                        'x': out['gaussian'][i, 0],
                        'y': out['gaussian'][i, 1],
                        'label': labels[i]
                    }

                predicted_labels = loss_dic['predicted_labels'].cpu().numpy()
                for (true, pred) in zip(labels, predicted_labels):
                    cm[self.labels.index(true), pred] += 1

                loss['total'] += total.item()
                loss['reconstruction'] += loss_rec.item()
                loss['gaussian'] += loss_gauss.item()
                loss['categorical'] += loss_cat.item()
                n_samples += x.size(0)

                if plot and (batch % plot_itvl == 0):
                    utils.save_image(
                        torch.cat([x[:8], x_reconst[:8]]),
                        f"{outdir}/VAE_epoch{epoch}_batch{batch}.png",
                        nrow=8
                    )
                    if verbose:
                        print('----------')
                        print(f'epoch: {epoch}, batch: {batch}')
                        true_preds = list(zip(labels, predicted_labels))
                        print(f'True-Preds: {true_preds}')
                        print('----------')

            for key in loss.keys():
                loss[key] /= n_samples

            if verbose:
                loss_info = ", ".join(
                    [f'Loss-{k}: {v:.3f}' for k, v in loss.items()])
                print(f'Test {loss_info}')

            out = loss
            # additional output key-value ↓
            out['cm'] = cm
            out['latent_features'] = latent_features

            return out
