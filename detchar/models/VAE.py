import datetime
import time
from tqdm import tqdm
import torch
from torch import nn
from torchvision import transforms
import numpy as np

from ..networks.Networks import VAENet
from ..losses.LossFunctions import LossFunctions

class VAE:
    def __init__(self, device, input_size, z_dim, y_dim):
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.net = VAENet(input_size, z_dim, y_dim)
        self.losses = LossFunctions()
        if torch.cuda.is_available():
            self.net = self.net.cuda()

        self.w_rec = 0.5
        self.w_gauss = 0.2
        self.w_cat = 0.3
        self.rec_type = 'mse'

    def init_model(self, train_loader, optimizer):
        self.train_loader = train_loader
        self.optimizer = optimizer

    def unlabeled_loss(self, x, out):
        # obtain network variables
        z, x_ = out['gaussian'], out['x_reconst']
        logits, prob_cat = out['logits'], out['prob_cat']
        y_mu, y_var = out['y_mean'], out['y_var']
        mu, var = out['mean'], out['var']

        loss_rec = self.losses.reconstruction_loss(x, x_, self.rec_type)
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {'total': loss_total,
                    'predicted_labels': predicted_labels,
                    'reconstruction': loss_rec,
                    'gaussian': loss_gauss,
                    'categorical': loss_cat}
        return loss_dic

    def fit_train(self, epoch):
        print(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")
        self.net.train()
        train_loss = 0
        samples_cnt = 0
        for batch_idx, (x, labels) in enumerate(self.train_loader):
            print(f'\rBatch: {batch_idx+1}', end='')
            x = x.to(self.device)
            self.optimizer.zero_grad()
            out = self.net(x)
            loss_dic = self.unlabeled_loss(x, out)
            print(labels)
            print(loss_dic['predicted_labels'])
            total = loss_dic['total']
            total.backward()
            self.optimizer.step()
            train_loss += total.item()
            samples_cnt += x.size(0)
        elapsed_t = time.time() - start_t
        print(f"\rLoss: {train_loss / samples_cnt:f}")
        print(f"Calc time: {elapsed_t} sec/epoch")
