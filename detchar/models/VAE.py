import datetime
import time
import os
from tqdm import tqdm
import torch
from torch import nn
from torchvision import utils, transforms
import numpy as np

from ..networks.Networks import VAENet
from ..losses.LossFunctions import LossFunctions

class VAE:
    def __init__(self, args):
        input_size = args.input_size
        z_dim = args.z_dim
        y_dim = args.y_dim

        self.device = args.device
        self.net = VAENet(input_size, z_dim, y_dim)
        self.losses = LossFunctions()

        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_categ

        self.rec_type = args.rec_type

    def init_model(self, train_loader, test_loader, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer

        if self.device == "cuda":
            self.net = self.net.cuda()
            torch.backends.cudnn.benchmark=True
        self.net.to(self.device)


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
        start_t = time.time()
        for batch_idx, (x, labels) in enumerate(self.train_loader):
            print(f'\rBatch: {batch_idx+1}', end='')
            x = x.to(self.device)
            self.optimizer.zero_grad()
            out = self.net(x)
            loss_dic = self.unlabeled_loss(x, out)
            total = loss_dic['total']
            total.backward()
            self.optimizer.step()
            train_loss += total.item()
            samples_cnt += x.size(0)
        elapsed_t = time.time() - start_t
        print(f"\rLoss: {train_loss / samples_cnt:f}")
        print(f"Calc time: {elapsed_t} sec/epoch")

    # Test
    def fit_test(self, epoch, outdir='result', interval=10):
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.net.eval()
        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(self.test_loader):
                x = x.to(self.device)
                out = self.net(x)
                x_reconst = out['x_reconst']
                loss_dic = self.unlabeled_loss(x, out)
                total = loss_dic['total']
                predicted_labels = loss_dic['predicted_labels']
                if batch_idx % interval == 0:
                    utils.save_image(
                        torch.cat([x[:8], x_reconst[:8]]),
                        f"{outdir}/VAE_epoch{epoch+1}_batch{batch_idx+1}.png",
                        nrow=8
                    )
                    print(f'True: {labels}\nPredict: {predicted_labels}')
