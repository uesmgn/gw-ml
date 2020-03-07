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

        self.labels = args.labels
        self.device = args.device
        self.net = VAENet(input_size, z_dim, y_dim)
        if self.device == "cuda":
            self.net = self.net.cuda()
            torch.backends.cudnn.benchmark=True
        self.net.to(self.device)
        self.losses = LossFunctions()

        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_categ

        self.rec_type = args.rec_type

    def init_model(self, train_loader, test_loader, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.net.optimizer = optimizer


    def unlabeled_loss(self, x, out):
        # obtain network variables
        z, x_ = out['gaussian'], out['x_reconst']
        logits, prob_cat, y = out['logits'], out['prob_cat'], out['categorical']
        y_mu, y_var = out['y_mean'], out['y_var']
        mu, var = out['mean'], out['var']

        loss_rec = self.losses.reconstruction_loss(x, x_, self.rec_type)
        loss_gauss = self.losses.gaussian_loss(z, mu, var, y_mu, y_var)
        loss_cat = -self.losses.entropy(logits, prob_cat) - np.log(0.1)
        loss_total = self.w_rec * loss_rec + self.w_gauss * loss_gauss + self.w_cat * loss_cat
        # EDIT:
        # _, predicted_labels = torch.max(y, dim=1)
        _, predicted_labels = torch.max(logits, dim=1)

        loss_dic = {'total': loss_total,
                    'predicted_labels': predicted_labels,
                    'reconstruction': loss_rec,
                    'gaussian': loss_gauss,
                    'categorical': loss_cat}
        return loss_dic

    def fit_train(self, epoch, temp):
        print(f"\nEpoch: {epoch:d} {datetime.datetime.now()}")
        self.net.train()
        train_loss = 0
        rec_loss = 0
        gauss_loss = 0
        cat_loss = 0
        samples_cnt = 0
        start_t = time.time()

        for batch_idx, (x, labels) in enumerate(self.train_loader):
            # print(f'\rBatch: {batch_idx+1}', end='')
            x = x.to(self.device)
            self.net.optimizer.zero_grad()
            out = self.net(x, temperature=temp)
            loss_dic = self.unlabeled_loss(x, out)
            total = loss_dic['total']
            loss_rec = loss_dic['reconstruction']
            loss_gauss = loss_dic['gaussian']
            loss_cat = loss_dic['categorical']
            total.backward()
            self.net.optimizer.step()

            train_loss += total.item()
            rec_loss += loss_rec.item()
            gauss_loss += loss_gauss.item()
            cat_loss += loss_cat.item()
            samples_cnt += x.size(0)

        elapsed_t = time.time() - start_t
        print(f"\rLoss: {train_loss / samples_cnt:f}")
        print(f"Calc time: {elapsed_t} sec/epoch")
        return {
            'total': train_loss/samples_cnt,
            'reconstruction': rec_loss/samples_cnt,
            'gaussian': gauss_loss/samples_cnt,
            'categorical': cat_loss/samples_cnt
        }

    # Test
    def fit_test(self, epoch, temp, outdir='result', interval=50):
        self.net.eval()
        test_loss = 0
        rec_loss = 0
        gauss_loss = 0
        cat_loss = 0
        samples_cnt = 0
        cm = np.zeros([len(self.labels), self.net.y_dim])
        for batch_idx, (x, labels) in enumerate(self.test_loader):
            x = x.to(self.device)
            out = self.net(x, temperature=temp)
            x_reconst = out['x_reconst']
            loss_dic = self.unlabeled_loss(x, out)
            total = loss_dic['total']
            loss_rec = loss_dic['reconstruction']
            loss_gauss = loss_dic['gaussian']
            loss_cat = loss_dic['categorical']
            # 0 - y_dim
            predicted_labels = loss_dic['predicted_labels'].cpu().numpy()
            for (true, pred) in zip(labels, predicted_labels):
                cm[self.labels.index(true), pred] += 1
            test_loss += total.item()
            rec_loss += loss_rec.item()
            gauss_loss += loss_gauss.item()
            cat_loss += loss_cat.item()
            samples_cnt += x.size(0)
            if batch_idx % interval == 0:
                utils.save_image(
                    torch.cat([x[:8], x_reconst[:8]]),
                    f"{outdir}/VAE_epoch{epoch}_batch{batch_idx+1}.png",
                    nrow=8
                )
                true_preds = list(zip(labels, predicted_labels))
                print(f'True-Preds: {true_preds}')
                loss_rec = loss_dic['reconstruction']
                loss_gauss = loss_dic['gaussian']
                loss_cat = loss_dic['categorical']
                print(f'Loss-reconst: {loss_rec}, Loss-gauss: {loss_gauss}, Loss-categorical: {loss_cat},')
        return {
            'total': test_loss/samples_cnt,
            'reconstruction': rec_loss/samples_cnt,
            'gaussian': gauss_loss/samples_cnt,
            'categorical': cat_loss/samples_cnt,
            'cm': cm
        }
