import datetime
import time
import os
from tqdm import tqdm
import torch
from torch import nn
from torchvision import utils, transforms
import numpy as np
from collections import defaultdict

from ..losses.LossFunctions import LossFunctions


class VAE:
    def __init__(self, args, net):
        self.input_size = args.input_size
        self.z_dim = args.z_dim
        self.y_dim = args.y_dim
        self.rec_type = args.rec_type
        self.labels = args.labels
        self.device = args.device
        self.net = net
        self.losses = LossFunctions()

        if torch.cuda.is_available():
            self.net.to(self.device)
            torch.backends.cudnn.benchmark = True

        self.__initialized = False

    def init_model(self, loader,
                   optimizer, scheduler=None):
        self.loader = loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.enable_scheduler = scheduler is not None
        self.__initialized = True

    def unlabeled_loss(self, x, out):
        # obtain network variables
        x_reconst = out['x_reconst']
        z, z_mu, z_logvar = out['z'], out['z_mu'], out['z_logvar']
        y_logits, y_prob, y = out['y_logits'], out['y_prob'], out['y']
        y_mu, y_logvar = out['y_mu'], out['y_logvar']
        z_var = z_logvar.exp()
        y_var = y_logvar.exp()

        loss_reconst = self.losses.reconstruction_loss(
            x, x_reconst, self.rec_type)
        # loss_gaussian = self.losses.gaussian_kl_loss(z, z_mu, z_logvar)
        loss_gaussian = self.losses.gaussian_loss(
            z, z_mu, z_var, y_mu, y_var)
        # loss_categorical = 100 * self.losses.categorical_kl_loss(y_prob)
        loss_categorical = -self.losses.entropy(y_logits, y_prob) - np.log(0.1)
        loss_total = loss_reconst + loss_gaussian + loss_categorical

        return {'total': loss_total,
                'reconst': loss_reconst,
                'gaussian': loss_gaussian,
                'categorical': loss_categorical}

    def fit(self, epoch, temp=1.0):
        assert self.__initialized

        self.net.train()

        loss = defaultdict(lambda: 0)
        n_samples = 0
        latents = torch.Tensor().to(self.device)
        latent_labels = []
        cm = np.zeros([len(self.labels), self.y_dim])

        for batch_idx, (x, labels) in enumerate(self.loader):
            batch = batch_idx + 1
            x = x.to(self.device)
            self.optimizer.zero_grad()
            net_out = self.net(x, temp=temp)

            z = net_out['z']
            x_reconst = net_out['x_reconst']
            y = net_out['y']
            _, preds = torch.max(y, dim=1)
            preds = preds.cpu().numpy()
            unlabeled_loss = self.unlabeled_loss(x, net_out)

            total = unlabeled_loss['total']
            loss_reconst = unlabeled_loss['reconst']
            loss_gaussian = unlabeled_loss['gaussian']
            loss_categorical = unlabeled_loss['categorical']
            total.backward()
            self.optimizer.step()
            loss['loss_total'] += total.item()
            loss['loss_reconst'] += loss_reconst.item()
            loss['loss_gaussian'] += loss_gaussian.item()
            loss['loss_categorical'] += loss_categorical.item()
            n_samples += x.size(0)

            for (true, pred) in zip(labels, preds):
                cm[self.labels.index(true), pred] += 1
            latents = torch.cat(
                [latents, z], dim=0)

            latent_labels += [[true, pred] for true, pred in zip(list(labels), preds)]

        if self.enable_scheduler:
            self.scheduler.step()

        for k in loss.keys():
            loss[k] /= n_samples

        print(", ".join([f'{k}: {v:.3f}' for k, v in loss.items()]))

        out = dict(loss)
        out['latents'] = latents.cpu().detach().numpy()
        out['true'], out['pred'] = np.array(latent_labels).T
        out['cm'] = cm
        return out
    #
    # # Test
    # def test(self, epoch, temp=1.0):
    #     assert self.__initialized
    #
    #     self.net.eval()
    #
    #     loss = defaultdict(lambda: 0)
    #     n_samples = 0
    #     latents = torch.Tensor()
    #     latents = latents.to(self.device)
    #     latent_labels = []
    #
    #     originals = torch.Tensor().to(self.device)
    #     reconsts = torch.Tensor().to(self.device)
    #     flags = np.zeros(len(self.labels))
    #     cm = np.zeros([len(self.labels), self.y_dim])
    #     predicted_labels = np.array([])
    #
    #     with torch.no_grad():
    #         for batch_idx, (x, labels) in enumerate(self.test_loader):
    #             batch = batch_idx + 1
    #             x = x.to(self.device)
    #             out = self.net(x, temp=temp, reparameterize=False)
    #             z = out['z']
    #             latents = torch.cat([latents, z], 0)
    #             latent_labels += labels
    #             x_reconst = out['x_reconst']
    #
    #             if not flags.all():
    #                 for i, label in enumerate(labels):
    #                     label_idx = self.labels.index(label)
    #                     if not flags[label_idx]:
    #                         originals = torch.cat([originals, x[i:i + 1]])
    #                         reconsts = torch.cat(
    #                             [reconsts, x_reconst[i:i + 1]])
    #                         flags[label_idx] = 1
    #
    #             loss_dic = self.unlabeled_loss(x, out)
    #             total = loss_dic['total']
    #             loss_reconst = loss_dic['reconst']
    #             loss_gaussian = loss_dic['gaussian']
    #             loss_categorical = loss_dic['categorical']
    #             preds = loss_dic['predicted_labels'].cpu().numpy()
    #             for (true, pred) in zip(labels, preds):
    #                 cm[self.labels.index(true), pred] += 1
    #
    #             predicted_labels = np.concatenate([predicted_labels, preds])
    #
    #             loss['loss_total'] += total.item()
    #             loss['loss_reconst'] += loss_reconst.item()
    #             loss['loss_gaussian'] += loss_gaussian.item()
    #             loss['loss_categorical'] += loss_categorical.item()
    #
    #             n_samples += x.size(0)
    #
    #         for k in loss.keys():
    #             loss[k] /= n_samples
    #
    #         loss_info = ", ".join([f'{k}: {v:.3f}' for k, v in loss.items()])
    #         print(f'Test {loss_info}')
    #
    #         latents = latents.cpu().numpy()
    #         comparison = torch.cat([originals[:12], reconsts[:12]]).cpu()
    #
    #         out = dict(loss)
    #         out['latents'] = latents
    #         out['labels'] = latent_labels
    #         out['comparison'] = comparison
    #         out['cm'] = cm
    #         out['predicted_labels'] = predicted_labels
    #
    #         return out
