import torch
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

        self.net.to(self.device)
        if torch.cuda.is_available():
            net = torch.nn.DataParallel(net)
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

            latents = torch.cat(
                [latents, z], dim=0)

            latent_labels += [[true, pred]
                              for true, pred in zip(list(labels), preds)]

        if self.enable_scheduler:
            self.scheduler.step()

        for k in loss.keys():
            loss[k] /= n_samples

        print(", ".join([f'{k}: {v:.3f}' for k, v in loss.items()]))

        out = dict(loss)
        out['latents'] = latents.cpu().detach().numpy()
        out['true'], out['pred'] = np.array(latent_labels).T.astype(str)
        return out
