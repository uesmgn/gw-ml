import torch
from collections import defaultdict
from ..losses.LossFunctions import LossFunctions
from ..networks.VAENet import VAENet

class VAE:
    def __init__(self, args):
        self.input_size = args.input_size
        self.z_dim = args.z_dim
        self.y_dim = args.y_dim
        self.w_rec = args.w_rec
        self.w_gauss = args.w_gauss
        self.w_cat = args.w_cat
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

    def init_model(self, train_loader, test_loader, optimizer):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.__initialized = True

    def get_loss(self, x, out):
        # obtain network variables
        z, x_reconst = out['z'], out['x_reconst']
        z_mu, z_var = out['z_mu'], out['z_var']

        loss_rec = self.losses.reconstruction_loss(x, x_reconst)
        kl_divergence = self.losses.gaussian_kl_divergence(z, z_mu, z_var)
        total = loss_rec - kl_divergence

        loss_dic = {'total': total,
                    'loss_rec': loss_rec,
                    'kl': kl_divergence}

        return loss_dic

    def train(self, epoch, verbose=True):
        assert self.__initialized
        self.net.train()

        loss = defaultdict(lambda: 0)
        n_samples = 0

        for b, (x, labels) in enumerate(self.train_loader):
            batch = b + 1
            x = x.to(self.device)
            self.optimizer.zero_grad()
            out = self.net(x)

            loss_dic = self.get_loss(x, out)
            total = loss_dic['total']
            loss_rec = loss_dic['loss_rec']
            kl = loss_dic['kl']

            total.backward()
            self.optimizer.step()

            loss['total'] += total.item()
            loss['loss_rec'] += loss_rec.item()
            loss['kl'] += kl.item()
            n_samples += x.size(0)
        print('')

        for key in loss.keys():
            loss[key] /= n_samples

        if verbose:
            loss_info = ", ".join(
                [f'Loss-{k}: {v:.3f}' for k, v in loss.items()])
            print(f'Train {loss_info}')

        out = loss
        # additional output key-value ↓

        return out
