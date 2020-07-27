import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils import data
from collections import  defaultdict
import numpy as np
from attrdict import AttrDict as attrdict

import net.models as models
import datasets
import utils as ut
from utils.clustering import decomposition, metrics, functional
from utils.plotlib import plot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLAGS = attrdict(
    batch_size=32,
    num_epochs=5000,
    num_workers=4,
    log_step=10,
    eval_step=20,
    save_step=100,
    lr=1e-3,
    dataset='gravityspy',
    use_fp16=True,
    x_size=96,
    x_dim=64,
    z_dim=64,
    filter_size=3
)

setup_transform = transforms.Compose([
                transforms.CenterCrop(486),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

data_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(FLAGS.x_size),
                transforms.ToTensor()
            ])

dataset = getattr(datasets, FLAGS.dataset)(root=ROOT, transform=data_transform,
                                           setup_transform=setup_transform,
                                           download=True)

target_labels = torch.Tensor([0, 1, 3, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 19, 21]).to(torch.long)
dataset.get_by_keys(target_labels)
targets_dict = dataset.targets_dict
train_set, test_set = dataset.split_dataset()
labeled_set, unlabeled_set = train_set.uniform_label_sampler(target_labels, num_per_class=50)
grid_set, _ = dataset.uniform_label_sampler(target_labels, num_per_class=1)
n_steps = len(unlabeled_set) // FLAGS.batch_size
labeled_batch = len(labeled_set) // n_steps

print(f'dataset length: {len(dataset)}')
print(f'train_set length: {len(train_set)}')
print(f'test_set length: {len(test_set)}')
print(f'unlabeled_set length: {len(unlabeled_set)}')
print(f'labeled_set length: {len(labeled_set)}')
print(f'unlabeled_batch: {FLAGS.batch_size}')
print(f'labeled_batch: {labeled_batch}')

alpha = len(dataset) * 0.1

def _data_loader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, shuffle=True, drop_last=True):
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           drop_last=drop_last)
def _to_acronym(arr, dict):
    return np.array([f'{l}-{ut.acronym(dict[l])}' for l in arr])

data_loader = _data_loader(dataset)
unlabeled_loader = _data_loader(unlabeled_set)
labeled_loader = _data_loader(labeled_set, batch_size=labeled_batch)
test_loader = _data_loader(test_set)
grid_loader = _data_loader(grid_set, batch_size=len(grid_set), shuffle=False)

resnet = models.resnet.ResNet(num_blocks=(1,1,1,1))
model = models.resvae.ResVAE_M2(resnet, x_dim=FLAGS.x_dim, z_dim=FLAGS.z_dim, y_dim=len(target_labels),
                                filter_size=FLAGS.filter_size, verbose=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if FLAGS.use_fp16:
    model = ut.network_to_half(model)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

stats = defaultdict(lambda: list())
outdir = f'result_{int(time.time())}'

for epoch in range(1, FLAGS.num_epochs):
    loss = defaultdict(lambda: 0)
    model.train()
    for step, ((ux, _), (lx, target)) in enumerate(zip(unlabeled_loader, labeled_loader)):
        target_index = torch.cat([(target_labels == t).nonzero().view(-1) for t in target.to(torch.long)])
        step_loss = model(ux, lx, target_index, alpha)
        optim.zero_grad()
        step_loss.backward()
        optim.step()
        loss['m2_loss'] += step_loss.item()
    scheduler.step()
    print(f'epoch: {epoch} -', ', '.join([f'{k}: {v:.3f}' for k, v in loss.items()]))

    for k, v in loss.items():
        stats[k].append(v)

    if epoch % FLAGS.eval_step == 0:

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        features = torch.Tensor().to(device)
        target_stack = np.array([]).astype(np.integer)
        pred_index_stack = np.array([]).astype(np.integer)
        acc = 0
        N = 0

        model.eval()
        with torch.no_grad():
            for step, (x, target) in enumerate(test_loader):
                target_idx = torch.cat([(target_labels == t).nonzero().view(-1) for t in target.to(torch.long)]).to(device)
                out_params = model(x, return_params=True)
                features = torch.cat((features, out_params['z']), 0)
                target_stack = np.append(target_stack, list(target.numpy()))
                y_pred = out_params['y_pred']
                acc += torch.nonzero(target_idx==y_pred).size(0)
                N += y_pred.size(0)
                pred_index_stack = np.append(pred_index_stack, list(y_pred.cpu().numpy()))
        pred_stack = target_labels.repeat(pred_index_stack.size, 1).gather(1, torch.Tensor(pred_index_stack).to(torch.long).unsqueeze(1)).view(-1).numpy()
        features = features.cpu().numpy()
        acc = acc / N * 100
        print(f'acc = {acc:.3f} at epoch {epoch}')
        stats['test_acc'].append(test_acc)

        if features.shape[1] > 2:
            features = decomposition.TSNE().fit_transform(features)
        if features.shape[1] != 2:
            raise ValueError('features can not visualize')

        target_acronym = _to_acronym(target_stack, targets_dict)
        pred_acronym = _to_acronym(pred_stack, targets_dict)

        plot.scatter(features[:, 0], features[:, 1], target_acronym,
                     out=f'{outdir}/latent_true_{epoch}.png',
                     title=f'latent features at epoch {epoch} - true label')
        plot.scatter(features[:, 0], features[:, 1], pred_acronym,
                     out=f'{outdir}/latent_pred_{epoch}.png',
                     title=f'latent features at epoch {epoch} - pred label')
        cm, cm_xlabels, cm_ylabels = metrics.confusion_matrix(
                    pred_stack, target_stack, target_labels.numpy(), target_labels.numpy(), return_labels=True)
        cm_figsize = (len(cm_xlabels) / 1.2, len(cm_ylabels) / 1.5)
        cm_xlabels_acronym = _to_acronym(cm_xlabels, targets_dict)
        cm_ylabels_acronym = _to_acronym(cm_ylabels, targets_dict)
        plot.plot_confusion_matrix(cm, cm_xlabels_acronym, cm_ylabels_acronym,
                                   out=f'{outdir}/cm_{epoch}.png',
                                   xlabel='predicted', ylabel='true',
                                   figsize=cm_figsize)
        for k, v in stats.items():
            data = np.stack((list(range(1, len(v)+1)), v), 1)
            plot.plot(data, out=f'{outdir}/{k}_{epoch}.png', title=f'{k}', xmin=1, xmax=len(v), xlabel='epoch', ylabel=k)
