import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data
from collections import  defaultdict
from itertools import cycle
import numpy as np
from attrdict import AttrDict as attrdict
import apex
from apex import amp, optimizers
from apex.parallel import DistributedDataParallel as DDP

import net.models as models
import datasets
import utils as ut
from utils.clustering import decomposition, metrics, functional
from utils.plotlib import plot


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLAGS = attrdict(
    batch_size=128,
    num_epochs=5000,
    num_workers=1,
    log_step=10,
    eval_step=5,
    save_step=100,
    lr=1e-1,
    momentum=0.9,
    weight_decay=1e-4,
    opt_level='O1',
    dataset='gravityspy',
    local_rank=0,
    x_size=160,
    x_dim=256,
    z_dim=256,
    filter_size=5,
    gp_output_size=(1, 1),
    num_blocks=(1, 1, 1, 1),
    planes=(32, 64, 96, 128),
)

def _data_loader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, shuffle=True, drop_last=True):
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           drop_last=drop_last)
def _to_acronym(arr, dict):
    return np.array([f'{l}-{ut.acronym(dict[l])}' for l in arr])

def _to_values(arr, dic):
    ret = [dic[x] for x in arr]
    return ret

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True

FLAGS.distributed = False
if 'WORLD_SIZE' in os.environ:
    FLAGS.distributed = int(os.environ['WORLD_SIZE']) > 1

torch.cuda.set_device(FLAGS.local_rank)

if FLAGS.distributed:
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    world_size = torch.distributed.get_world_size()
    print(f'world size: {world_size}')

assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

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
targets_dict = {k: f'{k}-{ut.acronym(v)}' for k, v in dataset.targets_dict.items()}
train_set, test_set = dataset.split_dataset()

print(f'dataset length: {len(dataset)}')
print(f'train_set length: {len(train_set)}')
print(f'test_set length: {len(test_set)}')
print(f'target: {targets_dict}')

train_loader = _data_loader(train_set)
test_loader = _data_loader(test_set)

model = models.resnet.ResNet(in_planes=1, num_classes=len(target_labels),
                             num_blocks=FLAGS.num_blocks, planes=FLAGS.planes)
# ------------
# model = apex.parallel.convert_syncbn_model(model)
model = model.cuda()
# optim = optimizers.FusedAdam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
optim = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
# optim = torch.optim.SGD(model.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
model, optim = amp.initialize(model, optim, opt_level=FLAGS.opt_level)

stats = defaultdict(lambda: list())
criterion = nn.CrossEntropyLoss()

for epoch in range(1, FLAGS.num_epochs):
    loss = defaultdict(lambda: 0)
    model.train()

    for step, (x, target) in enumerate(train_loader):
        x, target = x.to(device), target.to(device)
        out = model(x)
        step_loss = criterion(out, target)
        optim.zero_grad()
        with amp.scale_loss(step_loss, optim) as scaled_loss:
            scaled_loss.backward()
        optim.step()
        loss['CE_loss'] += step_loss.item()
    print(f'epoch: {epoch} -', ', '.join([f'{k}: {v:.3f}' for k, v in loss.items()]))

    for k, v in loss.items():
        stats[k].append([epoch, v])

    if epoch % FLAGS.eval_step == 0:

        if not os.path.exists(outdir):
            os.mkdir(outdir)

        target_stack = np.array([]).astype(np.integer)
        pred_stack = np.array([]).astype(np.integer)
        acc = 0
        N = 0

        model.eval()
        with torch.no_grad():
            for step, (x, target) in enumerate(test_loader):
                x = x.to(device)
                out = model(x)
                _, y_pred = torch.max(out, -1)
                target_stack = np.append(target_stack, list(target.numpy()))
                y_pred = y_pred.cpu()
                acc += torch.nonzero(target==y_pred).size(0)
                N += y_pred.size(0)
                pred_stack = np.append(pred_stack, list(y_pred.numpy()))
        acc = acc / N * 100
        print(f'acc = {acc:.3f} at epoch {epoch}')
        stats['test_acc'].append([epoch, acc])

        trues = _to_values(target_stack, targets_dict)
        preds = _to_values(pred_stack, targets_dict)

        cm, cm_xlabels, cm_ylabels = metrics.confusion_matrix(
                    pred_stack, target_stack, targets_dict.keys(), targets_dict.keys(), return_labels=True)
        cm_figsize = (len(cm_xlabels) / 1.2, len(cm_ylabels) / 1.5)
        cm_ylabels = _to_values(cm_ylabels, targets_dict)
        plot.plot_confusion_matrix(cm, cm_xlabels, cm_ylabels,
                                   out=f'{outdir}/cm_{epoch}.png',
                                   xlabel='predicted', ylabel='true',
                                   figsize=cm_figsize)
        for k, v in stats.items():
            xx = np.array(v)[:, 0]
            plot.plot(v, out=f'{outdir}/{k}_{epoch}.png', title=f'{k}', xmin=min(xx), xmax=max(xx), xlabel='epoch', ylabel=k)

        os.mkdir('.pretrain', exist_ok=True)
        torch.save(model.state_dict(), f'.pretrain/{model.__class__.__name__}.pt')
