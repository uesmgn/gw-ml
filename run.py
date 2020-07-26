import os
import torch
from torchvision import transforms
from torch.utils import data
from collections import  defaultdict
import numpy as np
from attrdict import AttrDict as attrdict

import net.models as models
import datasets
import utils.plotlib.plot as plot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLAGS = attrdict(
    x_dim=486,
    batch_size=32,
    num_epochs=5000,
    num_workers=4,
    log_step=10,
    eval_step=20,
    save_step=100,
    lr=1e-3,
    dataset='gravityspy'
)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark=True

setup_transform = transforms.Compose([
                transforms.CenterCrop(FLAGS.x_dim),
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

data_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(192),
                transforms.ToTensor()
            ])

dataset = getattr(datasets, FLAGS.dataset)(root=ROOT, transform=data_transform,
                                           setup_transform=setup_transform,
                                           download=True)

target_labels = torch.Tensor([0, 5, 6, 10, 15, 16, 17, 19, 21]).to(torch.long)
dataset.get_by_keys(target_labels)
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

def _data_loader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, shuffle=True, drop_last=True):
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           drop_last=drop_last)

data_loader = _data_loader(dataset)
unlabeled_loader = _data_loader(unlabeled_set)
labeled_loader = _data_loader(labeled_set, batch_size=labeled_batch)
test_loader = _data_loader(test_set)
grid_loader = _data_loader(grid_set, batch_size=len(grid_set), shuffle=False)

resnet = models.resnet.ResNet(num_blocks=(2,2,2,2))
model = models.resvae.ResVAE_M1(resnet, device=device, scale_factor=6, verbose=True)
optim = torch.optim.Adam(m1.parameters(), lr=FLAGS.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

stats = defaultdict(lambda: list())
outdir = f'result_{int(time.time())}'

for epoch in range(1, FLAGS.num_epochs):
    loss = defaultdict(lambda: 0)
    model.train()
    for step, (x, target) in enumerate(data_loader):
        target_index = torch.cat([(target_labels == t).nonzero().view(-1) for t in target.to(torch.long)])
        step_loss = model.criterion(x)
        optim.zero_grad()
        step_loss.backward()
        optim.step()
        loss['m1loss'] += step_loss.item()
    scheduler.step()
    print(f'epoch: {epoch} -', ', '.join([f'{k}: {v:.3f}' for k, v in loss.items()]))

    for k, v in loss.items():
        stats[k].append(v)

    if epoch % FLAGS.eval_step == 0:
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        for k, v in stats.items():
            data = np.stack((list(range(1, len(v)+1)), v), 1)
            plot.plot(data, out=f'{outdir}/{k}_{epoch}.png', title=f'{k}', xmin=1, xmax=len(v), xlabel='epoch', ylabel=k)
