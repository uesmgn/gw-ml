import os
import args_parse
from attrdict import AttrDict as attrdict
import pandas as pd
import timeout_decorator
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.test.test_utils as test_utils
import torch_xla.distributed.xla_multiprocessing as xmp

import datasets
import net

try:
    TPU_IP = os.environ['TPU_IP']
except:
    print('The environment variable $TPU_IP does not exist')
    exit(1)
os.environ['XRT_TPU_CONFIG'] = f'tpu_worker;0;{TPU_IP}:8470'
# os.environ['XLA_USE_BF16'] = '1'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    'cvae', 'vae_439'
]

DATASETS = [
    'gravityspy'
]

FLAGS = args_parse.parse_common_options(
    num_cores=8,
    batch_size=1024,
    num_epochs=100,
    num_workers=4,
    log_steps=10,
    lr=1e-3,
    model={
        'choices': MODELS,
        'default': 'vae_439',
    },
    dataset={
        'choices': DATASETS,
        'default': 'gravityspy',
    }
)

MODEL_PARAMS = attrdict(
    x_channel = 1,
    x_dim = (439, 439),
    z_dim = 512,
    y_dim = 20,
    bottle_channel = 32,
    channels = (64, 128, 192, 64),
    kernels = (3, 3, 3, 3),
    poolings = (3, 3, 3, 3),
    pool_func = 'max',
    act_func = 'ReLU'
)

def train_logger(device, step, loss, tracker, epoch=None, summary_writer=None):
  test_utils.print_training_update(
      device,
      step,
      loss.item(),
      tracker.rate(),
      tracker.global_rate(),
      epoch=epoch,
      summary_writer=summary_writer)

def train(loader, model_params):

    torch.manual_seed(42)

    @timeout_decorator.timeout(30)
    def get_xla_device():
        return xm.xla_device()

    try:
        device = get_xla_device()
    except:
        print('timed out in loading xla device.')
        print(f'check TPU_IP {TPU_IP} is appropriate or not.')
        exit(1)
    if FLAGS.verbose:
        print(f'device: {device}')

    model = getattr(net.models, FLAGS.model)(model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr)

    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()
        for step, (data, target) in enumerate(loader):
            params = model(data, target, return_params=True)
            loss = model.criterion(**params)
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)
            if step % FLAGS.log_steps == 0:
                xm.add_step_closure(
                    train_logger, args=(device, step, loss, tracker, epoch))

    parallel_loader = pl.ParallelLoader(loader, [device])

    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loop_fn(parallel_loader.per_device_loader(device), epoch)
        xm.master_print('Finished training epoch {}'.format(epoch))


def _mp_fn(index):

    model_params = MODEL_PARAMS

    torch.set_default_tensor_type('torch.FloatTensor')

    if FLAGS.fake_data:
        fake_dataset_len = 120000
        loader = xu.SampleGenerator(
                    data=(torch.zeros(FLAGS.batch_size, 1, *model_params.x_dim),
                          torch.zeros(FLAGS.batch_size, dtype=torch.int64)),
                    sample_count=fake_dataset_len // FLAGS.batch_size // xm.xrt_world_size())
    else:
        data_transform = transforms.Compose([
            transforms.CenterCrop(model_params.x_dim),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        dataset = getattr(datasets, FLAGS.dataset)(root=ROOT,
                                                   transform=data_transform,
                                                   download=True)
        sampler = None
        if xm.xrt_world_size() > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True)

        loader = DataLoader(dataset,
                            batch_size=FLAGS.batch_size,
                            sampler=sampler,
                            num_workers=FLAGS.num_workers,
                            shuffle=False if sampler else True,
                            drop_last=True)

    train(loader, model_params)

if __name__ == '__main__':
    xmp.spawn(_mp_fn, nprocs=FLAGS.num_cores)
