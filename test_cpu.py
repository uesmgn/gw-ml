import os
import args_parse
from attrdict import AttrDict as attrdict
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from gwspy.dataset import Dataset

BASEDIR = os.path.dirname(os.path.abspath(__file__))

FLAGS = args_parse.parse_common_options(
    num_cores=8,
    batch_size=1024,
    num_epochs=100,
    num_workers=4,
    log_steps=10,
    lr=1e-3
)

DEFAULT = attrdict(
    dataset_json = f'{BASEDIR}/dataset.json'
)

NKWARGS = attrdict(
    x_dim = 486,
    x_shape = (1, 486, 486),
    z_dim = 512,
    y_dim = 20,
    bottle_channel = 32,
    channels = (64, 128, 192, 64),
    kernels = (3, 3, 3, 3),
    poolings = (3, 3, 3, 3),
    pool_func = 'max',
    act_func = 'ReLU'
)

def train_gwspy():

    dataset_df = pd.read_json(DEFAULT.dataset_json)

    data_transform = transforms.Compose([
        transforms.CenterCrop(NKWARGS.x_dim),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = Dataset(dataset_df, transform=data_transform)
    # treatment dataset
    dataset.sample('label', min_value_count=200)
    loader = DataLoader(dataset,
                        batch_size=FLAGS.batch_size,
                        num_workers=FLAGS.num_workers,
                        shuffle=True,
                        drop_last=True)

    torch.manual_seed(42)

    # specify device
    device = 'cpu'
    if FLAGS.verbose:
        print(f'device: {device}')

    # unique labels
    true_unique = dataset.unique_column('label')
    pred_unique = np.array(range(NKWARGS.y_dim)).astype(np.int32)

    if FLAGS.verbose:
        print(f'true_unique: {true_unique}')
        print(f'pred_unique: {pred_unique}')


if __name__ == '__main__':
    train_gwspy()
