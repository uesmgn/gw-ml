import argparse
import configparser
import json
import datetime
import time
import os
import random
import multiprocessing as mp
from collections import defaultdict
from pprint import  pprint
from multiprocessing import Manager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import pandas as pd
import numpy as np

from gwspy.dataset import Dataset
from net.models import *
from net.criterion import criterion
from net.utils import get_middle_dim

from utils.clustering import decomposition, metrics, functional
from utils.plotlib import plot as plt

# ----------------------------------------------------------------
import os
import  args_parse
import torch
from torchvision import transforms

import datasets
import datasets.utils as du


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    'vae_439'
]

DATASETS = [
    'gravityspy'
]

FLAGS = args_parse.parse_common_options(
    x_dim=439,
    batch_size=64,
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

def main():
    device_ids = list(range(torch.cuda.device_count()))
    device = f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu'

    if FLAGS.fake_data:
        fake_data_len = 100000
        loader = du.SampleGenerator(
            data_shape=(FLAGS.batch_size, 1, FLAGS.x_dim, FLAGS.x_dim),
            target_shape=(FLAGS.batch_size, 1),
            sample_count=fake_data_len // FLAGS.batch_size
        )
    else:
        data_transform = transforms.Compose([
            transforms.CenterCrop(FLAGS.x_dim),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        dataset = getattr(datasets, FLAGS.dataset)(root=ROOT,
                                                   transform=data_transform,
                                                   download=True)
        loader = DataLoader(dataset,
                            batch_size=FLAGS.batch_size,
                            num_workers=FLAGS.num_workers,
                            shuffle=True,
                            drop_last=True)




if __name__ == '__main__':
    main()
