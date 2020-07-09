# This module cannot import any other PyTorch/XLA module. Only Python core modules.
import argparse
import os


def parse_common_options(x_dim=None,
                         batch_size=64,
                         num_epochs=100,
                         num_workers=4,
                         log_steps=10,
                         lr=None,
                         model=None,
                         dataset=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-x', '--x_dim', type=int, default=x_dim)
    parser.add_argument('-b', '--batch_size', type=int, default=batch_size)
    parser.add_argument('-e', '--num_epochs', type=int, default=num_epochs)
    parser.add_argument('-w', '--num_workers', type=int, default=num_workers)
    parser.add_argument('--log_steps', type=int, default=log_steps)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-f', '--fake_data', action='store_true')
    if model:
        parser.add_argument('--model', **model)
    if dataset:
        parser.add_argument('--dataset', **dataset)
    args, leftovers = parser.parse_known_args()
    return args
