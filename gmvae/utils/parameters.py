import torch.optim as optim
from .functions import activation


def get_activation(trial):
    activation_names = ['Tanh', 'ReLU', 'ELU']
    activation_name = trial.suggest_categorical('activation', activation_names)
    act_func = activation(activation_name)
    return act_func

def get_pooling(trial):
    pooling_names = ['max', 'avg']
    pooling_name = trial.suggest_categorical('pooling', pooling_names)
    return pooling_name

def get_optimizer(trial, model):
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    return optimizer

def get_kernels(trial, low=3, high=7, q=2, size=4):
    kernels = [int(trial.suggest_discrete_uniform(f'kernel_{i}', low, high, q))
               for i in range(size)]
    return kernels

def get_channels(trial, low=32, high=192, q=16, size=4):
    channels = [int(trial.suggest_discrete_uniform(f'channel_{i}', 32, 192, 16))
                for i in range(size)]
    return channels

def get_dim(trial, name, low=16, high=512, q=16):
    dim = int(trial.suggest_discrete_uniform(f'{name}_dim', low, high, q))
    return dim
