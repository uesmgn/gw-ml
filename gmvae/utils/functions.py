import torch
import torch.nn as nn

def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    x = mean + eps * std
    return x

def activation(type='ReLU'):
    if type in ('Tanh', 'tanh'):
        return nn.Tanh()
    elif type in ('Sigmoid', 'sigmoid'):
        return nn.Sigmoid()
    elif type in ('ELU', 'elu'):
        return nn.ELU()
    else:
        return nn.ReLU()
