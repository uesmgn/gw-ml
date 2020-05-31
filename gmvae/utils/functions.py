import torch

def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    x = mean + eps * std
    return x

def activation(type='ReLU'):
    if type in ('Tanh', 'tanh'):
        return torch.nn.Tanh()
    elif type in ('Sigmoid', 'sigmoid'):
        return torch.nn.Sigmoid()
    elif type in ('ELU', 'elu'):
        return torch.nn.ELU()
    else:
        return torch.nn.ReLU()
