import torch

def reparameterize(mean, var):
    std = torch.pow(var, 0.5)
    eps = torch.randn_like(std)
    x = mean + eps * std
    return x

def activation(type='ReLU'):
    if type in ('Tanh', 'tanh'):
        return torch.nn.Tanh()
    elif type in ('Sigmoid', 'sigmoid'):
        return torch.nn.Sigmoid()
    elif type in ('Softmax', 'softmax'):
        return torch.nn.Softmax(dim=-1)
    elif type in ('ELU', 'elu'):
        return torch.nn.ELU()
    elif type in ('LeakyReLU', 'lrelu'):
        return torch.nn.LeakyReLU(0.1)
    elif type in ('PReLU', 'prelu'):
        return torch.nn.PReLU()
    else:
        return torch.nn.ReLU()
