import torch

ACTIVATIONS = ['Tanh', 'Sigmoid', 'Softmax',
               'ELU', 'LeakyReLU', 'PReLU',
               'ReLU']

def get_activation(type='ReLU'):
    if type in ('Tanh', 'tanh'):
        return torch.nn.Tanh()
    elif type in ('Sigmoid', 'sigmoid'):
        return torch.nn.Sigmoid()
    elif type in ('Softmax', 'softmax'):
        return torch.nn.Softmax(dim=-1)
    elif type in ('ELU', 'elu'):
        return torch.nn.ELU(inplace=True)
    elif type in ('LeakyReLU', 'lrelu'):
        return torch.nn.LeakyReLU(0.1)
    elif type in ('PReLU', 'prelu'):
        return torch.nn.PReLU()
    elif type in ('ReLU', 'relu'):
        return torch.nn.ReLU(inplace=True)

def reparameterize(mean, var):
    if torch.is_tensor(var):
        std = torch.pow(var, 0.5)
    else:
        std = np.sqrt(var)
    eps = torch.randn_like(mean)
    x = mean + eps * std
    return x
