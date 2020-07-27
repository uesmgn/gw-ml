import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'Reshape',
    'Gaussian',
    'Clustering',
    'get_activation'
]


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.features = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x, reparameterize=True, eps=1e-10):
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        if reparameterize:
            x = self._reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var

    def _reparameterize(self, mean, var):
        if torch.is_tensor(var):
            std = torch.pow(var, 0.5)
        else:
            std = np.sqrt(var)
        eps = torch.randn_like(mean)
        x = mean + eps * std
        return x


class Clustering(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logits = self.logits(x)
        y = F.softmax(logits, -1)
        return logits, y

def get_activation(key='relu'):
    if key in ('Tanh', 'tanh'):
        return nn.Tanh()
    elif key in ('Sigmoid', 'sigmoid'):
        return nn.Sigmoid()
    elif key in ('Softmax', 'softmax'):
        return nn.Softmax(dim=-1)
    elif key in ('ELU', 'elu'):
        return nn.ELU(inplace=True)
    elif key in ('LeakyReLU', 'lrelu'):
        return nn.LeakyReLU(0.1)
    elif key in ('PReLU', 'prelu'):
        return nn.PReLU()
    elif key in ('ReLU', 'relu'):
        return nn.ReLU(inplace=True)
    else:
        raise ValueError(f'activation {key} is not supported.')
