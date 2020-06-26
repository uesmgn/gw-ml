import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Reshape',
    'Conv2dModule',
    'ConvTranspose2dModule',
    'Gaussian',
    'GaussianInput',
    'GaussianMixture',
    'GumbelSoftmax',
    'DownSample',
    'Upsample',
    'DenseModule'
]

eps = 1e-10


class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super().__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


class Conv2dModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=1,
                 stride=1,
                 act_func='ReLU'):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch,
                                kernel_size=kernel,
                                stride=stride,
                                padding=(kernel - stride) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(_activation(act_func))
        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class ConvTranspose2dModule(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel=1,
                 stride=1,
                 act_func='ReLU'):
        super().__init__()

        layers = []
        layers.append(nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=kernel,
                                         stride=stride,
                                         padding=(kernel - stride) // 2))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(_activation(act_func))
        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class Gaussian(nn.Module):
    def __init__(self, in_dim, out_dim, act_regur=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, out_dim * 2),)
        layers.append(_activation(act_regur))
        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        if self.training:
            x = _reparameterize(mean, var)
        else:
            x = mean
        return x, mean, var

class GaussianInput(nn.Module):
    def __init__(self, in_dim, out_dim, act_regur=None):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, out_dim * 2),)
        layers.append(_activation(act_regur))
        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        mean, logit = torch.split(x, x.shape[1] // 2, 1)
        var = F.softplus(logit) + eps
        return mean, var

class GaussianMixture(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, act_regur=None):
        super().__init__()
        self.layers = nn.ModuleList([
            _sequential(*[
                nn.Linear(in_dim, out_dim * 2),
                _activation(act_regur)]) for _ in range(n_components)
        ])

    def forward(self, x, pi):
        # x: (batch_size, dim)
        # pi: (batch_size, n_components)
        x_stack = []
        means_stack = []
        vars_stack = []
        for i, layer in enumerate(self.layers):
            h = layer(x)
            mean, logit = torch.split(h, h.shape[1] // 2, -1)
            var = F.softplus(logit) + eps
            dim = mean.shape[-1]
            p = pi[:,i].unsqueeze(-1).repeat(1, dim)
            mean = torch.pow(mean, p)
            var = torch.pow(var, p)
            if self.training:
                h = _reparameterize(mean, var)
            else:
                h = mean
            x_stack.append(h)
            means_stack.append(mean)
            vars_stack.append(var)
        x = torch.stack(x_stack, -1)
        means = torch.stack(means_stack, -1)
        vars = torch.stack(vars_stack, -1)
        return x, means, vars


class GumbelSoftmax(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.features = nn.Linear(in_dim, out_dim)

    def forward(self, x, tau=1., dim=-1, hard=False):
        logits = self.features(x)
        pi = logits.softmax(dim)
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y = gumbels.softmax(dim)

        if hard:
            index = y.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            y = y_hard - y.detach() + y

        return y


class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch,
                 kernel=3,
                 pooling=3,
                 pool_func='max',
                 act_func='ReLU'):
        super().__init__()
        layers = []
        assert kernel >= pooling
        if pool_func in ('avg',):
            layers.append(nn.AvgPool2d(kernel_size=kernel,
                                       stride=pooling,
                                       padding=(kernel - pooling) // 2))
        else:
            layers.append(nn.MaxPool2d(kernel_size=kernel,
                                       stride=pooling,
                                       padding=(kernel - pooling) // 2))

        layers.append(Conv2dModule(in_ch, out_ch,
                                    act_func=act_func))
        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch,
                 unpooling=3,
                 act_func='ReLU'):
        super().__init__()
        layers = []
        layers.append(ConvTranspose2dModule(in_ch, in_ch,
                                            kernel=unpooling,
                                            stride=unpooling,
                                            act_func=act_func))
        layers.append(ConvTranspose2dModule(in_ch, out_ch,
                                            act_func=act_func))
        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class DenseModule(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 n_layers=1,
                 hidden_dims=(64,),
                 act_trans='ReLU',
                 act_out=None):
        super().__init__()
        if n_layers > 0:
            assert len(hidden_dims) == n_layers
        layers = []
        if n_layers > 0:
            for i in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dims[i]))
                layers.append(_activation(act_trans))
            layers.append(nn.Linear(hidden_dims[-1], out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        layers.append(_activation(act_out))

        self.features = _sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x

class SafeBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.features = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum,
            affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        if x.shape[0] > 1 and x.dim() == 4:
            return self.features(x)
        return x

def _sequential(*layers):
    new_layers = []
    for layer in layers:
        if hasattr(layer, 'forward'):
            new_layers.append(layer)
    return nn.Sequential(*new_layers)

def _reparameterize(mean, var):
    if torch.is_tensor(var):
        std = torch.pow(var, 0.5)
    else:
        std = np.sqrt(var)
    eps = torch.randn_like(mean)
    x = mean + eps * std
    return x

def _activation(key='ReLU'):
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
    return None
