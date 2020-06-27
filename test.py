import  torch
import  torch.nn as nn
from net.models.cvae import *

kwargs = {
    'x_shape': (1, 486, 486),
    'y_dim': 16,
    'z_dim': 512,
    'bottle_channel': 32,
    'poolings': (3, 3, 3, 3),
    'kernels': (3, 3, 3, 3),
    'channels': (64, 128, 256, 64),
    'pool_func': 'max',
    'act_func': 'ReLU'
}
model = CVAE(**kwargs)
classifier = nn.Sequential(*list(model.children())[:2])
