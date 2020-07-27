import torch.nn as nn

__all__ = [
    'network_to_half',
    'network_to_half_with_fp16'
]

class tofp16(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.half()

def BN_convert_float(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(net):
    return BN_convert_float(net.half())

def network_to_half_with_fp16(net):
    return nn.Sequential(tofp16(), BN_convert_float(net.half()))
