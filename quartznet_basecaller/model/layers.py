"""
Bonito nn modules.
"""

import torch
from torch.nn import Module
from torch.nn.init import orthogonal_
from torch.nn.utils.fusion import fuse_conv_bn_eval


LAYERS = {}


def register(layer):
    layer.name = layer.__name__.lower()
    LAYERS[layer.name] = layer
    return layer


register(torch.nn.ReLU)
register(torch.nn.Tanh)


@register
class Permute(Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}

    def extra_repr(self):
        return 'dims={}'.format(self.dims)