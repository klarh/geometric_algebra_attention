import torch as pt

from .. import base
from . import geometric_algebra
from .internal import AttentionBase

class MultivectorAttention(AttentionBase, base.MultivectorAttention, pt.nn.Module):
    __doc__ = base.MultivectorAttention.__doc__

    def __init__(self, n_dim, *args, **kwargs):
        pt.nn.Module.__init__(self)
        base.MultivectorAttention.__init__(self, *args, **kwargs)

        self.n_dim = n_dim

        if type(self) == MultivectorAttention:
            self.init()
