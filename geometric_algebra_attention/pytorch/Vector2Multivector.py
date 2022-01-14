
import torch as pt

from .. import base
from .internal import AttentionBase

class Vector2Multivector(base.Vector2Multivector, pt.nn.Module):
    __doc__ = base.Vector2Multivector.__doc__

    math = AttentionBase.math

    def forward(self, inputs):
        return self._evaluate(inputs)
