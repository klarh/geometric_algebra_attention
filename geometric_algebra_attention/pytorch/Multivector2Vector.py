
import torch as pt

from .. import base

class Multivector2Vector(base.Multivector2Vector, pt.nn.Module):
    __doc__ = base.Multivector2Vector.__doc__

    def forward(self, inputs):
        return self._evaluate(inputs)
