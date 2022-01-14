
from .. import base
from .internal import AttentionBase

class Vector2Multivector(base.Vector2Multivector):
    __doc__ = base.Vector2Multivector.__doc__

    math = AttentionBase.math

    def __call__(self, inputs):
        return self._evaluate(inputs)
