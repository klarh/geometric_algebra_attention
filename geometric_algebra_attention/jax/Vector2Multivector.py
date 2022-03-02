
from .. import base
from .internal import AttentionBase

class Vector2Multivector(base.Vector2Multivector):
    __doc__ = base.Vector2Multivector.__doc__

    math = AttentionBase.math

    @classmethod
    def stax_init(cls, rng, input_shape):
        return input_shape, []

    @classmethod
    def stax_apply(cls, params, inputs, rng=None):
        return cls._evaluate(inputs)

    @property
    def stax_functions(self):
        return self.stax_init, stax_apply

    def __call__(self, inputs):
        return self.stax_apply(None, inputs)
