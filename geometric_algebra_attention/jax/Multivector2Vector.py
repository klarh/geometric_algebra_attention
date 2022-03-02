
from .. import base

class Multivector2Vector(base.Multivector2Vector):
    __doc__ = base.Multivector2Vector.__doc__

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
