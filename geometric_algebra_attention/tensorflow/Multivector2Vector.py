
from .. import base

class Multivector2Vector(base.Multivector2Vector):
    __doc__ = base.Multivector2Vector.__doc__

    def __call__(self, inputs):
        return self._evaluate(inputs)
