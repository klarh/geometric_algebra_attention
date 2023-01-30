
from tensorflow import keras

from .. import base

class Multivector2Vector(base.Multivector2Vector, keras.layers.Layer):
    __doc__ = base.Multivector2Vector.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return self._evaluate(inputs)

keras.utils.get_custom_objects()['Multivector2Vector'] = Multivector2Vector
