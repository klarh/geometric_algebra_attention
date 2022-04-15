
from tensorflow import keras

from .. import base

class Multivector2Vector(base.Multivector2Vector, keras.layers.Layer):
    __doc__ = base.Multivector2Vector.__doc__

    def call(self, inputs, mask=None):
        return self._evaluate(inputs)

keras.utils.get_custom_objects()['Multivector2Vector'] = Multivector2Vector
