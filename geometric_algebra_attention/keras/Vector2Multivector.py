
from tensorflow import keras

from .. import base
from .internal import AttentionBase

class Vector2Multivector(base.Vector2Multivector, keras.layers.Layer):
    __doc__ = base.Vector2Multivector.__doc__

    math = AttentionBase.math

    def call(self, inputs, mask=None):
        return self._evaluate(inputs)

keras.utils.get_custom_objects()['Vector2Multivector'] = Vector2Multivector
