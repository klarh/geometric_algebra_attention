from tensorflow import keras

from .. import base
from .Vector2VectorAttention import Vector2VectorAttention


class TiedVectorAttention(base.TiedVectorAttention, Vector2VectorAttention):
    __doc__ = base.TiedVectorAttention.__doc__

    def compute_mask(self, *args, **kwargs):
        result = super().compute_mask(*args, **kwargs)
        return result, result

keras.utils.get_custom_objects()["TiedVectorAttention"] = TiedVectorAttention
