from tensorflow import keras

from .. import base
from .Vector2VectorAttention import Vector2VectorAttention


class TiedVectorAttention(base.TiedVectorAttention, Vector2VectorAttention):
    __doc__ = base.TiedVectorAttention.__doc__


keras.utils.get_custom_objects()["TiedVectorAttention"] = TiedVectorAttention
