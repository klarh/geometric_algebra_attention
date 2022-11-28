from .. import base
from .Vector2VectorAttention import Vector2VectorAttention


class TiedVectorAttention(base.TiedVectorAttention, Vector2VectorAttention):
    __doc__ = base.TiedVectorAttention.__doc__
