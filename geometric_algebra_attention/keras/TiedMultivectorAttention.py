from tensorflow import keras

from .. import base
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention


class TiedMultivectorAttention(
    base.TiedMultivectorAttention, Multivector2MultivectorAttention
):
    __doc__ = base.TiedMultivectorAttention.__doc__


keras.utils.get_custom_objects()["TiedMultivectorAttention"] = TiedMultivectorAttention
