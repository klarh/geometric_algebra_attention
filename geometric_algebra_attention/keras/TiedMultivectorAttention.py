from tensorflow import keras

from .. import base
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention


class TiedMultivectorAttention(
    base.TiedMultivectorAttention, Multivector2MultivectorAttention
):
    __doc__ = base.TiedMultivectorAttention.__doc__

    def compute_mask(self, *args, **kwargs):
        result = super().compute_mask(*args, **kwargs)
        return result, result

keras.utils.get_custom_objects()["TiedMultivectorAttention"] = TiedMultivectorAttention
