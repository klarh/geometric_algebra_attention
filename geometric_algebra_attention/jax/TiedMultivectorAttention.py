from .. import base
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention


class TiedMultivectorAttention(
    base.TiedMultivectorAttention, Multivector2MultivectorAttention
):
    __doc__ = base.TiedMultivectorAttention.__doc__
