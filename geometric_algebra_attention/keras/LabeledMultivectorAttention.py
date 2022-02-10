
from .. import base
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention

class LabeledMultivectorAttention(base.LabeledMultivectorAttention, Multivector2MultivectorAttention):
    __doc__ = base.LabeledMultivectorAttention.__doc__

    def build(self, input_shape):
        modified_shape = input_shape[1]
        return super().build(modified_shape)
