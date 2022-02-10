
from .. import base
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention

class LabeledMultivectorAttention(base.LabeledMultivectorAttention, Multivector2MultivectorAttention):
    __doc__ = base.LabeledMultivectorAttention.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if type(self) == LabeledMultivectorAttention:
            self.init()
