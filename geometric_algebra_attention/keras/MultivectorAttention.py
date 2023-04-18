
from tensorflow import keras

from .. import base
from .internal import AttentionBase

class MultivectorAttention(
        AttentionBase, base.MultivectorAttention, keras.layers.Layer):
    __doc__ = base.MultivectorAttention.__doc__

    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='single',
                 include_normalized_products=False,
                 linear_mode='partial', linear_terms=0,
                 **kwargs):
        keras.layers.Layer.__init__(self, **kwargs)
        base.MultivectorAttention.__init__(
            self, score_net, value_net, reduce, merge_fun, join_fun, rank,
            invariant_mode, covariant_mode, include_normalized_products,
            linear_mode, linear_terms,
        )

keras.utils.get_custom_objects()['MultivectorAttention'] = MultivectorAttention
