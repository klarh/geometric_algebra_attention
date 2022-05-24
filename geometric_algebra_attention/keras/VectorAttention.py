
from tensorflow import keras

from .. import base
from .internal import AttentionBase

class VectorAttention(AttentionBase, base.VectorAttention, keras.layers.Layer):
    __doc__ = base.VectorAttention.__doc__

    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='single',
                 include_normalized_products=False, **kwargs):
        keras.layers.Layer.__init__(self, **kwargs)
        base.VectorAttention.__init__(
            self, score_net, value_net, reduce, merge_fun, join_fun, rank,
            invariant_mode, covariant_mode, include_normalized_products)

keras.utils.get_custom_objects()['VectorAttention'] = VectorAttention
