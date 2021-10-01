
from .. import base
from .VectorAttention import VectorAttention

class Vector2VectorAttention(base.Vector2VectorAttention, VectorAttention):
    def __init__(self, n_dim, score_net, value_net, scale_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='partial',
                 **kwargs):
        base.Vector2VectorAttention.__init__(self, scale_net=scale_net)
        VectorAttention.__init__(
            self, n_dim=n_dim, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            invariant_mode=invariant_mode, covariant_mode=covariant_mode,
            **kwargs)
