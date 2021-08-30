
from .. import base
from .VectorAttention import VectorAttention

class Vector2VectorAttention(base.Vector2VectorAttention, VectorAttention):
    def __init__(self, n_dim, score_net, value_net, scale_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 use_product_vectors=True, use_input_vectors=False,
                 learn_vector_projection=False, **kwargs):
        # needed for initialization ordering issues
        self.rank = rank

        base.Vector2VectorAttention.__init__(
            self, scale_net=scale_net, use_product_vectors=use_product_vectors,
            use_input_vectors=use_input_vectors,
            learn_vector_projection=learn_vector_projection)
        VectorAttention.__init__(
            self, n_dim=n_dim, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            **kwargs)
