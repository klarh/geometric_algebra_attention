
from .. import base
from .VectorAttention import VectorAttention

class Vector2VectorAttention(base.Vector2VectorAttention, VectorAttention):
    __doc__ = base.Vector2VectorAttention.__doc__

    def __init__(self, n_dim, score_net, value_net, scale_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='partial',
                 include_normalized_products=False,
                 linear_mode='partial', linear_terms=0,
                 convex_covariants=False,
                 **kwargs):
        VectorAttention.__init__(
            self, n_dim=n_dim, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            invariant_mode=invariant_mode, covariant_mode=covariant_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
            **kwargs)
        base.Vector2VectorAttention.__init__(
            self, scale_net=scale_net, convex_covariants=convex_covariants)

        if type(self) == Vector2VectorAttention:
            self.init()
