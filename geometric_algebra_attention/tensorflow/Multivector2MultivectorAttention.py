
from .. import base
from .MultivectorAttention import MultivectorAttention

class Multivector2MultivectorAttention(base.Multivector2MultivectorAttention, MultivectorAttention):
    __doc__ = base.Multivector2MultivectorAttention.__doc__

    def __init__(self, n_dim, score_net, value_net, scale_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='partial',
                 include_normalized_products=False,
                 convex_covariants=False, **kwargs):
        base.Multivector2MultivectorAttention.__init__(
            self, scale_net=scale_net, convex_covariants=convex_covariants)
        MultivectorAttention.__init__(
            self, n_dim=n_dim, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            invariant_mode=invariant_mode, covariant_mode=covariant_mode,
            include_normalized_products=include_normalized_products,
            **kwargs)
