from .. import base
from .Multivector2MultivectorAttention import Multivector2MultivectorAttention


class TiedMultivectorAttention(
    base.TiedMultivectorAttention, Multivector2MultivectorAttention
):
    __doc__ = base.TiedMultivectorAttention.__doc__

    def __init__(
        self,
        n_dim,
        score_net,
        value_net,
        scale_net,
        reduce=True,
        merge_fun="mean",
        join_fun="mean",
        rank=2,
        invariant_mode="single",
        covariant_mode="partial",
        include_normalized_products=False,
        convex_covariants=False,
        **kwargs
    ):
        Multivector2MultivectorAttention.__init__(
            self,
            n_dim=n_dim,
            score_net=score_net,
            value_net=value_net,
            scale_net=scale_net,
            reduce=reduce,
            merge_fun=merge_fun,
            join_fun=join_fun,
            rank=rank,
            invariant_mode=invariant_mode,
            covariant_mode=covariant_mode,
            include_normalized_products=include_normalized_products,
            convex_covariants=convex_covariants,
            **kwargs
        )

        if type(self) == TiedMultivectorAttention:
            self.init()
