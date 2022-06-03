import functools
import operator

import jax

from .. import base
from .VectorAttention import VectorAttention

class Vector2VectorAttention(base.Vector2VectorAttention, VectorAttention):
    __doc__ = base.Vector2VectorAttention.__doc__

    def __init__(self, score_net, value_net, scale_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='partial',
                 include_normalized_products=False,
                 convex_covariants=False, **kwargs):
        self.scale_net_params = self.scale_net_fn = None
        base.Vector2VectorAttention.__init__(
            self, scale_net=scale_net, convex_covariants=convex_covariants)
        VectorAttention.__init__(
            self, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            invariant_mode=invariant_mode, covariant_mode=covariant_mode,
            include_normalized_products=include_normalized_products,
            **kwargs)

    def stax_init(self, rng, input_shape):
        super().stax_init(rng, input_shape)

        rng, next_rng = jax.random.split(rng)
        _, self.scale_net_params = self.scale_net_init(next_rng, (self.n_dim,))

        return input_shape, self.params

    @property
    def params(self):
        result = super().params
        result['scale_net_params'] = self.scale_net_params
        return result

    @params.setter
    def params(self, values):
        for (name, val) in values.items():
            setattr(self, name, val)

    @property
    def scale_net(self):
        """Get and set the parameters for the scale-generating function.

        See the main jax module documentation for more details about
        these functions.
        """
        kwargs = {}
        if getattr(self, '_last_rng', None) is not None:
            mixin = functools.reduce(operator.mul, map(int, 'scale'.encode()))
            kwargs['rng'] = jax.random.fold_in(self._last_rng, mixin)
        return functools.partial(self.scale_net_fn, self.scale_net_params, **kwargs)

    @scale_net.setter
    def scale_net(self, value):
        self.scale_net_init, self.scale_net_fn = value
