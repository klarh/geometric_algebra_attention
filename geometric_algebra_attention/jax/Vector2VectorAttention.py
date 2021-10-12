import functools

import jax

from .. import base
from .VectorAttention import VectorAttention

class Vector2VectorAttention(base.Vector2VectorAttention, VectorAttention):
    def __init__(self, score_net, value_net, scale_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2,
                 invariant_mode='single', covariant_mode='partial',
                 **kwargs):
        self.scale_net_params = self.scale_net_fn = None
        base.Vector2VectorAttention.__init__(self, scale_net=scale_net)
        VectorAttention.__init__(
            self, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            invariant_mode=invariant_mode, covariant_mode=covariant_mode,
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
        return functools.partial(self.scale_net_fn, self.scale_net_params)

    @scale_net.setter
    def scale_net(self, value):
        self.scale_net_init, self.scale_net_fn = value
