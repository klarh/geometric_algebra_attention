
from .. import base
from .VectorAttention import VectorAttention

class Vector2VectorAttention(base.Vector2VectorAttention, VectorAttention):
    def __init__(self, score_net, value_net, scale_net, reduce=True,
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
            self, score_net=score_net, value_net=value_net,
            reduce=reduce, merge_fun=merge_fun, join_fun=join_fun, rank=rank,
            **kwargs)

    def init_fun(self, rng, input_shape):
        super().init_fun(rng, input_shape)

        rng, next_rng = jax.random.split(rng)
        _, self.value_net_params = self.value_net_init(next_rng, (self.n_dim,))

        return input_shape, self.params

    @property
    def params(self):
        result = super().params
        result['scale_net_params'] = self.scale_net_params
        return result

    @property
    def scale_net(self):
        return functools.partial(self.scale_net_fn, self.scale_net_params)

    @scale_net.setter
    def scale_net(self, value):
        self.scale_net_init, self.scale_net_fn = value
