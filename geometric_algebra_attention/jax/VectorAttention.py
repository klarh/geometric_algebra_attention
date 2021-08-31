
import functools
import operator

import jax
import jax.numpy as jnp

from .. import base
from . import geometric_algebra

class VectorAttention(base.VectorAttention):
    algebra = geometric_algebra

    math = base.Namespace(
        all=jnp.all,
        any=jnp.any,
        asarray=jnp.asarray,
        concat=jnp.concatenate,
        logical_and=jnp.logical_and,
        pow=jnp.power,
        product=jnp.prod,
        reshape=jnp.reshape,
        shape=jnp.shape,
        softmax=jax.nn.softmax,
        sqrt=jnp.sqrt,
        sum=jnp.sum,
        tensordot=jnp.tensordot,
        where=jnp.where,
    )

    @property
    def layer_functions(self):
        return self.init_fun, type(self).apply_fun

    def init_fun(self, rng, input_shape):
        input_shapes = self._parse_inputs(input_shape)
        self.n_dim = input_shapes.values[-1]

        weight_sets = self._build_weight_definitions(self.n_dim)
        for (name, defs) in weight_sets.groups.items():
            weights = [
                rng.normal(stddev=def_.stdev, size=def_.shape)
                for def_ in defs]
            setattr(self, name, weights)

        for (name, def_) in weight_sets.singles.items():
            weight = rng.normal(stddev=def_.stdev, size=def_.shape)
            setattr(self, name, weight)

        return input_shape, self

    def apply_fun(self, inputs, rng=None):
        result = self._evaluate(inputs)
        return result.output

    def _calculate_attention(self, scores, values, old_shape):
        dims, reduce_axes = self._get_reduction()

        last_dim = functools.reduce(operator.mul, old_shape[dims:], 1)
        shape = old_shape[:dims] + (last_dim,)
        scores = self.math.reshape(scores, shape)
        attention = self.math.reshape(self.math.softmax(scores), old_shape)
        output = self.math.sum(attention*values, reduce_axes)

        return attention, output
