
import functools
import operator

import jax
import jax.numpy as jnp

from .. import base
from . import geometric_algebra

class AttentionBase:
    algebra = geometric_algebra

    math = base.Namespace(
        all=jnp.all,
        any=jnp.any,
        asarray=jnp.asarray,
        bool_to_int=lambda x: x.astype(jnp.int8),
        clip=jnp.clip,
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
        zeros_like=jnp.zeros_like,
    )

    @property
    def stax_functions(self):
        """Returns the operations of this layer in the API of stax."""
        return self.stax_init, self.stax_apply

    def stax_init(self, rng, input_shape):
        """Initialize the parameters of this layer."""
        v_shape = input_shape[1]
        self.n_dim = v_shape[-1]

        rng, next_rng = jax.random.split(rng)
        _, self.score_net_params = self.score_net_init(next_rng, (self.n_dim,))

        rng, next_rng = jax.random.split(rng)
        _, self.value_net_params = self.value_net_init(next_rng, (self.invariant_dims,))

        def rngs(rng):
            next_rng = rng
            while True:
                rng, next_rng = jax.random.split(next_rng)
                yield rng

        rng_groups, rng_singles = jax.random.split(rng)

        weight_sets = self._build_weight_definitions(self.n_dim)
        for (name, defs) in weight_sets.groups.items():
            weights = [
                jax.random.normal(rng, shape=def_.shape)*def_.stdev
                for (def_, rng) in zip(defs, rngs(rng))]
            setattr(self, name, weights)

        for ((name, def_), rng) in zip(weight_sets.singles.items(), rngs(rng_singles)):
            weight = jax.random.normal(rng, shape=def_.shape)*def_.stdev
            setattr(self, name, weight)

        return v_shape, self.params

    def stax_apply(self, params, inputs, rng=None):
        """Apply the operation of this layer, given a set of layer parameters."""
        self.params = params
        self._last_rng = rng
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

    @property
    def params(self):
        """Get and set the parameters of this layer."""
        result = {}
        weight_sets = self._build_weight_definitions(self.n_dim)
        for name in weight_sets.groups:
            result[name] = getattr(self, name)
        for name in weight_sets.singles:
            result[name] = getattr(self, name)
        result['score_net_params'] = self.score_net_params
        result['value_net_params'] = self.value_net_params
        return result

    @params.setter
    def params(self, values):
        for (name, val) in values.items():
            setattr(self, name, val)

    @property
    def score_net(self):
        """Get and set the parameters for the score-generating function.

        See the main jax module documentation for more details about
        these functions.
        """
        kwargs = {}
        if getattr(self, '_last_rng', None) is not None:
            mixin = functools.reduce(operator.mul, map(int, 'score'.encode()))
            kwargs['rng'] = jax.random.fold_in(self._last_rng, mixin)
        return functools.partial(self.score_net_fn, self.score_net_params, **kwargs)

    @score_net.setter
    def score_net(self, value):
        self.score_net_init, self.score_net_fn = value

    @property
    def value_net(self):
        """Get and set the parameters for the value-generating function.

        See the main jax module documentation for more details about
        these functions.
        """
        kwargs = {}
        if getattr(self, '_last_rng', None) is not None:
            mixin = functools.reduce(operator.mul, map(int, 'value'.encode()))
            kwargs['rng'] = jax.random.fold_in(self._last_rng, mixin)
        return functools.partial(self.value_net_fn, self.value_net_params, **kwargs)

    @value_net.setter
    def value_net(self, value):
        self.value_net_init, self.value_net_fn = value
