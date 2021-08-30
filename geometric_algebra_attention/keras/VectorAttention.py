
from tensorflow import keras

from .. import base
from ..tensorflow.VectorAttention import VectorAttention as TFAttention

class VectorAttention(base.VectorAttention, keras.layers.Layer):
    algebra = TFAttention.algebra

    math = TFAttention.math

    def __init__(self, score_net, value_net, reduce=True,
                 merge_fun='mean', join_fun='mean', rank=2, **kwargs):
        keras.layers.Layer.__init__(self, **kwargs)
        base.VectorAttention.__init__(
            self, score_net, value_net, reduce, merge_fun, join_fun, rank)

    def build(self, input_shape):
        n_dim = input_shape[1][-1]
        weight_sets = self._build_weight_definitions(n_dim)
        for (name, defs) in weight_sets.groups.items():
            weights = [
                self.add_weight(
                    name=def_.name,
                    initializer=keras.initializers.RandomNormal(stddev=def_.stdev),
                    shape=def_.shape)
                for def_ in defs]
            setattr(self, name, weights)

        for (name, def_) in weight_sets.singles.items():
            weight = self.add_weight(
                name=def_.name,
                initializer=keras.initializers.RandomNormal(stddev=def_.stdev),
                shape=def_.shape)
            setattr(self, name, weight)

    def call(self, inputs, return_invariants=False, return_attention=False, mask=None):
        intermediates = self._evaluate(inputs, mask)
        result = [intermediates.output]
        if return_invariants:
            result.append(intermediates.invariants)
        if return_attention:
            result.append(intermediates.attention)

        if len(result) > 1:
            return tuple(result)
        else:
            return result[0]

    def compute_mask(self, inputs, mask=None):
        if not self.reduce or mask is None:
            return mask

        parsed_mask = self._parse_inputs(mask)
        position_mask = parsed_mask.positions
        value_mask = parsed_mask.values
        if position_mask is not None:
            return self.math.any(position_mask, axis=-1)
        else:
            return self.math.any(value_mask, axis=-1)
