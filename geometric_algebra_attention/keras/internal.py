
from tensorflow import keras

from ..tensorflow.VectorAttention import VectorAttention as TFAttention

class AttentionBase:
    algebra = TFAttention.algebra

    math = TFAttention.math

    def build(self, input_shape):
        """Store the shape of weights given the shape of a set of input values."""
        # scenario 1: inputs are [(x1, v1), (x2, v2, w2), ...]
        try:
            n_dim = input_shape[0][1][-1]
        except TypeError:
        # scenario 2: inputs are (x, v) or (x, v, w)
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
        """Evaluate the geometric algebra attention calculation for this layer."""
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
        """Calculate the output mask of this layer given input shapes and masks."""
        if not self.reduce or mask is None:
            return mask

        parsed_mask = self._parse_inputs(mask)
        position_mask = parsed_mask[0].positions
        value_mask = parsed_mask[0].values
        if position_mask is not None:
            return self.math.any(position_mask, axis=-1)
        else:
            return self.math.any(value_mask, axis=-1)

    @classmethod
    def from_config(cls, config):
        new_config = dict(config)
        for key in ('score_net', 'value_net'):
            new_config[key] = keras.models.Sequential.from_config(new_config[key])
        return cls(**new_config)

    def get_config(self):
        result = super().get_config()
        result['score_net'] = self.score_net.get_config()
        result['value_net'] = self.value_net.get_config()
        result['reduce'] = self.reduce
        result['merge_fun'] = self.merge_fun
        result['join_fun'] = self.join_fun
        result['rank'] = self.rank
        result['invariant_mode'] = self.invariant_mode
        result['covariant_mode'] = self.covariant_mode
        result['include_normalized_products'] = self.include_normalized_products
        return result
