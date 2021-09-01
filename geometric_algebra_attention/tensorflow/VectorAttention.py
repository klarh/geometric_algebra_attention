import tensorflow as tf

from .. import base
from . import geometric_algebra

class VectorAttention(base.VectorAttention):
    algebra = geometric_algebra

    math = base.Namespace(
        all=tf.reduce_all,
        any=tf.reduce_any,
        asarray=tf.convert_to_tensor,
        concat=tf.concat,
        logical_and=tf.logical_and,
        pow=tf.pow,
        product=tf.reduce_prod,
        reshape=tf.reshape,
        shape=tf.shape,
        softmax=tf.nn.softmax,
        sqrt=tf.sqrt,
        sum=tf.reduce_sum,
        tensordot=tf.tensordot,
        where=tf.where,
    )

    def __init__(self, n_dim, *args, **kwargs):
        self.n_dim = n_dim

        super().__init__(*args, **kwargs)

        weight_sets = self._build_weight_definitions(n_dim)
        for (name, defs) in weight_sets.groups.items():
            weights = [tf.Variable(
                tf.random.normal(def_.shape, stddev=def_.stdev), name=def_.name,
                trainable=True)
                       for def_ in defs]
            setattr(self, name, weights)

        for (name, def_) in weight_sets.singles.items():
            weight = tf.Variable(
                tf.random.normal(def_.shape, stddev=def_.stdev), name=def_.name,
                trainable=True)
            setattr(self, name, weight)

    def __call__(self, inputs, return_attention=False):
        intermediates = self._evaluate(inputs)
        result = [intermediates.output]

        if return_attention:
            result.append(intermediates.attention)

        if len(result) == 1:
            return result[0]

        return tuple(result)