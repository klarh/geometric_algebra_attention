
import functools
import unittest

import jax
from jax.experimental.stax import serial, Dense, Relu
import numpy as np
from geometric_algebra_attention.jax import (
    VectorAttention, Vector2VectorAttention, LabeledVectorAttention)

from test_internals import AllTests

class JaxTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single', reduce=True):
        rng = jax.random.PRNGKey(13)
        score = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(1)
            )

        value = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(self.DIM)
            )

        result_init, result_raw = VectorAttention(
            score, value, rank=rank, reduce=reduce, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def value_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                         join_fun='mean', invar_mode='single', reduce=True):
        net = self.get_value_layer(key, rank, merge_fun, join_fun, invar_mode, reduce)
        return np.asarray(net((r, v))).copy()

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single', covar_mode='single'):
        rng = jax.random.PRNGKey(13)
        score = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(1)
            )

        value = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(self.DIM)
            )

        scale = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(1)
            )

        result_init, result_raw = Vector2VectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode,
            covariant_mode=covar_mode).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def vector_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                          join_fun='mean', invar_mode='single',
                          covar_mode='single'):
        net = self.get_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode)
        return np.asarray(net((r, v))).copy()

    @functools.lru_cache(maxsize=2)
    def get_label_vector_layer(self, key=None):
        rng = jax.random.PRNGKey(13)
        score = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(1)
            )

        value = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(self.DIM)
            )

        scale = serial(
            Dense(2*self.DIM),
            Relu,
            Dense(1)
            )

        result_init, result_raw = LabeledVectorAttention(
            score, value, scale).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def label_vector_prediction(self, r, v, v2, key=None):
        net = self.get_label_vector_layer(key)
        return np.asarray(net((v2, (r, v)))).copy()

if __name__ == '__main__':
    unittest.main()
