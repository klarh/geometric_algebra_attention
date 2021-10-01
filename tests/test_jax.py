
import functools
import unittest

import jax
from jax.experimental.stax import serial, Dense, Relu
from geometric_algebra_attention.jax import (
    VectorAttention, Vector2VectorAttention, LabeledVectorAttention)

from test_internals import AllTests

class JaxTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(self, key=None):
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

        result_init, result_raw = VectorAttention(score, value).layer_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def value_prediction(self, r, v, key=None):
        rng = jax.random.PRNGKey(13)
        net = self.get_value_layer(key)
        return net((r, v))

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None):
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
            score, value, scale).layer_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def vector_prediction(self, r, v, key=None):
        rng = jax.random.PRNGKey(13)
        net = self.get_vector_layer(key)
        return net((r, v))

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
            score, value, scale).layer_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def label_vector_prediction(self, r, v, v2, key=None):
        rng = jax.random.PRNGKey(13)
        net = self.get_label_vector_layer(key)
        return net((v2, (r, v)))

if __name__ == '__main__':
    unittest.main()
