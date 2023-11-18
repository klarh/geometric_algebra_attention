
import functools
import unittest

import jax
try:
    from jax.example_libraries.stax import serial, Dense, Relu
except ImportError:
    from jax.experimental.stax import serial, Dense, Relu
import numpy as np
from geometric_algebra_attention import jax as gala

from test_internals import AllTests, deferred_class

@deferred_class
class JaxTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', reduce=True, linear_mode='partial', linear_terms=0):
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

        result_init, result_raw = gala.VectorAttention(
            score, value, rank=rank, reduce=reduce, linear_mode=linear_mode,
            linear_terms=linear_terms, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def value_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', reduce=True,
            linear_mode='partial', linear_terms=0):
        net = self.get_value_layer(
            key, rank, merge_fun, join_fun, invar_mode, reduce,
            linear_mode, linear_terms)
        return np.asarray(net((r, v))).copy()

    @functools.lru_cache(maxsize=2)
    def get_value_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', reduce=True, linear_mode='partial', linear_terms=0):
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

        result_init, result_raw = gala.MultivectorAttention(
            score, value, rank=rank, reduce=reduce, linear_mode=linear_mode,
            linear_terms=linear_terms, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def value_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', reduce=True,
            linear_mode='partial', linear_terms=0):
        r = gala.Vector2Multivector()(r)
        net = self.get_value_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, reduce, linear_mode, linear_terms)
        return np.asarray(net((r, v))).copy()

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single', covar_mode='single',
                         include_normalized_products=False,
                         linear_mode='partial', linear_terms=0,
                         ):
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

        result_init, result_raw = gala.Vector2VectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode,
            covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        ).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def vector_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                          join_fun='mean', invar_mode='single',
                          covar_mode='single', include_normalized_products=False,
                          linear_mode='partial', linear_terms=0,
                          ):
        net = self.get_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        return np.asarray(net((r, v))).copy()

    @functools.lru_cache(maxsize=2)
    def get_vector_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
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

        result_init, result_raw = gala.Multivector2MultivectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode,
            covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        ).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def vector_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single',
            covar_mode='single', include_normalized_products=False,
            linear_mode='partial', linear_terms=0,
    ):
        r = gala.Vector2Multivector()(r)
        net = self.get_vector_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        return gala.Multivector2Vector()(np.asarray(net((r, v)))).copy()

    @functools.lru_cache(maxsize=2)
    def get_tied_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single', include_normalized_products=False,
            linear_mode='partial', linear_terms=0,
    ):
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

        result_init, result_raw = gala.TiedMultivectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode,
            covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        ).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def tied_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single',
            covar_mode='single', include_normalized_products=False,
            linear_mode='partial', linear_terms=0):
        r = gala.Vector2Multivector()(r)
        net = self.get_tied_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        result = list(net((r, v)))
        result[0] = gala.Multivector2Vector()(result[0])
        return tuple(np.array(arr) for arr in result)

    @functools.lru_cache(maxsize=2)
    def get_tied_vector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
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

        result_init, result_raw = gala.TiedVectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode,
            covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        ).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def tied_vector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single',
            covar_mode='single', include_normalized_products=False,
            linear_mode='partial', linear_terms=0):
        net = self.get_tied_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        return tuple(np.array(arr) for arr in net((r, v)))

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

        result_init, result_raw = gala.LabeledVectorAttention(
            score, value, scale).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def label_vector_prediction(self, r, v, v2, key=None):
        net = self.get_label_vector_layer(key)
        return np.asarray(net((v2, (r, v)))).copy()

    @functools.lru_cache(maxsize=2)
    def get_label_multivector_layer(self, key=None):
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

        result_init, result_raw = gala.LabeledMultivectorAttention(
            score, value, scale).stax_functions
        _, result_params = result_init(rng, (None, (self.DIM,)))
        return functools.partial(result_raw, result_params)

    def label_multivector_prediction(self, r, v, v2, key=None):
        net = self.get_label_multivector_layer(key)
        r = gala.Vector2Multivector()(r)
        return gala.Multivector2Vector()(np.asarray(net((v2, (r, v))))).copy()

if __name__ == '__main__':
    unittest.main()
