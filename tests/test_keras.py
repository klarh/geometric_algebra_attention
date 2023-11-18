
import functools
import unittest

import hypothesis
from hypothesis.extra import numpy as hnp
import numpy as np
import numpy.testing as npt
import tensorflow as tf
from tensorflow import keras
from geometric_algebra_attention import keras as gala

from test_internals import AllTests, TFRandom, deferred_class, finite_dtype, point_cloud

hypothesis.register_random(TFRandom)

@deferred_class
class KerasTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', reduce=True, linear_mode='partial', linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        return gala.VectorAttention(
            score, value, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode,
            reduce=reduce, linear_mode=linear_mode, linear_terms=linear_terms)

    def value_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', reduce=True,
            linear_mode='partial', linear_terms=0):
        net = self.get_value_layer(
            key, rank, merge_fun, join_fun, invar_mode, reduce,
            linear_mode, linear_terms)
        return net((r, v)).numpy()

    @functools.lru_cache(maxsize=2)
    def get_value_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', reduce=True, linear_mode='partial',
            linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        return gala.MultivectorAttention(
            score, value, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, reduce=reduce,
            linear_mode=linear_mode, linear_terms=linear_terms)

    def value_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', reduce=True,
            linear_mode='partial', linear_terms=0):
        r = gala.Vector2Multivector()(r)
        net = self.get_value_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, reduce, linear_mode, linear_terms)
        return net((r, v)).numpy()

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        scale = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        return gala.Vector2VectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products, linear_mode=linear_mode, linear_terms=linear_terms)

    def vector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        net = self.get_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        return net((r, v)).numpy()

    @functools.lru_cache(maxsize=2)
    def get_vector_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        scale = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        return gala.Multivector2MultivectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms)

    def vector_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        r = gala.Vector2Multivector()(r)
        net = self.get_vector_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        return gala.Multivector2Vector()(net((r, v))).numpy()

    @functools.lru_cache(maxsize=2)
    def get_tied_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        scale = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        return gala.TiedMultivectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms)

    def tied_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        r = gala.Vector2Multivector()(r)
        net = self.get_tied_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        result = list(net((r, v)))
        result[0] = gala.Multivector2Vector()(result[0])
        return tuple(arr.numpy() for arr in result)

    @functools.lru_cache(maxsize=2)
    def get_tied_vector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        scale = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        return gala.TiedVectorAttention(
            score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms)

    def tied_vector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        net = self.get_tied_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode=linear_mode,
            linear_terms=linear_terms)
        return tuple(arr.numpy() for arr in net((r, v)))

    @functools.lru_cache(maxsize=2)
    def get_label_vector_layer(self, key=None):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        scale = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        return gala.LabeledVectorAttention(score, value, scale)

    def label_vector_prediction(self, r, v, v2, key=None):
        net = self.get_label_vector_layer(key)
        return net((v2, (r, v))).numpy()

    @functools.lru_cache(maxsize=2)
    def get_label_multivector_layer(self, key=None):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        scale = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        return gala.LabeledMultivectorAttention(score, value, scale)

    def label_multivector_prediction(self, r, v, v2, key=None):
        r = gala.Vector2Multivector()(r)
        net = self.get_label_multivector_layer(key)
        return gala.Multivector2Vector()(net((v2, (r, v)))).numpy()

    @hypothesis.given(
        hnp.arrays(np.float32, hnp.array_shapes(min_dims=2), elements=finite_dtype))
    def basic_momentum(self, x):
        hypothesis.assume(np.all(np.abs(x) > 1e-3))
        hypothesis.assume(len(np.unique(np.round(x, 3))) > 1)
        hypothesis.assume(x[..., 0].size > 1)

        layer = gala.MomentumNormalization(momentum=.1)
        mean = lambda arr: np.mean(arr, axis=tuple(range(0, arr.ndim - 1)))
        std = lambda arr: np.std(arr, axis=tuple(range(0, arr.ndim - 1)))
        hypothesis.assume(np.all(std(x) > 1e-3))

        @tf.function
        def f(x):
            return layer(x, training=True)

        for _ in range(32):
            output = f(x)

        npt.assert_allclose(mean(output), 0., rtol=1e-2, atol=1e-2)
        npt.assert_allclose(std(output), 1., rtol=1e-2, atol=1e-2)

    @hypothesis.given(
        hnp.arrays(np.float32, hnp.array_shapes(min_dims=2), elements=finite_dtype))
    def basic_momentum_layer(self, x):
        hypothesis.assume(np.all(np.abs(x) > 1e-3))
        hypothesis.assume(len(np.unique(np.round(x, 3))) > 1)
        hypothesis.assume(x[..., 0].size > 1)

        layer = gala.MomentumLayerNormalization(momentum=.1)
        norm = lambda arr: np.mean(np.linalg.norm(arr, axis=-1))

        @tf.function
        def f(x):
            return layer(x, training=True)

        for _ in range(32):
            output = f(x)

        npt.assert_allclose(norm(output), 1., rtol=1e-2, atol=1e-2)

    @hypothesis.given(point_cloud(weights=True))
    def basic_mask(self, cloud):
        (r, v, w) = cloud
        mask = np.argsort(w) > 1

        layer = self.get_value_layer('basic_mask', reduce=False)
        first_result = layer((r, v), mask=mask).numpy()
        r[~mask] += 1
        v[~mask] += 1
        second_result = layer((r, v), mask=mask).numpy()
        second_result[~mask] = first_result[~mask]
        npt.assert_allclose(first_result, second_result)

if __name__ == '__main__':
    unittest.main()
