
import functools
import unittest

import hypothesis
import tensorflow as tf
from tensorflow import keras
from geometric_algebra_attention import tensorflow as gala

from test_internals import deferred_class, AllTests, TFRandom

hypothesis.register_random(TFRandom)

@deferred_class
class TensorflowTests(AllTests, unittest.TestCase):
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
            self.DIM, score, value, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, reduce=reduce,
            linear_mode=linear_mode, linear_terms=linear_terms)

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
            invar_mode='single', reduce=True, linear_mode='partial', linear_terms=0):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        return gala.MultivectorAttention(
            self.DIM, score, value, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, reduce=reduce,
            linear_mode=linear_mode, linear_terms=linear_terms)

    def value_multivector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', reduce=True,
            linear_mode='partial', linear_terms=0):
        r = gala.Vector2Multivector()(r)
        net = self.get_value_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, reduce,
            linear_mode, linear_terms)
        return net((r, v)).numpy()

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single', covar_mode='single',
                         include_normalized_products=False,
                         linear_mode='partial', linear_terms=0):
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
            self.DIM, score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        )

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
            self.DIM, score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        )

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
            self.DIM, score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        )

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
            self.DIM, score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode,
            include_normalized_products=include_normalized_products,
            linear_mode=linear_mode, linear_terms=linear_terms,
        )

    def tied_vector_prediction(
            self, r, v, key=None, rank=2, merge_fun='mean',
            join_fun='mean', invar_mode='single', covar_mode='single',
            include_normalized_products=False, linear_mode='partial', linear_terms=0):
        net = self.get_tied_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
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

        return gala.LabeledVectorAttention(self.DIM, score, value, scale)

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

        return gala.LabeledMultivectorAttention(self.DIM, score, value, scale)

    def label_multivector_prediction(self, r, v, v2, key=None):
        r = gala.Vector2Multivector()(r)
        net = self.get_label_multivector_layer(key)
        return gala.Multivector2Vector()(net((v2, (r, v)))).numpy()

if __name__ == '__main__':
    unittest.main()
