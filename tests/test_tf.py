
import functools
import unittest

import tensorflow as tf
from tensorflow import keras
from geometric_algebra_attention.tensorflow import (
    VectorAttention, Vector2VectorAttention, LabeledVectorAttention)

from test_internals import AllTests

class TensorflowTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single'):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        return VectorAttention(self.DIM, score, value, rank=rank, merge_fun=merge_fun,
                               join_fun=join_fun, invariant_mode=invar_mode)

    def value_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                         join_fun='mean', invar_mode='single'):
        net = self.get_value_layer(key, rank, merge_fun, join_fun, invar_mode)
        return net((r, v))

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None):
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

        return Vector2VectorAttention(self.DIM, score, value, scale)

    def vector_prediction(self, r, v, key=None):
        net = self.get_vector_layer(key)
        return net((r, v))

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

        return LabeledVectorAttention(self.DIM, score, value, scale)

    def label_vector_prediction(self, r, v, v2, key=None):
        net = self.get_label_vector_layer(key)
        return net((v2, (r, v)))

if __name__ == '__main__':
    unittest.main()
