
import functools
import unittest

import tensorflow as tf
from tensorflow import keras
from geometric_algebra_attention.keras import (
    VectorAttention, Vector2VectorAttention, LabeledVectorAttention)

from test_internals import AllTests

class KerasTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(self, key=None):
        score = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(1)
        ])

        value = keras.models.Sequential([
            keras.layers.Dense(2*self.DIM, activation='relu'),
            keras.layers.Dense(self.DIM)
        ])

        return VectorAttention(score, value)

    @tf.function
    def value_prediction(self, r, v, key=None):
        net = self.get_value_layer(key)
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

        return Vector2VectorAttention(score, value, scale)

    @tf.function
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

        return LabeledVectorAttention(score, value, scale)

    @tf.function
    def label_vector_prediction(self, r, v, v2, key=None):
        net = self.get_label_vector_layer(key)
        return net((r, v, v2))

if __name__ == '__main__':
    unittest.main()
