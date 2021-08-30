
import functools
import unittest

import torch as pt
from geometric_algebra_attention.pytorch import (
    VectorAttention, Vector2VectorAttention, LabeledVectorAttention)

from test_internals import AllTests

class PytorchTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(self, key=None):
        score = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        value = pt.nn.Sequential(
            pt.nn.Linear(2, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, self.DIM)
        )

        return VectorAttention(self.DIM, score, value)

    def value_prediction(self, r, v, key=None):
        r, v = map(pt.as_tensor, (r, v))
        net = self.get_value_layer(key)
        return net.forward((r, v)).detach().numpy()

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None):
        score = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        value = pt.nn.Sequential(
            pt.nn.Linear(2, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, self.DIM)
        )

        scale = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        return Vector2VectorAttention(self.DIM, score, value, scale)

    def vector_prediction(self, r, v, key=None):
        r, v = map(pt.as_tensor, (r, v))
        net = self.get_vector_layer(key)
        return net.forward((r, v)).detach().numpy()

    @functools.lru_cache(maxsize=2)
    def get_label_vector_layer(self, key=None):
        score = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        value = pt.nn.Sequential(
            pt.nn.Linear(2, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, self.DIM)
        )

        scale = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        return LabeledVectorAttention(self.DIM, score, value, scale)

    def label_vector_prediction(self, r, v, v2, key=None):
        r, v, v2 = map(pt.as_tensor, (r, v, v2))
        net = self.get_label_vector_layer(key)
        return net.forward((r, v, v2)).detach().numpy()

if __name__ == '__main__':
    unittest.main()
