
import functools
import unittest

import hypothesis
import torch as pt
from geometric_algebra_attention.pytorch import (
    MultivectorAttention, VectorAttention, Vector2Multivector,
    Vector2VectorAttention, LabeledVectorAttention)

from test_internals import AllTests

@hypothesis.register_random
class TorchRandom:
    @staticmethod
    def seed(seed):
        pt.random.manual_seed(seed)

    @staticmethod
    def getstate():
        return pt.random.get_rng_state()

    @staticmethod
    def setstate(state):
        pt.random.set_rng_state(state)

class PytorchTests(AllTests, unittest.TestCase):
    @functools.lru_cache(maxsize=2)
    def get_value_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single', reduce=True):
        score = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        invar_dims = VectorAttention.get_invariant_dims(rank, invar_mode)
        value = pt.nn.Sequential(
            pt.nn.Linear(invar_dims, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, self.DIM)
        )

        return VectorAttention(self.DIM, score, value, rank=rank, merge_fun=merge_fun,
                               join_fun=join_fun, invariant_mode=invar_mode, reduce=reduce)

    def value_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                         join_fun='mean', invar_mode='single', reduce=True):
        r, v = map(pt.as_tensor, (r, v))
        net = self.get_value_layer(key, rank, merge_fun, join_fun, invar_mode, reduce)
        return net.forward((r, v)).detach().numpy()

    @functools.lru_cache(maxsize=2)
    def get_value_multivector_layer(
            self, key=None, rank=2, merge_fun='mean', join_fun='mean',
            invar_mode='single', reduce=True):
        score = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        invar_dims = MultivectorAttention.get_invariant_dims(rank, invar_mode)
        value = pt.nn.Sequential(
            pt.nn.Linear(invar_dims, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, self.DIM)
        )

        return MultivectorAttention(
            self.DIM, score, value, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, reduce=reduce)

    def value_multivector_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                         join_fun='mean', invar_mode='single', reduce=True):
        r, v = map(pt.as_tensor, (r, v))
        r = Vector2Multivector().forward(r)
        net = self.get_value_multivector_layer(
            key, rank, merge_fun, join_fun, invar_mode, reduce)
        return net.forward((r, v)).detach().numpy()

    @functools.lru_cache(maxsize=2)
    def get_vector_layer(self, key=None, rank=2, merge_fun='mean', join_fun='mean',
                         invar_mode='single', covar_mode='single'):
        score = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        invar_dims = VectorAttention.get_invariant_dims(rank, invar_mode)
        value = pt.nn.Sequential(
            pt.nn.Linear(invar_dims, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, self.DIM)
        )

        scale = pt.nn.Sequential(
            pt.nn.Linear(self.DIM, 2*self.DIM),
            pt.nn.ReLU(),
            pt.nn.Linear(2*self.DIM, 1)
        )

        return Vector2VectorAttention(
            self.DIM, score, value, scale, rank=rank, merge_fun=merge_fun,
            join_fun=join_fun, invariant_mode=invar_mode, covariant_mode=covar_mode)

    def vector_prediction(self, r, v, key=None, rank=2, merge_fun='mean',
                          join_fun='mean', invar_mode='single', covar_mode='single'):
        r, v = map(pt.as_tensor, (r, v))
        net = self.get_vector_layer(
            key, rank, merge_fun, join_fun, invar_mode, covar_mode)
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
        return net.forward((v2, (r, v))).detach().numpy()

if __name__ == '__main__':
    unittest.main()
