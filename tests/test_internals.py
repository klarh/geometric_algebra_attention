
import rowan
import numpy as np
from numpy import testing as npt
from hypothesis import assume, given, settings
from hypothesis import strategies as hs
from hypothesis.extra import numpy as hnp

DIM = 7

INVARIANT_MODES = ['full', 'partial', 'single']

LINEAR_MODES = ['full', 'partial']

MERGE_MODES = ['mean', 'concat']

finite_dtype = hnp.from_dtype(
    np.dtype('float32'), min_value=-1, max_value=1,
    allow_nan=False, allow_infinity=False)

weight_dtype = hnp.from_dtype(
    np.dtype('float32'), min_value=np.float32(1e-3), max_value=10,
    allow_nan=False, allow_infinity=False)

@hs.composite
def bond_weights(draw, shape):
    w = draw(hnp.arrays(np.float32, shape, elements=weight_dtype))
    w /= np.sum(w)
    return w

@hs.composite
def point_cloud(draw, weights=False):
    N = draw(hs.integers(min_value=4, max_value=12))
    r = draw(hnp.arrays(np.float32, (N, 3), elements=finite_dtype))
    v = draw(hnp.arrays(np.float32, (N, DIM), elements=finite_dtype))

    result = [r, v]

    if weights:
        w = draw(bond_weights((N,)))
        result.append(w)

    return tuple(result)

@hs.composite
def unit_quaternions(draw):
    result = draw(hnp.arrays(np.float32, (4,), elements=finite_dtype))
    assume(np.sum(np.square(result)) > 1e-5)
    return result/np.linalg.norm(result)

class AllTests:
    DIM = DIM

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_rotation_invariance_value(self, q, rv, rank, merge_fun, join_fun,
                                       invar_mode, linear_mode, linear_terms):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_invariance'
        prediction1 = self.value_prediction(
            r, v, key, rank, merge_fun, join_fun,
            invar_mode, linear_mode=linear_mode, linear_terms=linear_terms)
        prediction2 = self.value_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode,
            linear_mode=linear_mode, linear_terms=linear_terms)

        self.assertEqual(v[0].shape, prediction1.shape)
        err = np.max(np.square(prediction1 - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        hs.integers(0, 128),
        hs.integers(0, 128),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_permutation_equivariance_value(self, swap_i, swap_j, rank, merge_fun,
                                            join_fun, invar_mode, linear_mode,
                                            linear_terms):
        np.random.seed(13)
        r = np.random.normal(size=(7, 3)).astype(np.float32)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        v = np.zeros((r.shape[0], self.DIM), dtype=np.float32)
        v[:, 0] = np.arange(len(r))

        swap_i = swap_i%len(r)
        swap_j = swap_j%len(r)
        rprime, vprime = r.copy(), v.copy()
        rprime[swap_i], rprime[swap_j] = r[swap_j], r[swap_i]
        vprime[swap_i], vprime[swap_j] = v[swap_j], v[swap_i]

        key = 'permutation_equivariance'
        prediction1 = self.value_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, reduce=False,
            linear_mode=linear_mode, linear_terms=linear_terms)
        prediction2 = self.value_prediction(
            rprime, vprime, key, rank, merge_fun, join_fun, invar_mode,
            reduce=False, linear_mode=linear_mode, linear_terms=linear_terms)

        self.assertEqual(v.shape, prediction1.shape)
        temp = prediction2[swap_i].copy()
        prediction2[swap_i] = prediction2[swap_j]
        prediction2[swap_j] = temp
        err = np.max(np.square(prediction1 - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_rotation_invariance_multivector_value(
            self, q, rv, rank, merge_fun,
            join_fun, invar_mode, linear_mode, linear_terms):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_invariance'
        prediction1 = self.value_multivector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode,
            linear_mode=linear_mode, linear_terms=linear_terms)
        prediction2 = self.value_multivector_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode,
            linear_mode=linear_mode, linear_terms=linear_terms)

        self.assertEqual(v[0].shape, prediction1.shape)
        err = np.max(np.square(prediction1 - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        hs.integers(0, 128),
        hs.integers(0, 128),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_permutation_equivariance_multivector_value(
            self, swap_i, swap_j, rank, merge_fun, join_fun, invar_mode,
            linear_mode, linear_terms):
        np.random.seed(13)
        r = np.random.normal(size=(7, 3)).astype(np.float32)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        v = np.zeros((r.shape[0], self.DIM), dtype=np.float32)
        v[:, 0] = np.arange(len(r))

        swap_i = swap_i%len(r)
        swap_j = swap_j%len(r)
        rprime, vprime = r.copy(), v.copy()
        rprime[swap_i], rprime[swap_j] = r[swap_j], r[swap_i]
        vprime[swap_i], vprime[swap_j] = v[swap_j], v[swap_i]

        key = 'permutation_equivariance'
        prediction1 = self.value_multivector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, reduce=False,
            linear_mode=linear_mode, linear_terms=linear_terms)
        prediction2 = self.value_multivector_prediction(
            rprime, vprime, key, rank, merge_fun, join_fun, invar_mode,
            reduce=False, linear_mode=linear_mode, linear_terms=linear_terms)

        self.assertEqual(v.shape, prediction1.shape)
        temp = prediction2[swap_i].copy()
        prediction2[swap_i] = prediction2[swap_j]
        prediction2[swap_j] = temp
        err = np.max(np.square(prediction1 - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.booleans(),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_rotation_covariance_vector(
            self, q, rv, rank, merge_fun, join_fun,
            invar_mode, covar_mode, include_normalized_products,
            linear_mode, linear_terms):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_covariance'
        prediction1 = self.vector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode=linear_mode,
            linear_terms=linear_terms)
        prediction1_prime = rowan.rotate(q, prediction1)
        prediction2 = self.vector_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode=linear_mode,
            linear_terms=linear_terms)

        err = np.max(np.square(prediction1_prime - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.booleans(),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_rotation_covariance_multivector(
            self, q, rv, rank, merge_fun, join_fun,
            invar_mode, covar_mode, include_normalized_products,
            linear_mode, linear_terms):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_covariance'
        prediction1 = self.vector_multivector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode=linear_mode,
            linear_terms=linear_terms)
        prediction1_prime = rowan.rotate(q, prediction1)
        prediction2 = self.vector_multivector_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode=linear_mode,
            linear_terms=linear_terms)

        err = np.max(np.square(prediction1_prime - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        point_cloud())
    def test_rotation_covariance_label_vector(self, q, rv, rv2):
        r, v = rv
        v2 = rv2[1]
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_covariance_label'
        prediction1 = self.label_vector_prediction(r, v, v2, key)
        prediction1_prime = rowan.rotate(q[None], prediction1)
        prediction2 = self.label_vector_prediction(rprime, v, v2, key)

        err = np.max(np.square(prediction1_prime - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        point_cloud())
    def test_rotation_covariance_label_multivector(self, q, rv, rv2):
        r, v = rv
        v2 = rv2[1]
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_covariance_label'
        prediction1 = self.label_multivector_prediction(r, v, v2, key)
        prediction1_prime = rowan.rotate(q[None], prediction1)
        prediction2 = self.label_multivector_prediction(rprime, v, v2, key)

        err = np.max(np.square(prediction1_prime - prediction2))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.booleans(),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_rotation_equivariance_tied_vector(
            self, q, rv, rank, merge_fun, join_fun,
            invar_mode, covar_mode, include_normalized_products,
            linear_mode, linear_terms):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_equivariance_tied_vector'
        (pred1_covar, pred1_invar) = self.tied_vector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        pred1_prime = rowan.rotate(q, pred1_covar)
        (pred2_covar, pred2_invar) = self.tied_vector_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)

        err = np.max(np.square(pred1_prime - pred2_covar))
        self.assertLess(err, 1e-5)

        err = np.max(np.square(pred1_invar - pred2_invar))
        self.assertLess(err, 1e-5)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.booleans(),
        hs.sampled_from(LINEAR_MODES),
        hs.integers(0, 4),
    )
    def test_rotation_equivariance_tied_multivector(
            self, q, rv, rank, merge_fun, join_fun,
            invar_mode, covar_mode, include_normalized_products,
            linear_mode, linear_terms):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_equivariance_tied_multivector'
        (pred1_covar, pred1_invar) = self.tied_multivector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)
        pred1_prime = rowan.rotate(q, pred1_covar)
        (pred2_covar, pred2_invar) = self.tied_multivector_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode,
            include_normalized_products, linear_mode, linear_terms)

        err = np.max(np.square(pred1_prime - pred2_covar))
        self.assertLess(err, 1e-5)

        err = np.max(np.square(pred1_invar - pred2_invar))
        self.assertLess(err, 1e-5)

class TFRandom:
    @staticmethod
    def seed(seed):
        import tensorflow as tf
        tf.random.set_seed(seed)

    @staticmethod
    def getstate():
        import tensorflow as tf
        gen = tf.random.get_global_generator()
        return gen.state, gen.algorithm

    @staticmethod
    def setstate(state):
        import tensorflow as tf
        gen = tf.random.Generator.from_state(*state)
        tf.random.set_global_generator(gen)
