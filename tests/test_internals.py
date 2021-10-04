
import rowan
import numpy as np
from numpy import testing as npt
from hypothesis import assume, given, settings
from hypothesis import strategies as hs
from hypothesis.extra import numpy as hnp

DIM = 8

INVARIANT_MODES = ['full', 'partial', 'single']

MERGE_MODES = ['mean', 'concat']

finite_dtype = hnp.from_dtype(
    np.dtype('float32'), min_value=-4, max_value=4,
    allow_nan=False, allow_infinity=False)

weight_dtype = hnp.from_dtype(
    np.dtype('float32'), min_value=0, max_value=10,
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
        hs.sampled_from(INVARIANT_MODES))
    def test_rotation_invariance_value(self, q, rv, rank, merge_fun, join_fun, invar_mode):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_invariance'
        prediction1 = self.value_prediction(r, v, key, rank, merge_fun, join_fun, invar_mode)
        prediction2 = self.value_prediction(rprime, v, key, rank, merge_fun, join_fun, invar_mode)

        delta = max(1e-3, np.mean(np.square(v)))
        npt.assert_allclose(prediction1 - prediction2, 0, atol=1e-2*delta)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        hs.integers(1, 3),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(MERGE_MODES),
        hs.sampled_from(INVARIANT_MODES),
        hs.sampled_from(INVARIANT_MODES))
    def test_rotation_covariance_vector(self, q, rv, rank, merge_fun, join_fun,
                                        invar_mode, covar_mode):
        r, v = rv
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_covariance'
        prediction1 = self.vector_prediction(
            r, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode)
        prediction1_prime = rowan.rotate(q[None], prediction1)
        prediction2 = self.vector_prediction(
            rprime, v, key, rank, merge_fun, join_fun, invar_mode, covar_mode)

        delta = max(1e-3, np.mean(np.square(r)))
        npt.assert_allclose(prediction1_prime - prediction2, 0, atol=1e-2*delta)

    @settings(deadline=None)
    @given(
        unit_quaternions(),
        point_cloud(),
        point_cloud())
    def test_rotation_covariance_label(self, q, rv, rv2):
        r, v = rv
        v2 = rv2[1]
        rprime = rowan.rotate(q[None], r).astype(np.float32)

        key = 'rotation_covariance_label'
        prediction1 = self.label_vector_prediction(r, v, v2, key)
        prediction1_prime = rowan.rotate(q[None], prediction1)
        prediction2 = self.label_vector_prediction(rprime, v, v2, key)

        npt.assert_allclose(prediction1_prime - prediction2, 0, atol=1e-4)
