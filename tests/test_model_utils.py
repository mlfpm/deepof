# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.model_utils

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import deepof.model_utils


@settings(deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)
    )
)
def test_far_away_uniform_initialiser(shape):
    far = deepof.model_utils.far_away_uniform_initialiser(shape, 0, 15, 100)
    assert far.shape == shape


# @settings(deadline=None)
# @given()
# def test_compute_kernel():
#     pass
#

# @settings(deadline=None)
# @given()
# def test_compute_mmd():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_onecyclescheduler():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_far_away_uniform_initialiser():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_uncorrelated_features_constraint():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_mcdropout():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_kldivergence_layer():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_dense_transpose():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_mmdiscrepancy_layer():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_gaussian_mixture_overlap():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_dead_neuron_control():
#     pass
#
#
# @settings(deadline=None)
# @given()
# def test_entropy_regulariser():
#     pass
