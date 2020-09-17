# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.model_utils

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.model_utils
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


@settings(deadline=None)
@given(
    shape=st.tuples(
        st.integers(min_value=2, max_value=10), st.integers(min_value=2, max_value=10)
    )
)
def test_far_away_uniform_initialiser(shape):
    far = deepof.model_utils.far_away_uniform_initialiser(shape, 0, 15, 100)
    random = tf.random.uniform(shape, 0, 15)
    assert far.shape == shape
    assert tf.abs(tf.norm(tf.math.subtract(far[1:], far[:1]))) > tf.abs(
        tf.norm(tf.math.subtract(random[1:], random[:1]))
    )


@settings(deadline=None)
@given(
    tensor=arrays(
            shape=(10, 10),
            dtype=float,
            unique=True,
            elements=st.floats(min_value=-300, max_value=300),
        ),
)
def test_compute_mmd(tensor):

    tensor1 = tf.cast(tf.convert_to_tensor(tensor), dtype=tf.float32)
    tensor2 = tf.random.uniform(tensor1.shape, -300, 300, dtype=tf.float32)

    mmd_kernel = deepof.model_utils.compute_mmd(tuple([tensor1, tensor2]))
    null_kernel = deepof.model_utils.compute_mmd(tuple([tensor1, tensor1]))

    assert type(mmd_kernel) == EagerTensor
    assert null_kernel == 0


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
