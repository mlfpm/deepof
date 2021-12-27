# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.hypermodels. Checks that all hyperparameter
tuning models are building properly in all possible configurations

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from keras_tuner import HyperParameters
import deepof.hypermodels
import tensorflow as tf

tf.config.run_functions_eagerly(True)


@settings(deadline=None, max_examples=10)
@given(
    encoding_size=st.integers(min_value=2, max_value=16),
    loss=st.one_of(st.just("ELBO"), st.just("MMD"), st.just("ELBO+MMD")),
    number_of_components=st.integers(min_value=1, max_value=5),
)
def test_GMVAE_hypermodel_build(
    encoding_size,
    loss,
    number_of_components,
):
    deepof.hypermodels.GMVAE(
        encoding=encoding_size,
        batch_size=64,
        input_shape=(
            100,
            15,
            10,
        ),
        loss=loss,
        number_of_components=number_of_components,
        next_sequence_prediction=True,
    ).build(hp=HyperParameters())
