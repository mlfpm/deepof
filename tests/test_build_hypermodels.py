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


@settings(deadline=None, max_examples=10)
@given(
    latent_dim=st.integers(min_value=2, max_value=16).filter(lambda x: x % 2 == 0),
    n_components=st.integers(min_value=2, max_value=16).filter(lambda x: x % 2 == 0),
)
def test_VQVAE_hypermodel_build(
    latent_dim,
    n_components,
):
    deepof.hypermodels.VQVAE(
        latent_dim=latent_dim,
        input_shape=(
            100,
            15,
            10,
        ),
        n_components=n_components,
    ).build(hp=HyperParameters())


@settings(deadline=None, max_examples=10)
@given(
    latent_dim=st.integers(min_value=2, max_value=16).filter(lambda x: x % 2 == 0),
    loss=st.one_of(
        st.just("SELBO"), st.just("SIWAE"), st.just("MMD"), st.just("SELBO+MMD")
    ),
    n_components=st.integers(min_value=1, max_value=5).filter(lambda x: x % 2 == 0),
)
def test_GMVAE_hypermodel_build(
    latent_dim,
    loss,
    n_components,
):
    deepof.hypermodels.GMVAE(
        latent_dim=latent_dim,
        batch_size=64,
        input_shape=(
            100,
            15,
            10,
        ),
        latent_loss=loss,
        n_components=n_components,
        next_sequence_prediction=True,
    ).build(hp=HyperParameters())
