# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import deepof.unsupervised_utils
import deepof.models


@settings(deadline=None, max_examples=25)
@given(
    kl_warmup_epochs=st.integers(min_value=0, max_value=1),
    montecarlo_kl=st.integers(min_value=10, max_value=20),
    n_components=st.integers(min_value=2, max_value=4).filter(lambda x: x % 2 == 0),
    annealing_mode=st.one_of(st.just("linear"), st.just("sigmoid")),
)
def test_GMVAE_build(kl_warmup_epochs, montecarlo_kl, n_components, annealing_mode):
    gmvae = deepof.models.GMVAE(
        input_shape=(1000, 15, 10),
        batch_size=64,
        kl_warmup_epochs=kl_warmup_epochs,
        montecarlo_kl=montecarlo_kl,
        n_components=n_components,
        next_sequence_prediction=True,
        phenotype_prediction=True,
        supervised_prediction=True,
        n_cluster_loss=1.0,
        kl_annealing_mode=annealing_mode,
    )
    gmvae.build((1000, 15, 10))
    gmvae.compile()


@settings(deadline=None, max_examples=25)
@given(
    latent_dim=st.integers(min_value=4, max_value=16).filter(lambda x: x % 2 == 0),
    n_components=st.integers(min_value=4, max_value=16).filter(lambda x: x % 2 == 0),
)
def test_VQVAE_build(latent_dim, n_components):
    vqvae = deepof.models.VQVAE(
        input_shape=(1000, 15, 10), latent_dim=latent_dim, n_components=n_components
    )
    vqvae.build((1000, 15, 10))
    vqvae.compile()
