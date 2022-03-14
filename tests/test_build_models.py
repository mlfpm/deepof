# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import deepof.model_utils
import deepof.models


@settings(deadline=None, max_examples=25)
@given(
    latent_dim=st.integers(min_value=4, max_value=16).filter(lambda x: x % 2 == 0),
    n_components=st.integers(min_value=4, max_value=16).filter(lambda x: x % 2 == 0),
)
def test_VQVAE_build(
    latent_dim,
    n_components,
):
    vqvae = deepof.models.VQVAE(
        input_shape=(1000, 15, 10),
        latent_dim=latent_dim,
        n_components=n_components,
    )
    vqvae.build((1000, 15, 10))
    vqvae.compile()
