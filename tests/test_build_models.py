# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import deepof.models
import deepof.model_utils
import tensorflow as tf

# tf.config.experimental_run_functions_eagerly(True)


@settings(deadline=None, max_examples=25)
@given(
    loss=st.one_of(st.just("ELBO"), st.just("MMD"), st.just("ELBO+MMD")),
    kl_warmup_epochs=st.integers(min_value=0, max_value=1),
    mmd_warmup_epochs=st.integers(min_value=0, max_value=1),
    montecarlo_kl=st.integers(min_value=1, max_value=2),
    number_of_components=st.integers(min_value=1, max_value=2),
    annealing_mode=st.one_of(st.just("linear"), st.just("sigmoid")),
)
def test_GMVAE_build(
    loss,
    kl_warmup_epochs,
    mmd_warmup_epochs,
    montecarlo_kl,
    number_of_components,
    annealing_mode,
):
    deepof.models.GMVAE(
        loss=loss,
        kl_warmup_epochs=kl_warmup_epochs,
        mmd_warmup_epochs=mmd_warmup_epochs,
        montecarlo_kl=montecarlo_kl,
        number_of_components=number_of_components,
        next_sequence_prediction=True,
        phenotype_prediction=True,
        supervised_prediction=True,
        overlap_loss=True,
        kl_annealing_mode=annealing_mode,
        mmd_annealing_mode=annealing_mode,
    ).build(
        input_shape=(
            100,
            15,
            10,
        )
    )
