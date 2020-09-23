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
from kerastuner import HyperParameters
import deepof.hypermodels
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)


@settings(deadline=None,
          max_examples=10)
@given(
    input_shape=st.tuples(
        st.integers(min_value=100, max_value=1000),
        st.integers(min_value=5, max_value=15),
        st.integers(min_value=5, max_value=15),
    )
)
def test_SEQ_2_SEQ_AE_hypermodel_build(input_shape):
    deepof.hypermodels.SEQ_2_SEQ_AE(input_shape=input_shape).build(hp=HyperParameters())


@settings(deadline=None,
          max_examples=10)
@given(
    loss=st.one_of(st.just("ELBO"), st.just("MMD"), st.just("ELBO+MMD")),
    kl_warmup_epochs=st.integers(min_value=1, max_value=5),
    mmd_warmup_epochs=st.integers(min_value=1, max_value=5),
    number_of_components=st.integers(min_value=1, max_value=5),
    entropy_reg_weight=st.one_of(st.just(0.0), st.just(1.0)),
)
def test_SEQ_2_SEQ_GMVAE_hypermodel_build(
    loss,
    kl_warmup_epochs,
    mmd_warmup_epochs,
    number_of_components,
    entropy_reg_weight,
):
    deepof.hypermodels.SEQ_2_SEQ_GMVAE(
        input_shape=(100, 15, 10,),
        batch_size=10,
        loss=loss,
        kl_warmup_epochs=kl_warmup_epochs,
        mmd_warmup_epochs=mmd_warmup_epochs,
        number_of_components=number_of_components,
        predictor=True,
    ).build(hp=HyperParameters())
