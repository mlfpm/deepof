# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.models
import deepof.model_utils
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework.ops import EagerTensor


@settings(deadline=None)
@given(
    input_shape=st.tuples(
        st.integers(min_value=100, max_value=1000),
        st.integers(min_value=5, max_value=15),
        st.integers(min_value=5, max_value=15),
    )
)
def test_SEQ_2_SEQ_AE_build(input_shape):
    deepof.models.SEQ_2_SEQ_AE(input_shape=input_shape)


@settings(deadline=None)
@given(
    input_shape=st.tuples(
        st.integers(min_value=100, max_value=1000),
        st.integers(min_value=5, max_value=15),
        st.integers(min_value=5, max_value=15),
    ),
    loss=st.one_of(st.just("ELBO"), st.just("MMD"), st.just("ELBO+MMD")),
    kl_warmup_epochs=st.integers(min_value=0, max_value=5),
    mmd_warmup_epochs=st.integers(min_value=0, max_value=5),
    number_of_components=st.integers(min_value=1, max_value=5),
    predictor=st.booleans(),
    overlap_loss=st.booleans(),
    entropy_reg_weight=st.floats(min_value=0.0, max_value=1.0),
)
def test_SEQ_2_SEQ_GMVAE_build(
    input_shape,
    loss,
    kl_warmup_epochs,
    mmd_warmup_epochs,
    number_of_components,
    predictor,
    overlap_loss,
    entropy_reg_weight,
):
    deepof.models.SEQ_2_SEQ_GMVAE(
        input_shape=input_shape,
        loss=loss,
        kl_warmup_epochs=kl_warmup_epochs,
        mmd_warmup_epochs=mmd_warmup_epochs,
        number_of_components=number_of_components,
        predictor=predictor,
        overlap_loss=overlap_loss,
        entropy_reg_weight=entropy_reg_weight,
    )
