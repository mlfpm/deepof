# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.train_utils

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.model_utils
import deepof.train_utils
import keras
import os
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

# For coverage.py to work with @tf.function decorated functions and methods,
# graph execution is disabled when running this script with pytest
tf.config.experimental_run_functions_eagerly(True)


def test_load_hparams():
    assert type(deepof.train_utils.load_hparams(None)) == dict
    assert (
        type(
            deepof.train_utils.load_hparams(
                os.path.join("tests", "test_examples", "Others", "test_hparams.pkl")
            )
        )
        == dict
    )


def test_load_treatments():
    assert deepof.train_utils.load_treatments(".") is None
    assert (
        type(
            deepof.train_utils.load_treatments(
                os.path.join("tests", "test_examples", "Others")
            )
        )
        == dict
    )


@given(
    X_train=arrays(
        shape=st.tuples(st.integers(min_value=1, max_value=1000)), dtype=float
    ),
    batch_size=st.integers(min_value=128, max_value=512),
    k=st.integers(min_value=1, max_value=50),
    kl_wu=st.integers(min_value=0, max_value=25),
    loss=st.one_of(st.just("test_A"), st.just("test_B")),
    mmd_wu=st.integers(min_value=0, max_value=25),
    predictor=st.floats(min_value=0.0, max_value=1.0),
    variational=st.booleans(),
)
def test_get_callbacks(
    X_train, batch_size, variational, predictor, k, loss, kl_wu, mmd_wu
):
    runID, tbc, cpc, cycle1c = deepof.train_utils.get_callbacks(
        X_train, batch_size, variational, predictor, k, loss, kl_wu, mmd_wu,
    )
    assert type(runID) == str
    assert type(tbc) == keras.callbacks.tensorboard_v2.TensorBoard
    assert type(cpc) == tf.python.keras.callbacks.ModelCheckpoint
    assert type(cycle1c) == deepof.model_utils.one_cycle_scheduler


@settings(max_examples=1)
@given(
    train=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=100),
            st.integers(min_value=2, max_value=15),
            st.integers(min_value=2, max_value=10),
        ),
    ),
    test=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=100),
            st.integers(min_value=2, max_value=15),
            st.integers(min_value=2, max_value=10),
        ),
    ),
    batch_size=st.integers(min_value=128, max_value=512),
    hypermodel=st.one_of(st.just("S2SAE"), st.just("S2SGMVAE")),
    k=st.integers(min_value=1, max_value=10),
    kl_wu=st.integers(min_value=0, max_value=10),
    loss=st.one_of(st.just("ELBO"), st.just("MMD")),
    mmd_wu=st.integers(min_value=0, max_value=10),
    overlap_loss=st.floats(min_value=0.0, max_value=1.0),
    predictor=st.floats(min_value=0.0, max_value=1.0),
    project_name=st.text(min_size=5),
)
def test_tune_search(
    train,
    test,
    batch_size,
    hypermodel,
    k,
    kl_wu,
    loss,
    mmd_wu,
    overlap_loss,
    predictor,
    project_name,
):
    callbacks = list(
        deepof.train_utils.get_callbacks(
            train,
            batch_size,
            hypermodel == "S2SGMVAE",
            predictor,
            k,
            loss,
            kl_wu,
            mmd_wu,
        )
    )

    deepof.train_utils.tune_search(
        train,
        test,
        1,
        hypermodel,
        k,
        kl_wu,
        loss,
        mmd_wu,
        overlap_loss,
        predictor,
        project_name,
        callbacks,
        n_epochs=1,
    )
