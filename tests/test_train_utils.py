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
import os
import tensorflow as tf


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
        shape=st.tuples(st.integers(min_value=1, max_value=1000)),
        dtype=float,
        elements=st.floats(min_value=0.0, max_value=1,),
    ),
    batch_size=st.integers(min_value=128, max_value=512),
    loss=st.one_of(st.just("test_A"), st.just("test_B")),
    predictor=st.floats(min_value=0.0, max_value=1.0),
    variational=st.booleans(),
)
def test_get_callbacks(
    X_train, batch_size, variational, predictor, loss,
):
    runID, tbc, cpc, cycle1c = deepof.train_utils.get_callbacks(
        X_train, batch_size, variational, predictor, loss,
    )
    assert type(runID) == str
    assert type(tbc) == tf.keras.callbacks.TensorBoard
    assert type(cpc) == tf.python.keras.callbacks.ModelCheckpoint
    assert type(cycle1c) == deepof.model_utils.one_cycle_scheduler


@settings(max_examples=1, deadline=None)
@given(
    train=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=100),
            st.integers(min_value=2, max_value=15),
            st.integers(min_value=2, max_value=10),
        ),
        elements=st.floats(min_value=0.0, max_value=1,),
    ),
    batch_size=st.integers(min_value=128, max_value=512),
    hypermodel=st.just("S2SGMVAE"),
    k=st.integers(min_value=1, max_value=10),
    loss=st.one_of(st.just("ELBO"), st.just("MMD")),
    overlap_loss=st.floats(min_value=0.0, max_value=1.0),
    predictor=st.floats(min_value=0.0, max_value=1.0),
)
def test_tune_search(
    train, batch_size, hypermodel, k, loss, overlap_loss, predictor,
):
    callbacks = list(
        deepof.train_utils.get_callbacks(
            train,
            batch_size,
            hypermodel == "S2SGMVAE",
            predictor,
            loss,
        )
    )[1:]

    deepof.train_utils.tune_search(
        train,
        train,
        1,
        hypermodel,
        k,
        loss,
        overlap_loss,
        predictor,
        "test_run",
        callbacks,
        n_epochs=1,
    )
