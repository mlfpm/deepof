# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.train_utils

"""

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import deepof.data
import deepof.model_utils
import deepof.train_utils
import numpy as np
import os
import tensorflow as tf


def test_load_treatments():
    assert deepof.train_utils.load_treatments("tests") is None
    assert isinstance(
        deepof.train_utils.load_treatments(
            os.path.join("tests", "test_examples", "test_single_topview", "Others")
        ),
        dict,
    )


@given(
    X_train=arrays(
        shape=st.tuples(st.integers(min_value=1, max_value=1000)),
        dtype=float,
        elements=st.floats(
            min_value=0.0,
            max_value=1,
        ),
    ),
    batch_size=st.integers(min_value=128, max_value=512),
    loss=st.one_of(st.just("test_A"), st.just("test_B")),
    next_sequence_prediction=st.floats(min_value=0.0, max_value=1.0),
    phenotype_prediction=st.floats(min_value=0.0, max_value=1.0),
    rule_based_prediction=st.floats(min_value=0.0, max_value=1.0),
    variational=st.booleans(),
)
def test_get_callbacks(
    X_train,
    batch_size,
    variational,
    next_sequence_prediction,
    phenotype_prediction,
    rule_based_prediction,
    loss,
):
    callbacks = deepof.train_utils.get_callbacks(
        X_train=X_train,
        batch_size=batch_size,
        variational=variational,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=rule_based_prediction,
        loss=loss,
        X_val=X_train,
        cp=True,
        reg_cat_clusters=False,
        reg_cluster_variance=False,
        logparam={"encoding": 2, "k": 15},
    )
    assert np.any([isinstance(i, str) for i in callbacks])
    assert np.any(
        [isinstance(i, tf.keras.callbacks.ModelCheckpoint) for i in callbacks]
    )
    assert np.any(
        [isinstance(i, deepof.model_utils.neighbor_latent_entropy) for i in callbacks]
    )
    assert np.any(
        [isinstance(i, deepof.model_utils.one_cycle_scheduler) for i in callbacks]
    )


@settings(max_examples=1, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    loss=st.one_of(st.just("ELBO"), st.just("MMD"), st.just("ELBO+MMD")),
    next_sequence_prediction=st.one_of(st.just(1.0), st.just(1.0)),
    phenotype_prediction=st.one_of(st.just(1.0), st.just(1.0)),
    rule_based_prediction=st.one_of(st.just(1.0), st.just(1.0)),
    variational=st.one_of(st.just(True), st.just(True)),
)
def test_autoencoder_fitting(
    loss,
    next_sequence_prediction,
    phenotype_prediction,
    rule_based_prediction,
    variational,
):
    X_train = np.random.uniform(-1, 1, [20, 5, 6])
    y_train = np.round(np.random.uniform(0, 1, [20, 1]))
    if rule_based_prediction:
        y_train = np.concatenate(
            [y_train, np.round(np.random.uniform(0, 1, [20, 6]), 1)], axis=1
        )

    if next_sequence_prediction:
        y_train = y_train[1:]

    preprocessed_data = (X_train, y_train, X_train, y_train)

    prun = deepof.data.project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
    ).run()

    prun.deep_unsupervised_embedding(
        preprocessed_data,
        batch_size=100,
        encoding_size=2,
        epochs=1,
        kl_warmup=1,
        log_history=True,
        log_hparams=True,
        mmd_warmup=1,
        n_components=2,
        loss=loss,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=rule_based_prediction,
        variational=variational,
        entropy_samples=10,
        entropy_knn=5,
    )


@settings(max_examples=1, deadline=None)
@given(
    X_train=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=100),
            st.integers(min_value=2, max_value=15),
            st.integers(min_value=2, max_value=10),
        ),
        elements=st.floats(
            min_value=0.0,
            max_value=1,
        ),
    ),
    batch_size=st.integers(min_value=128, max_value=512),
    encoding_size=st.integers(min_value=1, max_value=16),
    hpt_type=st.one_of(st.just("bayopt"), st.just("hyperband")),
    hypermodel=st.just("S2SGMVAE"),
    k=st.integers(min_value=1, max_value=10),
    loss=st.one_of(st.just("ELBO"), st.just("MMD")),
    overlap_loss=st.floats(min_value=0.0, max_value=1.0),
    next_sequence_prediction=st.floats(min_value=0.0, max_value=1.0),
    phenotype_prediction=st.floats(min_value=0.0, max_value=1.0),
    rule_based_prediction=st.floats(min_value=0.0, max_value=1.0),
)
def test_tune_search(
    X_train,
    batch_size,
    encoding_size,
    hpt_type,
    hypermodel,
    k,
    loss,
    overlap_loss,
    next_sequence_prediction,
    phenotype_prediction,
    rule_based_prediction,
):
    callbacks = list(
        deepof.train_utils.get_callbacks(
            X_train=X_train,
            batch_size=batch_size,
            variational=(hypermodel == "S2SGMVAE"),
            next_sequence_prediction=next_sequence_prediction,
            phenotype_prediction=phenotype_prediction,
            rule_based_prediction=rule_based_prediction,
            loss=loss,
            X_val=X_train,
            cp=False,
            reg_cat_clusters=True,
            reg_cluster_variance=True,
            entropy_samples=10,
            entropy_knn=5,
            logparam={"encoding": 2, "k": 15},
        )
    )[1:]

    y_train = tf.random.uniform(shape=(X_train.shape[1], 1), maxval=1.0)

    deepof.train_utils.tune_search(
        data=[X_train, y_train, X_train, y_train],
        encoding_size=encoding_size,
        hpt_type=hpt_type,
        hypertun_trials=1,
        hypermodel=hypermodel,
        k=k,
        kl_warmup_epochs=0,
        loss=loss,
        mmd_warmup_epochs=0,
        overlap_loss=overlap_loss,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=rule_based_prediction,
        project_name="test_run",
        callbacks=callbacks,
        n_epochs=1,
    )
