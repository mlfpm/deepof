# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.train_utils

"""

import os

import numpy as np
import tensorflow as tf
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import deepof.data
import deepof.model_utils
import deepof.train_utils


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
        shape=st.tuples(st.integers(min_value=1, max_value=1000), st.just(24)),
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
    overlap_loss=st.floats(min_value=0.0, max_value=1.0),
)
def test_get_callbacks(
    X_train,
    batch_size,
    next_sequence_prediction,
    phenotype_prediction,
    rule_based_prediction,
    overlap_loss,
    loss,
):
    callbacks = deepof.train_utils.get_callbacks(
        X_train=X_train,
        batch_size=batch_size,
        phenotype_prediction=phenotype_prediction,
        next_sequence_prediction=next_sequence_prediction,
        supervised_prediction=rule_based_prediction,
        overlap_loss=overlap_loss,
        loss=loss,
        input_type=False,
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
        [isinstance(i, deepof.model_utils.one_cycle_scheduler) for i in callbacks]
    )


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    loss=st.one_of(st.just("ELBO"), st.just("MMD")),
    next_sequence_prediction=st.one_of(st.just(0.0), st.just(1.0)),
    phenotype_prediction=st.one_of(st.just(0.0), st.just(1.0)),
    rule_based_prediction=st.one_of(st.just(0.0), st.just(1.0)),
)
def test_autoencoder_fitting(
    loss,
    next_sequence_prediction,
    rule_based_prediction,
    phenotype_prediction,
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

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=380,
        video_format=".mp4",
    ).run()

    prun.deep_unsupervised_embedding(
        preprocessed_data,
        batch_size=10,
        encoding_size=2,
        epochs=2,
        kl_warmup=1,
        log_history=True,
        log_hparams=True,
        mmd_warmup=1,
        n_components=2,
        loss=loss,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=rule_based_prediction,
        entropy_knn=5,
    )


@settings(max_examples=12, deadline=None)
@given(
    X_train=arrays(
        dtype=float,
        shape=(100, 8, 6),
        elements=st.floats(
            min_value=0.0,
            max_value=1,
        ),
    ),
    y_train=st.data(),
    hpt_type=st.one_of(st.just("bayopt"), st.just("hyperband")),
    loss=st.one_of(st.just("ELBO"), st.just("MMD")),
    overlap_loss=st.one_of(st.just(1.0)),
    predictor_branch=st.one_of(
        st.just("next_seq"), st.just("pheno"), st.just("supervised")
    ),
)
def test_tune_search(
    X_train,
    y_train,
    hpt_type,
    loss,
    overlap_loss,
    predictor_branch,
):

    next_sequence_prediction = 1.0 if predictor_branch == "next_seq" else 0.0
    phenotype_prediction = 1.0 if predictor_branch == "pheno" else 0.0
    supervised_prediction = 1.0 if predictor_branch == "supervised" else 0.0

    callbacks = list(
        deepof.train_utils.get_callbacks(
            X_train=X_train,
            batch_size=25,
            phenotype_prediction=np.round(phenotype_prediction, 2),
            next_sequence_prediction=np.round(next_sequence_prediction, 2),
            supervised_prediction=np.round(supervised_prediction, 2),
            loss=loss,
            input_type=False,
            cp=False,
            reg_cat_clusters=True,
            reg_cluster_variance=True,
            overlap_loss=overlap_loss,
            entropy_knn=5,
            logparam={"encoding": 2, "k": 5},
        )
    )[1:]

    y_train = y_train.draw(
        arrays(
            dtype=np.float32,
            elements=st.floats(min_value=0.0, max_value=1.0, width=32),
            shape=(100, 1),
        )
    )

    deepof.train_utils.tune_search(
        data=[X_train, y_train, X_train, y_train],
        encoding_size=2,
        hpt_type=hpt_type,
        hypertun_trials=1,
        k=5,
        kl_warmup_epochs=0,
        loss=loss,
        mmd_warmup_epochs=0,
        overlap_loss=overlap_loss,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        rule_based_prediction=supervised_prediction,
        project_name="test_run",
        callbacks=callbacks,
        n_epochs=1,
    )
