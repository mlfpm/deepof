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


def test_find_learning_rate():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )
    test_model.build(X.shape)

    deepof.train_utils.find_learning_rate(test_model, X, y)


@given(
    embedding_model=st.one_of(st.just("VQVAE"), st.just("GMVAE")),
    loss=st.one_of(st.just("test_A"), st.just("test_B")),
    next_sequence_prediction=st.floats(min_value=0.0, max_value=1.0),
    phenotype_prediction=st.floats(min_value=0.0, max_value=1.0),
    supervised_prediction=st.floats(min_value=0.0, max_value=1.0),
    n_cluster_loss=st.floats(min_value=0.0, max_value=1.0),
)
def test_get_callbacks(
    embedding_model,
    next_sequence_prediction,
    phenotype_prediction,
    supervised_prediction,
    n_cluster_loss,
    loss,
):
    callbacks = deepof.train_utils.get_callbacks(
        embedding_model=embedding_model,
        phenotype_prediction=phenotype_prediction,
        next_sequence_prediction=next_sequence_prediction,
        supervised_prediction=supervised_prediction,
        n_cluster_loss=n_cluster_loss,
        latent_loss=loss,
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


@settings(max_examples=32, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    embedding_model=st.one_of(st.just("VQVAE"), st.just("GMVAE")),
    loss=st.one_of(st.just("SELBO"), st.just("SIWAE"), st.just("MMD")),
    next_sequence_prediction=st.one_of(st.just(0.0), st.just(0.5)),
    phenotype_prediction=st.one_of(st.just(0.0), st.just(0.5)),
    supervised_prediction=st.one_of(st.just(0.0), st.just(0.5)),
)
def test_autoencoder_fitting(
    embedding_model,
    loss,
    next_sequence_prediction,
    supervised_prediction,
    phenotype_prediction,
):

    X_train = np.ones([20, 5, 6]).astype(float)
    y_train = np.ones([20, 1]).astype(float)

    if supervised_prediction:
        y_train = np.concatenate([y_train, np.ones([20, 6]).astype(float)], axis=1)

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
        embedding_model=embedding_model,
        batch_size=1,
        latent_dim=2,
        epochs=1,
        kl_warmup=1,
        log_history=True,
        log_hparams=True,
        mmd_warmup=1,
        n_components=5,
        latent_loss=loss,
        n_cluster_loss=0.1,
        gram_loss=0.1,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        supervised_prediction=supervised_prediction,
        entropy_knn=5,
    )


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
    stateful_step_count=1,
)
@given(
    embedding_model=st.one_of(st.just("VQVAE"), st.just("GMVAE")),
    hpt_type=st.one_of(st.just("bayopt"), st.just("hyperband")),
    loss=st.one_of(st.just("SELBO"), st.just("SIWAE"), st.just("MMD")),
)
def test_tune_search(
    embedding_model,
    hpt_type,
    loss,
):

    n_cluster_loss = 0.1
    next_sequence_prediction = 0.1
    phenotype_prediction = 0.1
    supervised_prediction = 0.1

    X_train = np.ones([100, 5, 6]).astype(float)
    y_train = np.ones([100, 1]).astype(float)

    callbacks = list(
        deepof.train_utils.get_callbacks(
            embedding_model=embedding_model,
            phenotype_prediction=phenotype_prediction,
            next_sequence_prediction=next_sequence_prediction,
            supervised_prediction=supervised_prediction,
            latent_loss=loss,
            input_type=False,
            cp=False,
            reg_cat_clusters=True,
            reg_cluster_variance=True,
            n_cluster_loss=n_cluster_loss,
            gram_loss=0.1,
            entropy_knn=5,
            outpath="unsupervised_tuner_search",
            logparam={"encoding": 16, "k": 5},
        )
    )[1:]

    deepof.train_utils.tune_search(
        data=[X_train, y_train, X_train, y_train],
        batch_size=25,
        embedding_model=embedding_model,
        encoding_size=16,
        hpt_type=hpt_type,
        hypertun_trials=1,
        k=5,
        kl_warmup_epochs=0,
        loss=loss,
        mmd_warmup_epochs=0,
        n_cluster_loss=n_cluster_loss,
        gram_loss=0.1,
        next_sequence_prediction=next_sequence_prediction,
        phenotype_prediction=phenotype_prediction,
        supervised_prediction=supervised_prediction,
        project_name="test_run",
        callbacks=callbacks,
        n_epochs=1,
    )
