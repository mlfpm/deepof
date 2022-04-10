# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.model_utils

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


def test_load_treatments():
    assert deepof.model_utils.load_treatments("tests") is None
    assert isinstance(
        deepof.model_utils.load_treatments(
            os.path.join("tests", "test_examples", "test_single_topview", "Others")
        ),
        dict,
    )


def test_find_learning_rate():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])
    dataset = tf.data.Dataset.from_tensors((X, y))

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1, input_shape=X.shape[1:]))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.SGD(),
    )

    deepof.model_utils.find_learning_rate(test_model, data=dataset)


@given(
    encoding=st.integers(min_value=1, max_value=10),
    k=st.integers(min_value=1, max_value=10),
)
def test_get_callbacks(
    encoding,
    k,
):
    callbacks = deepof.model_utils.get_callbacks(
        input_type=False,
        cp=True,
        logparam={"latent_dim": encoding, "n_components": k},
    )
    assert np.any([isinstance(i, str) for i in callbacks])
    assert np.any(
        [isinstance(i, tf.keras.callbacks.ModelCheckpoint) for i in callbacks]
    )


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    embedding=st.integers(min_value=2, max_value=8).filter(lambda x: x % 2 == 0),
    k=st.just(10),
    pheno_prediction=st.one_of(st.just(0.0), st.just(1.0)),
)
def test_autoencoder_fitting(
    embedding,
    k,
    pheno_prediction,
):

    X_train = np.ones([20, 5, 6]).astype(float)
    y_train = np.ones([20, 1]).astype(float)

    preprocessed_data = (X_train, y_train, X_train, y_train)

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular-autodetect",
        arena_dims=380,
        video_format=".mp4",
    ).run()

    prun.deep_unsupervised_embedding(
        preprocessed_data,
        batch_size=1,
        latent_dim=embedding,
        epochs=1,
        log_history=True,
        log_hparams=True,
        n_components=k,
        gram_loss=0.1,
        phenotype_prediction=pheno_prediction,
    )


@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
    stateful_step_count=1,
)
@given(
    hpt_type=st.one_of(st.just("bayopt"), st.just("hyperband")),
    pheno_loss=st.one_of(st.just(0.0), st.just(1.0)),
)
def test_tune_search(
    hpt_type,
    pheno_loss,
):

    X_train = np.ones([100, 5, 6]).astype(float)
    y_train = np.ones([100, 1]).astype(float)

    callbacks = list(
        deepof.model_utils.get_callbacks(
            input_type=False,
            cp=False,
            gram_loss=0.1,
            outpath="unsupervised_tuner_search",
            logparam={"latent_dim": 16, "n_components": 5},
        )
    )[1:]

    deepof.model_utils.tune_search(
        data=[X_train, y_train, X_train, y_train],
        batch_size=25,
        encoding_size=16,
        hpt_type=hpt_type,
        hypertun_trials=1,
        k=5,
        gram_loss=0.1,
        phenotype_prediction=pheno_loss,
        project_name="test_run",
        callbacks=callbacks,
        n_epochs=1,
    )
