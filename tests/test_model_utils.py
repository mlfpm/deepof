# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.model_utils

"""

import os

import networkx as nx
import numpy as np
import tensorflow as tf
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from shutil import rmtree

import deepof.data
import deepof.model_utils


def test_find_learning_rate():
    X = np.random.uniform(0, 10, [1500, 5])
    y = np.random.randint(0, 2, [1500, 1])
    dataset = tf.data.Dataset.from_tensors((X, y))

    test_model = tf.keras.Sequential()
    test_model.add(tf.keras.layers.Dense(1, input_shape=X.shape[1:]))

    test_model.compile(
        loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD()
    )

    deepof.model_utils.find_learning_rate(test_model, data=dataset)


@given(
    embedding_model=st.sampled_from(["VQVAE", "GMVAE", "Contrastive"]),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    encoding=st.integers(min_value=1, max_value=10),
    k=st.integers(min_value=1, max_value=10),
)
def test_get_callbacks(encoding, k, embedding_model, encoder_type):
    callbacks = deepof.model_utils.get_callbacks(
        encoder_type=encoder_type,
        embedding_model=embedding_model,
        input_type=False,
        cp=True,
        logparam={"latent_dim": encoding, "n_components": k},
    )
    assert np.any([isinstance(i, str) for i in callbacks])
    assert np.any(
        [isinstance(i, tf.keras.callbacks.ModelCheckpoint) for i in callbacks]
    )


@settings(deadline=None)
@given(
    soft_counts=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
            st.integers(min_value=1, max_value=10),
        ),
        elements=st.floats(min_value=0, max_value=1),
    ),
)
def test_get_hard_counts(soft_counts):
    hard_counts = deepof.model_utils.get_hard_counts(soft_counts.astype(np.float32))
    assert isinstance(hard_counts, tf.Tensor)


@settings(max_examples=18, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    embedding_model=st.sampled_from(["VQVAE", "VaDE", "Contrastive"]),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    use_graph=st.booleans(),
)
def test_model_embedding_fitting(
    embedding_model,
    encoder_type,
    use_graph,
):

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
    ).create(force=True)
    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    X_train = np.ones([20, 5, 6]).astype(float)
    y_train = np.array([20, 1]).astype(float)

    if not use_graph:
        preprocessed_data = (X_train, y_train, X_train, y_train)
    else:
        preprocessed_data = (X_train, X_train, y_train, X_train, X_train, y_train)

    prun.deep_unsupervised_embedding(
        preprocessed_data,
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(6, 6)
        ).todense(),
        embedding_model=embedding_model,
        encoder_type=encoder_type,
        batch_size=10,
        latent_dim=4,
        epochs=1,
        log_history=True,
        log_hparams=True,
        n_components=10,
        kmeans_loss=0.1,
    )


@settings(
    max_examples=36,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
    stateful_step_count=1,
)
@given(
    embedding_model=st.sampled_from(["VQVAE", "VaDE", "Contrastive"]),
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
    hpt_type=st.sampled_from(["bayopt", "hyperband"]),
    use_graph=st.booleans(),
)
def test_tune_search(hpt_type, encoder_type, embedding_model, use_graph):

    X_train = np.ones([20, 5, 6]).astype(float)
    y_train = np.array([20, 1]).astype(float)

    if not use_graph:
        preprocessed_data = (X_train, y_train, X_train, y_train)
    else:
        preprocessed_data = (X_train, X_train, y_train, X_train, X_train, y_train)

    callbacks = list(
        deepof.model_utils.get_callbacks(
            input_type=False,
            embedding_model=embedding_model,
            encoder_type=encoder_type,
            cp=False,
            kmeans_loss=0.1,
            outpath="unsupervised_tuner_search",
            logparam={"latent_dim": 16, "n_components": 5},
        )
    )[1:]

    deepof.model_utils.tune_search(
        preprocessed_object=preprocessed_data,
        adjacency_matrix=nx.adjacency_matrix(
            nx.generators.random_graphs.dense_gnm_random_graph(6, 6)
        ).todense(),
        embedding_model=embedding_model,
        batch_size=5,
        encoding_size=4,
        hpt_type=hpt_type,
        hypertun_trials=1,
        k=5,
        project_name="test_run",
        callbacks=callbacks,
        n_epochs=1,
    )
