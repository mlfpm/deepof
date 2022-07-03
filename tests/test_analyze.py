# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.analyze

"""
import os
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import tensorflow as tf
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import deepof.analyze
import deepof.data


def test_get_time_on_cluster():

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Define a test breaks dictionary
    breaks = {i: [10] * 100 for i in range(10)}

    toc = deepof.analyze.get_time_on_cluster(soft_counts, breaks)

    # Assert that both the soft counts and breaks are correctly aggregated
    assert toc.shape[0] * 100 == np.concatenate(list(soft_counts.values())).shape[0]
    assert toc.shape[1] == np.concatenate(list(soft_counts.values())).shape[1]


@given(
    reduce_dim=st.booleans(), agg=st.sampled_from(["mean", "median"]),
)
def test_get_aggregated_embedding(reduce_dim, agg):

    # Define a test embedding dixtionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}

    aggregated_embeddings = deepof.analyze.get_aggregated_embedding(
        embedding, reduce_dim, agg
    )

    assert aggregated_embeddings.shape[0] == len(embedding)


@given(
    bin_size=st.integers(min_value=15, max_value=20),
    bin_index=st.integers(min_value=0, max_value=1),
)
def test_select_time_bin(bin_size, bin_index):

    # Define a test embedding dixtionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Create a dictionary of breaks, whose sums add up to the number of chunks
    breaks = {i: np.array([10] * 100) for i in range(10)}

    embedding, soft_counts, breaks = deepof.analyze.select_time_bin(
        embedding, soft_counts, breaks, bin_size, bin_index
    )

    # Assert that the embedding and soft counts are correctly binned
    assert list(embedding.values())[0].shape[0] > 0
    assert list(soft_counts.values())[0].shape[0] > 0
    assert list(breaks.values())[0].shape[0] > 0


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    scan_mode=st.sampled_from(["growing-window", "per-bin"]),
    agg=st.sampled_from(["time_on_cluster", "mean", "median"]),
    metric=st.sampled_from(["auc-linear", "wasserstein"]),
)
def test_condition_distance_binning(scan_mode, agg, metric):

    # Define a test embedding dixtionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Create a dictionary of breaks, whose sums add up to the number of chunks
    breaks = {i: np.array([10] * 100) for i in range(10)}

    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}

    distance_binning = deepof.analyze.condition_distance_binning(
        embedding=embedding,
        soft_counts=soft_counts,
        breaks=breaks,
        exp_conditions=exp_conditions,
        start_bin=11,
        end_bin=100,
        step_bin=11,
        scan_mode=scan_mode,
        agg=agg,
        metric=metric,
        n_jobs=1,
    )

    # Assert that the embedding and soft counts are correctly binned
    assert np.argmax(distance_binning) >= 0


@given(
    bin_size=st.integers(min_value=11, max_value=100), normalize=st.booleans(),
)
def test_cluster_enrichment_across_conditions(bin_size, normalize):

    # Define a test embedding dixtionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Create a dictionary of breaks, whose sums add up to the number of chunks
    breaks = {i: np.array([10] * 100) for i in range(10)}

    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}

    enrichment = deepof.analyze.cluster_enrichment_across_conditions(
        embedding,
        soft_counts,
        breaks,
        exp_conditions,
        bin_size=bin_size,
        bin_index=0,
        normalize=normalize,
    )

    assert isinstance(enrichment, pd.DataFrame)
    assert enrichment.shape[0] > 0
    assert enrichment.shape[1] == 3


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_states=st.integers(min_value=2, max_value=25),)
def test_get_transitions(n_states):

    sequence = np.random.choice(range(n_states), 1000, replace=True)
    transitions = deepof.analyze.get_transitions(sequence, n_states=n_states)

    assert transitions.shape[0] == n_states
    assert transitions.shape[1] == n_states


@given(
    bin_size=st.integers(min_value=11, max_value=100),
    aggregate=st.booleans(),
    normalize=st.booleans(),
    steady_state_entropy=st.booleans(),
)
def test_compute_transition_matrix_per_condition(
    bin_size, aggregate, normalize, steady_state_entropy
):

    # Define a test embedding dixtionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Create a dictionary of breaks, whose sums add up to the number of chunks
    breaks = {i: np.array([10] * 100) for i in range(10)}

    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}

    transitions = deepof.analyze.compute_transition_matrix_per_condition(
        embedding,
        soft_counts,
        breaks,
        exp_conditions,
        bin_size=bin_size,
        bin_index=0,
        aggregate=aggregate,
        normalize=normalize,
    )

    assert isinstance(transitions, dict)
    if aggregate:
        assert len(transitions) == len(set(exp_conditions.values()))
    else:
        assert len(transitions) == len(exp_conditions)

    # Get steady states
    steady_states = deepof.analyze.compute_steady_state(
        transitions, return_entropy=steady_state_entropy
    )

    assert isinstance(steady_states, dict)
    assert isinstance(list(steady_states.values())[0], float) == steady_state_entropy

    if steady_state_entropy:
        print(steady_states)


def test_extract_kinematic_features():

    # Set up testing names for our random features
    body_part_names = ["test_name_{}".format(i) for i in range(4)]

    # Define a matrix of soft counts
    counts = np.random.normal(size=(100, 10))
    soft_counts = counts / counts.sum(axis=1)[:, None]

    # And extract hard counts from them
    hard_counts = np.argmax(soft_counts, axis=1)

    # Define a table with chunk data
    chunked_dataset = np.random.uniform(size=(100, 25, 4))

    kinematic_features = deepof.analyze.extract_kinematic_features(
        chunked_dataset, pd.Series(hard_counts), body_part_names
    )

    assert isinstance(kinematic_features, pd.DataFrame)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    mode=st.one_of(st.just("single"), st.just("multi")),
    exclude=st.one_of(st.just(tuple([""])), st.just(["Tail_1"])),
    sampler=st.data(),
)
def test_align_deepof_kinematics_with_unsupervised_labels(mode, exclude, sampler):

    prun = deepof.data.Project(
        path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        arena="circular-autodetect",
        arena_dims=380,
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).run()

    # get breaks
    breaks = {i: np.array([10] * 10) for i in ["test", "test2"]}

    # extract kinematic features
    kinematics = deepof.analyze.align_deepof_kinematics_with_unsupervised_labels(
        prun,
        breaks,
        kin_derivative=sampler.draw(st.integers(min_value=1, max_value=2)),
        include_distances=sampler.draw(st.booleans()),
        include_angles=sampler.draw(st.booleans()),
        animal_id=(
            sampler.draw(st.sampled_from(["B", "W"])) if mode == "multi" else None
        ),
    )

    # check that the output is a DataFrame
    assert isinstance(kinematics, pd.DataFrame)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    mode=st.one_of(st.just("single"), st.just("multi")), sampler=st.data(),
)
def test_align_deepof_supervised_and_unsupervised_labels(mode, sampler):

    prun = deepof.data.Project(
        path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        arena="circular-autodetect",
        arena_dims=380,
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).run()

    # get breaks
    breaks = {i: np.array([10] * 10) for i in ["test", "test2"]}

    # get supervised annotations from project
    supervised_annotations = prun.supervised_annotation()

    # align supervised and unsupervised labels
    aligned_labels = deepof.analyze.align_deepof_supervised_and_unsupervised_labels(
        supervised_annotations,
        breaks,
        animal_id=(
            sampler.draw(st.sampled_from(["B", "W"])) if mode == "multi" else None
        ),
        aggregate=sampler.draw(st.one_of(st.just(np.mean), st.just(np.median))),
    )

    assert isinstance(aligned_labels, pd.DataFrame)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    mode=st.one_of(st.just("single"), st.just("multi")), sampler=st.data(),
)
def test_annotate_time_chunks(mode, sampler):
    prun = deepof.data.Project(
        path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        arena="circular-autodetect",
        arena_dims=380,
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).run()

    # get breaks and soft_counts
    breaks = {}
    soft_counts = {}
    for i in ["test", "test2"]:
        counts = np.random.normal(size=(10, 10))
        breaks[i] = np.array([10] * 10)
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # get supervised annotations from project
    supervised_annotations = prun.supervised_annotation()

    # annotate time chunks
    time_chunks, hard_counts = deepof.analyze.annotate_time_chunks(
        prun,
        soft_counts,
        breaks,
        supervised_annotations,
        animal_id=(
            sampler.draw(st.sampled_from(["B", "W"])) if mode == "multi" else None
        ),
        kin_derivative=sampler.draw(st.integers(min_value=1, max_value=2)),
        include_distances=sampler.draw(st.booleans()),
        include_angles=sampler.draw(st.booleans()),
    )

    assert isinstance(time_chunks, pd.DataFrame)
    assert isinstance(hard_counts, pd.Series)


@given(folds=st.integers(min_value=2, max_value=10))
def test_chunk_cv_splitter(folds):

    # Create an example stats matrix with indices as the first feature
    chunk_stats = (
        pd.DataFrame(
            np.random.rand(1000, 3),
            columns=["chunk_id", "mean", "std"],
            index=range(1000),
        )
        .reset_index()
        .values
    )

    assert chunk_stats.shape == (1000, 4)

    # Create a dictionary of breaks, whose sums add up to the number of chunks
    breaks = {i: np.array([10] * 100) for i in range(10)}

    # Compute folds
    cv_splitter = deepof.analyze.chunk_cv_splitter(chunk_stats, breaks, folds)
    assert len(cv_splitter) == folds
