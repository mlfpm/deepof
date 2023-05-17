# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.post_hoc

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
from hypothesis.extra import numpy as hnp
from shutil import rmtree

import deepof.post_hoc
import deepof.data


@settings(deadline=None, max_examples=25)
@given(states=st.sampled_from([3, "aic", "bic", "priors"]))
def test_recluster(states):

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_single_topview",
            "Videos",
        ),
        table_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_single_topview",
            "Tables",
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=[""],
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True)
    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    # Define a test embedding dictionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}

    if states == "priors":
        # Define a test matrix of soft counts
        soft_counts = {}
        for i in range(10):
            counts = np.abs(np.random.normal(size=(100, 2)))
            soft_counts[i] = counts / counts.sum(axis=1)[:, None]
    else:
        soft_counts = None

    deepof.post_hoc.recluster(
        prun,
        embedding,
        soft_counts,
        states=states,
        min_states=2,
        max_states=3,
        save=False,
    )


def test_get_time_on_cluster():

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Define a test breaks dictionary
    breaks = {i: [10] * 100 for i in range(10)}

    toc = deepof.post_hoc.get_time_on_cluster(soft_counts, breaks)

    # Assert that both the soft counts and breaks are correctly aggregated
    assert toc.shape[0] * 100 == np.concatenate(list(soft_counts.values())).shape[0]
    assert toc.shape[1] == np.concatenate(list(soft_counts.values())).shape[1]


@given(reduce_dim=st.booleans(), agg=st.sampled_from(["mean", "median"]))
def test_get_aggregated_embedding(reduce_dim, agg):

    # Define a test embedding dictionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}

    aggregated_embeddings = deepof.post_hoc.get_aggregated_embedding(
        embedding, reduce_dim, agg
    )

    assert aggregated_embeddings.shape[0] == len(embedding)


@given(
    bin_size=st.integers(min_value=15, max_value=20),
    bin_index=st.integers(min_value=0, max_value=1),
    supervised=st.booleans(),
)
def test_select_time_bin(bin_size, bin_index, supervised):

    # Define a test embedding dictionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}
    if supervised:
        embedding = {i: pd.DataFrame(embedding[i].numpy()) for i in range(10)}

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Create a dictionary of breaks, whose sums add up to the number of chunks
    breaks = {i: np.array([10] * 100) for i in range(10)}

    embedding, soft_counts, breaks, supervised_annots = deepof.post_hoc.select_time_bin(
        embedding=(embedding if not supervised else None),
        soft_counts=(soft_counts if not supervised else None),
        breaks=(breaks if not supervised else None),
        supervised_annotations=(None if not supervised else embedding),
        bin_size=bin_size,
        bin_index=bin_index,
    )

    # Assert that all returned objects are binned
    if not supervised:
        assert list(embedding.values())[0].shape[0] > 0
        assert list(soft_counts.values())[0].shape[0] > 0
        assert list(breaks.values())[0].shape[0] > 0

    else:
        assert list(supervised_annots.values())[0].shape[0] > 0


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    scan_mode=st.sampled_from(["growing-window", "per-bin"]),
    agg=st.sampled_from(["time_on_cluster", "mean", "median"]),
    metric=st.sampled_from(["auc", "wasserstein"]),
)
def test_condition_distance_binning(scan_mode, agg, metric):

    # Define a test embedding dictionary
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

    distance_binning = deepof.post_hoc.condition_distance_binning(
        embedding=embedding,
        soft_counts=soft_counts,
        breaks=breaks,
        exp_conditions=exp_conditions,
        start_bin=11,
        end_bin=99,
        step_bin=11,
        scan_mode=scan_mode,
        agg=agg,
        metric=metric,
        n_jobs=1,
    )

    assert isinstance(distance_binning, np.ndarray)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    input_data=hnp.arrays(
        shape=(10, 2),
        dtype=np.float32,
        elements=st.floats(min_value=0.0, max_value=1.0, width=32),
    )
)
def test_fit_normative_global_model(input_data):
    normative_model = deepof.post_hoc.fit_normative_global_model(
        pd.DataFrame(input_data)
    )
    assert isinstance(normative_model.sample(10), np.ndarray)


@given(
    bin_size=st.integers(min_value=11, max_value=100),
    normalize=st.booleans(),
    supervised=st.booleans(),
)
def test_cluster_enrichment_across_conditions(bin_size, normalize, supervised):

    # Define a test embedding dictionary
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

    enrichment = deepof.post_hoc.enrichment_across_conditions(
        embedding,
        soft_counts,
        breaks,
        supervised_annotations=(
            None
            if not supervised
            else {i: pd.DataFrame(embedding[i].numpy()) for i in range(10)}
        ),
        exp_conditions=exp_conditions,
        bin_size=bin_size,
        bin_index=0,
        normalize=normalize,
    )

    assert isinstance(enrichment, pd.DataFrame)
    assert enrichment.shape[0] > 0
    assert enrichment.shape[1] == 3


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(n_states=st.integers(min_value=2, max_value=25))
def test_get_transitions(n_states):

    sequence = np.random.choice(range(n_states), 1000, replace=True)
    transitions = deepof.post_hoc.get_transitions(sequence, n_states=n_states)

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

    # Define a test embedding dictionary
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

    transitions = deepof.post_hoc.compute_transition_matrix_per_condition(
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
    steady_states = deepof.post_hoc.compute_steady_state(
        transitions, return_entropy=steady_state_entropy
    )

    assert isinstance(steady_states, dict)
    assert isinstance(list(steady_states.values())[0], float) == steady_state_entropy

    if steady_state_entropy:
        print(steady_states)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    mode=st.one_of(st.just("single"), st.just("multi")),
    exclude=st.one_of(st.just(tuple([""])), st.just(["Tail_1"])),
    sampler=st.data(),
)
def test_align_deepof_kinematics_with_unsupervised_labels(mode, exclude, sampler):

    prun = deepof.data.Project(
        project_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        video_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "Videos",
        ),
        table_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "Tables",
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True)
    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "deepof_project",
        )
    )

    # get breaks
    breaks = {i: np.array([10] * 10) for i in ["test", "test2"]}

    # extract kinematic features
    kinematics = deepof.post_hoc.align_deepof_kinematics_with_unsupervised_labels(
        prun,
        kin_derivative=sampler.draw(st.integers(min_value=1, max_value=2)),
        include_distances=sampler.draw(st.booleans()),
        include_angles=sampler.draw(st.booleans()),
        animal_id=(
            sampler.draw(st.sampled_from(["B", "W"])) if mode == "multi" else None
        ),
    )

    # check that the output is a DataFrame
    assert isinstance(kinematics, dict)


def test_chunk_summary_statistics():

    # Set up testing names for our random features
    body_part_names = ["test_name_{}".format(i) for i in range(4)]

    # Define a matrix of soft counts
    counts = np.random.normal(size=(100, 10))
    soft_counts = counts / counts.sum(axis=1)[:, None]

    # And extract hard counts from them
    hard_counts = np.argmax(soft_counts, axis=1)

    # Define a table with chunk data
    chunked_dataset = np.random.uniform(size=(100, 25, 4))

    kinematic_features = deepof.post_hoc.chunk_summary_statistics(
        chunked_dataset, body_part_names
    )

    assert isinstance(kinematic_features, pd.DataFrame)


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    mode=st.one_of(st.just("single"), st.just("multi"), st.just("madlc")),
    sampler=st.data(),
)
def test_shap_pipeline(mode, sampler):

    np.random.seed(42)
    prun = deepof.data.Project(
        project_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        video_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "Videos",
        ),
        table_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "Tables",
        ),
        exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True)
    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "deepof_project",
        )
    )

    # get breaks and soft_counts
    breaks = {}
    soft_counts = {}
    for i in ["test", "test2"]:
        counts = np.random.normal(size=(100 - 5 + 1, 10))
        breaks[i] = np.array([1] * (100 - 5 + 1))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # get supervised annotations from project
    supervised_annotations = prun.supervised_annotation()

    # annotate time chunks
    time_chunks, hard_counts, breaks = deepof.post_hoc.annotate_time_chunks(
        prun,
        soft_counts,
        breaks,
        supervised_annotations,
        window_size=5,
        animal_id=(
            sampler.draw(st.sampled_from(["B", "W"])) if mode != "multi" else None
        ),
        kin_derivative=sampler.draw(st.integers(min_value=1, max_value=2)),
        include_distances=sampler.draw(st.booleans()),
        include_angles=sampler.draw(st.booleans()),
        aggregate=sampler.draw(st.one_of(st.just("mean"), st.just("seglearn"))),
    )

    assert isinstance(time_chunks, pd.DataFrame)
    assert isinstance(hard_counts, pd.Series)
    assert isinstance(breaks, dict)


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
    cv_splitter = deepof.post_hoc.chunk_cv_splitter(chunk_stats, breaks, folds)
    assert len(cv_splitter) == folds
