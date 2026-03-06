# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.post_hoc

"""
import os
from shutil import rmtree
from typing import Optional, Any, Dict, NewType, Union, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import deepof.data
import deepof.post_hoc
from deepof.data import TableDict


@settings(deadline=None, max_examples=25)
@given(states=st.sampled_from([3, 2, "aic", "bic"]))
def test_get_contrastive_soft_counts(states):

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
        video_scale="380 mm",
        video_format=".mp4",
        animal_ids=[""],
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True, test=True)

    # Define a test embedding dictionary
    embeddings = {i: np.random.normal(size=(100, 10)) for i in range(10)}

    embeddings=deepof.data.TableDict(
        embeddings,
        typ="unsupervised_embedding",
        table_path=None, 
        exp_conditions=None,
    )
    
    if isinstance(states, int):
        # Define a test matrix of soft counts
        soft_counts = {}
        for i in range(10):
            counts = np.abs(np.random.normal(size=(100, states)))
            soft_counts[i] = counts / counts.sum(axis=1)[:, None]
    else:
        soft_counts = None

    soft_counts_out = deepof.post_hoc.get_contrastive_soft_counts(
        prun,
        embeddings,
        states=states,
        min_states=2,
        max_states=3,
        soft_counts=soft_counts,
    )

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )
    
    # For each key, soft_counts have 100 rows (since embeddings have 100 rows), rows should sum to 1 each, so 100 rows sum to 100
    assert all([np.sum(soft_counts_out[key])==100.0 for key in soft_counts_out.keys()])

    # Check if Ideal states determined based on different modes stay consistent 
    # (i.e. this is what teh tests currently return, if these results based on teh same inputs change, we have an issue)
    if states != "bic" and states != 2:
        assert all([np.shape(soft_counts_out[key])[1]==3 for key in soft_counts_out.keys()])
    else:
        assert all([np.shape(soft_counts_out[key])[1]==2 for key in soft_counts_out.keys()])


@settings(deadline=None, max_examples=25)
@given(K_pose=st.sampled_from([3, 2]), M_bins=st.sampled_from([1, 2]), window_size=st.sampled_from([6, 12]),distance_bp=st.sampled_from(["Nose", "Center"]),exp_type=st.sampled_from(["test_single_topview", "test_multi_topview"]))
def test_get_contrastive_soft_counts_gmm(K_pose,M_bins,window_size,distance_bp,exp_type):


    animal_ids = [""]
    if not exp_type=="test_single_topview":
        animal_ids=["B","W"]

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", exp_type),
        video_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            exp_type,
            "Videos",
        ),
        table_path=os.path.join(
            ".",
            "tests",
            "test_examples",
            exp_type,
            "Tables",
        ),
        animal_ids=animal_ids,
        arena="circular-autodetect",
        video_scale="380 mm",
        video_format=".mp4",
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True, test=True)

    # Define a test embedding dictionary
    keys=prun._tables.keys()
    embeddings = {key: np.random.normal(size=(100-window_size+1, 10)) for key in keys}

    embeddings=deepof.data.TableDict(
        embeddings,
        typ="unsupervised_embedding",
        table_path=None, 
        exp_conditions=None,
    )
    
    soft_counts_out = deepof.post_hoc.get_contrastive_soft_counts_gmm(
        prun,
        embeddings,
        animal_ids=animal_ids,
        window_size=window_size,
        K_pose=K_pose,
        M_bins=M_bins,
        distance_bp=distance_bp,
    )

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", exp_type, "deepof_project"
        )
    )
    
    # For each key, soft_counts have 100 rows (since embeddings have 100 rows), rows should sum to 1 each, so 100 rows sum to 100
    assert all([np.round(np.sum(soft_counts_out[key]))==100-window_size+1 for key in soft_counts_out.keys()])

    # Check if Ideal states determined based on different modes stay consistent 
    # (i.e. this is what teh tests currently return, if these results based on teh same inputs change, we have an issue)
    if exp_type=="test_single_topview":
        assert all([np.shape(soft_counts_out[key])[1]==K_pose for key in soft_counts_out.keys()])
    else:
        assert all([np.shape(soft_counts_out[key])[1]==K_pose*M_bins for key in soft_counts_out.keys()])


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
        video_scale="380 mm",
        video_format=".mp4",
        animal_ids=[""],
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True, test=True)

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

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )


def test_get_time_on_cluster():

    # Define a test matrix of soft counts
    soft_counts = {}
    bin_info = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]
        bin_info[i] = {}
        bin_info[i]["time"]=np.array([0,1,2,3,4,5,6,7,8]) 
        bin_info[i][""]=np.array([True]*5+[False]*4)

    roi_number=1

    toc = deepof.post_hoc.get_time_on_cluster(soft_counts,False)
    toc2 = deepof.post_hoc.get_time_on_cluster(soft_counts,False, False, bin_info,roi_number)

    # Assert that both the soft counts and breaks are correctly aggregated
    assert toc.shape[0] * 100 == np.concatenate(list(soft_counts.values())).shape[0]
    assert toc.shape[1] == np.concatenate(list(soft_counts.values())).shape[1] # one shorter due to binning
    assert np.sum(np.sum(toc)) > np.sum(np.sum(toc2))
    assert all(np.sum(toc, axis=1)==100)    
    assert all(np.sum(toc2, axis=1)==5)


@given(
    reduce_dim=st.booleans(),
    agg=st.sampled_from(["mean", "median"]),
    bins = st.lists(
        st.integers(min_value=0, max_value=99),
        min_size=10,
        max_size=100,
        unique=True
    ),
    roi_number = st.integers(min_value=1, max_value=2),
    animals_in_roi = st.one_of(st.just(["A"]),st.just(["A","B"]))
)
def test_get_aggregated_embedding(reduce_dim, agg, bins, roi_number, animals_in_roi):

    # Define a test embedding dictionary
    embedding = {i: np.random.normal(size=(100, 10)) for i in range(10)}

    # Create local_bin_info
    bin_info = {i: {} for i in range(10)}
    for key in bin_info:
        local_bin_info = {"time": np.array(bins)}
        for k, animal_id in enumerate(animals_in_roi):
            local_bin_info[animal_id] = np.ones(len(bins)).astype(bool)
            if key==0:
                local_bin_info[animal_id] = np.zeros(len(bins)).astype(bool)
        bin_info[key]=local_bin_info

    aggregated_embeddings = deepof.post_hoc.get_aggregated_embedding(
        embedding, reduce_dim, agg, bin_info, roi_number=roi_number, animals_in_roi = animals_in_roi
    )

    assert aggregated_embeddings.shape[0] == len(embedding)
    # PCA reduced to 2 dimensions
    if reduce_dim:
        assert aggregated_embeddings.shape[1] == 2
    # We purposefully set one experiment to all false in teh roi bins, so this experiment should have been removed
    else:
        assert aggregated_embeddings.dropna().shape[0] == 9


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    scan_mode=st.sampled_from(["growing_window", "per-bin", "precomputed"]),
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

    # fallback for not defined scan mode
    precomputed_bins=(np.ones(9)*11).astype(int)
    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}

    distance_binning = deepof.post_hoc.condition_distance_binning(
        embedding=embedding,
        soft_counts=soft_counts,
        exp_conditions=exp_conditions,
        start_bin=11,
        end_bin=99,
        step_bin=11,
        scan_mode=scan_mode,
        precomputed_bins=precomputed_bins,
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

    # Define a test embedding dictionary and derive supervised annotations from it
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))
    supervised_annotations= TableDict({i: pd.DataFrame(embedding[i].numpy()) for i in range(10)}, typ='supervised')
    for key in supervised_annotations.keys():
        supervised_annotations[key].columns=pd.Index(["speed","1","2","3","4","5","6","7","8","9"])


    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    soft_counts=TableDict(soft_counts, typ='soft_counts')

    bin_info={i: {"time": np.arange(0, bin_size)} for i in range(10)}


    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}

    enrichment = deepof.post_hoc.enrichment_across_conditions(
        soft_counts,
        supervised_annotations=(
            None
            if not supervised
            else supervised_annotations
        ),
        exp_conditions=exp_conditions,
        bin_info=bin_info,
        normalize=normalize,
    )

    assert isinstance(enrichment, pd.DataFrame)
    assert enrichment.shape[0] > 0
    assert enrichment.shape[1] == 4


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
    silence_diagonal=st.booleans(),
    steady_state_entropy=st.booleans(),
)
def test_compute_transition_matrix_per_condition(
    bin_size, silence_diagonal, aggregate, normalize, steady_state_entropy
):

    # Define a test embedding dictionary
    embedding = {i: tf.random.normal(shape=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))

    # Define a test matrix of soft counts
    soft_counts = {}
    for i in range(10):
        counts = np.random.normal(size=(100, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}

    #convert into used formats
    soft_counts=TableDict(soft_counts, typ='soft_counts')
    bin_info={i: {"time": np.arange(0, bin_size)} for i in range(10)}

    
    transitions = deepof.post_hoc.compute_transition_matrix_per_condition(
        soft_counts,
        exp_conditions,
        silence_diagonal,
        bin_info=bin_info,
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
        video_scale="380 mm",
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True, test=True)

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

    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "deepof_project",
        )
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

    # np.random.seed(42)
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
        video_scale="380 mm",
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
    ).create(force=True, test=True)

    # get breaks and soft_counts
    soft_counts = {}
    for i in ["test", "test2"]:
        counts = np.random.normal(size=(100 - 5 + 1, 10))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]

    # get supervised annotations from project
    supervised_annotations = prun.supervised_annotation()

    #convert into used formats
    soft_counts=TableDict(soft_counts, typ='soft_counts')

    # annotate time chunks
    time_chunks, hard_counts, breaks = deepof.post_hoc.annotate_time_chunks(
        prun,
        soft_counts,
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

    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            "deepof_project",
        )
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
