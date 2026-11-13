# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.post_hoc

"""
import os
from typing import Optional, Any, Dict, NewType, Union, Tuple, List

import numpy as np
import pytest
import pandas as pd
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import deepof.data
import deepof.post_hoc
from deepof.data import TableDict
from tests.test_objects.test_objects import (
    ANIMAL_IDS, TEST_CONDITIONS, create_project, random_embeddings, random_soft_counts
)



@settings(deadline=None, max_examples=25)
@given(
    N_clusters_per_gate=st.sampled_from([[3], [2,2], [2,3]]),
    M_gates=st.sampled_from([1, 2]), 
    window_size=st.sampled_from([6, 12]),
    distance_bp=st.sampled_from(["Nose", "Center"]),
    sample_size=st.one_of(st.just(100),st.just(200000)),
    exp_type=st.sampled_from(["test_single_topview", "test_multi_topview"]),
    use_supervised_gates=st.booleans(),
    )
def test_get_contrastive_soft_counts_gmm(N_clusters_per_gate,M_gates,window_size,distance_bp,sample_size,exp_type, use_supervised_gates):


    animal_ids = ANIMAL_IDS[exp_type]
    animal_pair = "" if exp_type == "test_single_topview" else ("B", "W")

    prun = create_project(exp_type, animal_ids=animal_ids, exp_conditions=TEST_CONDITIONS, frame_rate=25)

    supervised_annotations= None
    if use_supervised_gates:
        supervised_annotations = prun.supervised_annotation()

    # Define a test embedding dictionary
    embeddings = random_embeddings(prun._tables.keys(), 100 - window_size + 1, 10)

    if use_supervised_gates:
        gate_edges = None
        # overwrite distance_bp with behavior gate
        if not exp_type=="test_single_topview":
            distance_bp = "B_moving"
        else:
            distance_bp = "moving"
        animal_pair = 'behavior_combinations'
    else:
        gate_edges=deepof.post_hoc.compute_gate_edges(
            prun,
            animal_ids,
            window_size=window_size,
            M_gates=M_gates,
            embedding_gates=distance_bp,
        )
    
    soft_counts_out = deepof.post_hoc.get_contrastive_soft_counts_gmm(
        prun,
        embeddings,
        animal_ids=animal_ids,
        window_size=window_size,
        N_clusters_per_gate=N_clusters_per_gate,
        M_gates=M_gates,
        supervised_annotations=supervised_annotations,
        embedding_gates=distance_bp,
        gate_edges=gate_edges,
        sample_size=sample_size,
    )

    # For each key, soft_counts have 100 rows (since embeddings have 100 rows), rows should sum to 1 each, so 100 rows sum to 100
    assert all([np.round(np.sum(soft_counts_out[animal_pair][key]))==100-window_size+1 for key in soft_counts_out[animal_pair].keys()])

    # Check if the states determined correspond to the requested states
    total_clusters = np.sum(N_clusters_per_gate) if M_gates == len(N_clusters_per_gate) else np.sum(N_clusters_per_gate[0]*M_gates) 
    if exp_type=="test_single_topview" and not use_supervised_gates:
        # Distance gates need animal pairs, so a single animal has one gate; behavior gates keep M_gates
        total_clusters = N_clusters_per_gate[0]
    assert all([np.shape(soft_counts_out[animal_pair][key])[1]==total_clusters for key in soft_counts_out[animal_pair].keys()])

@settings(deadline=None, max_examples=25)
@given(
    N_clusters_per_gate=st.sampled_from([[3], [2,2]]),
    M_gates=st.sampled_from([1, 2]), 
    window_size=st.sampled_from([6, 12]),
    distance_bp=st.sampled_from(["Nose", "Center"]),
    sample_size=st.one_of(st.just(100),st.just(200000)),
    exp_type=st.sampled_from(["test_single_topview", "test_multi_topview"]),
    use_supervised_gates=st.booleans(),
    macro_method=st.sampled_from(["bace", "pcca", "kmeans", "modularity", "pcca_counts"]),
    split_factor=st.sampled_from([None, 1.0]),
    decode_method=st.sampled_from(["viterbi_macro", "forward_backward_macro", "viterbi_micro",
                                   "forward_backward_micro", "lookup"]),
    temporal_smooth_win=st.sampled_from([None, 3]),
    )
def test_get_contrastive_soft_counts_msm_pcca(N_clusters_per_gate,M_gates,window_size,distance_bp,sample_size,exp_type, use_supervised_gates, macro_method, split_factor, decode_method, temporal_smooth_win):


    animal_ids = ANIMAL_IDS[exp_type]
    animal_pair = "" if exp_type == "test_single_topview" else ("B", "W")

    prun = create_project(exp_type, animal_ids=animal_ids, exp_conditions=TEST_CONDITIONS, frame_rate=25)

    supervised_annotations= None
    if use_supervised_gates:
        supervised_annotations = prun.supervised_annotation()

    # Define a test embedding dictionary
    embeddings = random_embeddings(prun._tables.keys(), 100 - window_size + 1, 10)

    if use_supervised_gates:
        gate_edges = None
        # overwrite distance_bp with behavior gate
        if not exp_type=="test_single_topview":
            distance_bp = "B_moving"
        else:
            distance_bp = "moving"
        animal_pair = 'behavior_combinations'
    else:
        gate_edges=deepof.post_hoc.compute_gate_edges(
            prun,
            animal_ids,
            window_size=window_size,
            M_gates=M_gates,
            embedding_gates=distance_bp,
        )

    
    soft_counts_out = deepof.post_hoc.get_contrastive_soft_counts_msm_pcca(
        prun,
        embeddings,
        animal_ids=animal_ids,
        window_size=window_size,
        N_clusters_per_gate=N_clusters_per_gate,
        M_gates=M_gates,
        supervised_annotations=supervised_annotations,
        embedding_gates=distance_bp,
        gate_edges=gate_edges,
        sample_size=sample_size,
        macro_method=macro_method,
        split_factor=split_factor,
        decode_method=decode_method,
        temporal_smooth_win=temporal_smooth_win,
    )

    # For each key, soft_counts have 100 rows (since embeddings have 100 rows), rows should sum to 1 each, so 100 rows sum to 100
    assert all([np.round(np.sum(soft_counts_out[animal_pair][key]))==100-window_size+1 for key in soft_counts_out[animal_pair].keys()])

    # Check if the states determined correspond to the requested states
    total_clusters = np.sum(N_clusters_per_gate) if M_gates == len(N_clusters_per_gate) else np.sum(N_clusters_per_gate[0]*M_gates) 
    if exp_type=="test_single_topview" and not use_supervised_gates:
        # Distance gates need animal pairs, so a single animal has one gate; behavior gates keep M_gates
        total_clusters = N_clusters_per_gate[0]
    widths = [np.shape(soft_counts_out[animal_pair][key])[1] for key in soft_counts_out[animal_pair].keys()]
    if split_factor and macro_method == "pcca":  # split states can only add columns
        assert len(set(widths)) == 1 and widths[0] >= total_clusters
    else:
        assert all(w == total_clusters for w in widths)


@pytest.mark.parametrize("macro_method,split_factor", [("pcca", 1.0), ("pcca", 3.0), ("kmeans", None), ("bace", None), ("modularity", None),
                                                       ("pcca_counts", None)])
def test_regroup_macrostates(macro_method, split_factor):
    # 12 microstates on a line: PCCA state 0 owns 10 of them (83%), state 1 the other two
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    P = rng.random((12, 12)) + np.eye(12) * 5
    chi = np.zeros((12, 2), dtype=np.float32)
    chi[:10, 0], chi[10:, 1] = 1.0, 1.0
    model = {"active_syms": np.arange(12), "kmeans": SimpleNamespace(cluster_centers_=np.arange(12.0)[:, None]),
             "pi_micro": np.full(12, 1 / 12), "P_micro": P / P.sum(1, keepdims=True), "chi": chi, "n_micro_eff": 12,
             "counts": P * 100}
    out = deepof.post_hoc._regroup_macrostates(model, 2, macro_method=macro_method, split_factor=split_factor)
    if split_factor == 3.0:  # 83% < 3 / 2: nothing to split, only crisp memberships
        assert np.array_equal(out["chi"], chi) and np.allclose(out["P_macro"].sum(1), 1)
        return
    if macro_method == "pcca":  # share 0.83 * 2 states -> 2 sub-states, inserted in place of state 0
        assert out["chi"].shape == (12, 3) and np.array_equal(out["chi"][:, 2], chi[:, 1])
    elif macro_method == "kmeans":  # geometric halves of the line instead of the PCCA states
        assert out["chi"].shape == (12, 2) and sorted(out["chi"].sum(0)) == [6.0, 6.0]
    else:  # bace, modularity, pcca_counts: two non-empty states from the transitions
        assert out["chi"].shape == (12, 2) and np.bincount(out["chi"].argmax(1), minlength=2).min() >= 1
    assert np.allclose(out["chi"].sum(1), 1) and np.allclose(out["P_macro"].sum(1), 1)
    assert np.isclose(out["pi_macro"].sum(), 1) and out["micro2macro"].shape == (12, out["chi"].shape[1])


def test_social_share_column_weights():
    # 8 own columns separate the rows in two groups, 2 social columns in two other groups: the microstates follow the
    # own columns when all columns count equally and the social columns when these get most of the distance
    import deepof.clustering.model_utils_new as mu
    rng = np.random.default_rng(0)
    own, social = np.repeat([0, 1], 200), np.tile([0, 1], 200)
    X = np.concatenate([own[:, None] + 0.05 * rng.normal(size=(400, 8)), 0.01 * social[:, None] + 0.0005 * rng.normal(size=(400, 2))], axis=1)
    def micro(weights):
        scaler, kmeans, _ = deepof.post_hoc._fit_microstates_on_sample(
            X, seed=0, n_micro=2, min_micro_per_macro=1, N_clusters_per_gate=2, column_weights=weights)
        return kmeans.predict(scaler.transform(X))
    agree = lambda a, b: max(np.mean(a == b), np.mean(a != b))
    assert mu._social_column_weights(None, 2, 10) is None and agree(micro(None), own) == 1
    weights = mu._social_column_weights(0.9, 2, 10)
    assert np.isclose(2 * weights[-1] ** 2 / (weights ** 2).sum(), 0.9) and agree(micro(weights), social) == 1
    with pytest.raises(ValueError):
        mu._social_column_weights(0.5, 0, 10)


def test_bace_groups():
    # Two blocks of microstates that exchange within and rarely between; the weak state 6 behaves like block 0.
    # The states are nested: 2 states are a merge of the 3 states
    flux = np.full((7, 7), 1.0)
    flux[:3, :3] = flux[3:6, 3:6] = 50.0
    np.fill_diagonal(flux, 500.0)
    flux[6], flux[:, 6], flux[6, 6] = 0.01, 0.01, 2.0
    flux[6, :3] = flux[:3, 6] = 1.0
    bace = lambda k: deepof.post_hoc._bace_groups(flux, k)
    two, three = bace(2), bace(3)
    assert len(set(two[:3])) == 1 and len(set(two[3:6])) == 1 and two[0] != two[3] and two[6] == two[0]
    assert len(set(three)) == 3 and all(len(set(two[three == s])) == 1 for s in range(3))
    assert np.array_equal(bace(9), np.arange(7))
    # Modularity: three blocks of four microstates that exchange within and rarely between
    blocks = np.full((12, 12), 1.0)
    for s in range(0, 12, 4):
        blocks[s:s + 4, s:s + 4] = 50.0
    np.fill_diagonal(blocks, 100.0)
    assert np.array_equal(deepof.post_hoc._modularity_groups(blocks, 3), np.repeat([0, 1, 2], 4))
    # Observed counts between the active microstates only (symbol 2 is not active), in their order
    counts = deepof.post_hoc._transition_counts([np.array([5, 5, 2, 0, 5]), np.array([0, 0])], np.array([0, 5]), 1)
    assert np.array_equal(counts, [[1, 1], [0, 1]])


def test_regroup_macrostates_crisp_share():
    # Fuzzy memberships: state 0 owns 10 of 12 microstates but lends 0.4 of each to state 2, which owns none.
    # Its crisp share (83%) is split, state 2 is dropped (it would decode empty)
    from types import SimpleNamespace
    chi = np.zeros((12, 3), dtype=np.float32)
    chi[:10, 0], chi[:10, 2], chi[10:, 1] = 0.6, 0.4, 1.0
    model = {"active_syms": np.arange(12), "kmeans": SimpleNamespace(cluster_centers_=np.arange(12.0)[:, None]),
             "pi_micro": np.full(12, 1 / 12), "P_micro": np.full((12, 12), 1 / 12), "chi": chi, "n_micro_eff": 12}
    out = deepof.post_hoc._regroup_macrostates(model, 3, split_factor=1.0)  # threshold 1/3 < 0.83 (soft: 0.5)
    assert out["chi"].shape == (12, 3)  # 0 -> round(0.83 * 3) = 2 sub-states, 1 kept, 2 dropped
    assert np.array_equal(out["chi"][:, 2], chi[:, 1]) and set(np.unique(out["chi"])) == {0.0, 1.0}  # crisp
    assert np.all(np.isin(np.arange(3), out["chi"].argmax(1)))  # no empty state


@settings(deadline=None, max_examples=25)
@given(
    m_soft_counts=st.integers(5,10),
    quality_threshold=st.floats(0,1,allow_nan=False),
    frac_bps_below=st.floats(0,1,allow_nan=False),
    window_size=st.integers(5,10),
    exp_type=st.sampled_from(["test_single_topview", "test_multi_topview"]),
    )
def test_add_chaos_gates(m_soft_counts,quality_threshold,frac_bps_below,window_size,exp_type):

    prun = create_project(exp_type, animal_ids=ANIMAL_IDS[exp_type], exp_conditions=TEST_CONDITIONS)

    keys = prun._tables.keys()
    n_rows = 100 - window_size + 1
    soft_counts = {"behavior_combinations": random_soft_counts(keys, n_rows, m_soft_counts)}

    supervised_chaos=deepof.post_hoc.get_supervised_chaos(coordinates=prun,quality_threshold=quality_threshold,frac_bps_below=frac_bps_below)
    m_soft_counts_chaos=supervised_chaos[list(keys)[0]].shape[1]
    # doubled columns to simulate the no-chaos / chaos gating split
    chaos = random_soft_counts(keys, n_rows, m_soft_counts_chaos)
    soft_counts_chaos = {"behavior_combinations": {key: np.concatenate([tab, tab], axis=1) for key, tab in chaos.items()}}

    combined_soft_counts=deepof.post_hoc.add_chaos_gates(
        coordinates=prun,
        soft_counts_dict=soft_counts,
        soft_counts_chaos_dict=soft_counts_chaos,
        supervised_chaos=supervised_chaos,
        window_size=window_size,
    ) 

    # All behaviors occur in evaluation data (only anychaos gets added)
    assert all(
        sc.shape[1] == (m_soft_counts + m_soft_counts_chaos)
        for sc in combined_soft_counts["behavior_combinations"].values()
    )

    # Soft counts still sum to 1
    assert all(
        np.isclose(np.sum(sc), n_rows, atol=1e-05)
        for sc in combined_soft_counts["behavior_combinations"].values()
    )


@settings(deadline=None, max_examples=25)
@given(states=st.sampled_from([3, "aic", "bic", "priors"]))
def test_recluster(states):

    prun = create_project(animal_ids=[""], exp_conditions=TEST_CONDITIONS)

    # Define a test embedding dictionary
    embedding = {i: np.random.normal(size=(100, 10)) for i in range(10)}

    soft_counts = random_soft_counts(range(10), 100, 2, table_dict=False) if states == "priors" else None

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

    soft_counts = random_soft_counts(range(10), 100, 10, table_dict=False)
    bin_info = {i: {"time": np.arange(9), "": np.array([True] * 5 + [False] * 4)} for i in range(10)}

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
    # We purposefully set one experiment to all false in the roi bins, so this experiment should have been removed
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
    embedding = {i: np.random.normal(size=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))

    soft_counts = random_soft_counts(range(10), 100, 10, table_dict=False)

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
    embedding = {i: np.random.normal(size=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))
    supervised_annotations= TableDict({i: pd.DataFrame(embedding[i]) for i in range(10)}, typ='supervised')
    for key in supervised_annotations.keys():
        supervised_annotations[key].columns=pd.Index(["speed","1","2","3","4","5","6","7","8","9"])


    soft_counts = random_soft_counts(range(10), 100, 10)

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
    embedding = {i: np.random.normal(size=(100, 10)) for i in range(10)}
    assert np.all(np.isfinite(embedding[0]))

    soft_counts = random_soft_counts(range(10), 100, 10)

    # Create test experimental conditions
    exp_conditions = {i: i > 4 for i in range(10)}
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

    prun = create_project(
        "test_{}_topview".format(mode),
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        exclude_bodyparts=exclude,
        exp_conditions=TEST_CONDITIONS,
    )

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

    prun = create_project(
        "test_{}_topview".format(mode),
        exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        exp_conditions=TEST_CONDITIONS,
    )

    # get soft counts
    soft_counts = random_soft_counts(["test", "test2"], 100 - 5 + 1, 10)

    # get supervised annotations from project
    supervised_annotations = prun.supervised_annotation()

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

def test_get_gate_entry_ignores_pair_order():
    gates = {("res", "int"): 1, "": 2}
    assert deepof.post_hoc.get_gate_entry(gates, ("int", "res")) == 1
    assert deepof.post_hoc.get_gate_entry(gates, ("res", "int")) == 1
    assert deepof.post_hoc.get_gate_entry(gates, "") == 2
    with pytest.raises(KeyError):
        deepof.post_hoc.get_gate_entry(gates, ("res", "other"))


@pytest.mark.parametrize("method", ["bace", "msm", "kmeans", "combined", "modularity", "pcca_counts"])
def test_cluster_number_selection(method):
    # Three sticky, well separated states: every criterion is computed and the timescale gap finds them
    rng = np.random.default_rng(0)
    means = np.array([[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]])
    embeddings = {}
    for v in range(2):
        states = np.repeat(rng.integers(3, size=30), 60)
        z = (means[states] + rng.normal(0, 0.5, (len(states), 2))).astype(np.float32)
        z[100] = np.nan  # non-finite rows split the segments
        embeddings[f"video{v}"] = z
    df = deepof.post_hoc.cluster_number_selection(embeddings, method=method, k_range=[2, 3, 4, 5], n_micro=20, n_repeats=2)
    assert list(df["K"]) == [2, 3, 4, 5] and df[["explained_dispersion", "metastability", "stability"]].notna().all().all()
    assert set(df.attrs["suggested"]) == {"explained_dispersion", "metastability", "stability", "timescale_gap"}
    assert df.attrs["suggested"]["timescale_gap"] == 3
    with pytest.raises(ValueError):
        deepof.post_hoc.cluster_number_selection(embeddings, method=method, k_range=[1, 2])


def test_msm_pcca_restricted_gates():
    # Restricted to one gate, the frames outside it are dropped (with their embeddings) or set to NaN; invalid gate
    # indices and settings are refused
    prun = create_project("test_multi_topview", animal_ids=ANIMAL_IDS["test_multi_topview"],
                          exp_conditions=TEST_CONDITIONS, frame_rate=25)
    window_size = 6
    embeddings = random_embeddings(prun._tables.keys(), 100 - window_size + 1, 10)
    kwargs = dict(animal_ids=["B", "W"], window_size=window_size, N_clusters_per_gate=[2, 2], M_gates=2,
                  embedding_gates="Center", sample_size=100)
    kwargs["gate_edges"] = deepof.post_hoc.compute_gate_edges(prun, ["B", "W"], window_size=window_size, M_gates=2,
                                                             embedding_gates="Center")
    full = deepof.post_hoc.get_contrastive_soft_counts_msm_pcca(prun, embeddings, **kwargs)[("B", "W")]
    dropped, kept_embeddings = deepof.post_hoc.get_contrastive_soft_counts_msm_pcca(
        prun, embeddings, restrict_return_gates=1, **kwargs)
    masked = deepof.post_hoc.get_contrastive_soft_counts_msm_pcca(
        prun, embeddings, restrict_return_gates=[1], drop_out_of_gate=False, **kwargs)[("B", "W")]
    for key in embeddings.keys():
        rows = dropped[("B", "W")][key]
        nan_rows = np.isnan(masked[key]).all(axis=1)
        assert len(rows) == (~nan_rows).sum() < len(full[key]) and len(kept_embeddings[("B", "W")][key]) == len(rows)
        assert np.allclose(masked[key][~nan_rows].sum(axis=1), 1) and np.allclose(rows.sum(axis=1), 1)
    for wrong, error in [({"restrict_return_gates": 2}, ValueError), ({"restrict_return_gates": 0, "drop_out_of_gate": "yes"}, TypeError),
                         ({"macro_method": "spectral"}, ValueError), ({"split_factor": 0}, ValueError)]:
        with pytest.raises(error):
            deepof.post_hoc.get_contrastive_soft_counts_msm_pcca(prun, embeddings, **kwargs, **wrong)
