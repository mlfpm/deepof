# @author NoCreativeIdeaForGoodUserName
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals_utils

"""

import os
import string
import random

import cv2
import pytest

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import settings, example
from hypothesis import strategies as st
from hypothesis import reproduce_failure
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from types import SimpleNamespace
import warnings

import deepof.data
import deepof.export_video
import deepof.utils
import deepof.visuals_utils
from deepof.data import TableDict
from deepof.visuals_utils import (
    calculate_average_arena,
    _filter_embeddings,
    _get_polygon_coords,
    _process_animation_data,
    create_bin_pairs,
    cohend,
    _preprocess_time_bins,
    _apply_rois_to_bin_info,
    _preprocess_embedding_evaluation,
)
from deepof.utils import (
    connect_mouse,
    get_supervised_behaviors_in_roi,
    get_unsupervised_behaviors_in_roi,
    get_behavior_frames_in_roi,
    seconds_to_time,
    time_to_seconds,
)
from tests.test_objects.test_objects import (
    ANIMAL_IDS,
    TEST_CONDITIONS,
    create_project,
    make_project,
    project_dir,
    random_embeddings,
    random_soft_counts,
    random_supervised,
    get_supervised_tables,
    CUSTOM_BEHAVIORS,
    CUSTOM_BEHAVIOR_CONTEXT,
)


# TESTING SOME AUXILIARY FUNCTIONS #


@given(
    second=st.floats(min_value=0, max_value=100000),
    full_second=st.integers(min_value=0, max_value=100000),
)
def test_time_conversion(second, full_second):
    assert full_second == time_to_seconds(seconds_to_time(float(full_second)))
    second = np.round(second * 10**9) / 10**9 #up to 9 digits allowed
    second_second=time_to_seconds(seconds_to_time(float(second), cut_milliseconds=False)) 
    assert second == second_second #this pun is intended and necessary


@given(
    r=st.integers(min_value=0, max_value=255),
    g=st.integers(min_value=0, max_value=255),
    b=st.integers(min_value=0, max_value=255),
)
def test_color_conversion(r, g, b):
    bgr_color = (b, g, r)
    rgb_color = (r, g, b)
    
    # Test BGR -> hex -> BGR round trip
    assert bgr_color == deepof.visuals_utils.hex_to_BGR(deepof.visuals_utils.BGR_to_hex(bgr_color))
    
    # Test that RGB_to_hex and BGR_to_hex produce the same hex for equivalent colors
    assert deepof.visuals_utils.RGB_to_hex(rgb_color) == deepof.visuals_utils.BGR_to_hex(bgr_color)


@settings(max_examples=2, deadline=None)
@given(

    experiment_type=st.one_of(
        st.just("test_multi_topview"),
        st.just("test_single_topview"),
    ),
    named_single_mouse=st.booleans()
)
def test_get_behavior_colors(experiment_type,named_single_mouse):

    if experiment_type == "test_multi_topview":
        animal_ids=["B","W"]
    elif experiment_type == "test_single_topview":
        animal_ids = None
    if named_single_mouse and experiment_type == "test_single_topview":
        aid_addon = "mouse1"

    prun = create_project(experiment_type, exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"], animal_ids=animal_ids)

    supervised = prun.supervised_annotation(custom_behaviors=CUSTOM_BEHAVIORS, custom_behavior_context=CUSTOM_BEHAVIOR_CONTEXT)
    behaviors=list(supervised['test'].keys())

    # remove continuous behaviors (they currently do not get a color attributed)
    continuous_behaviors, _=deepof.visuals_utils.generate_behavior_combinations(animal_ids,False,False,False,True)
    behaviors=list(set(behaviors)-set(continuous_behaviors))

    df=supervised['test']
    if named_single_mouse and experiment_type == "test_single_topview":
        behaviors = [aid_addon+"_"+beh for beh in behaviors]
        animal_ids=aid_addon
        df.columns = [aid_addon+"_"+beh for beh in df.columns]

    colors_a = deepof.visuals_utils.get_behavior_colors(behaviors,animal_ids, custom_behaviors=CUSTOM_BEHAVIORS)
    colors_b = deepof.visuals_utils.get_behavior_colors(behaviors,df, custom_behaviors=CUSTOM_BEHAVIORS)

    #check if all supervised behaviors have a color
    assert not None in colors_a
    #check if generated Colors stay the same independent of animal id retrieval
    assert colors_a == colors_b


def test_get_behavior_colors_fallback():
    # Behaviors unknown to deepOF (e.g. labels of other datasets) get distinct fallback colors, known ones keep theirs
    colors = deepof.visuals_utils.get_behavior_colors(["attack", "B_moving", "mount"], ["B", "W"])
    known = deepof.visuals_utils.get_behavior_colors(["B_moving"], ["B", "W"])
    assert colors[1] == known[0] and len({colors[0], colors[2], colors[1]}) == 3



@settings(deadline=None)
@given(
    all_vertices=st.lists(
        st.lists(
            st.tuples(
                st.floats(min_value=0, max_value=1000),
                st.floats(min_value=0, max_value=1000),
            ),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=10,
    ),
    num_points=st.integers(min_value=1, max_value=10000),
)
def test_calculate_average_arena(all_vertices, num_points):
    all_vertices

    all_vertices_dict={}
    for index, element in enumerate(all_vertices):
        all_vertices_dict[index] = element

    max_length = max(len(lst) for lst in all_vertices_dict.values()) + 1
    if num_points > max_length:
        avg_arena = calculate_average_arena(all_vertices_dict, num_points)
        assert len(avg_arena) == num_points



@given(
    keys=st.lists(
        st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=1, max_size=10),
        min_size=1, max_size=10, unique=True), 
    exp_condition=st.one_of(st.just('Cond1'),st.just('Cond2')) 
)
def test_filter_embeddings(keys,exp_condition):
    

    class Pseudo_Coordinates:
        def __init__(self, keys):
            self._exp_conditions = {}
 
            #create random exp conditions
            for i, key in enumerate(keys):
                if (i+1)%2==0:
                    self._exp_conditions[key]=pd.DataFrame([['even','blubb']],columns=['Cond1','Cond2'])
                else:
                    self._exp_conditions[key]=pd.DataFrame([['odd','blobb']],columns=['Cond1','Cond2'])

        @property
        def get_exp_conditions(self):
            """Return the stored dictionary with experimental conditions per subject."""
            return self._exp_conditions
    
    coordinates=Pseudo_Coordinates(keys)

    # Define a test embedding dictionary
    embeddings = {i: np.random.normal(size=(100, 10)) for i in keys}
    soft_counts = random_soft_counts(keys, 100, 2, table_dict=False)
    supervised_annotations= TableDict({i: pd.DataFrame(embeddings[i]) for i in keys}, typ='supervised')

    embeddings, soft_counts, supervised_annotations, concat_hue = _filter_embeddings(
    coordinates,
    embeddings,
    soft_counts,
    supervised_annotations,
    exp_condition,
    )

    N_keys=len(embeddings.keys())
    if exp_condition=='Cond1':
        comp_list=['odd' if i % 2 == 0 else 'even' for i in range(N_keys)]
        assert concat_hue == comp_list
    else:
        comp_list=['blobb' if i % 2 == 0 else 'blubb' for i in range(N_keys)]
        assert concat_hue == comp_list
    assert embeddings.keys()==soft_counts.keys()
    assert embeddings.keys()==supervised_annotations.keys()


@settings(deadline=None, max_examples=25)
@given(
    include_behaviors=st.one_of(st.just(None),st.just(["B_moving"]),st.just(["B_moving","B_stat-active","B_stat-passive"])),
    window_size=st.integers(5,10),
    alignment_mode=st.one_of(st.just("any"),st.just("center")),
    minimum_number_of_positives=st.integers(5,100),
    normalize=st.booleans(),
    exp_type=st.sampled_from(["test_single_topview", "test_multi_topview"]),
    )
def test_preprocess_embedding_evaluation(include_behaviors,window_size,alignment_mode,minimum_number_of_positives,normalize,exp_type):


    prun = create_project(exp_type, animal_ids=ANIMAL_IDS[exp_type], exp_conditions=TEST_CONDITIONS)

    # Define test embeddings and supervised annotations (12 unnamed columns if no behaviors are given)
    keys = prun._tables.keys()
    embeddings = random_embeddings(keys, 100 - window_size + 1, 10)
    supervised_annotations = random_supervised(keys, 100, include_behaviors if include_behaviors is not None else 12)

    eval=_preprocess_embedding_evaluation(
        coordinates=prun,
        embeddings=embeddings,
        supervised_annotations=supervised_annotations,
        include_behaviors=include_behaviors,
        window_size=window_size,
        alignment_mode=alignment_mode,
        minimum_number_of_positives = minimum_number_of_positives,
        normalize=normalize,
    ) 

    # empty case: as none of the default behaviors are in our supervised behaviors, we get an empty dataframe back
    if include_behaviors is None:
        assert isinstance( eval, pd.DataFrame)
        assert len(eval)==0     
    else:
        # All behaviors occur in evaluation data
        assert all([beh in list(eval.behavior) for beh in include_behaviors])

        # Number of total findows corresponds to widnows fitting into frames in project 
        assert eval.n_windows[0]==(100-window_size+1)*2

        # Parameters are in expected range
        assert all([ np.isnan(trace_cov_pos_norm_global) or (trace_cov_pos_norm_global>=0 and trace_cov_pos_norm_global<=1.1) for trace_cov_pos_norm_global in list(eval.trace_cov_pos_norm_global)]) 
        assert all([ np.isnan(ap_mean) or (ap_mean>=0 and ap_mean<=1) for ap_mean in list(eval.ap_mean)]) 
        assert all([ np.isnan(pos_knn_agree_mean) or (pos_knn_agree_mean>=0 and pos_knn_agree_mean<=1) for pos_knn_agree_mean in list(eval.pos_knn_agree_mean)])


@given(
    template=st.one_of(st.just("deepof_14"),st.just("deepof_11"),st.just("deepof_8")), #deepof_8 does not have all required body parts
    animal_id=st.one_of(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=1, max_size=20), st.just(None)),
)
def test_get_polygon_coords(template,animal_id):
    
    # Get body parts and coords
    features=connect_mouse(animal_ids=animal_id, graph_preset=template).nodes()
    features=list(features)
    coordinates = ['x', 'y']

    # Create a MultiIndex for the columns
    multi_index_columns = pd.MultiIndex.from_product(
        [features, coordinates],
        names=['Feature', 'Coordinate']
    )

    # Generate random data and dataframe
    data = np.random.rand(len(features)*len(coordinates),len(features)*len(coordinates))
    df = pd.DataFrame(data, columns=multi_index_columns)
    
    [head, body, tail]=_get_polygon_coords(df,animal_id)

    if template=="deepof_14":
        assert head.shape[1]==8
        assert body.shape[1]==12
        assert tail.shape[1]==8
    elif template=="deepof_11":
        assert head.shape[1]==8
        assert body.shape[1]==12
        assert tail.shape[1]==4
    elif template=="deepof_8":
        assert head.shape[1]==6
        assert body.shape[1]==6
        assert tail.shape[1]==4


@settings(max_examples=20, deadline=None)
@given(
    min_confidence=st.floats(min_value=0.0, max_value=0.5),
    min_bout_duration=st.integers(min_value=1, max_value=5),
    selected_cluster=st.integers(min_value=0, max_value=1),
)
def test_process_animation_data(min_confidence,min_bout_duration,selected_cluster):
    
    animal_id="test_"
    # Get body parts and coords
    features=connect_mouse("deepof_14").nodes()
    features=[feature[10:] for feature in features]
    coordinates = ['x', 'y']

    # Create a MultiIndex for the columns
    multi_index_columns = pd.MultiIndex.from_product(
        [features, coordinates],
        names=['Feature', 'Coordinate']
    )

    # Generate random data and dataframe
    data = np.random.rand(len(features)*len(coordinates),len(features)*len(coordinates))
    coords = pd.DataFrame(data, columns=multi_index_columns)
    

    # add animal ids
    coords.columns = pd.MultiIndex.from_tuples(
        [(f"{animal_id}{feature}", coordinate) for feature, coordinate in coords.columns]
    )

    #create random embeddings and soft counts
    cur_embeddings = np.random.normal(size=(len(features)*len(coordinates), 10))
    counts = np.abs(np.random.normal(size=(len(features)*len(coordinates), 2)))
    cur_soft_counts = counts / counts.sum(axis=1)[:, None]
    cur_soft_counts[0:6,:]=[1,0] # add strongly confident bouts to not trigger assert
    cur_soft_counts[-1:-6:-1,:]=[0,1]
    
    (   
        coords,
        cur_embeddings,
        cluster_embedding,
        concat_embedding,
        hard_counts,
    ) = _process_animation_data(
        coords,
        cur_embeddings=cur_embeddings,
        cur_soft_counts=cur_soft_counts,
        min_confidence=min_confidence,
        min_bout_duration=min_bout_duration,
        selected_cluster=selected_cluster
    )

    assert coords.shape[0]==np.sum(hard_counts==selected_cluster) #data from correct cluster was selected for coords
    assert cur_embeddings[0].shape[0]>=concat_embedding.shape[0] #concatenated embeddings are of equal size or smaller than original
    assert cur_embeddings[0].shape[1]==concat_embedding.shape[1] #embeddings were reshaped to 2D
    assert cluster_embedding[0].shape[0]==coords.shape[0]  #data from correct cluster was selected for cluster_embedding


@given(
    L_array=st.integers(min_value=1, max_value=100000),
    N_time_bins=st.integers(min_value=1, max_value=100),
)
def test_create_bin_pairs(L_array, N_time_bins):
    assert all(np.diff(create_bin_pairs(L_array, N_time_bins)) >= 0)


@given(
    array_a=st.lists(
        elements=st.floats(min_value=-10e10, max_value=-0.00001), min_size=1, max_size=500
    ),
    array_b=st.lists(
        elements=st.floats(min_value=-10e10, max_value=-0.00001), min_size=1, max_size=500
    ),
    array_c=st.lists(
        elements=st.floats(min_value=0.00001, max_value=10e10), min_size=1, max_size=500
    ),
    array_d=st.lists(
        elements=st.floats(min_value=0.00001, max_value=10e10), min_size=1, max_size=500
    ),
)
def test_cohend(array_a, array_b, array_c, array_d):
    # tests for symmetry, scaling and constant invariance of cohends d
    assert (
        cohend(np.array(array_a) * 2, np.array(array_b) * 2)
        + cohend(np.array(array_b) + 1, np.array(array_a) + 1)
        < 10e-5
    )
    assert (
        cohend(np.array(array_a) * 2, np.array(array_c) * 2)
        + cohend(np.array(array_c) + 1, np.array(array_a) + 1)
        < 10e-5
    )
    assert (
        cohend(np.array(array_c) * 2, np.array(array_d) * 2)
        + cohend(np.array(array_d) + 1, np.array(array_c) + 1)
        < 10e-5
    )


# define pseudo coordinates object only containing properties necessary for testing bin preprocessing
class Pseudo_Coordinates:
    def __init__(self, start_times_raw, frame_rate):
        self._frame_rate = frame_rate
        self._start_times = {}
        self._table_lengths = {}  
        start_marker = None,
        
        # set start time as time strings
        for i, start_time in enumerate(start_times_raw):
            start_time = seconds_to_time(start_time)
            self._start_times[f"key{i + 1}"] = start_time

        # set lengths as a minimum of start time + 10 seconds
        for i, start_time in enumerate(start_times_raw):
            min_length = 120 * frame_rate
            self._table_lengths[f"key{i + 1}"] = int(time_to_seconds(self._start_times[f"key{i + 1}"])* frame_rate) + int(min_length)


    def add_table_lengths(self, lengths):
        """Add multiple table lengths with keys 'key1', 'key2', etc."""
        for i, length in enumerate(lengths):
            self._table_lengths[f"key{i + 1}"] = int(length)

    def get_start_times(self, start_marker=None):
        return self._start_times

    def get_table_lengths(self, tab_dict_for_binning=None, start_marker=None):
        return self._table_lengths


def test_preprocess_time_bins_align_lengths():
    # By default all experiments are cut to the shortest one; with align_lengths=False they keep their full range
    coords = Pseudo_Coordinates([0, 0, 0], frame_rate=10)
    coords.add_table_lengths([96, 21364, 500])
    lengths = lambda bin_info: [len(bins) for bins in bin_info.values()]
    assert lengths(_preprocess_time_bins(coordinates=coords, samples_max=None)) == [96, 96, 96]
    assert lengths(_preprocess_time_bins(coordinates=coords, samples_max=None, align_lengths=False)) == [96, 21364, 500]
    # Cutting instead of downsampling keeps the frames consecutive
    cut = _preprocess_time_bins(coordinates=coords, samples_max=200, down_sample=False, align_lengths=False)
    assert np.array_equal(cut["key2"], np.arange(200))


@settings(deadline=None, max_examples=200)    
@given(
    start_times_raw=st.lists(
        elements=st.integers(min_value=0, max_value=120), min_size=5, max_size=50
    ),
    frame_rate=st.floats(min_value=1, max_value=60),
    bin_size=st.floats(min_value=1, max_value=120),
    bin_index=st.floats(min_value=0, max_value=100),
    is_int=st.booleans(),
    has_precomputed_bins=st.booleans(),
    samples_max=st.integers(min_value=10, max_value=2000),
    experiment_id=st.one_of(
        st.just(None),
        st.just("key1"),
    ),
    makes_sense=st.one_of(
        st.just("yes"),
        st.just("no"),
        st.just("no bins")
    )
)
def test_preprocess_time_bins(
    start_times_raw, frame_rate, bin_size, bin_index, is_int, has_precomputed_bins, samples_max, experiment_id, makes_sense
    ):
    
    # Only allow up to 8 decimales for float inputs 
    # (because of time string conversion limitations this otherwise leads to 1-index deviations 
    # in requested and required result, causing the test to fail)
    bin_size=np.round(bin_size, decimals=8)
    bin_index=np.round(bin_index, decimals=8)

    # Create Pseudo_Coordinates
    coords = Pseudo_Coordinates(start_times_raw,frame_rate)
    precomputed_bins=None

    # Simulate precomputed bin input 
    # (_preprocess_time_bins just skips them, so I don't know why I put the effort in that)
    if has_precomputed_bins:
        precomputed_bins=np.array([False]*int((bin_index+bin_size)*frame_rate+10))
        start=int(bin_index*frame_rate)
        stop=start+int(bin_size*frame_rate)
        precomputed_bins[start:stop]=True

    # Simulate index and time string user inputs
    if is_int:
        bin_size_user = int(bin_size)
        max_bin_no=(120*frame_rate)/np.round(bin_size_user*frame_rate)-1
        bin_index_user = int(np.min([int(bin_index),np.max([0,max_bin_no])]))
    else:
        bin_index_user = seconds_to_time(bin_index, False)
        bin_size_user = seconds_to_time(bin_size, False)

    # Simulate "special" inputs"
    if makes_sense=="no":
        bin_index_user="Banana!"
    elif makes_sense=="no bins":
        bin_size_user=None
        bin_index_user=None
        precomputed_bins=None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_info = _preprocess_time_bins(
        coordinates=coords, bin_size=bin_size_user, bin_index=bin_index_user, precomputed_bins=precomputed_bins, samples_max=samples_max, experiment_id=experiment_id,
        )

    for key in bin_info.keys():
        lengths=coords.get_table_lengths()
        assert isinstance(bin_info[key], np.ndarray)
        if (len(bin_info[key])>0):
            assert bin_info[key][-1] <= lengths[key]
            assert bin_info[key][0] >= 0


@settings(deadline=None)
@given(
    mode=st.one_of(st.just("single"), st.just("multi")),
    bin_size=st.one_of(st.just(None), st.just(100), st.just(50)),
    in_roi_criterion=st.one_of(st.just("Center"), st.just("Nose"), st.just("all"), st.just(["Spine_1","Center","Spine_2"])),
    invert_roi=st.booleans(),
    use_numba=st.booleans(),  # intended to be so low that numba runs (10) or not
)
def test_apply_rois(mode, bin_size, in_roi_criterion, invert_roi,use_numba):

    fast_implementations_threshold = 100000
    if use_numba:
        fast_implementations_threshold = 10

    example = "test_{}_topview".format(mode)
    prun = make_project(example, animal_ids=ANIMAL_IDS[example], fast_implementations_threshold=fast_implementations_threshold)

    #also use large table handling 
    if use_numba:
        prun.very_large_project=True
    
    prun = prun.create(force=True, test=True)
 
    bin_info_time=None
    if bin_size is not None:
        bin_info_time={i: np.arange(0, bin_size) for i in prun._tables.keys()}

    bin_info_roi1=_apply_rois_to_bin_info(coordinates=prun, roi_numbers=1, bin_info_time=bin_info_time,in_roi_criterion=in_roi_criterion, invert_roi=invert_roi)
    bin_info_roi2=_apply_rois_to_bin_info(coordinates=prun, roi_numbers=2, bin_info_time=bin_info_time,in_roi_criterion=in_roi_criterion, invert_roi=invert_roi)

    # bin info is a two level dictionary
    assert isinstance(bin_info_roi1, dict) 
    assert isinstance(bin_info_roi1[list(bin_info_roi1.keys())[0]], dict)
    # There are always more or an equal amount of frames in which the animal is in the larger roi (roi 1) as compared to it being in the smaller roi (roi2) 
    if not invert_roi:
        for key in bin_info_roi1.keys():
            for roi in bin_info_roi1[key].keys():
                assert np.sum(bin_info_roi1[key][roi]) >= np.sum(bin_info_roi2[key][roi]) 
    # The opposite is true if ROIs are inverted
    else:
        for key in bin_info_roi1.keys():
            for roi in bin_info_roi1[key].keys():
                assert np.sum(bin_info_roi1[key][roi]) <= np.sum(bin_info_roi2[key][roi]) 
    

@settings(deadline=None)
@given(
    animal_ids=st.text(alphabet=string.ascii_letters, min_size=0, max_size=3).filter(lambda s: len(s) == len(set(s))),
    bins = st.lists(
        st.integers(min_value=0, max_value=99),
        min_size=10,
        max_size=100,
        unique=True
    ).map(sorted),
    supervised_behavior = st.booleans(),
    roi_mode = st.one_of(st.just("mousewise"),st.just("behaviorwise"))
)
def test_get_rois(animal_ids, bins, supervised_behavior, roi_mode):

    animal_ids=list(animal_ids)
    if len(animal_ids)==0:
        animal_ids=['']

    num_rows = 100
    num_cols = 10

    # Generate column names with random animal_id combinations
    column_names = []
    for col_num in range(1, num_cols + 1):
        k = random.randint(1, len(animal_ids))
        subset = random.sample(animal_ids, k)
        subset.sort()
        if animal_ids[0] != '':
            prefix = '_'.join(subset)
            column_names.append(f"{prefix}_{col_num}")
        else:
            column_names.append(f"{col_num}")

    # Create supervised DataFrame with binary values
    cur_supervised = pd.DataFrame(
        np.random.randint(2, size=(num_rows, num_cols)),
        columns=column_names
    )

    # Convert DataFrame to numpy array for unsupervised data
    cur_unsupervised = cur_supervised.to_numpy().astype(float)

    # Create local_bin_info
    local_bin_info = {"time": np.array(bins)}
    for k, animal_id in enumerate(animal_ids):
        local_bin_info[animal_id] = cur_unsupervised[bins,k].astype(bool)

    # Determine random behavior from column names or as unsupervised dummy
    if supervised_behavior:
        behavior=column_names[0]
    else:
        behavior="_"

    # get ROIs with different methods
    cur_supervised_filtered = get_supervised_behaviors_in_roi(cur_supervised.iloc[bins], local_bin_info, animal_ids,roi_mode=roi_mode)
    cur_unsupervised_filtered = get_unsupervised_behaviors_in_roi(cur_unsupervised[bins], local_bin_info, animal_ids)
    frames = get_behavior_frames_in_roi(behavior, local_bin_info, animal_ids)

    # In the not supervised case, frames represent the non-nan positions in cur_unsupervised after filtering
    if not supervised_behavior:
        assert (cur_unsupervised[frames] == cur_unsupervised_filtered[~np.isnan(cur_unsupervised_filtered).any(axis=1)]).all()
    # if there is only one supervised behavior, frames represent the non-nan positions in cur_supervised after filtering
    elif len(animal_ids)==1:
        assert (cur_supervised.iloc[frames] == cur_supervised_filtered.dropna()).all().all()
    # For multiple animals, selecting frames based on a random behavior is always greater or equal the number of non-nan supervised rows
    # (because the more rows will be set to nan the more mice are involved in a behavior)
    else:
        assert (len(cur_supervised.iloc[frames]) >= len(cur_supervised_filtered.dropna()))
    # unsupervised always only filters by one animal, supervised can filter by combinations, respectively supervised can filter out more but not less
    assert (len((cur_unsupervised_filtered[~np.isnan(cur_unsupervised_filtered).any(axis=1)]))) >= len(cur_supervised_filtered.dropna())


@settings(deadline=None)
@given(
    animal_ids=st.text(alphabet=string.ascii_letters, min_size=0, max_size=3).filter(lambda s: len(s) == len(set(s))),
    bins = st.lists(
        st.integers(min_value=0, max_value=99),
        min_size=10,
        max_size=100,
        unique=True
    ).map(sorted),
)
def test_mousewise_behaviorwise(animal_ids, bins):

    animal_ids=list(animal_ids)
    mono_mouse = False
    if len(animal_ids)==0:
        animal_ids=['']
        mono_mouse=True

    num_rows = 100
    num_cols = 10

    # Generate column names with random animal_id combinations
    column_names = []
    for col_num in range(1, num_cols + 1):
        k = random.randint(1, len(animal_ids))
        subset = random.sample(animal_ids, k)
        subset.sort()
        if animal_ids[0] != '':
            prefix = '_'.join(subset)
            column_names.append(f"{prefix}_{col_num}")
        else:
            column_names.append(f"{col_num}")

    # Create supervised DataFrame with binary values
    cur_supervised = pd.DataFrame(
        np.random.randint(2, size=(num_rows, num_cols)),
        columns=column_names
    )

    # Convert DataFrame to numpy array for bin info generation
    cur_unsupervised = cur_supervised.to_numpy().astype(float)

    # Create local_bin_info
    local_bin_info = {"time": np.array(bins)}
    for k, animal_id in enumerate(animal_ids):
        local_bin_info[animal_id] = cur_unsupervised[bins,k].astype(bool)

    # get ROIs with different methods
    # will set full rows of cur_supervised to NaN if not all required mice are present
    cur_supervised_filtered_mousewise = get_supervised_behaviors_in_roi(cur_supervised.iloc[bins], local_bin_info, animal_ids,roi_mode="mousewise")
    # will set individual values of cur_supervised to NaN based on not all mice for behavior being present
    cur_supervised_filtered_behaviorwise = get_supervised_behaviors_in_roi(cur_supervised.iloc[bins], local_bin_info, animal_ids,roi_mode="behaviorwise")

    # Since mousewise sets full rows to NaN, all behaviors will always have an equal number of NaNs
    assert len(set(cur_supervised_filtered_mousewise.isna().sum()))==1 

    #If there is only one mouse, mousewise and behaviorwise are identical
    if mono_mouse:
        assert cur_supervised_filtered_mousewise.equals(cur_supervised_filtered_behaviorwise)
    # Otherwise, with multiple mice...
    else:
        positions_all_mice = [c for c in cur_supervised_filtered_mousewise.columns if set(animal_ids) <= set(c.split('_'))]
        positions_not_all_mice = [c for c in cur_supervised_filtered_mousewise.columns if not set(animal_ids) <= set(c.split('_'))]
        # ...if all mice are involved in a behavior, mousewise and behaviorwise are identical
        cur_supervised_filtered_mousewise[positions_all_mice].equals(cur_supervised_filtered_behaviorwise[positions_all_mice])
        # ...if not all mice are involved, there will always be at least as many NaNs in the mousewise case or more as it filters behaviors more strictly
        cur_supervised_filtered_mousewise[positions_not_all_mice].isna().sum().sum() >= cur_supervised_filtered_behaviorwise[positions_not_all_mice].isna().sum().sum()
        

@settings(deadline=None)
@given(
    max_val=st.integers(min_value=1, max_value=99),
    preceding_behavior=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=100, max_value=100))),
    proximate_behavior=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=100, max_value=100))),
    frame_rate=st.floats(min_value=1, max_value=100),
    delta_T=st.floats(min_value=0.0, max_value=100),
)
def test_calculate_FSTTC(max_val,preceding_behavior,proximate_behavior,frame_rate,delta_T):

    preceding_behavior=preceding_behavior[0:max_val]
    proximate_behavior=proximate_behavior[0:max_val]
    fsttc=deepof.visuals_utils.calculate_FSTTC(preceding_behavior,proximate_behavior,frame_rate,delta_T)

    # The FSTTC can only reach values in the range between -1 and 1
    assert(1 >= fsttc and fsttc >=-1)


@settings(deadline=None)
@given(
    max_val=st.integers(min_value=1, max_value=99),
    preceding_behavior=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=100, max_value=100))),
    proximate_behavior=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=100, max_value=100))),
    frame_rate=st.floats(min_value=1, max_value=100),
    delta_T=st.floats(min_value=0.0, max_value=100),
)
def test_calculate_simple_association(max_val,preceding_behavior,proximate_behavior,frame_rate,delta_T):

    preceding_behavior=preceding_behavior[0:max_val]
    proximate_behavior=proximate_behavior[0:max_val]
    Q=deepof.visuals_utils.calculate_simple_association(preceding_behavior,proximate_behavior,frame_rate,delta_T)

    # Yule's coefficient Q can only reach values in the range between -1 and 1
    assert(1 >= Q and Q >=-1)

@given(
    binary_table=get_supervised_tables(n_min=1, n_max=1, m_min=1, m_max=100),
)
def test_contiguous_segments(binary_table):
    # yields slices for contiguous True blocks

    binary_array=binary_table.to_numpy()
    slices=deepof.visuals_utils.contiguous_segments(binary_array)

    assert len(slices) <= np.sum(binary_array)
    if (binary_array==1).any():
        assert np.where(binary_array)[1][0] == slices[0].start


@given(
    fps=st.floats(10.0, 120.0, allow_nan=False, allow_infinity=False, width=32),
    mm_to_px=st.floats(0.10000000149011612, 10.0, allow_nan=False, allow_infinity=False, width=32),
    value=st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False, width=32),
)
def test_scale_units(fps, mm_to_px, value):
    key = "k"
    coordinates = SimpleNamespace(_frame_rate=fps, _scales={key: (0.0, 0.0, mm_to_px, 1.0)})

    out, out_u = deepof.visuals_utils.scale_units(coordinates, key, value, None)
    # unit=None returns unchanged
    assert out == value and out_u is None

    # known conversions (fps and mm_to_px dependent)
    out, out_u = deepof.visuals_utils.scale_units(coordinates, key, 100.0, "frames", None, "s")
    assert np.allclose(out, 100.0 / fps, rtol=0.0, atol=1e-9) and out_u == "s"

    out, out_u = deepof.visuals_utils.scale_units(coordinates, key, 10.0, "mm", "px", None)
    assert np.allclose(out, 10.0 * mm_to_px, rtol=0.0, atol=1e-9) and out_u == "px"

    out, out_u = deepof.visuals_utils.scale_units(coordinates, key, 1.0, "mm/s", "px", "frames")
    assert np.allclose(out, (1.0*mm_to_px) / fps, rtol=0.0, atol=1e-9) and out_u == "px/frames"

    # known conversions (fps and mm_to_px independent)
    out, out_u = deepof.visuals_utils.scale_units(coordinates, key, 10.0, "mm", "cm", None)
    assert np.allclose(out, 1.0, rtol=0.0, atol=1e-9) and out_u == "cm"

    out, out_u = deepof.visuals_utils.scale_units(coordinates, key, 60.0, "s", None, "min")
    assert np.allclose(out, 1.0, rtol=0.0, atol=1e-9) and out_u == "min"

    # round-trip: A->B->A preserves value
    out1, _ = deepof.visuals_utils.scale_units(coordinates, key, value, "mm", "px", None)
    out2, _ = deepof.visuals_utils.scale_units(coordinates, key, out1, "px", "mm", None)
    assert np.allclose(out2, value, rtol=0.0, atol=1e-9)

    out1, _ = deepof.visuals_utils.scale_units(coordinates, key, value, "frames", None, "s")
    out2, _ = deepof.visuals_utils.scale_units(coordinates, key, out1, "s", None, "frames")
    assert np.allclose(out2, value, rtol=0.0, atol=1e-9)

    out1, _ = deepof.visuals_utils.scale_units(coordinates, key, value, "mm/s", "px", "frames")
    out2, _ = deepof.visuals_utils.scale_units(coordinates, key, out1, "px/frames", "mm", "s")
    assert np.allclose(out2, value, rtol=0.0, atol=1e-9)


@given(n=st.integers(1, 100))
def test_get_square_shape_for_gridlike_plot(n):
    r, c = deepof.visuals_utils.get_square_shape_for_gridlike_plot(n)
    assert r * c == n
    if n == 12: assert (r, c) == (4, 3)


######
# Regression testing of plot Data for consitency
######


def test_mouse_roi_interaction():
    prun = create_project("test_multi_topview", animal_ids=["B", "W"], bodypart_graph="deepof_11", exp_conditions=None, frame_rate=25)

    roi = np.array([[158.61861862, 154.05405405],
                    [276.15615616, 152.91291291],
                    [276.15615616, 260.18018018],
                    [158.61861862, 260.18018018]])
    prun._roi_dicts = {"test": {1: roi}, "test2": {1: roi}}

    # Create exp_conditions with pandas DataFrames containing "CSDS" column
    prun._exp_conditions = {
        "test": pd.DataFrame({"CSDS": ["test_cond1"]}),
        "test2": pd.DataFrame({"CSDS": ["test_cond2"]}),
    }

    ref_path = os.path.join(".", "tests", "test_examples", "test_data", "mouse_roi_interaction")
    os.makedirs(ref_path, exist_ok=True)

    # FOV mode with experiment_ids
    effect_fov, group_fov = deepof.visuals.return_mouse_roi_interaction(
        prun, animal_id="B", roi_number=1, N_time_bins=20, mode="fov",
        experiment_ids={"a": ["test"], "b": ["test2"]}, error_bars="std", unit_distance="pixel",
    )
    
    ref_fov_e = os.path.join(ref_path, "fov_effect.csv")
    ref_fov_g = os.path.join(ref_path, "fov_group.csv")

    if not os.path.exists(ref_fov_e):
        effect_fov.to_csv(ref_fov_e, index=False)
        group_fov.to_csv(ref_fov_g, index=False)
        
    # Output data matches reference (FOV mode)
    pd.testing.assert_frame_equal(effect_fov, pd.read_csv(ref_fov_e), atol=1e-9, check_like=True, check_dtype=False)
    pd.testing.assert_frame_equal(group_fov, pd.read_csv(ref_fov_g), atol=1e-9, check_like=True, check_dtype=False)

    # Distance mode with custom bins
    effect_dist, group_dist = deepof.visuals.return_mouse_roi_interaction(
        prun, bodyparts="B_Nose", mode="distance",
        custom_time_bins=[[0, 2], [3, 6], [7, 22], [21, 99]],
        hide_time_bins=[False, True, False, True],
        exp_condition="CSDS" ,condition_values=["test_cond1","test_cond2"],
        experiment_ids="test",
    )
    
    ref_dist_e = os.path.join(ref_path, "distance_effect.csv")
    ref_dist_g = os.path.join(ref_path, "distance_group.csv")

    if not os.path.exists(ref_dist_e):
        effect_dist.to_csv(ref_dist_e, index=False)
        group_dist.to_csv(ref_dist_g, index=False)
    
    # Output data matches reference (distance mode)
    pd.testing.assert_frame_equal(effect_dist, pd.read_csv(ref_dist_e), atol=1e-9, check_like=True, check_dtype=False)
    pd.testing.assert_frame_equal(group_dist, pd.read_csv(ref_dist_g), atol=1e-9, check_like=True, check_dtype=False)
 
    #Test raw data extraction
    raw_data_fov = deepof.visuals.return_mouse_roi_interaction(
        prun, animal_id="B", roi_number=1, mode="fov",
        experiment_ids={"a": ["test"], "b": ["test2"]}, error_bars="sem", unit_distance="m", get_raw_data=True, fov_angle_deg=150,
    )

    ref_raw_e = os.path.join(ref_path, "raw_data.csv")

    if not os.path.exists(ref_raw_e):
        raw_data_fov.to_csv(ref_raw_e, index=False)

    pd.testing.assert_frame_equal(raw_data_fov, pd.read_csv(ref_raw_e), atol=1e-9, check_like=True, check_dtype=False)


def test_transitions():
    prun = create_project("test_multi_topview", animal_ids=["B", "W"], bodypart_graph="deepof_11", exp_conditions=None, frame_rate=25)

    # Create exp_conditions with pandas DataFrames containing "CSDS" column
    prun._exp_conditions = {
        "test": pd.DataFrame({"CSDS": ["test_cond1"]}),
        "test2": pd.DataFrame({"CSDS": ["test_cond2"]}),
    }

    # Generate supervised annotations (replace with actual method if different)
    supervised_annotation = prun.supervised_annotation(custom_behaviors=CUSTOM_BEHAVIORS, custom_behavior_context=CUSTOM_BEHAVIOR_CONTEXT)

    ref_path = os.path.join(".", "tests", "test_examples", "test_data", "transitions")

    # Call 1: silence_diagonal=True, normalize=False, delta_T=2
    result1 = deepof.visuals.return_transitions(
        prun,
        supervised_annotations=supervised_annotation,
        visualization="heatmaps",
        bin_size=6,
        bin_index=0,
        exp_condition="CSDS",
        delta_T=2,
        normalize=False,
        silence_diagonal=True,
    )
    
    ref1 = os.path.join(ref_path, "transitions_silenced.csv")  
    
    pd.testing.assert_frame_equal(result1, pd.read_csv(ref1, index_col=0), atol=1e-9, check_like=True, check_dtype=False)

    # Call 2: normalize=True, diagonal_behavior_counting="Events"
    result2 = deepof.visuals.return_transitions(
        prun,
        supervised_annotations=supervised_annotation,
        visualization="networks",
        bin_size=3,
        bin_index=1,
        exp_condition="CSDS",
        delta_T=1,
        normalize=True,
        silence_diagonal=False,
        diagonal_behavior_counting="Events",
    )
    
    ref2 = os.path.join(ref_path, "transitions_normalized.csv")
    
    pd.testing.assert_frame_equal(result2, pd.read_csv(ref2, index_col=0), atol=1e-9, check_like=True, check_dtype=False)


def test_sample_bouts():
    import deepof.export_video
    # three bouts at the start, middle and end of a long recording
    frames = np.concatenate([np.arange(0, 50), np.arange(10000, 10050), np.arange(90000, 90050)])
    picked = deepof.export_video._sample_bouts(frames, 80, random_state=3)
    assert len(picked) == 80 and np.all(np.diff(picked) > 0) and np.isin(picked, frames).all()
    assert picked.max() >= 10000  # not only the beginning of the recording
    assert np.array_equal(picked, deepof.export_video._sample_bouts(frames, 80, random_state=3))  # reproducible
    assert np.array_equal(deepof.export_video._sample_bouts(frames, np.inf), frames)


def test_cluster_enrichment_stats():
    # Animal-level time shares per cluster; a cluster preferred by one condition differs significantly, corrected
    # p-values are valid probabilities for every cluster
    rng = np.random.default_rng(0)
    rows = []
    for animal in range(12):
        condition = "wt" if animal < 6 else "ko"
        shares = [0.25] * 4 if condition == "wt" else [0.7, 0.1, 0.1, 0.1]
        rows += [(f"m{animal}", c, condition) for c in rng.choice(4, size=300, p=shares)]
    df = pd.DataFrame(rows, columns=["experiment_id", "Cluster", "genotype"])
    props = deepof.visuals_utils._compute_animal_proportions(df, "genotype")
    assert props.shape == (12, 5) and np.allclose(props[[0, 1, 2, 3]].sum(axis=1), 1)
    p = deepof.visuals_utils.cluster_enrichment_stats(df, "genotype")
    assert set(p) == {0, 1, 2, 3} and all(0 <= v <= 1 for v in p.values()) and p[0] < 0.05


@given(
    frame_rate=st.sampled_from([10, 25.0]),
    length=st.integers(50, 2000),
    bin_size=st.integers(1, 20),
    step_fraction=st.sampled_from([None, 0.5, 1.0]),
)
def test_build_valid_multibins(frame_rate, length, bin_size, step_fraction):
    # Fixed-size (or sliding) bins of bin_size seconds cover all frames, the last one possibly shorter; time strings
    # give the same bins; custom bins are converted to frames, overlapping custom bins are warned about
    coords = SimpleNamespace(_frame_rate=frame_rate)
    bin_step = None if step_fraction is None else max(1, int(bin_size * step_fraction))
    bins, hidden = deepof.visuals_utils.build_valid_multibins(coords, 4, length, min_bins_required=1, bin_size=bin_size,
                                                              bin_step=bin_step)
    size, step = int(round(bin_size * frame_rate)), int(round((bin_step or bin_size) * frame_rate))
    assert bins[0][0] == 0 and bins[-1][1] == length - 1 and len(hidden) == len(bins) and not hidden.any()
    assert all(b - a + 1 <= size for a, b in bins) and all(b[0] - a[0] == step for a, b in zip(bins, bins[1:]))
    as_strings = deepof.visuals_utils.build_valid_multibins(
        coords, 4, length, min_bins_required=1, bin_size=deepof.utils.seconds_to_time(bin_size),
        bin_step=None if bin_step is None else deepof.utils.seconds_to_time(bin_step))[0]
    assert as_strings == bins


def test_build_valid_multibins_custom():
    coords = SimpleNamespace(_frame_rate=10)
    bins, hidden = deepof.visuals_utils.build_valid_multibins(
        coords, 4, 1000, custom_time_bins=[[0, 9], ["00:00:01", "00:00:02"]], hide_time_bins=[False, True],
        min_bins_required=1)
    assert bins == [[0, 9], [10, 20]] and hidden.tolist() == [False, True]
    with pytest.warns(UserWarning, match="overlapping"):
        deepof.visuals_utils.build_valid_multibins(coords, 4, 1000, custom_time_bins=[[0, 20], [10, 30]],
                                                   min_bins_required=1)


def test_animal_level_roi_units():
    # Pooled body parts measure every animal with the condition value, prefixed ones only their animal; fov uses the
    # head of every animal (or of the given one); prefixed and pooled body parts can not be mixed
    coords = SimpleNamespace(_animal_ids=["B", "W"], get_animal_conditions={
        "v1": pd.DataFrame({"genotype": ["wt", "ko"]}, index=["B", "W"]),
        "v2": pd.DataFrame({"genotype": ["wt", "wt"]}, index=["B", "W"]),
    })
    units = lambda mode, bodyparts=None, animal_id=None: deepof.visuals_utils._animal_level_roi_units(
        coords, "genotype", ["wt", "ko"], mode, bodyparts, animal_id)
    assert units("distance", ["Nose"]) == {
        "wt": {"v1": [("B", ["B_Nose"])], "v2": [("B", ["B_Nose"]), ("W", ["W_Nose"])]},
        "ko": {"v1": [("W", ["W_Nose"])]}}
    assert units("distance", ["W_Nose", "W_Center"]) == {
        "wt": {"v2": [("W", ["W_Nose", "W_Center"])]}, "ko": {"v1": [("W", ["W_Nose", "W_Center"])]}}
    head = lambda a: [f"{a}_Left_ear", f"{a}_Nose", f"{a}_Right_ear"]
    assert units("fov")["ko"] == {"v1": [("W", head("W"))]}
    assert units("fov", animal_id="B") == {"wt": {"v1": [("B", head("B"))], "v2": [("B", head("B"))]}, "ko": {}}
    with pytest.raises(ValueError, match="all with animal prefix"):
        units("distance", ["B_Nose", "Nose"])


def test_return_supervised_summary():
    # One row per video and time bin; binary behaviors as time or as share of the bin (the same counts), continuous
    # ones as mean and std; inside a ROI never more than overall; the table is saved in the project
    coords = create_project("test_multi_topview", animal_ids=["B", "W"], exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
                            exp_conditions={"test": pd.DataFrame({"CSDS": ["a"]}),
                                            "test2": pd.DataFrame({"CSDS": ["b"]})})
    supervised = coords.supervised_annotation()
    summary = lambda **kw: deepof.visuals_utils._return_supervised_summary(coords, supervised, **kw)
    frames = summary(N_time_bins=3, unit_time="frames", save_table=True)
    shares = summary(N_time_bins=3, binary_units="fraction", save_table=False)
    assert len(frames) == 6 and sorted(frames["bin_number"].unique()) == [0, 1, 2]
    assert os.listdir(os.path.join(project_dir(coords), "Out_tables"))
    behaviors = [c for c in shares.columns if c.startswith("B_") and "[" not in c]
    timed = [f"{b} [frames]" for b in behaviors]
    assert behaviors and ((shares[behaviors] >= 0) & (shares[behaviors] <= 1)).all().all()
    ratio = frames[timed].to_numpy() / np.maximum(shares[behaviors].to_numpy(), 1e-12)
    lengths = ratio[shares[behaviors].to_numpy() > 0]
    assert np.allclose(lengths, np.round(lengths))  # frames / share = number of frames of the bin
    assert any("_mean [" in c for c in frames.columns) and any("_std [" in c for c in frames.columns)
    in_roi = summary(N_time_bins=3, unit_time="frames", roi_number=1, animals_in_roi=["B"], save_table=False)
    assert (in_roi[timed].to_numpy() <= frames[timed].to_numpy() + 1e-9).all()


def _frame_counts(folder):
    """{file name: number of frames} of the videos in folder."""
    counts = {}
    for name in os.listdir(folder):
        cap = cv2.VideoCapture(os.path.join(folder, name))
        counts[name] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return counts


def test_output_videos_per_cluster(tmp_path):
    # Artificial soft counts with known clusters (frames 0-29: cluster 0, 30-59: cluster 1, 60-99: cluster 2): every
    # cluster video holds exactly the frames of its cluster from the selected videos, also with a condition filter, a
    # confidence threshold, time bins, a frame limit per video and a minimum bout duration
    coords = create_project(exp_conditions={"test": pd.DataFrame({"CSDS": ["a"]}), "test2": pd.DataFrame({"CSDS": ["b"]})})
    n = min(coords.get_table_lengths().values())
    labels = np.minimum(np.arange(n) // 30, 2)
    confidence = {"test": 0.8, "test2": 0.4}
    soft_counts = {}
    for key, conf in confidence.items():
        sc = np.full((n, 3), (1 - conf) / 2, dtype=np.float32)
        sc[np.arange(n), labels] = conf
        soft_counts[key] = sc
    soft_counts = deepof.data.TableDict(soft_counts, typ="soft_counts")
    expected = np.bincount(labels, minlength=3)

    def export(name, **kwargs):
        folder = str(tmp_path / name)
        deepof.export_video.output_videos_per_cluster(
            coords, exp_conditions=kwargs.pop("exp_conditions", None), behavior_dict=soft_counts, behaviors=None,
            behaviors_renamed=None, single_output_resolution=(160, 120), out_path=folder,
            min_bout_duration=kwargs.pop("min_bout_duration", 1),
            video_export_config=deepof.export_video.VideoExportConfig(supervised_export=False), **kwargs)
        counts = _frame_counts(folder)
        return [next(v for k, v in counts.items() if k.startswith(f"Behavior=Cluster_{c}_")) for c in range(3)]

    assert export("all") == list(2 * expected)
    assert export("condition", exp_conditions={"CSDS": "a"}) == list(expected)
    assert export("confident", min_confidence=0.5) == list(expected)  # test2 is below the threshold
    bins = {key: {"time": np.arange(40, n)} for key in confidence}
    assert export("bins", bin_info=bins) == list(2 * np.bincount(labels[40:], minlength=3))
    assert export("limit", frame_limit_per_video=10) == [20, 20, 20]
    assert export("long_bouts", min_bout_duration=35) == [0, 0, 2 * expected[2]]  # only the 40 frame bouts


def test_output_annotated_video(tmp_path):
    # All overlays (arena, ROI, body parts, animal labels, time, counter, binary and continuous behaviors) on selected
    # frames, written in the requested size
    coords = create_project("test_multi_topview", animal_ids=["B", "W"], exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"])
    supervised = coords.supervised_annotation()
    tab = supervised["test"]
    behaviors = [c for c in tab.columns if c.startswith("B_")][:3] + ["B_speed"]
    config = deepof.export_video.VideoExportConfig(
        display_video_name=True, display_time=True, display_counter=True, display_arena=True, display_markers=True,
        display_mouse_labels=True, display_roi=1, display_loading_bar=False)
    deepof.export_video.output_annotated_video(
        coords, "test", tab, behaviors, video_export_config=config, frames=np.arange(5, 25), v_width=200, v_height=150,
        out_path=str(tmp_path), behaviors_renamed=[b.upper() for b in behaviors])
    (name, count), = _frame_counts(str(tmp_path)).items()
    cap = cv2.VideoCapture(str(tmp_path / name))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert count == 20 and size == (200, 150)
