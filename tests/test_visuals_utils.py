# @author NoCreativeIdeaForGoodUserName
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals_utils

"""

import os
import string
import random

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import settings, example
from hypothesis import strategies as st
from hypothesis import reproduce_failure
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from shutil import rmtree
import warnings

import deepof.data
from deepof.data import TableDict
from deepof.utils import connect_mouse
from deepof.visuals_utils import (
    time_to_seconds,
    seconds_to_time,
    calculate_average_arena,
    _filter_embeddings,
    _get_polygon_coords,
    _process_animation_data,
    create_bin_pairs,
    cohend,
    _preprocess_time_bins,
    _apply_rois_to_bin_info,
    get_supervised_behaviors_in_roi,
    get_unsupervised_behaviors_in_roi,
    get_beheavior_frames_in_roi,
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


@settings(max_examples=2, deadline=None)
@given(

    experiment_type=st.one_of(
        st.just("test_multi_topview"),
        st.just("test_single_topview"),
    )
)
def test_get_behavior_colors(experiment_type):

    if experiment_type == "test_multi_topview":
        animal_ids=["B","W"] 
    else:
        animal_ids = None

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", experiment_type),
        video_path=os.path.join(
            ".", "tests", "test_examples", experiment_type, "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", experiment_type, "Tables"
        ),
        arena="circular-autodetect",
        exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
        video_scale=380,
        animal_ids=animal_ids,
        video_format=".mp4",
        table_format=".h5",
    ).create(force=True, test=True)

    supervised = prun.supervised_annotation()
    behaviors=list(supervised['test'].keys())

    colors_a = deepof.visuals_utils.get_behavior_colors(behaviors,animal_ids)
    colors_b = deepof.visuals_utils.get_behavior_colors(behaviors,supervised['test'])

    #check if all supervised behaviors have a color
    assert not None in colors_a
    #check if generated Colors stay the same independent of animal id retrieval
    assert colors_a == colors_b


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
    soft_counts = {}
    for i in keys:
        counts = np.abs(np.random.normal(size=(100, 2)))
        soft_counts[i] = counts / counts.sum(axis=1)[:, None]
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


@given(
    template=st.one_of(st.just("deepof_14"),st.just("deepof_11")), #deepof_8 does not have all required body parts
    animal_id=st.one_of(st.text(alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', min_size=1, max_size=20), st.just(None)),
)
def test_get_polygon_coords(template,animal_id):
    
    # Get body parts and coords
    features=connect_mouse(template).nodes()
    features=[feature[10:] for feature in features]
    coordinates = ['x', 'y']

    # Create a MultiIndex for the columns
    multi_index_columns = pd.MultiIndex.from_product(
        [features, coordinates],
        names=['Feature', 'Coordinate']
    )

    # Generate random data and dataframe
    data = np.random.rand(len(features)*len(coordinates),len(features)*len(coordinates))
    df = pd.DataFrame(data, columns=multi_index_columns)
    
    #to include None-case in testing for that 1 line of extra coverage
    if animal_id is None:
        a_id=""
    else:
        a_id=animal_id+"_"

    # add animal ids
    df.columns = pd.MultiIndex.from_tuples(
        [(f"{a_id}{feature}", coordinate) for feature, coordinate in df.columns]
    )

    [head, body, tail]=_get_polygon_coords(df,animal_id)

    assert head.shape[1]==8
    assert body.shape[1]==12
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
    cur_embeddings = np.random.normal(size=(len(features)*len(coordinates)-5, 10))
    counts = np.abs(np.random.normal(size=(len(features)*len(coordinates)-5, 2)))
    cur_soft_counts = counts / counts.sum(axis=1)[:, None]

    
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
        elements=st.floats(min_value=-10e10, max_value=-0.00001), min_size=5, max_size=500
    ),
    array_b=st.lists(
        elements=st.floats(min_value=-10e10, max_value=-0.00001), min_size=5, max_size=500
    ),
    array_c=st.lists(
        elements=st.floats(min_value=0.00001, max_value=10e10), min_size=5, max_size=500
    ),
    array_d=st.lists(
        elements=st.floats(min_value=0.00001, max_value=10e10), min_size=5, max_size=500
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
        
        # set start time as time strings
        for i, start_time in enumerate(start_times_raw):
            start_time = seconds_to_time(start_time)
            self._start_times[f"key{i + 1}"] = start_time

        # set lengths as a minimum of start time + 10 seconds
        for i, start_time in enumerate(start_times_raw):
            min_length = 120 * frame_rate
            self._table_lengths[f"key{i + 1}"] = int(min_length)


    def add_table_lengths(self, lengths):
        """Add multiple table lengths with keys 'key1', 'key2', etc."""
        for i, length in enumerate(lengths):
            self._table_lengths[f"key{i + 1}"] = int(length)

    def get_start_times(self):
        return self._start_times

    def get_table_lengths(self):
        return self._table_lengths


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
    makes_sense=st.one_of(
        st.just("yes"),
        st.just("no"),
        st.just("no bins")
    )
)
def test_preprocess_time_bins(
    start_times_raw, frame_rate, bin_size, bin_index, is_int, has_precomputed_bins, samples_max, makes_sense
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
        coordinates=coords, bin_size=bin_size_user, bin_index=bin_index_user, precomputed_bins=precomputed_bins, samples_max=samples_max,
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
    bin_size=st.one_of(st.just(100), st.just(50)),
    in_roi_criterion=st.one_of(st.just("Center"), st.just("Nose")),
    use_numba=st.booleans(),  # intended to be so low that numba runs (10) or not
)
def test_apply_rois(mode, bin_size, in_roi_criterion, use_numba):

    fast_implementations_threshold = 100000
    if use_numba:
        fast_implementations_threshold = 10

    if mode == "multi":
        animal_ids = ["B", "W"]
    else:
        animal_ids = [""]

    prun = deepof.data.Project(
        project_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode), "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode), "Tables"
        ),
        project_name=f"deepof_project_roi_test",
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
        fast_implementations_threshold=fast_implementations_threshold,
    )

    #also use large table handling 
    if use_numba:
        prun.very_large_project=True
    
    prun = prun.create(force=True, test=True)
 
    bin_info_time={i: np.arange(0, bin_size) for i in prun._tables.keys()}

    bin_info_roi1=_apply_rois_to_bin_info(coordinates=prun, roi_number=1, bin_info_time=bin_info_time,in_roi_criterion=in_roi_criterion)
    bin_info_roi2=_apply_rois_to_bin_info(coordinates=prun, roi_number=2, bin_info_time=bin_info_time,in_roi_criterion=in_roi_criterion)

    # bin info is a two level dictionary
    assert isinstance(bin_info_roi1, dict) 
    assert isinstance(bin_info_roi1[list(bin_info_roi1.keys())[0]], dict)
    # There are always more or an equal amount of frames in which the animal is in the larger roi (roi 1) as compared to it being in the smaller roi (roi2) 
    for key in bin_info_roi1.keys():
        for roi in bin_info_roi1[key].keys():
            assert np.sum(bin_info_roi1[key][roi]) >= np.sum(bin_info_roi2[key][roi]) 
    

@settings(deadline=None)
@given(

    animal_ids=st.text(alphabet=string.ascii_letters, min_size=0, max_size=3),
    bins = st.lists(
        st.integers(min_value=0, max_value=99),
        min_size=10,
        max_size=100,
        unique=True
    ).map(sorted),
    supervised_behavior = st.booleans(),
)
def test_get_rois(animal_ids, bins, supervised_behavior):

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
    cur_supervised_filtered = get_supervised_behaviors_in_roi(cur_supervised.iloc[bins], local_bin_info, animal_ids)
    cur_unsupervised_filtered = get_unsupervised_behaviors_in_roi(cur_unsupervised[bins], local_bin_info, animal_ids)
    frames = get_beheavior_frames_in_roi(behavior, local_bin_info, animal_ids)

    # In the not supervised case, frames represent the non-nan positions in cur_unsupervised after filtering
    if not supervised_behavior:
        assert (cur_unsupervised[frames] == cur_unsupervised_filtered[~np.isnan(cur_unsupervised_filtered).any(axis=1)]).all()
    # if there is only one supervised behavior, frames represent the non-nan positions in cur_supervised after filtering
    elif len(animal_ids)==1:
        assert (cur_supervised.iloc[frames] == cur_supervised_filtered.dropna()).all().all()
    # For multiple animals, selecting frames based on a random behavior is always greator or equal the number of non-nan supervised rows
    # (because the more rows will be set to nan the more mice are involved in a behavior)
    else:
        assert (len(cur_supervised.iloc[frames]) >= len(cur_supervised_filtered.dropna()))
    # unsupervised always onyl filters by one animal, supervised can filter by combinations, respectively supervised can filter out more but not less
    assert (len((cur_unsupervised_filtered[~np.isnan(cur_unsupervised_filtered).any(axis=1)]))) >= len(cur_supervised_filtered.dropna())


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

    # The FSTTC can only reach values in teh range between -1 and 1
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
    fsttc=deepof.visuals_utils.calculate_simple_association(preceding_behavior,proximate_behavior,frame_rate,delta_T)

    # Yule's coefficient Q can only reach values in the range between -1 and 1
    assert(1 >= fsttc and fsttc >=-1)