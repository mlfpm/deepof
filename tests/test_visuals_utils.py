# @author NoCreativeIdeaForGoodUserName
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals

"""

import os
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis import reproduce_failure
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from scipy.spatial import distance
from shutil import rmtree
import warnings

import deepof.data
from deepof.visuals_utils import (
    calculate_average_arena,
    time_to_seconds,
    seconds_to_time,
    create_bin_pairs,
    cohend,
    _preprocess_time_bins,
)

# TESTING SOME AUXILIARY FUNCTIONS #


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
    second=st.floats(min_value=0, max_value=100000),
    full_second=st.integers(min_value=0, max_value=100000),
)
def test_time_conversion(second, full_second):
    assert full_second == time_to_seconds(seconds_to_time(float(full_second)))
    second = np.round(second * 10**9) / 10**9


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


#define pseudo coordinates object only containing properties necessary for testing bin preprocessing
class Pseudo_Coordinates:
    def __init__(self, start_times_raw, frame_rate):
        self._frame_rate = frame_rate
        self._start_times = {}
        self._table_lengths = {}  
        
        #set start time as time strings
        for i, start_time in enumerate(start_times_raw):
            start_time=seconds_to_time(start_time)
            self._start_times[f'key{i + 1}'] = start_time

        #set lengths as a minimum of start time + 10 seconds
        for i, start_time in enumerate(start_times_raw):
            min_length=120*frame_rate
            self._table_lengths[f'key{i + 1}'] = int(min_length)


    def add_table_lengths(self, lengths):
        """Add multiple table lengths with keys 'key1', 'key2', etc."""
        for i, length in enumerate(lengths):
            self._table_lengths[f'key{i + 1}'] = int(length)

    def get_start_times(self):
        return self._start_times

    def get_table_lengths(self):
        return self._table_lengths

    
@given(
    start_times_raw=st.lists(
        elements=st.integers(min_value=0, max_value=120), min_size=5, max_size=50
    ),
    frame_rate=st.floats(min_value=1, max_value=60),
    bin_size=st.floats(min_value=1, max_value=120),
    bin_index=st.floats(min_value=0, max_value=100),
    is_int=st.booleans(),
    has_precomputed_bins=st.booleans(),

)
def test_preprocess_time_bins(start_times_raw, frame_rate,bin_size,bin_index,is_int,has_precomputed_bins):
    
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bin_info = _preprocess_time_bins(
        coordinates=coords, bin_size=bin_size_user, bin_index=bin_index_user, precomputed_bins=precomputed_bins
        )

    for key in bin_info.keys():
        lengths=coords.get_table_lengths()
        assert isinstance(bin_info[key], np.ndarray)
        if (len(bin_info[key])>0):
            assert bin_info[key][-1] <= lengths[key]
            assert bin_info[key][0] >= 0
    