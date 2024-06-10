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
import tensorflow as tf
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from scipy.spatial import distance
from shutil import rmtree

import deepof.data
import deepof.visuals

# TESTING SOME AUXILIARY FUNCTIONS #

@settings(deadline=None)
@given(
    all_vertices=st.lists(
        st.lists(
        st.tuples(
            st.floats(min_value=0, max_value=1000),
            st.floats(min_value=0, max_value=1000),
        ), min_size=1, max_size=10
        ), min_size=1, max_size=10
    ),
    num_points=st.integers(min_value=1, max_value=10000),
)
def test_calculate_average_arena(all_vertices, num_points):
    max_length = max(len(lst) for lst in all_vertices)+1
    if num_points > max_length:
        avg_arena=deepof.visuals.calculate_average_arena(all_vertices, num_points)
        assert len(avg_arena)==num_points



    