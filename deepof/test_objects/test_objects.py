# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

"""

premade objects to import for tests (the types of tables and stuff we use often in DeepOF)

"""

import os
from itertools import combinations
from shutil import rmtree

import networkx as nx
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from scipy.spatial import distance
from shapely.geometry import Point, Polygon


import deepof.data
import deepof.utils
import deepof.arena_utils

# Generate soft counts
@st.composite
def get_soft_counts(draw, n_min=1, n_max=50, m_min=1, m_max=10):
    n = draw(st.integers(n_min, n_max)); m = draw(st.integers(m_min, m_max))
    raw = draw(arrays(np.float32, (n, m), elements=st.floats(0.0010000000474974513, 1.0, allow_nan=False, allow_infinity=False, width=32)))
    return raw / raw.sum(axis=1, keepdims=True)

# Generate supervised tables
@st.composite
def get_supervised_tables(draw, n_min=1, n_max=50, m_min=1, m_max=10):
    n = draw(st.integers(n_min, n_max))
    m = draw(st.integers(m_min, m_max))
    data = draw(arrays(np.int8, (n, m), elements=st.integers(min_value=0, max_value=1)))
    cols = [f"col_{i}" for i in range(m)]  # unique column names
    return pd.DataFrame(data, columns=cols)