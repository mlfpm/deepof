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

# Generate embeddings
@st.composite
def get_embeddings_tab_dict(draw, keys, n_min=1, n_max=50, m_min=1, m_max=10):

    n = draw(st.integers(n_min, n_max)); m = draw(st.integers(m_min, m_max))
    emb_dict={}
    for key in keys:
        emb_dict[key]=draw(arrays(np.float32, (n, m), elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=32)))

    return deepof.data.TableDict(
        emb_dict,
        typ="unsupervised_embedding",
        table_path=None, 
        exp_conditions=None,
    )


# Generate supervised tables
@st.composite
def get_supervised_tables(draw, n_min=1, n_max=50, m_min=1, m_max=10):
    n = draw(st.integers(n_min, n_max))
    m = draw(st.integers(m_min, m_max))
    data = draw(arrays(np.int8, (n, m), elements=st.integers(min_value=0, max_value=1)))
    cols = [f"col_{i}" for i in range(m)]  # unique column names
    return pd.DataFrame(data, columns=cols)


# Generate embeddings
@st.composite
def get_embeddings_tab_dict(draw, keys, n_min=1, n_max=50, m_min=1, m_max=10):

    n = draw(st.integers(n_min, n_max)); m = draw(st.integers(m_min, m_max))
    emb_dict={}
    for key in keys:
        emb_dict[key]=draw(arrays(np.float32, (n, m), elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False, width=32)))

    return deepof.data.TableDict(
        emb_dict,
        typ="unsupervised_embedding",
        table_path=None, 
        exp_conditions=None,
    )


# Generate supervised tabdicts
@st.composite
def get_supervised_tab_dict(draw, keys, col_names=None, n_min=1, n_max=50, m_min=1, m_max=10):

    if col_names is not None:
        assert len(col_names)==m_min and len(col_names)==m_max, "For this test-object if custom column names are given, number of columns must match with number of names"
    n = draw(st.integers(n_min, n_max)); m = draw(st.integers(m_min, m_max))
    sup_dict={}
    for key in keys:
        data=draw(arrays(np.int8, (n, m), elements=st.integers(min_value=0, max_value=1)))
        if col_names is None:
            cols = [f"col_{i}" for i in range(m)]
        else:
            cols=col_names
        sup_dict[key]=pd.DataFrame(data, columns=cols)

    return deepof.data.TableDict(
        sup_dict,
        typ="supervised_annotations",
        table_path=None, 
        exp_conditions=None,
    )


def get_embeddings_tab_dict_instance(keys, n_min=1, n_max=50, m_min=1, m_max=10):
    n = np.random.randint(n_min, n_max + 1)
    m = np.random.randint(m_min, m_max + 1)

    emb_dict = {}
    for key in keys:
        emb_dict[key] = np.random.uniform(-3.0, 3.0, size=(n, m)).astype(np.float32)

    return deepof.data.TableDict(
        emb_dict,
        typ="unsupervised_embedding",
        table_path=None,
        exp_conditions=None,
    )


def get_soft_counts_tab_dict_instance(keys, n_min=1, n_max=50, m_min=1, m_max=10):
    n = np.random.randint(n_min, n_max + 1)
    m = np.random.randint(m_min, m_max + 1)

    soft_counts_dict = {}
    for key in keys:

        soft_counts_raw = np.random.uniform(0, 1.0, size=(n, m)).astype(np.float32)
        soft_counts_dict[key] = soft_counts_raw / soft_counts_raw.sum(axis=1, keepdims=True)

    return deepof.data.TableDict(
        soft_counts_dict,
        typ="soft_counts",
        table_path=None,
        exp_conditions=None,
    )


def get_supervised_tab_dict_instance(keys, col_names=None, n_min=1, n_max=50, m_min=1, m_max=10):
    n = np.random.randint(n_min, n_max + 1)
    m = np.random.randint(m_min, m_max + 1)

    if col_names is not None:
        assert len(col_names) == m, \
            f"Expected {m} column names, got {len(col_names)}"

    sup_dict = {}
    for key in keys:
        data = np.random.randint(0, 2, size=(n, m)).astype(np.int8)
        if col_names is None:
            cols = [f"col_{i}" for i in range(m)]
        else:
            cols = col_names
        sup_dict[key] = pd.DataFrame(data, columns=cols)

    return deepof.data.TableDict(
        sup_dict,
        typ="supervised_annotations",
        table_path=None,
        exp_conditions=None,
    )