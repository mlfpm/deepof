# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

"""

premade objects to import for tests (the types of tables and stuff we use often in DeepOF)

"""

import os
from itertools import combinations
from shutil import rmtree

from deepof.annotation_utils import DeepOF_behavior
from deepof.annotation_utils import Behavior_scope, Behavior_output
from deepof.annotation_utils import animal_ids, BehaviorContext
from deepof.annotation_utils import postprocess_identity

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

# Custom behaviors to import for tests

def mouse_nose_mid_distance(ctx: BehaviorContext, mice_pair: animal_ids):
     
    # As this is going to be a function for paired mouse behavior, we expect mice_pair to contain two animal ids.
    a, b = mice_pair 

    # Now we also need the tracked bodypart positions. We can get them from the "raw_coords" field (a pandas dataTable) in our BehaviorContext ctx
    pos_dframe=ctx.raw_coords
    
    # We are specifically interested in the Noses. So we get the "nsoe" bodypart for both of our animals.
    # The bp function here simply applies the corret formatting. We could also simply write str(a)+"_Nose" 
    nose_m1=ctx.bp(a, "Nose")
    nose_m2=ctx.bp(b, "Nose")

    # And here we calculate all frames within the table in which the noses of both mice are farther away that our "close_contact_tol" 
    # from the supervised parameters but also closer than 3 times that tolerance. Here we create a binary output array
    middle_contact = (
        (np.linalg.norm(pos_dframe[nose_m1] - pos_dframe[nose_m2], axis=1) > float(ctx.params["close_contact_tol"])) & 
        (np.linalg.norm(pos_dframe[nose_m1] - pos_dframe[nose_m2], axis=1) <= 5*float(ctx.params["close_contact_tol"]))
    )
    return middle_contact

def mouse_compression(ctx: BehaviorContext, mouse: animal_ids):    

    # As this is going to be a function for single mouse behavior, we expect mouse to contain only one animal id.
    a=mouse

    # We again use the coordinates data_frame
    pos_dframe=ctx.raw_coords

    # We additionally use the lieklyhoods dataframe which contains the the tracking accuracy for each bodypart as a percentage
    likely_dframe=ctx.likelihoods

    # We extract the bodypart names
    m1_nose=ctx.bp(a,"Nose")
    m1_tailbase=ctx.bp(a,"Tail_base")

    # We calculate the distance 
    mouse_compression = np.linalg.norm(pos_dframe[m1_nose]-pos_dframe[m1_tailbase],axis=1)

    # We set all our "compression" values to 0 at frames where the nose or tail base of the mouse is not accurately detected 
    # i.e. the likelyhood value is below the threshold we defined
    mouse_compression = mouse_compression*(likely_dframe[m1_nose]>ctx.extra['likelyhood_threshold'])
    mouse_compression = mouse_compression*(likely_dframe[m1_tailbase]>ctx.extra['likelyhood_threshold'])

    return mouse_compression

mouse_nose_mid_distance_behavior=DeepOF_behavior(
    name="nose2nose-mid", # The name for our behavior.
    scope=Behavior_scope.PAIR_NONDIRECTIONAL, # just like deepofs nose2nose, our behavior is pairwise and nondirectional
    output_type=Behavior_output.BINARY, # We choose the binary output type
    compute=mouse_nose_mid_distance, # This is the computation function we just defined
)

mouse_compression_behavior=DeepOF_behavior(
    name="is-compressed",
    scope=Behavior_scope.INDIVIDUAL,
    output_type=Behavior_output.CONTINUOUS,
    compute=mouse_compression,
    postprocess=postprocess_identity, # here we use the identity-postprocessing function from deepof i.e. no postprocessing is applied at all.
)

CUSTOM_BEHAVIORS=[mouse_nose_mid_distance_behavior, mouse_compression_behavior]

# packaging our parameters
CUSTOM_BEHAVIOR_CONTEXT={

    'likelyhood_threshold' : 0.5,
}

