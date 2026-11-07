# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

"""

premade objects to import for tests (the types of tables and stuff we use often in DeepOF)

Example projects: create_project("test_multi_topview", animal_ids=["B", "W"], ...) creates a project on one of the
example data sets in tests/test_examples with the settings most tests share. The project folder is removed after the
test, also when it fails (tests/conftest.py); so are the folders made with output_dir.

"""

import itertools
import os
from contextlib import ExitStack, contextmanager
from shutil import rmtree
from unittest import mock

import numpy as np
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import deepof.data
from deepof.annotation_utils import DeepOF_behavior
from deepof.annotation_utils import Behavior_scope, Behavior_output
from deepof.annotation_utils import animal_ids, BehaviorContext
from deepof.annotation_utils import postprocess_identity


# Example projects

EXAMPLES_PATH = os.path.join(".", "tests", "test_examples")
# animal ids of the example data sets (test_madlc_topview: ["mouse_black_tail", "mouse_white_tail"])
ANIMAL_IDS = {"test_single_topview": [""], "test_multi_topview": ["B", "W"]}
# experimental conditions for the two videos of every example data set
TEST_CONDITIONS = {"test": "test_cond", "test2": "test_cond"}

_folders_to_remove = set()
_project_counter = itertools.count()


def example_path(example: str, *parts: str) -> str:
    """Path of an example data set (e.g. "test_single_topview") in tests/test_examples, or of a file / folder in it."""
    return os.path.join(EXAMPLES_PATH, example, *parts)


def make_project(example: str = "test_single_topview", **kwargs) -> deepof.data.Project:
    """deepof.data.Project on an example data set. Keyword arguments are passed on and override the shared settings.

    The default project name contains the process id, so that test files can run in parallel. The project folder is
    registered for removal after the test (remove_test_folders, called by tests/conftest.py).
    """
    settings = dict(
        project_path=example_path(example),
        video_path=example_path(example, "Videos"),
        table_path=example_path(example, "Tables"),
        project_name=f"deepof_project_{os.getpid()}",
        arena="circular-autodetect",
        video_scale="380 mm",
        video_format=".mp4",
        table_format=".h5",
    )
    settings.update(kwargs)
    _folders_to_remove.add(os.path.join(settings["project_path"], settings["project_name"]))
    return deepof.data.Project(**settings)


def unique_project_name() -> str:
    """A project name not used before in this process. For tests that create a project per hypothesis example in
    very large project mode: there, a folder can not be created again in the same session (DuckDB file still open)."""
    return f"deepof_project_{os.getpid()}_{next(_project_counter)}"


def create_project(example: str = "test_single_topview", **kwargs) -> deepof.data.Coordinates:
    """make_project(example, **kwargs).create(force=True, test=True) (test mode: fixed arenas, no arena detection)."""
    return make_project(example, **kwargs).create(force=True, test=True)


def project_dir(coordinates: deepof.data.Coordinates) -> str:
    """Folder of a created project."""
    return os.path.join(coordinates._project_path, coordinates._project_name)


def output_dir(name: str) -> str:
    """Empty folder tests/test_examples/test_data/<name> for the output of a test, removed after the test."""
    path = example_path("test_data", name)
    rmtree(path, ignore_errors=True)
    os.makedirs(path)
    _folders_to_remove.add(path)
    return path


def remove_test_folders():
    """Remove the folders of all projects made with make_project / create_project and of output_dir."""
    for path in _folders_to_remove:
        rmtree(path, ignore_errors=True)
    _folders_to_remove.clear()


@contextmanager
def patched(*replacements):
    """Replace module attributes within the with block, also restored when the test fails:
    with patched((deepof.clustering.training, "step_vade", wrapped_step), ...): ..."""
    with ExitStack() as stack:
        for target, name, new in replacements:
            stack.enter_context(mock.patch.object(target, name, new))
        yield


# Random tables with fixed sizes

def random_soft_counts(keys, n_rows: int, n_states: int, table_dict: bool = True):
    """Soft counts (rows sum to 1) per key, as TableDict (typ "soft_counts") or plain dictionary."""
    soft_counts = {}
    for key in keys:
        counts = np.random.uniform(0.001, 1.0, size=(n_rows, n_states)).astype(np.float32)
        soft_counts[key] = counts / counts.sum(axis=1, keepdims=True)
    if not table_dict:
        return soft_counts
    return deepof.data.TableDict(soft_counts, typ="soft_counts", table_path=None, exp_conditions=None)


def random_embeddings(keys, n_rows: int, dim: int, table_dict: bool = True):
    """Embeddings (uniform in [-3, 3]) per key, as TableDict (typ "unsupervised_embedding") or plain dictionary."""
    embeddings = {key: np.random.uniform(-3.0, 3.0, size=(n_rows, dim)).astype(np.float32) for key in keys}
    if not table_dict:
        return embeddings
    return deepof.data.TableDict(embeddings, typ="unsupervised_embedding", table_path=None, exp_conditions=None)


def random_supervised(keys, n_rows: int, columns) -> deepof.data.TableDict:
    """Binary supervised annotations per key; columns: number of columns ("col_<i>") or list of behavior names."""
    names = [f"col_{i}" for i in range(columns)] if isinstance(columns, int) else list(columns)
    tables = {
        key: pd.DataFrame(np.random.randint(0, 2, size=(n_rows, len(names))).astype(np.int8), columns=names)
        for key in keys
    }
    return deepof.data.TableDict(tables, typ="supervised_annotations", table_path=None, exp_conditions=None)


# Hypothesis strategies

@st.composite
def get_soft_counts(draw, n_min=1, n_max=50, m_min=1, m_max=10):
    """Soft count matrix (rows sum to 1) of random size."""
    n = draw(st.integers(n_min, n_max))
    m = draw(st.integers(m_min, m_max))
    raw = draw(arrays(np.float32, (n, m), elements=st.floats(0.0010000000474974513, 1.0, allow_nan=False, allow_infinity=False, width=32)))
    return raw / raw.sum(axis=1, keepdims=True)


@st.composite
def get_supervised_tables(draw, n_min=1, n_max=50, m_min=1, m_max=10):
    """Binary supervised table of random size (columns "col_<i>")."""
    n = draw(st.integers(n_min, n_max))
    m = draw(st.integers(m_min, m_max))
    data = draw(arrays(np.int8, (n, m), elements=st.integers(min_value=0, max_value=1)))
    cols = [f"col_{i}" for i in range(m)]  # unique column names
    return pd.DataFrame(data, columns=cols)


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

    # Optional factor per video (checks that the context passes the video key and the extra parameters)
    mouse_compression = mouse_compression*ctx.extra.get('bias_correction_dict', {}).get(ctx.key, 1)

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

