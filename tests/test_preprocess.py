# @author lucasmiranda42

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from scipy.spatial import distance
from deepof.utils import *
import deepof.preprocess
import pytest


@settings(deadline=None)
@given(
    table_type=st.integers(min_value=0, max_value=2),
    arena_type=st.integers(min_value=0, max_value=1),
)
def test_project_init(table_type, arena_type):

    table_type = [".h5", ".csv", ".foo"][table_type]
    arena_type = ["circular", "foo"][arena_type]

    if arena_type == "foo":
        with pytest.raises(NotImplementedError):
            prun = deepof.preprocess.project(
                path=os.path.join(".", "tests", "test_examples"),
                arena=arena_type,
                arena_dims=[380],
                angles=False,
                video_format=".mp4",
                table_format=table_type,
            )
    else:
        prun = deepof.preprocess.project(
            path=os.path.join(".", "tests", "test_examples"),
            arena=arena_type,
            arena_dims=[380],
            angles=False,
            video_format=".mp4",
            table_format=table_type,
        )

    if table_type != ".foo" and arena_type != "foo":

        assert type(prun) == deepof.preprocess.project
        assert type(prun.load_tables(verbose=True)) == tuple
        assert type(prun.get_scale) == np.ndarray
        print(prun)

    elif table_type == ".foo" and arena_type != "foo":
        with pytest.raises(NotImplementedError):
            prun.load_tables(verbose=True)


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_distances(nodes, ego):

    nodes = ["All", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.preprocess.project(
        path=os.path.join(".", "tests", "test_examples"),
        arena="circular",
        arena_dims=[380],
        angles=False,
        video_format=".mp4",
        table_format=".h5",
        distances=nodes,
        ego=ego,
    )

    prun = prun.get_distances(prun.load_tables()[0], verbose=True)

    assert type(prun) == dict


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_angles(nodes, ego):

    nodes = ["All", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.preprocess.project(
        path=os.path.join(".", "tests", "test_examples"),
        arena="circular",
        arena_dims=[380],
        video_format=".mp4",
        table_format=".h5",
        distances=nodes,
        ego=ego,
    )

    prun = prun.get_angles(prun.load_tables()[0], verbose=True)

    assert type(prun) == dict


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_run(nodes, ego):

    nodes = ["All", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.preprocess.project(
        path=os.path.join(".", "tests", "test_examples"),
        arena="circular",
        arena_dims=[380],
        video_format=".mp4",
        table_format=".h5",
        distances=nodes,
        ego=ego,
    ).run(verbose=True)

    assert type(prun) == deepof.preprocess.coordinates
