# @author lucasmiranda42

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from collections import defaultdict
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


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
    sampler=st.data(),
)
def test_get_table_dicts(nodes, ego, sampler):

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

    coords = prun.get_coords(
        center=sampler.draw(st.one_of(st.just("arena"), st.just("Center"))),
        polar=sampler.draw(st.booleans()),
        length=sampler.draw(st.one_of(st.just(False), st.just("00:10:00"))),
        align=sampler.draw(st.one_of(st.just(False), st.just("Nose"))),
    )
    speeds = prun.get_coords(
        center=sampler.draw(st.one_of(st.just("arena"), st.just("Center"))),
        polar=sampler.draw(st.booleans()),
        length=sampler.draw(st.one_of(st.just(False), st.just("00:10:00"))),
        speed=sampler.draw(st.integers(min_value=0, max_value=5)),
    )
    distances = prun.get_distances(
        length=sampler.draw(st.one_of(st.just(False), st.just("00:10:00"))),
        speed=sampler.draw(st.integers(min_value=0, max_value=5)),
    )
    angles = prun.get_angles(
        degrees=sampler.draw(st.booleans()),
        length=sampler.draw(st.one_of(st.just(False), st.just("00:10:00"))),
        speed=sampler.draw(st.integers(min_value=0, max_value=5)),
    )

    # deepof.coordinates testing

    assert type(coords) == deepof.preprocess.table_dict
    assert type(speeds) == deepof.preprocess.table_dict
    assert type(distances) == deepof.preprocess.table_dict
    assert type(angles) == deepof.preprocess.table_dict
    assert type(prun.get_videos()) == list
    assert prun.get_exp_conditions is None
    assert type(prun.get_quality()) == defaultdict
    assert type(prun.get_arenas) == tuple

    # deepof.table_dict testing

    table = sampler.draw(
        st.one_of(st.just(coords), st.just(speeds), st.just(distances), st.just(angles))
    )

    #table.filter()