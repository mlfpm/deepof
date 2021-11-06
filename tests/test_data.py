# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.preprocess

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from collections import defaultdict
import deepof.utils
import deepof.data
import matplotlib.figure
import numpy as np
import os
import pytest


@settings(deadline=None)
@given(
    table_type=st.integers(min_value=0, max_value=1),
    arena_type=st.integers(min_value=0, max_value=1),
)
def test_project_init(table_type, arena_type):

    table_type = [".h5", ".csv"][table_type]
    arena_type = ["circular", "foo"][arena_type]

    if arena_type == "foo":
        with pytest.raises(NotImplementedError):
            prun = deepof.data.Project(
                path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
                arena=arena_type,
                arena_dims=tuple([380]),
                video_format=".mp4",
                table_format=table_type,
            ).run()
    else:
        prun = deepof.data.Project(
            path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
            arena=arena_type,
            arena_dims=tuple([380]),
            video_format=".mp4",
            table_format=table_type,
        )

    if table_type != ".foo" and arena_type != "foo":

        assert isinstance(prun, deepof.data.Project)
        assert isinstance(prun.load_tables(verbose=True), tuple)


def test_project_properties():

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
    )

    assert prun.subset_condition is None
    prun.subset_condition = "testing"
    assert prun.subset_condition == "testing"

    assert prun.distances == "all"
    prun.distances = "testing"
    assert prun.distances == "testing"

    assert not prun.ego
    prun.ego = "testing"
    assert prun.ego == "testing"

    assert prun.angles
    prun.angles = False
    assert not prun.angles


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_distances(nodes, ego):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
    )
    tables, _ = prun.load_tables(verbose=True)
    prun.scales, prun.arena_params, prun.video_resolution = prun.get_arena(
        tables=tables
    )
    prun.distances = nodes
    prun.ego = ego
    prun = prun.get_distances(prun.load_tables()[0], verbose=True)

    assert isinstance(prun, dict)


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_angles(nodes, ego):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
    )

    prun.distances = nodes
    prun.ego = ego
    prun = prun.get_angles(prun.load_tables()[0], verbose=True)

    assert isinstance(prun, dict)


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_run(nodes, ego):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
    )

    prun.distances = nodes
    prun.ego = ego
    prun = prun.run(verbose=True)

    assert isinstance(prun, deepof.data.Coordinates)


def test_get_rule_based_annotation():

    prun = deepof.data.Project(
        path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
    ).run()

    prun = prun.supervised_annotation()

    assert isinstance(prun, deepof.data.TableDict)
    assert prun._type == "rule-based"


@settings(max_examples=10, deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    mode=st.one_of(st.just("single"), st.just("multi")),
    ego=st.integers(min_value=0, max_value=2),
    exclude=st.one_of(st.just(tuple([""])), st.just(["Tail_1"])),
    sampler=st.data(),
)
def test_get_table_dicts(nodes, mode, ego, exclude, sampler):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.data.Project(
        path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        arena="circular",
        arena_dims=380,
        video_format=".mp4",
        animal_ids=(["B", "W"] if mode == "multi" else [""]),
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={"test": "test_cond"},
    )

    if mode == "single":
        prun.distances = nodes
        prun.ego = ego

    prun = prun.run(verbose=False)

    algn = sampler.draw(st.one_of(st.just(False), st.just("Spine_1")))
    polar = st.one_of(st.just(True), st.just(False))
    speed = sampler.draw(st.integers(min_value=1, max_value=3))
    propagate = sampler.draw(st.booleans())
    propagate_annots = False
    if exclude == tuple([""]) and nodes == "all" and not ego:
        propagate_annots = sampler.draw(
            st.one_of(st.just(prun.supervised_annotation()), st.just(False))
        )

    coords = prun.get_coords(
        center=sampler.draw(st.one_of(st.just("arena"), st.just("Center"))),
        polar=polar,
        align=algn,
        propagate_labels=propagate,
        propagate_annotations=propagate_annots,
    )
    speeds = prun.get_coords(
        center=sampler.draw(st.one_of(st.just("arena"), st.just("Center"))),
        polar=sampler.draw(st.booleans()),
        speed=(speed if not ego and nodes == "all" else 0),
        propagate_labels=propagate,
        propagate_annotations=propagate_annots,
    )
    distances = prun.get_distances(
        speed=sampler.draw(st.integers(min_value=0, max_value=2)),
        propagate_labels=propagate,
        propagate_annotations=propagate_annots,
    )
    angles = prun.get_angles(
        degrees=sampler.draw(st.booleans()),
        speed=sampler.draw(st.integers(min_value=0, max_value=2)),
        propagate_labels=propagate,
        propagate_annotations=propagate_annots,
    )

    # deepof.coordinates testing

    assert isinstance(coords, deepof.data.TableDict)
    assert isinstance(speeds, deepof.data.TableDict)
    assert isinstance(distances, deepof.data.TableDict)
    assert isinstance(angles, deepof.data.TableDict)
    assert isinstance(prun.get_videos(), list)
    assert prun.get_exp_conditions is not None
    assert isinstance(prun.get_quality(), defaultdict)
    assert isinstance(prun.get_arenas, tuple)

    # deepof.table_dict testing

    table = sampler.draw(
        st.one_of(
            st.just(coords), st.just(speeds), st.just(distances), st.just(angles)
        ),
        st.just(deepof.data.merge_tables(coords, speeds, distances, angles)),
    )

    assert table.filter_videos(["test"]) == table
    tset = table.get_training_set(
        test_videos=sampler.draw(st.integers(min_value=0, max_value=len(table) - 1))
    )
    assert len(tset) == 4
    assert isinstance(tset[0], np.ndarray)

    if table._type == "coords" and algn == "Nose" and polar is False and speed == 0:

        assert isinstance(
            table.plot_heatmaps(bodyparts=["Spine_1"]), matplotlib.figure.Figure
        )

        align = sampler.draw(
            st.one_of(st.just(False), st.just("all"), st.just("center"))
        )

    else:
        align = False

    selected_id = None
    if mode == "multi" and nodes == "all" and not ego:
        selected_id = sampler.draw(st.one_of(st.just(None), st.just("B")))

    prep = table.preprocess(
        window_size=11,
        window_step=1,
        scale=sampler.draw(st.one_of(st.just("standard"), st.just("minmax"))),
        test_videos=sampler.draw(st.integers(min_value=0, max_value=len(table) - 1)),
        verbose=True,
        conv_filter=sampler.draw(st.one_of(st.just(None), st.just("gaussian"))),
        sigma=sampler.draw(st.floats(min_value=0.5, max_value=5.0)),
        shift=sampler.draw(st.floats(min_value=-1.0, max_value=1.0)),
        shuffle=sampler.draw(st.booleans()),
        align=align,
        selected_id=selected_id,
    )

    assert isinstance(prep[0], np.ndarray)

    # deepof dimensionality reduction testing

    table = deepof.data.TableDict(
        dict(table, **{"test1": table["test"]}), typ=table._type
    )

    print(table)

    assert isinstance(table.random_projection(n_components=2), tuple)
    assert isinstance(table.pca(n_components=2), tuple)
    assert isinstance(table.tsne(n_components=2), tuple)
