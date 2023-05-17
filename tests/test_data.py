# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.preprocess

"""

import os
from collections import defaultdict
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest
import string
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import deepof.data
import deepof.utils


@settings(max_examples=2, deadline=None)
@given(
    table_type=st.one_of(st.just(".h5"), st.just(".csv")),
)
def test_project_init(table_type):

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        project_name=f"test_{table_type[1:]}",
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format=table_type,
    )

    assert isinstance(prun, deepof.data.Project)
    assert isinstance(prun.load_tables(verbose=True), tuple)

    prun = prun.create()
    assert isinstance(prun, deepof.data.Coordinates)
    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_single_topview",
            f"test_{table_type[1:]}",
        )
    )


def test_project_properties():

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
    )

    assert prun.distances == "all"
    prun.distances = "testing"
    assert prun.distances == "testing"

    assert not prun.ego
    prun.ego = "testing"
    assert prun.ego == "testing"

    assert prun.angles
    prun.angles = False
    assert not prun.angles


def test_project_filters():

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
    ).create(force=True)
    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    # Update experimental conditions with mock values
    prun._exp_conditions = {
        key: pd.DataFrame(
            {"CSDS": np.random.choice(["Case", "Control"], size=1)[0]}, index=[0]
        )
        for key in prun.get_coords().keys()
    }

    coords = prun.get_coords()
    assert isinstance(coords.filter_id("B"), dict)
    assert isinstance(coords.filter_videos(coords.keys()), dict)
    assert isinstance(coords.filter_condition(exp_filters={"CSDS": "Control"}), dict)


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_distances(nodes, ego):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
    )
    prun.create(force=True)

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
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
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
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
    )

    prun.distances = nodes
    prun.ego = ego
    prun = prun.create(force=True)
    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    assert isinstance(prun, deepof.data.Coordinates)


def test_get_supervised_annotation():

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
    ).create(force=True)
    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    prun = prun.supervised_annotation()

    assert isinstance(prun, deepof.data.TableDict)
    assert prun._type == "supervised"


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    mode=st.one_of(st.just("single"), st.just("multi"), st.just("madlc")),
    ego=st.integers(min_value=0, max_value=1),
    exclude=st.one_of(st.just(tuple([""])), st.just(["Tail_tip"])),
    sampler=st.data(),
    random_id=st.text(alphabet=string.ascii_letters, min_size=50, max_size=50),
)
def test_get_table_dicts(nodes, mode, ego, exclude, sampler, random_id):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    if mode == "multi":
        animal_ids = ["B", "W"]
    elif mode == "madlc":
        animal_ids = ["mouse_black_tail", "mouse_white_tail"]
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
        project_name=f"deepof_project_{random_id}",
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={
            "test": pd.DataFrame({"CSDS": "test_cond"}, index=[0]),
            "test2": pd.DataFrame({"CSDS": "test_cond"}, index=[0]),
        },
    )

    if mode == "single":
        prun.distances = nodes
        prun.ego = ego

    prun = prun.create(force=True)
    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            f"deepof_project_{random_id}",
        )
    )

    center = sampler.draw(st.one_of(st.just("arena"), st.just("Center")))
    algn = sampler.draw(st.one_of(st.just(False), st.just("Spine_1")))
    polar = sampler.draw(st.booleans())
    speed = sampler.draw(st.integers(min_value=1, max_value=3))
    propagate = sampler.draw(st.sampled_from(["CSDS", False]))
    propagate_annots = False

    selected_id = None
    if mode == "multi" and nodes == "all" and not ego:
        selected_id = "B"
    elif mode == "madlc" and nodes == "all" and not ego:
        selected_id = "mouse_black_tail"

    coords = prun.get_coords(
        center=center,
        polar=polar,
        align=(algn if center == "Center" and not polar else False),
        propagate_labels=propagate,
        propagate_annotations=propagate_annots,
        selected_id=selected_id,
    )
    speeds = prun.get_coords(
        speed=(speed if not ego and nodes == "all" else 0),
        propagate_labels=propagate,
        selected_id=selected_id,
    )
    distances = prun.get_distances(
        speed=sampler.draw(st.integers(min_value=0, max_value=2)),
        propagate_labels=propagate,
        selected_id=selected_id,
    )
    angles = prun.get_angles(
        degrees=sampler.draw(st.booleans()),
        speed=sampler.draw(st.integers(min_value=0, max_value=2)),
        propagate_labels=propagate,
        selected_id=selected_id,
    )
    areas = prun.get_areas()
    merged = coords.merge(speeds, distances, angles, areas)

    # deepof.coordinates testing
    assert isinstance(coords, deepof.data.TableDict)
    assert isinstance(speeds, deepof.data.TableDict)
    assert isinstance(distances, deepof.data.TableDict)
    assert isinstance(angles, deepof.data.TableDict)
    assert isinstance(areas, deepof.data.TableDict)
    assert isinstance(merged, deepof.data.TableDict)
    assert isinstance(prun.get_videos(), list)
    assert prun.get_exp_conditions is not None
    assert isinstance(prun.get_quality(), deepof.data.TableDict)
    assert isinstance(prun.get_arenas, tuple)

    # deepof.table testing
    prep = coords.preprocess(
        window_size=11,
        window_step=1,
        automatic_changepoints=(
            False if not any([propagate_annots, propagate]) else "linear"
        ),
        scale=sampler.draw(
            st.one_of(st.just("standard"), st.just("minmax"), st.just("robust"))
        ),
        test_videos=1,
        verbose=2,
        filter_low_variance=1e-3 * (not any([propagate_annots, propagate])),
        interpolate_normalized=5 * (not any([propagate_annots, propagate])),
        shuffle=sampler.draw(st.booleans()),
    )

    assert isinstance(prep[0][0], np.ndarray)

    # deepof dimensionality reduction testing

    assert isinstance(coords.random_projection(n_components=2), tuple)
    assert isinstance(coords.pca(n_components=2), tuple)


@settings(deadline=None)
@given(
    mode=st.one_of(st.just("single"), st.just("multi"), st.just("madlc")),
    sampler=st.data(),
    random_id=st.text(alphabet=string.ascii_letters, min_size=50, max_size=50),
)
def test_get_graph_dataset(mode, sampler, random_id):

    if mode == "multi":
        animal_ids = ["B", "W"]
    elif mode == "madlc":
        animal_ids = ["mouse_black_tail", "mouse_white_tail"]
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
        project_name=f"deepof_project_{random_id}",
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
    ).create(force=True)
    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            f"deepof_project_{random_id}",
        )
    )

    graph_dset, adj_matrix, to_preprocess, global_scaler = prun.get_graph_dataset(
        animal_id=sampler.draw(st.one_of(st.just(None), st.just(animal_ids[0]))),
        automatic_changepoints=sampler.draw(
            st.one_of(st.just("linear"), st.just(False))
        ),
        scale=sampler.draw(
            st.one_of(
                st.just("standard"),
                st.just("minmax"),
                st.just("robust"),
                st.just(False),
            )
        ),
        test_videos=1,
    )

    assert isinstance(graph_dset, tuple)
    assert isinstance(adj_matrix, np.ndarray)
    assert isinstance(to_preprocess, deepof.data.TableDict)
