"""Testing module for deepof.skeleton (custom labelling schemes)."""

import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import deepof.custom_gui
import deepof.data
import deepof.skeleton
import deepof.utils
from deepof.config import SCHEMA_POSITIONS
from tests.test_objects.test_objects import example_path, make_project, project_dir

# CalMS21-like labelling scheme: neck lies between the deepOF ears and Spine_1, hips on the deepOF back hips
CALMS = {"nose": [425, 80], "left_ear": [372, 191], "right_ear": [467, 190], "neck": [421, 210],
         "left_hip": [350, 377], "right_hip": [490, 341], "tail_base": [424, 460]}


def edges(graph):
    return {tuple(sorted(e)) for e in nx.Graph(graph).edges}


def test_match_to_deepof():
    # far away points stay unmatched and do not take the place of close ones
    positions = {"snout": [426, 81], "left_hip": [349, 378], "left_paw": [340, 370], "tail": [600, 600]}
    assert deepof.skeleton.match_to_deepof(positions) == {"snout": "Nose", "left_hip": "Left_bhip"}


def test_resolve_skeleton():
    resolved = deepof.skeleton.resolve_skeleton(
        {"positions": CALMS, "derive": {"Center": ["neck", "left_hip", "right_hip", "tail_base"]}}
    )
    assert resolved["rename"] == {"nose": "Nose", "left_ear": "Left_ear", "right_ear": "Right_ear", "neck": "neck",
                                  "left_hip": "Left_bhip", "right_hip": "Right_bhip", "tail_base": "Tail_base"}
    assert resolved["derive"] == {"Center": ["neck", "Left_bhip", "Right_bhip", "Tail_base"]}
    assert resolved["positions"]["Center"] == pytest.approx(np.mean([CALMS[p] for p in ["neck", "left_hip", "right_hip", "tail_base"]], axis=0), abs=0.1)
    # connected, includes the derived point and is left/right symmetric
    graph = nx.Graph(resolved["graph"])
    assert nx.is_connected(graph) and "Center" in graph
    for a, b in graph.edges:
        assert graph.has_edge(a.replace("Left", "Right"), b.replace("Left", "Right"))

    # a body part placed away from its namesake deepOF body part is renamed
    with pytest.warns(UserWarning):
        assert deepof.skeleton.resolve_skeleton({"positions": {"Nose": [600, 300], "Tail_base": [425, 462]}})["rename"]["Nose"] == "Nose_extra"
    # deepOF positions give the deepOF graph
    full = deepof.skeleton.resolve_skeleton({"positions": {n: list(p) for n, p in SCHEMA_POSITIONS.items()}})
    assert edges(full["graph"]) == edges(deepof.utils.connect_mouse(graph_preset="deepof_14"))
    with pytest.raises(ValueError):
        deepof.skeleton.resolve_skeleton({"positions": CALMS, "rename": {"neck": "Nose"}})


def test_add_derived_points():
    columns = pd.MultiIndex.from_product([["B_a", "B_b"], ["x", "y", "likelihood"]], names=["bodyparts", "coords"])
    table = pd.DataFrame([[0, 0, 0.9, 2, 4, 0.5], [np.nan, 0, 1.0, 2, 4, 1.0]], columns=columns, dtype=float)
    derived, _ = deepof.skeleton.add_derived_points(table, {"m": ["a", "b"]}, ["B"])
    assert derived[("B_m", "x")].tolist()[0] == 1 and np.isnan(derived[("B_m", "x")].tolist()[1])
    assert derived[("B_m", "likelihood")].tolist() == [0.5, 1.0]
    # an existing body part of the same name is replaced (and reported)
    replaced, names = deepof.skeleton.add_derived_points(table, {"a": ["b"]}, ["B"])
    assert names == ["a"] and replaced[("B_a", "x")].tolist() == [2, 2] and table[("B_a", "x")].tolist()[0] == 0


def test_rename_track_bps_exact():
    tab = pd.DataFrame([["B_ear", "B_left_ear", "W_ear"]], index=["bodyparts"])
    renamed = deepof.utils.rename_track_bps(tab, {"ear": "Ear"}, ["B", "W"])
    assert renamed.loc["bodyparts"].tolist() == ["B_Ear", "B_left_ear", "W_Ear"]


def test_custom_skeleton_project():
    base_path = example_path("test_multi_topview")
    # deepOF tables relabelled: Spine_1 / Spine_2 become unknown points, the tables' Center is not used but derived
    positions = {n: list(p) for n, p in SCHEMA_POSITIONS.items() if n != "Center"}
    positions.update({"Spine_1": [440, 260], "Spine_2": [440, 360]})
    skeleton = {"positions": positions, "derive": {"Center": ["Spine_1", "Spine_2", "Left_bhip", "Right_bhip"]}}
    source_tables = {f: open(os.path.join(base_path, "Tables", f), "rb").read() for f in os.listdir(os.path.join(base_path, "Tables"))}
    prun = make_project("test_multi_topview", animal_ids=["B", "W"], bodypart_graph=skeleton, exclude_bodyparts=[])
    with pytest.warns(UserWarning, match="replaced by the derived points"):
        coords = prun.create(force=True, test=True)
    tab = coords.get_coords()[list(coords._tables)[0]]
    bodyparts = [c[0] for c in tab.columns]
    assert {"B_Spine_1_extra", "B_Spine_2_extra", "B_Center", "W_Center"} <= set(bodyparts)
    assert bodyparts.count("B_Center") == 2  # x and y, the tables' own Center was replaced
    # mean of the body parts, apart from frames changed by outlier removal / imputation of single points
    mean = tab[[(f"B_{p}", "x") for p in ["Spine_1_extra", "Spine_2_extra", "Left_bhip", "Right_bhip"]]].mean(axis=1)
    assert (tab[("B_Center", "x")] - mean).abs().median() < 1e-3
    assert all(open(os.path.join(base_path, "Tables", f), "rb").read() == content for f, content in source_tables.items())
    assert os.path.exists(os.path.join(project_dir(coords), "skeleton.json"))
    with pytest.raises(NotImplementedError):
        coords.supervised_annotation()
    coords.get_graph_dataset(center="Center", align="Nose", window_size=5, window_step=5, test_videos=0)


def test_skeleton_editor():
    editor = deepof.custom_gui.SkeletonEditor(list(CALMS) + ["pole"])
    click = lambda pos: editor.handle_click(int(pos[0] * editor.scale), int(pos[1] * editor.scale))
    for name in list(CALMS)[:2]:
        click(CALMS[name])
    editor.handle_key(ord("d"))  # undo
    assert editor.current == "left_ear"
    for name in list(CALMS)[1:]:
        click(CALMS[name])
    editor.handle_key(ord("s"))  # pole is not on the mouse
    assert editor.mode == "center"
    editor.handle_key(ord("y"))
    assert editor.mode == "review" and editor.derive == {"Center": ["neck", "Left_bhip", "Right_bhip", "Tail_base"]}
    assert editor.render().shape[1] == editor.schema.shape[1] + editor.PANEL_WIDTH

    # toggle an edge and undo it
    positions = editor._resolve()["positions"]
    had_edge = editor.graph.has_edge("Nose", "Tail_base")
    click(positions["Nose"]), click(positions["Tail_base"])
    assert editor.graph.has_edge("Nose", "Tail_base") != had_edge
    editor.handle_key(ord("d"))
    assert editor.graph.has_edge("Nose", "Tail_base") == had_edge
    assert editor.handle_key(ord("q")) == "done"

    resolved = deepof.skeleton.resolve_skeleton(editor.to_skeleton(), list(CALMS) + ["pole"])
    assert "pole" not in resolved["graph"] and "Center" in resolved["graph"]


def test_graph_bodyparts_missing_in_tables():
    prun = make_project(
        "test_multi_topview", animal_ids=["B", "W"], bodypart_graph={"Nose": ["Left_ear", "Paw"], "Left_ear": ["Center"]}
    )
    # raised while loading the first table, before arena detection
    with pytest.raises(ValueError, match="B_Paw"):
        prun.create(force=True, test=True)


def test_skeleton_editor_states():
    # Clicks into the panel are ignored; "n" answers the Center question, undo returns to it and then to the placement;
    # a body part away from the mouse is an extra one; removing all edges of a body part gives a graph that can not be
    # saved (undo restores the last edge); a click far from all body parts deselects; every mode renders; a complete
    # skeleton with a Center starts in the review
    editor = deepof.custom_gui.SkeletonEditor(list(CALMS) + ["pole"])
    click = lambda pos: editor.handle_click(int(pos[0] * editor.scale), int(pos[1] * editor.scale))
    width = editor.schema.shape[1] + editor.PANEL_WIDTH
    editor.handle_key(ord("d"))  # nothing to undo
    editor.handle_click(editor.schema.shape[1] + 5, 10)
    assert editor.positions == {} and editor.render().shape[1] == width
    editor.handle_key(ord("h"))
    assert not editor.show_help and editor.handle_key(27) == "cancel"
    for name in CALMS:
        click(CALMS[name])
    editor.handle_key(ord("s"))
    assert editor.mode == "center" and editor.render().shape[1] == width
    editor.handle_key(ord("n"))
    assert editor.mode == "review" and editor.derive == {}
    editor.handle_key(ord("d"))
    assert editor.mode == "center"
    editor.handle_key(ord("d"))
    assert editor.mode == "place" and editor.current == "pole"
    click((150, 300))
    assert editor.mode == "center" and editor.to_skeleton()["graph"] is None
    editor.handle_key(ord("y"))
    assert editor.mode == "review" and "pole" in editor.graph

    positions = editor._resolve()["positions"]
    neighbours = list(editor.graph["pole"])
    for other in neighbours:
        click(positions["pole"])
        assert editor.selected == "pole" and editor.render().shape[1] == width
        click(positions[other])
    assert editor.graph.degree("pole") == 0 and editor.handle_key(ord("q")) is None
    assert editor.render().shape[1] == width
    editor.handle_key(ord("d"))
    assert editor.graph.has_edge("pole", neighbours[-1]) and editor.handle_key(ord("q")) == "done"
    click(positions["pole"])
    click((positions["pole"][0], positions["pole"][1] + 300))
    assert editor.selected is None

    full = deepof.custom_gui.SkeletonEditor(
        ["a", "b", "c"], {"positions": {"a": SCHEMA_POSITIONS["Nose"], "b": SCHEMA_POSITIONS["Center"],
                                        "c": SCHEMA_POSITIONS["Tail_base"]}})
    assert full.mode == "review" and full.render().shape[1] == full.schema.shape[1] + full.PANEL_WIDTH


def test_skeleton_files_and_invalid_skeletons(tmp_path):
    # Skeletons are loaded from JSON files (presets and plain graphs are no skeletons); invalid entries, body parts,
    # derived points and graphs are refused; a table without derived points stays as it is
    skeleton = {"positions": CALMS, "derive": {"Center": ["neck", "left_hip", "right_hip", "tail_base"]}}
    path = str(tmp_path / "skeleton.json")
    deepof.skeleton.save_skeleton(skeleton, path)
    assert deepof.skeleton.load_skeleton(path) == skeleton and deepof.skeleton.load_skeleton("deepof_14") is None
    assert deepof.skeleton.match_to_deepof({}) == {}

    resolve = deepof.skeleton.resolve_skeleton
    for wrong, names, match in [
        ({"positions": CALMS, "edges": {}}, None, "Unknown skeleton entries"),
        ({"positions": CALMS}, ["nose"], "not part of the tracking body parts"),
        ({"positions": CALMS, "derive": {"Nose": ["neck", "tail_base"]}}, None, "already exists"),
        ({"positions": CALMS, "derive": {"Center": ["neck", "paw"]}}, None, "unknown body parts"),
        ({"positions": CALMS, "graph": {"nose": ["paw"]}}, None, "not part of the skeleton"),
        ({"positions": {"nose": CALMS["nose"]}, "rename": {"tail": "tail"}}, ["nose", "tail"], "no position"),
    ]:
        with pytest.raises(ValueError, match=match):
            resolve(wrong, names)
    # a table body part named like a derived point, but not placed, is replaced by it
    resolved = resolve({"positions": {n: list(p) for n, p in SCHEMA_POSITIONS.items() if n != "Center"},
                        "derive": {"Center": ["Spine_1", "Spine_2"]}}, list(SCHEMA_POSITIONS))
    assert resolved["derive"] == {"Center": ["Spine_1", "Spine_2"]} and "Center" in resolved["graph"]
    table = pd.DataFrame([[1.0, 2.0]], columns=pd.MultiIndex.from_tuples([("Nose", "x"), ("Nose", "y")],
                                                                        names=["bodyparts", "coords"]))
    assert deepof.skeleton.add_derived_points(table, {}, [""]) == (table, [])
