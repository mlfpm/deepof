"""Custom labelling schemes (skeletons) for deepOF projects.

A skeleton maps the body parts of a custom labelling scheme to deepOF. It is a dictionary (or a JSON file containing
one) with the following, all optional, entries:

    "positions": {body part: [x, y]}  Position of each body part in deepOF's mouse schema (deepof/assets/mouse_schema.png).
                                      Body parts placed on a deepOF body part (see deepof.config.SCHEMA_POSITIONS) are
                                      renamed to it, all others keep their names.
    "rename":    {body part: name}    Explicit renaming, applied on top of the renaming derived from the positions.
    "derive":    {name: [body parts]} Points added to the tables as the mean of other body parts, e.g. a "Center".
    "graph":     {body part: [neighbors]} Connectivity graph, in final body part names (table names are accepted where
                                      unambiguous). Body parts that are not part of the graph are excluded from the
                                      analysis. If missing, the graph is computed from the positions.

Instead of writing the skeleton by hand, it can be defined in a window by clicking on the mouse schema, see
define_skeleton (or use bodypart_graph="custom" in deepof.data.Project).

Only the unsupervised pipeline supports custom skeletons. Supervised annotations require all deepof_11 body parts.

"""

import copy
import json
import os
import re
import warnings
from itertools import combinations
from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import deepof.utils
from deepof.config import SCHEMA_MATCH_RADIUS, SCHEMA_POSITIONS

SKELETON_KEYS = ("positions", "rename", "derive", "graph")
PRESETS = ("deepof_14", "deepof_11", "deepof_8")
# Schema region (y range in pixels) of the trunk, used to suggest the points a center can be derived from
TRUNK_Y_RANGE = (200, 470)


def is_skeleton(bodypart_graph) -> bool:
    """Return True if bodypart_graph is a skeleton (or a path to a skeleton JSON file) rather than a preset or graph."""
    if isinstance(bodypart_graph, str):
        return bodypart_graph.lower().endswith(".json")
    return isinstance(bodypart_graph, dict) and any(key in bodypart_graph for key in SKELETON_KEYS)


def load_skeleton(bodypart_graph: Union[str, dict]) -> dict:
    """Return the skeleton given as dictionary or JSON file path, or None for presets and plain graphs."""
    if not is_skeleton(bodypart_graph):
        return None
    if isinstance(bodypart_graph, str):
        with open(bodypart_graph, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return copy.deepcopy(bodypart_graph)


def save_skeleton(skeleton: dict, path: str):
    """Save a skeleton as JSON file."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(skeleton, handle, indent=2)


def define_skeleton(bodypart_names: List[str], skeleton: Union[str, dict] = None, save_path: str = None) -> dict:  # pragma: no cover
    """Open a window to define a custom skeleton by placing the body parts on deepOF's mouse schema.

    Body parts placed inside a yellow circle are treated as the corresponding deepOF body part, all others are kept as
    additional body parts. If there is no "Center", it can be derived from other body parts. Finally, the proposed graph
    can be edited.

    Args:
        bodypart_names (list): body part names of the tracking tables.
        skeleton (Union[str, dict]): optional skeleton (or path to a skeleton JSON file) to start from.
        save_path (str): if given, the skeleton is saved as JSON file at this path.

    Returns:
        skeleton (dict): the skeleton, to be used as bodypart_graph in deepof.data.Project, or None if cancelled.
    """
    import deepof.custom_gui  # the GUI module itself uses this module

    result = deepof.custom_gui.define_skeleton_gui(bodypart_names, load_skeleton(skeleton) if skeleton else None)
    if result is not None and save_path is not None:
        save_skeleton(result, save_path)
    return result


def match_to_deepof(positions: Dict[str, tuple], radius: float = SCHEMA_MATCH_RADIUS) -> Dict[str, str]:
    """Match body part positions in the mouse schema one-to-one to deepOF body parts.

    Args:
        positions (dict): {body part: (x, y)} in schema pixels.
        radius (float): maximum distance between a position and the deepOF body part it is matched to.

    Returns:
        matches (dict): {body part: deepOF body part} for all body parts within radius of a deepOF body part.
    """
    if not positions:
        return {}
    names, targets = list(positions), list(SCHEMA_POSITIONS)
    pos = np.array([positions[n] for n in names], dtype=float)
    ref = np.array([SCHEMA_POSITIONS[t] for t in targets], dtype=float)
    cost = np.linalg.norm(pos[:, None, :] - ref[None, :, :], axis=-1)
    cost[cost > radius] = 1e6
    # one "unmatched" option per body part, so far away body parts do not take the place of close ones
    unmatched = np.full((len(names), len(names)), radius + 1.0)
    rows, cols = linear_sum_assignment(np.concatenate([cost, unmatched], axis=1))
    return {names[r]: targets[c] for r, c in zip(rows, cols) if c < len(targets) and cost[r, c] <= radius}


def auto_graph(positions: Dict[str, tuple]) -> Dict[str, list]:
    """Create a connectivity graph for body parts with known schema positions.

    If the body parts are exactly those of a deepOF preset, the preset is used. Otherwise, the graph consists of the
    deepof_14 edges between present body parts and the edges of the Gabriel graph of all positions (two body parts are
    connected if no other body part lies within the circle that has both as diameter). The Gabriel graph is always
    connected and follows the symmetry of the positions.

    Args:
        positions (dict): {body part: (x, y)} in schema pixels, for all body parts of the graph.

    Returns:
        graph (dict): {body part: [neighbors]}.
    """
    names = list(positions)
    for preset in PRESETS:
        preset_graph = deepof.utils.connect_mouse(graph_preset=preset)
        if set(names) == set(preset_graph.nodes):
            return nx.to_dict_of_lists(preset_graph)

    graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from(
        edge for edge in deepof.utils.connect_mouse(graph_preset="deepof_14").edges
        if edge[0] in positions and edge[1] in positions
    )
    pos = {n: np.asarray(positions[n], dtype=float) for n in names}
    for a, b in combinations(names, 2):
        middle, radius = (pos[a] + pos[b]) / 2, np.linalg.norm(pos[a] - pos[b]) / 2
        if not any(np.linalg.norm(pos[n] - middle) < radius for n in names if n not in (a, b)):
            graph.add_edge(a, b)
    # left/right symmetry, which the clicked (and the schema's) positions do not exactly have
    mirrored = [(_mirror(a, names), _mirror(b, names)) for a, b in list(graph.edges)]
    graph.add_edges_from([(a, b) for a, b in mirrored if a is not None and b is not None])
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return nx.to_dict_of_lists(graph)


def _mirror(name: str, names: List[str]) -> str:
    """Return the left/right counterpart of a body part (e.g. Left_ear <-> Right_ear), the body part itself if it has
    no side (e.g. Nose), or None if its counterpart is not part of names."""
    swap = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    swapped = re.sub("|".join(swap), lambda m: swap[m.group(0)], name)
    if swapped == name:
        return name
    return swapped if swapped in names else None


def suggest_derived_center(positions: Dict[str, tuple]) -> List[str]:
    """Return the body parts (final names) on the trunk, from which a missing "Center" can be derived."""
    return [n for n, (x, y) in positions.items() if TRUNK_Y_RANGE[0] <= y <= TRUNK_Y_RANGE[1]]


def resolve_skeleton(skeleton: dict, bodypart_names: List[str] = None) -> dict:
    """Resolve a skeleton into the renaming, derived points and graph used by the project.

    Args:
        skeleton (dict): skeleton as described in the module documentation.
        bodypart_names (list): body part names of the tracking tables. If None, the names given in the skeleton are used.

    Returns:
        resolved (dict): skeleton with entries "rename" ({table name: final name}, for all known body parts),
            "derive" and "graph" (in final names) and "positions" (final names, including deepOF body parts and
            derived points).
    """
    unknown = set(skeleton) - set(SKELETON_KEYS)
    if unknown:
        raise ValueError(f"Unknown skeleton entries {sorted(unknown)}. Valid entries are {list(SKELETON_KEYS)}.")
    raw_positions = {n: tuple(p) for n, p in (skeleton.get("positions") or {}).items()}
    explicit = dict(skeleton.get("rename") or {})
    if bodypart_names is None:
        bodypart_names = list(dict.fromkeys(list(raw_positions) + list(explicit)))
    missing = [n for n in list(raw_positions) + list(explicit) if n not in bodypart_names]
    if missing:
        raise ValueError(f"Skeleton body parts {missing} are not part of the tracking body parts {bodypart_names}.")

    matches = match_to_deepof(raw_positions)
    rename = {n: n for n in bodypart_names}
    rename.update(matches)
    rename.update(explicit)
    # a body part placed away from the deepOF body part it is named after would be mistaken for it
    for name in bodypart_names:
        if name in SCHEMA_POSITIONS and name in raw_positions and name not in matches and name not in explicit:
            rename[name] = f"{name}_extra"
            warnings.warn(
                f"\033[38;5;208mBody part \"{name}\" was not placed on the deepOF body part with the same name and "
                f"is renamed to \"{name}_extra\".\033[0m"
            )
    finals = list(rename.values())
    duplicates = sorted({n for n in finals if finals.count(n) > 1})
    if duplicates:
        raise ValueError(f"Several body parts would be named {duplicates}. Please check the skeleton.")

    positions = {rename[n]: p for n, p in raw_positions.items()}
    for name in finals:
        if name in SCHEMA_POSITIONS and name not in positions:
            positions[name] = SCHEMA_POSITIONS[name]

    derive = {}
    for new, parts in (skeleton.get("derive") or {}).items():
        parts = [rename.get(p, p) for p in parts]
        if new in finals:
            sources = [n for n, f in rename.items() if f == new]
            if any(n in raw_positions or n in explicit for n in sources):
                raise ValueError(f"Derived point \"{new}\" already exists as body part.")
            # an unused table body part of the same name is replaced by the derived point (see add_derived_points)
            finals = [f for f in finals if f != new]
        absent = [p for p in parts if p not in finals]
        if absent:
            raise ValueError(f"Derived point \"{new}\" uses unknown body parts {absent}.")
        derive[new] = parts
        if all(p in positions for p in parts):
            positions[new] = tuple(np.mean([positions[p] for p in parts], axis=0).round(1).tolist())

    nodes = finals + list(derive)
    nodes_set = set(nodes)
    if skeleton.get("graph"):
        # graph body parts are final names, or table names where these are not final names themselves
        final = lambda n: n if n in nodes_set else rename.get(n, n)
        graph = {final(n): [final(m) for m in nbs] for n, nbs in skeleton["graph"].items()}
        absent = sorted({n for n, nbs in graph.items() for n in [n] + nbs} - set(nodes))
        if absent:
            raise ValueError(f"Graph body parts {absent} are not part of the skeleton.")
    else:
        unplaced = [n for n in nodes if n not in positions]
        if unplaced:
            raise ValueError(
                f"Body parts {unplaced} have no position in the mouse schema, so no graph can be computed. "
                "Please provide their \"positions\" or a \"graph\"."
            )
        graph = auto_graph({n: positions[n] for n in nodes})

    if "Center" not in nodes:
        suggestion = suggest_derived_center({n: p for n, p in positions.items() if n in finals})
        warnings.warn(
            "\033[38;5;208mYour skeleton has no \"Center\" body part, which deepOF uses by default for centering and "
            "ROIs. You can add it as the mean of other body parts with the skeleton entry "
            f"\"derive\": {{\"Center\": {suggestion}}}.\033[0m"
        )
    return {"rename": rename, "derive": derive, "graph": graph, "positions": positions}


def add_derived_points(table: pd.DataFrame, derive: Dict[str, list], animal_ids: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Add derived points to a table with (bodyparts, coords) columns, as the mean of their body parts.

    Coordinates are NaN if any of the body parts is missing, the likelihood is the minimum of the body parts. Body
    parts of the table with the name of a derived point are replaced.

    Returns:
        table (pd.DataFrame): table with derived points.
        replaced (list): names of the derived points that replaced body parts of the table.
    """
    if not derive:
        return table, []
    existing = set(table.columns.get_level_values("bodyparts"))
    clashes = sorted({f"{aid}_{new}" if aid else new for aid in animal_ids for new in derive} & existing)
    new_columns = {}
    for aid in animal_ids:
        prefix = f"{aid}_" if aid else ""
        for new, parts in derive.items():
            for coord in ("x", "y"):
                new_columns[(prefix + new, coord)] = table[[(prefix + p, coord) for p in parts]].mean(axis=1, skipna=False)
            if "likelihood" in table.columns.get_level_values("coords"):
                new_columns[(prefix + new, "likelihood")] = table[[(prefix + p, "likelihood") for p in parts]].min(axis=1)
    added = pd.DataFrame(new_columns, index=table.index)
    added.columns = pd.MultiIndex.from_tuples(added.columns, names=table.columns.names)
    # only the project's tables are affected, the source tables are never written to
    table = table.drop(columns=clashes, level="bodyparts") if clashes else table
    replaced = sorted(n for n in derive if any(c == n or c.endswith("_" + n) for c in clashes))
    return pd.concat([table, added], axis=1), replaced


def check_supervised_support(skeleton: dict, graph: Union[str, dict]):
    """Raise an error if the supervised annotations do not support the project's custom skeleton."""
    if skeleton is None:
        return
    nodes = set(nx.Graph(graph).nodes) if not isinstance(graph, str) else set()
    # body parts required by the supervised annotations
    required = set(deepof.utils.connect_mouse(graph_preset="deepof_11").nodes)
    missing = sorted(required - nodes)
    if missing:
        raise NotImplementedError(
            "Supervised annotations are not available for custom skeletons, as they require all deepof_11 body parts. "
            f"Missing body parts: {missing}."
        )
