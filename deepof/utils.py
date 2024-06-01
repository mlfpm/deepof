# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Functions and general utilities for the deepof package."""
import copy
from copy import deepcopy
from dask_image.imread import imread
from difflib import get_close_matches
from itertools import combinations, product
from joblib import Parallel, delayed
from math import atan2, dist
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
from segment_anything import sam_model_registry, SamPredictor
from shapely.geometry import Polygon
from sklearn import mixture
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sktime.transformations.series.impute import Imputer
from tqdm import tqdm
from typing import Tuple, Any, List, Union, NewType
import argparse
import cv2
import h5py
import math
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
import os
import pandas as pd
import regex as re
import requests
import ruptures as rpt
import sleap_io as sio
import torch
import warnings

import deepof.data

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# CONNECTIVITY AND GRAPH REPRESENTATIONS


def connect_mouse(
    animal_ids=None, exclude_bodyparts: list = None, graph_preset: str = "deepof_14"
) -> nx.Graph:
    """Create a nx.Graph object with the connectivity of the bodyparts in the DLC topview model for a single mouse.

    Used later for angle computing, among others.

    Args:
        animal_ids (str): if more than one animal is tagged, specify the animal identyfier as a string.
        exclude_bodyparts (list): Remove the specified nodes from the graph.
        graph_preset (str): Connectivity preset to use. Currently supported: "deepof_14" and "deepof_8".

    Returns:
        connectivity (nx.Graph)

    """
    if animal_ids is None:
        animal_ids = [""]
    if not isinstance(animal_ids, list):
        animal_ids = [animal_ids]

    connectivities = []

    for animal_id in animal_ids:
        try:
            connectivity_dict = {
                "deepof_14": {
                    "Nose": ["Left_ear", "Right_ear"],
                    "Spine_1": ["Center", "Left_ear", "Right_ear"],
                    "Center": ["Left_fhip", "Right_fhip", "Spine_2"],
                    "Spine_2": ["Left_bhip", "Right_bhip", "Tail_base"],
                    "Tail_base": ["Tail_1"],
                    "Tail_1": ["Tail_2"],
                    "Tail_2": ["Tail_tip"],
                },
                "deepof_8": {
                    "Nose": ["Left_ear", "Right_ear"],
                    "Center": [
                        "Left_fhip",
                        "Right_fhip",
                        "Tail_base",
                        "Left_ear",
                        "Right_ear",
                    ],
                    "Tail_base": ["Tail_tip"],
                },
            }
            connectivity = nx.Graph(connectivity_dict[graph_preset])
        except TypeError:
            connectivity = nx.Graph(graph_preset)

        if animal_id:
            mapping = {
                node: "{}_{}".format(animal_id, node) for node in connectivity.nodes()
            }
            if exclude_bodyparts is not None:
                exclude = ["{}_{}".format(animal_id, exc) for exc in exclude_bodyparts]
            nx.relabel_nodes(connectivity, mapping, copy=False)
        else:
            exclude = exclude_bodyparts

        if exclude_bodyparts is not None:
            connectivity.remove_nodes_from(exclude)

        connectivities.append(connectivity)

    if len(connectivities) > 1:
        pass

    final_graph = connectivities[0]
    for g in range(1, len(connectivities)):
        final_graph = nx.compose(final_graph, connectivities[g])
        final_graph.add_edge(
            "{}_Nose".format(animal_ids[g - 1]), "{}_Nose".format(animal_ids[g])
        )
        final_graph.add_edge(
            "{}_Tail_base".format(animal_ids[g - 1]),
            "{}_Tail_base".format(animal_ids[g]),
        )
        final_graph.add_edge(
            "{}_Nose".format(animal_ids[g]), "{}_Tail_base".format(animal_ids[g - 1])
        )
        final_graph.add_edge(
            "{}_Nose".format(animal_ids[g - 1]), "{}_Tail_base".format(animal_ids[g])
        )

    return final_graph


def edges_to_weighted_adj(adj: np.ndarray, edges: np.ndarray):
    """Convert an edge feature matrix to a weighted adjacency matrix.

    Args:
        - adj (np.ndarray): binary adjacency matrix of the current graph.
        - edges (np.ndarray): edge feature matrix. Last two axes should be of shape nodes x features.

    """
    adj = np.repeat(np.expand_dims(adj.astype(float), axis=0), edges.shape[0], axis=0)
    if len(edges.shape) == 3:
        adj = np.repeat(np.expand_dims(adj, axis=1), edges.shape[1], axis=1)

    adj[np.where(adj)] = np.concatenate([edges, edges[:, ::-1]], axis=-2).flatten()

    return adj


def enumerate_all_bridges(G: nx.graph) -> list:
    """Enumerate all 3-node connected sequences in the given graph.

    Args:
        - G (nx.graph): Animal connectivity graph.

    Returns:
        bridges (list): List with all 3-node connected sequences in the provided graph.

    """
    degrees = dict(nx.degree(G))
    centers = [node for node in degrees.keys() if degrees[node] >= 2]

    bridges = []
    for center in centers:
        for comb in list(combinations(list(G[center].keys()), 2)):
            bridges.append([comb[0], center, comb[1]])

    return bridges


# QUALITY CONTROL AND PREPROCESSING #


def str2bool(v: str) -> bool:
    """

    Return the passed string as a boolean.

    Args:
        v (str): String to transform to boolean value.

    Returns:
        bool. If conversion is not possible, it raises an error

    """
    if isinstance(v, bool):
        return v  # pragma: no cover
    elif v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean compatible value expected.")


def compute_animal_presence_mask(
    quality: table_dict, threshold: float = 0.5
) -> table_dict:
    """Compute a mask of the animal presence in the video.

    Args:
        quality (table_dict): Dictionary with the quality of the tracking for each body part and animal.
        threshold (float): Threshold for the quality of the tracking. If the quality is below this threshold, the animal is considered to be absent.

    Returns:
        animal_presence_mask (table_dict): Dictionary with the animal presence mask for each bodypart and animal.

    """
    animal_presence_mask = {}

    for exp in quality.keys():
        animal_presence_mask[exp] = {}
        for animal_id in quality._animal_ids:
            animal_presence_mask[exp][animal_id] = (
                quality.filter_id(animal_id)[exp].median(axis=1) > threshold
            ).astype(int)

        animal_presence_mask[exp] = pd.DataFrame(animal_presence_mask[exp])

    return deepof.data.TableDict(
        animal_presence_mask, typ="animal_presence_mask", animal_ids=quality._animal_ids
    )


def iterative_imputation(project: project, tab_dict: dict, lik_dict: dict):
    """Perform iterative imputation on occluded body parts. Run per animal and experiment.

    Args:
        project (project): Project object.
        tab_dict (dict): Dictionary with the coordinates of the body parts.
        lik_dict (dict): Dictionary with the likelihood of the tracking for each body part and animal.

    Returns:
        tab_dict (dict): Dictionary with the coordinates of the body parts after imputation.

    """
    presence_masks = compute_animal_presence_mask(lik_dict)
    tab_dict = deepof.data.TableDict(
        tab_dict, typ="coords", animal_ids=project.animal_ids
    )
    imputed_tabs = copy.deepcopy(tab_dict)

    for animal_id in project.animal_ids:

        for k, tab in tab_dict.filter_id(animal_id).items():

            try:
                scaler = StandardScaler()
                imputed = Imputer(method="drift",).fit_transform(
                    scaler.fit_transform(
                        tab.iloc[np.where(presence_masks[k][animal_id].values)[0]]
                    )
                )

                imputed = pd.DataFrame(
                    scaler.inverse_transform(imputed),
                    index=tab.index[np.where(presence_masks[k][animal_id].values)[0]],
                    columns=tab.loc[:, tab.isnull().mean(axis=0) != 1.0].columns,
                )

                imputed_tabs[k].update(imputed)

                if tab.shape[1] != imputed.shape[1]:
                    warnings.warn(
                        "Some of the body parts have zero measurements. Iterative imputation skips these,"
                        " which could bring problems downstream. A possible solution could be to refine "
                        "DLC tracklets."
                    )

            except ValueError:
                warnings.warn(
                    f"Animal {animal_id} in experiment {k} has not enough data. Skipping imputation."
                )

    return imputed_tabs


def set_missing_animals(
    coordinates: project, tab_dict: dict, lik_dict: dict, animal_ids: list = None
):
    """Set the coordinates of the missing animals to NaN.

    Args:
        coordinates (project): Project object.
        tab_dict (dict): Dictionary with the coordinates of the body parts.
        lik_dict (dict): Dictionary with the likelihood of the tracking for each body part and animal.
        animal_ids (list): List with the animal ids to remove. If None, all the animals with missing data are processed.

    Returns:
        tab_dict (dict): Dictionary with the coordinates of the body parts after removing missing animals.

    """
    if animal_ids is None:
        try:
            animal_ids = coordinates.animal_ids
        except AttributeError:
            animal_ids = coordinates._animal_ids

    presence_masks = compute_animal_presence_mask(lik_dict)
    tab_dict = deepof.data.TableDict(tab_dict, typ="qc", animal_ids=animal_ids)

    for animal_id in animal_ids:
        for k, tab in tab_dict.filter_id(animal_id).items():
            try:
                missing_times = tab[presence_masks[k][animal_id] == 0]
            except KeyError:
                missing_times = tab[
                    presence_masks[k].sum(axis=1) < (len(animal_ids) - 1)
                ]

            tab_dict[k].loc[missing_times.index, missing_times.columns] = np.nan

    return tab_dict


def bp2polar(tab: pd.DataFrame) -> pd.DataFrame:
    """Return the DataFrame in polar coordinates.

    Args:
        tab (pandas.DataFrame): Table with cartesian coordinates.

    Returns:
        polar (pandas.DataFrame): Equivalent to input, but with values in polar coordinates.

    """
    tab_ = np.array(tab)
    complex_ = tab_[:, 0] + 1j * tab_[:, 1]
    polar = pd.DataFrame(np.array([abs(complex_), np.angle(complex_)]).T)
    polar.rename(columns={0: "rho", 1: "phi"}, inplace=True)
    return polar


def tab2polar(cartesian_df: pd.DataFrame) -> pd.DataFrame:
    """Return a pandas.DataFrame in which all the coordinates are polar.

    Args:
        cartesian_df (pandas.DataFrame): DataFrame containing tables with cartesian coordinates.

    Returns:
        result (pandas.DataFrame): Equivalent to input, but with values in polar coordinates.

    """
    result = []
    for df in list(cartesian_df.columns.levels[0]):
        result.append(bp2polar(cartesian_df[df]))
    result = pd.concat(result, axis=1)
    idx = pd.MultiIndex.from_product(
        [list(cartesian_df.columns.levels[0]), ["rho", "phi"]]
    )
    result.columns = idx
    result.index = cartesian_df.index
    return result


def compute_dist(
    pair_array: np.array, arena_abs: int = 1, arena_rel: int = 1
) -> pd.DataFrame:
    """Return a pandas.DataFrame with the scaled distances between a pair of body parts.

    Args:
        pair_array (numpy.array): np.array of shape N * 4 containing X, y positions over time for a given pair of body parts.
        arena_abs (int): Diameter of the real arena in cm.
        arena_rel (int): Diameter of the captured arena in pixels.

    Returns:
        result (pd.DataFrame): pandas.DataFrame with the absolute distances between a pair of body parts.

    """
    lim = 2 if pair_array.shape[1] == 4 else 1
    a, b = pair_array[:, :lim], pair_array[:, lim:]
    ab = a - b

    dist = np.sqrt(np.einsum("...i,...i", ab, ab))
    return pd.DataFrame(dist * arena_abs / arena_rel)


def bpart_distance(
    dataframe: pd.DataFrame, arena_abs: int = 1, arena_rel: int = 1
) -> pd.DataFrame:
    """Return a pandas.DataFrame with the scaled distances between all pairs of body parts.

    Args:
        dataframe (pandas.DataFrame): pd.DataFrame of shape N*(2*bp) containing X,y positions over time for a given set of bp body parts.
        arena_abs (int): Diameter of the real arena in cm.
        arena_rel (int): Diameter of the captured arena in pixels.

    Returns:
        result (pd.DataFrame): pandas.DataFrame with the absolute distances between all pairs of body parts.

    """
    indexes = combinations(dataframe.columns.levels[0], 2)
    dists = []
    for idx in indexes:
        dist = compute_dist(np.array(dataframe.loc[:, list(idx)]), arena_abs, arena_rel)
        dist.columns = [idx]
        dists.append(dist)

    return pd.concat(dists, axis=1)


def angle(bpart_array: np.array) -> np.array:
    """Return a numpy.ndarray with the angles between the provided instances.

    Args:
        bpart_array (numpy.array): 2D positions over time for a bodypart.

    Returns:
        ang (np.array): 1D angles between the three-point-instances.

    """
    a, b, c = bpart_array

    ba = a - b
    bc = c - b

    cosine_angle = np.einsum("...i,...i", ba, bc) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    ang = np.arccos(cosine_angle)

    return ang


def compute_areas(polygon_xy_stack):
    """Compute polygon areas for the provided stack of sets of data point-xy coordinates.

    Args:
        polygon_xy_stack: 3D numpy array [NPolygons (i.e. NFrames), Npoints, NDim (x,y)]

    Returns:
        areas: list areas for the provided xy coordinates.

    """

    # list of polygon areas, a list entry is set to np.nan if points forming the respective polygon are missing
    polygon_areas = [
        Polygon(polygon_xy_stack[i]).area
        if not np.isnan(polygon_xy_stack[i]).any()
        else np.nan
        for i in range(len(polygon_xy_stack))
    ]

    return polygon_areas


def rotate(
    p: np.array, angles: np.array, origin: np.array = np.array([0, 0])
) -> np.array:
    """Return a 2D numpy.ndarray with the initial values rotated by angles radians.

    Args:
        p (numpy.ndarray): 2D Array containing positions of bodyparts over time.
        angles (numpy.ndarray): Set of angles (in radians) to rotate p with.
        origin (numpy.ndarray): Rotation axis (zero vector by default).

    Returns:
        - rotated (numpy.ndarray): rotated positions over time

    """
    R = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]])

    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    rotated = np.squeeze((R @ (p.T - o.T) + o.T).T)

    return rotated


# noinspection PyArgumentList
def align_trajectories(data: np.array, mode: str = "all") -> np.array:
    """Remove rotational variance on the trajectories.

    Returns a numpy.array with the positions rotated in a way that the center (0 vector), and body part in the first
    column of data are aligned with the y-axis.

    Args:
        data (numpy.ndarray): 3D array containing positions of body parts over time, where shape is N (sliding window instances) * m (sliding window size) * l (features)
        mode (string): Specifies if *all* instances of each sliding window get aligned, or only the *center*

    Returns:
        aligned_trajs (np.ndarray): 2D aligned positions over time.

    """
    angles = np.zeros(data.shape[0])
    data = deepcopy(data)
    dshape = data.shape

    if mode == "center":
        center_time = (data.shape[1] - 1) // 2
        angles = np.arctan2(data[:, center_time, 0], data[:, center_time, 1])
    elif mode == "all":
        data = data.reshape(-1, dshape[-1], order="C")
        angles = np.arctan2(data[:, 0], data[:, 1])
    elif mode == "none":
        data = data.reshape(-1, dshape[-1], order="C")
        angles = np.zeros(data.shape[0])

    aligned_trajs = np.zeros(data.shape)

    for frame in range(data.shape[0]):
        aligned_trajs[frame] = rotate(
            data[frame].reshape([-1, 2], order="C"), angles[frame]
        ).reshape(data.shape[1:], order="C")

    if mode == "all" or mode == "none":
        aligned_trajs = aligned_trajs.reshape(dshape, order="C")

    return aligned_trajs


def load_table(
    tab: str,
    table_path: str,
    table_format: str,
    rename_bodyparts: list = None,
    animal_ids: list = None,
):
    """Loads a table into a structured pandas data frame.

    Supports inputs from both DeepLabCut and (S)LEAP.

    Args:
        tab (str): Name of the file containing the tracks.
        table_path (string): Full path to the file containing the tracks.
        table_format (str): type of the files to load, coming from either DeepLabCut (CSV and H5) and (S)LEAP (NPY).
        rename_bodyparts (list): list of names to use for the body parts in the provided tracking files. The order should match that of the columns in your DLC tables or the node dimensions on your (S)LEAP .npy files.

    Returns:
        loaded_tab (pd.DataFrame): Data frame containing the loaded tracks. Likelihood for (S)LEAP files is imputed as 1.0 (tracked values) or 0.0 (missing values).

    """

    if table_format == "h5":

        loaded_tab = pd.read_hdf(os.path.join(table_path, tab), dtype=float)

        # Adapt index to be compatible with downstream processing
        loaded_tab = loaded_tab.T.reset_index(drop=False).T
        loaded_tab.columns = loaded_tab.loc["scorer", :]
        loaded_tab = loaded_tab.iloc[1:]

    elif table_format == "csv":

        loaded_tab = pd.read_csv(
            os.path.join(table_path, tab),
            index_col=0,
            low_memory=False,
        )

    elif table_format in ["npy", "slp", "analysis.h5"]:

        if table_format == "analysis.h5":
            # Load sleap .h5 file from disk
            with h5py.File(os.path.join(table_path, tab), "r") as f:
                loaded_tab = np.stack(np.transpose(f["tracks"][:], [3, 0, 2, 1]))
                slp_bodyparts = [n.decode() for n in f["node_names"][:]]
                slp_animal_ids = [n.decode() for n in f["track_names"][:]]

        elif table_format == "slp":
            # Use sleap-io to convert .slp files into numpy arrays
            loaded_tab = sio.load_slp(os.path.join(table_path, tab))
            slp_bodyparts = [i.name for i in loaded_tab.skeletons[0].nodes]
            slp_animal_ids = [i.name for i in loaded_tab.tracks]
            loaded_tab = loaded_tab.numpy()

        else:
            # Load numpy array from disk
            loaded_tab = np.load(os.path.join(table_path, tab), "r")

            # Check that body part names are provided
            slp_bodyparts = rename_bodyparts
            if not animal_ids[0]:
                slp_animal_ids = [str(i) for i in range(loaded_tab.shape[1])]
            else:
                slp_animal_ids = animal_ids
        assert len(slp_bodyparts) == loaded_tab.shape[2], (
            "Some body part names appear to be in excess or missing.\n"
            " If you used the rename_bodyparts argument, check if you set it correctly.\n"
            " Otherwise, there might be an issue with the tables in your Tables-folder"
        )

        # Create the header as a multi index, using animals, body parts and coordinates
        if not animal_ids[0]:
            animal_ids = slp_animal_ids

        # Impute likelihood as a third dimension in the last axis,
        # with 1.0 if xy values are present and 0.0 otherwise
        likelihoods = np.expand_dims(
            np.all(np.isfinite(loaded_tab), axis=-1), axis=-1
        ).astype(float)
        loaded_tab = np.concatenate([loaded_tab, likelihoods], axis=-1)

        # Collapse nodes and animals to the desired shape
        loaded_tab = pd.DataFrame(loaded_tab.reshape(loaded_tab.shape[0], -1))

        multi_index = pd.MultiIndex.from_product(
            [["sleap_scorer"], slp_animal_ids, slp_bodyparts, ["x", "y", "likelihood"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        multi_index = pd.DataFrame(
            pd.DataFrame(multi_index).explode(0).values.reshape([-1, 4]).T,
            index=["scorer", "individuals", "bodyparts", "coords"],
        )

        loaded_tab = pd.concat([multi_index.iloc[1:], loaded_tab], axis=0)
        loaded_tab.columns = multi_index.loc["scorer"]

    if rename_bodyparts is not None:
        loaded_tab = rename_track_bps(
            loaded_tab,
            rename_bodyparts,
            (animal_ids if table_format in ["h5", "csv"] else [""]),
        )

    return loaded_tab


def rename_track_bps(
    loaded_tab: pd.DataFrame, rename_bodyparts: list, animal_ids: list
):
    """Renames all body parts in the provided dataframe.

    Args:
        loaded_tab (pd.DataFrame): Data frame containing the loaded tracks. Likelihood for (S)LEAP files is imputed as 1.0 (tracked values) or 0.0 (missing values).
        rename_bodyparts (list): list of names to use for the body parts in the provided tracking files. The order should match that of the columns in your DLC tables or the node dimensions on your (S)LEAP files.
        animal_ids (list): list of IDs to use for the animals present in the provided tracking files.

    Returns:
        renamed_tab (pd.DataFrame): Data frame with renamed body parts

    """
    renamed_tab = copy.deepcopy(loaded_tab)

    if not animal_ids[0]:
        current_bparts = loaded_tab.loc["bodyparts", :].unique()
    else:
        current_bparts = list(
            map(
                lambda x: "_".join(x.split("_")[1:]),
                loaded_tab.loc["bodyparts", :].unique(),
            )
        )

    for old, new in zip(current_bparts, rename_bodyparts):
        renamed_tab.replace(old, new, inplace=True, regex=True)

    return renamed_tab


def scale_table(
    coordinates: coordinates,
    feature_array: np.ndarray,
    scale: str,
    global_scaler: Any = None,
):
    """Scales features in a table controlling for both individual body size and interanimal variability.

    Args:
        coordinates (coordinates): a deepof coordinates object.
        feature_array (np.ndarray): array to scale. Should be shape (instances x features).
        scale (str): Data scaling method. Must be one of 'standard', 'robust' (default; recommended) and 'minmax'.
        global_scaler (Any): global scaler, fit in the whole dataset.

    """
    exp_temp = feature_array.to_numpy()

    annot_length = 0
    if coordinates._propagate_labels:
        exp_temp = exp_temp[:, :-1]
        annot_length += 1

    if coordinates._propagate_annotations:
        exp_temp = exp_temp[
            :, : -list(coordinates._propagate_annotations.values())[0].shape[1]
        ]
        annot_length += list(coordinates._propagate_annotations.values())[0].shape[1]

    if global_scaler is None:
        # Scale each modality separately using a custom function
        exp_temp = scale_animal(exp_temp, scale)
    else:
        # Scale all experiments together, to control for differential stats
        exp_temp = global_scaler.transform(exp_temp)

    current_tab = np.concatenate(
        [
            exp_temp,
            feature_array.copy().to_numpy()[:, feature_array.shape[1] - annot_length :],
        ],
        axis=1,
    )

    return current_tab


def scale_animal(feature_array: np.ndarray, scale: str):
    """Scales features in the provided array.

    Args:
        feature_array (np.ndarray): array to scale. Should be shape (instances x features).
        graph (nx.Graph): connectivity graph for the current animals.
        scale (str): Data scaling method. Must be one of 'standard', 'robust' (default; recommended) and 'minmax'.

    Returns:
        Scaled version of the input array, with features normalized by modality.
        List of scalers per modality.

    """
    scalers = []

    # number of body part sets to use for coords (x, y), speeds, and distances
    if scale == "standard":
        cur_scaler = StandardScaler()
    elif scale == "minmax":
        cur_scaler = MinMaxScaler()
    else:
        cur_scaler = RobustScaler()

    normalized_array = cur_scaler.fit_transform(feature_array)
    scalers.append(cur_scaler)

    return normalized_array


def kleinberg(
    offsets: list, s: float = np.e, gamma: float = 1.0, n=None, T=None, k=None
):
    """Apply Kleinberg's algorithm (described in 'Bursty and Hierarchical Structure in Streams').

    The algorithm models activity bursts in a time series as an
    infinite hidden Markov model.

    Taken from pybursts (https://github.com/romain-fontugne/pybursts/blob/master/pybursts/pybursts.py)
    and adapted for dependency compatibility reasons.

    Args:
        offsets (list): a list of time offsets (numeric)
        s (float): the base of the exponential distribution that is used for modeling the event frequencies
        gamma (float): coefficient for the transition costs between states
        n, T: to have a fixed cost function (not dependent of the given offsets). Which is needed if you want to compare bursts for different inputs.
        k: maximum burst level

    """
    if s <= 1:
        raise ValueError("s must be greater than 1!")
    if gamma <= 0:
        raise ValueError("gamma must be positive!")
    if not n is None and n <= 0:
        raise ValueError("n must be positive!")
    if not T is None and T <= 0:
        raise ValueError("T must be positive!")
    if len(offsets) < 1:
        raise ValueError("offsets must be non-empty!")

    offsets = np.array(offsets, dtype=object)

    if offsets.size == 1:
        bursts = np.array([0, offsets[0], offsets[0]], ndmin=2, dtype=object)
        return bursts

    offsets = np.sort(offsets)
    gaps = np.diff(offsets)

    if not np.all(gaps):
        raise ValueError("Input cannot contain events with zero time between!")

    if T is None:
        T = np.sum(gaps)

    if n is None:
        n = np.size(gaps)

    g_hat = T / n
    gamma_log_n = gamma * math.log(n)

    if k is None:
        k = int(math.ceil(float(1 + math.log(T, s) + math.log(1 / np.amin(gaps), s))))

    def tau(i, j):
        if i >= j:
            return 0
        else:
            return (j - i) * gamma_log_n

    alpha_function = np.vectorize(lambda x: s**x / g_hat)
    alpha = alpha_function(np.arange(k))

    def f(j, x):
        return alpha[j] * math.exp(-alpha[j] * x)

    C = np.repeat(float("inf"), k)
    C[0] = 0

    q = np.empty((k, 0))
    for t in range(np.size(gaps)):
        C_prime = np.repeat(float("inf"), k)
        q_prime = np.empty((k, t + 1))
        q_prime.fill(np.nan)

        for j in range(k):
            cost_function = np.vectorize(lambda x: C[x] + tau(x, j))
            cost = cost_function(np.arange(0, k))

            el = np.argmin(cost)

            if f(j, gaps[t]) > 0:
                C_prime[j] = cost[el] - math.log(f(j, gaps[t]))

            if t > 0:
                q_prime[j, :t] = q[el, :]

            q_prime[j, t] = j + 1

        C = C_prime
        q = q_prime

    j = np.argmin(C)
    q = q[j, :]

    prev_q = 0

    N = 0
    for t in range(np.size(gaps)):
        if q[t] > prev_q:
            N = N + q[t] - prev_q
        prev_q = q[t]

    bursts = np.array(
        [np.repeat(np.nan, N), np.repeat(offsets[0], N), np.repeat(offsets[0], N)],
        ndmin=2,
        dtype=object,
    ).transpose()

    burst_counter = -1
    prev_q = 0
    stack = np.zeros(int(N), dtype=int)
    stack_counter = -1
    for t in range(np.size(gaps)):
        if q[t] > prev_q:
            num_levels_opened = q[t] - prev_q
            for i in range(int(num_levels_opened)):
                burst_counter += 1
                bursts[burst_counter, 0] = prev_q + i
                bursts[burst_counter, 1] = offsets[t]
                stack_counter += 1
                stack[stack_counter] = int(burst_counter)
        elif q[t] < prev_q:
            num_levels_closed = prev_q - q[t]
            for i in range(int(num_levels_closed)):
                bursts[stack[stack_counter], 2] = offsets[t]
                stack_counter -= 1
        prev_q = q[t]

    while stack_counter >= 0:
        bursts[stack[stack_counter], 2] = offsets[np.size(gaps)]
        stack_counter -= 1

    return bursts


def smooth_boolean_array(a: np.array, scale: int = 1) -> np.array:
    """Return a boolean array in which isolated appearances of a feature are smoothed.

    Args:
        a (numpy.ndarray): Boolean instances.
        scale (int): Kleinberg scale parameter. Higher values result in stricter smoothing.

    Returns:
        a (numpy.ndarray): Smoothened boolean instances.

    """
    offsets = np.where(a)[0]
    if len(offsets) == 0:
        return a  # no detected activity

    bursts = kleinberg(offsets, gamma=0.01)
    a = np.zeros(np.size(a), dtype=bool)
    for i in bursts:
        if i[0] == scale:
            a[int(i[1]) : int(i[2])] = True

    return a


def split_with_breakpoints(a: np.ndarray, breakpoints: list) -> np.ndarray:
    """

    Split a numpy.ndarray at the given breakpoints.

    Args:
        a (np.ndarray): N (instances) * m (features) shape
        breakpoints (list): list of breakpoints obtained with ruptures

    Returns:
        split_a (np.ndarray): padded array of shape N (instances) * l (maximum break length) * m (features)

    """
    rpt_lengths = list(np.array(breakpoints)[1:] - np.array(breakpoints)[:-1])

    try:
        max_rpt_length = np.max([breakpoints[0], np.max(rpt_lengths)])
    except ValueError:
        max_rpt_length = breakpoints[0]

    # Reshape experiment data according to extracted ruptures
    split_a = np.split(np.expand_dims(a, axis=0), breakpoints[:-1], axis=1)

    split_a = [
        np.pad(
            i, ((0, 0), (0, max_rpt_length - i.shape[1]), (0, 0)), constant_values=0.0
        )
        for i in split_a
    ]
    split_a = np.concatenate(split_a, axis=0)

    return split_a


def rolling_window(
    a: np.ndarray,
    window_size: int,
    window_step: int,
    automatic_changepoints: str = False,
    precomputed_breaks: np.ndarray = None,
) -> np.ndarray:
    """Return a 3D numpy.array with a sliding-window extra dimension.

    Args:
        a (np.ndarray): N (instances) * m (features) shape
        window_size (int): Size of the window to apply
        window_step (int): Step of the window to apply
        automatic_changepoints (str): Changepoint detection algorithm to apply. If False, applies a fixed sliding window.
        precomputed_breaks (np.ndarray): Precomputed breaks to use, bypassing the changepoint detection algorithm. None by default (break points are computed).

    Returns:
        rolled_a (np.ndarray): N (sliding window instances) * l (sliding window size) * m (features)

    """
    breakpoints = None

    if automatic_changepoints:
        # Define change point detection model using ruptures
        # Remove dimensions with low variance (occurring when aligning the animals with the y axis)
        if precomputed_breaks is None:
            rpt_model = rpt.KernelCPD(
                kernel=automatic_changepoints, min_size=window_size, jump=window_step
            ).fit(VarianceThreshold(threshold=1e-3).fit_transform(a))

            # Extract change points from current experiment
            breakpoints = rpt_model.predict(pen=4.0)

        else:
            breakpoints = np.cumsum(precomputed_breaks)

        rolled_a = split_with_breakpoints(a, breakpoints)

    else:
        shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
        strides = (a.strides[0],) + a.strides
        rolled_a = np.lib.stride_tricks.as_strided(
            a, shape=shape, strides=strides, writeable=True
        )[::window_step]

    return rolled_a, breakpoints


def rupture_per_experiment(
    table_dict: table_dict,
    to_rupture: np.ndarray,
    rupture_indices: list,
    automatic_changepoints: str,
    window_size: int,
    window_step: int,
    precomputed_breaks: dict = None,
) -> np.ndarray:
    """Apply the rupture method independently to each experiment, and concatenate into a single dataset at the end.

    Returns a dataset and the rupture indices, adapted to be used in a concatenated version
    of the labels.

    Args:
        table_dict (deepof.data.table_dict): table_dict with all experiments.
        to_rupture (np.ndarray): Array with dataset to rupture.
        rupture_indices (list): Indices of tables to rupture. Useful to select training and test sets.
        automatic_changepoints (str): Rupture method to apply. If false, a sliding window of window_length * window_size is obtained. If one of "l1", "l2" or "rbf", different automatic change point detection algorithms are applied on each independent experiment.
        window_size (int): If automatic_changepoints is False, specifies the length of the sliding window. If not, it determines the minimum size of the obtained time series breaks.
        window_step (int): If automatic_changepoints is False, specifies the stride of the sliding window. If not, it determines the minimum step size of the obtained time series breaks.
        precomputed_breaks (dict): If provided, changepoint detection is prevented, and provided breaks are used instead.

    Returns:
        ruptured_dataset (np.ndarray): Dataset with all ruptures concatenated across the first axis.
        rupture_indices (list): Indices of ruptures.

    """
    # Generate a base ruptured training set and a set of breaks
    ruptured_dataset, break_indices = None, None
    cumulative_shape = 0
    # Iterate over all experiments and populate them
    for i, (key, tab) in enumerate(table_dict.items()):
        if i in rupture_indices:
            current_size = tab.shape[0]
            current_train, current_breaks = rolling_window(
                to_rupture[cumulative_shape : cumulative_shape + current_size],
                window_size,
                window_step,
                automatic_changepoints,
                (None if not precomputed_breaks else precomputed_breaks[key]),
            )
            # Add shape of the current tab as the last breakpoint,
            # to avoid skipping breakpoints between experiments
            if current_breaks is not None:
                current_breaks = np.array(current_breaks) + cumulative_shape

            cumulative_shape += current_size

            try:  # pragma: no cover
                # To concatenate the current ruptures with the ones obtained
                # until now, pad the smallest to the length of the largest
                # alongside axis 1 (temporal dimension) with zeros.
                if ruptured_dataset.shape[1] >= current_train.shape[1]:
                    current_train = np.pad(
                        current_train,
                        (
                            (0, 0),
                            (0, ruptured_dataset.shape[1] - current_train.shape[1]),
                            (0, 0),
                        ),
                    )
                elif ruptured_dataset.shape[1] < current_train.shape[1]:
                    ruptured_dataset = np.pad(
                        ruptured_dataset,
                        (
                            (0, 0),
                            (0, current_train.shape[1] - ruptured_dataset.shape[1]),
                            (0, 0),
                        ),
                    )

                # Once that's taken care of, concatenate ruptures alongside axis 0
                ruptured_dataset = np.concatenate([ruptured_dataset, current_train])
                if current_breaks is not None:
                    break_indices = np.concatenate([break_indices, current_breaks])
            except (ValueError, AttributeError):
                ruptured_dataset = current_train
                if current_breaks is not None:
                    break_indices = current_breaks

    return ruptured_dataset, break_indices


def smooth_mult_trajectory(
    series: np.array, alpha: int = 0, w_length: int = 11
) -> np.ndarray:
    """Return a smoothed a trajectory using a Savitzky-Golay 1D filter.

    Args:
        series (numpy.ndarray): 1D trajectory array with N (instances)
        alpha (int): 0 <= alpha < w_length; indicates the difference between the degree of the polynomial and the window length for the Savitzky-Golay filter used for smoothing. Higher values produce a worse fit, hence more smoothing.
        w_length (int): Length of the sliding window to which the filter fit. Higher values yield a coarser fit, hence more smoothing.

    Returns:
        smoothed_series (np.ndarray): smoothed version of the input, with equal shape

    """
    if alpha is None:
        return series

    smoothed_series = savgol_filter(
        series, polyorder=(w_length - alpha), window_length=w_length, axis=0
    )

    assert smoothed_series.shape == series.shape

    return smoothed_series


def moving_average(time_series: pd.Series, lag: int = 5) -> pd.Series:
    """Fast implementation of a moving average function.

    Args:
        time_series (pd.Series): Uni-variate time series to take the moving average of.
        lag (int): size of the convolution window used to compute the moving average.

    Returns:
        moving_avg (pd.Series): Uni-variate moving average over time_series.

    """
    moving_avg = np.convolve(time_series, np.ones(lag) / lag, mode="same")

    return moving_avg


def mask_outliers(
    time_series: pd.DataFrame,
    likelihood: pd.DataFrame,
    likelihood_tolerance: float,
    lag: int,
    n_std: int,
    mode: str,
) -> pd.DataFrame:
    """Return a mask over the bivariate trajectory of a body part, identifying as True all detected outliers.

    An outlier can be marked with one of two criteria: 1) the likelihood reported by DLC is below likelihood_tolerance,
    and/or 2) the deviation from a moving average model is greater than n_std.

    Args:
        time_series (pd.DataFrame): Bi-variate time series representing the x, y positions of a single body part
        likelihood (pd.DataFrame): Data frame with likelihood data per body part as extracted from deeplabcut
        likelihood_tolerance (float): Minimum tolerated likelihood, below which an outlier is called
        lag (int): Size of the convolution window used to compute the moving average
        n_std (int): Number of standard deviations over the moving average to be considered an outlier
        mode (str): If "and" (default) both x and y have to be marked in order to call an outlier. If "or", one is enough.

    Returns
        mask (pd.DataFrame): Bi-variate mask over time_series. True indicates an outlier.

    """
    moving_avg_x = moving_average(time_series["x"], lag)
    moving_avg_y = moving_average(time_series["y"], lag)

    residuals_x = time_series["x"] - moving_avg_x
    residuals_y = time_series["y"] - moving_avg_y

    outlier_mask_x = np.abs(residuals_x) > np.mean(
        residuals_x[lag:-lag]
    ) + n_std * np.std(residuals_x[lag:-lag])
    outlier_mask_y = np.abs(residuals_y) > np.mean(
        residuals_y[lag:-lag]
    ) + n_std * np.std(residuals_y[lag:-lag])
    outlier_mask_l = likelihood < likelihood_tolerance
    mask = None

    if mode == "and":
        mask = (outlier_mask_x & outlier_mask_y) | outlier_mask_l
    elif mode == "or":
        mask = (outlier_mask_x | outlier_mask_y) | outlier_mask_l

    return mask


def full_outlier_mask(
    experiment: pd.DataFrame,
    likelihood: pd.DataFrame,
    likelihood_tolerance: float,
    exclude: str,
    lag: int,
    n_std: int,
    mode: str,
) -> pd.DataFrame:
    """Iterate over all body parts of experiment, and outputs a dataframe where all x, y positions are replaced by a boolean mask, where True indicates an outlier.

    Args:
        experiment (pd.DataFrame): Data frame with time series representing the x, y positions of every body part
        likelihood (pd.DataFrame): Data frame with likelihood data per body part as extracted from deeplabcut
        likelihood_tolerance (float): Minimum tolerated likelihood, below which an outlier is called
        exclude (str): Body part to exclude from the analysis (to concatenate with bpart alignment)
        lag (int): Size of the convolution window used to compute the moving average
        n_std (int): Number of standard deviations over the moving average to be considered an outlier
        mode (str): If "and" (default) both x and y have to be marked in order to call an outlier. If "or", one is enough.

    Returns:
        full_mask (pd.DataFrame): Mask over all body parts in experiment. True indicates an outlier

    """
    body_parts = experiment.columns.levels[0]
    full_mask = experiment.copy()

    if exclude:
        full_mask.drop(exclude, axis=1, inplace=True)

    for bpart in body_parts:
        if bpart != exclude:
            mask = mask_outliers(
                experiment[bpart],
                likelihood[bpart],
                likelihood_tolerance,
                lag,
                n_std,
                mode,
            )

            full_mask.loc[:, (bpart, "x")] = mask
            full_mask.loc[:, (bpart, "y")] = mask
            continue

    return full_mask


def interpolate_outliers(
    experiment: pd.DataFrame,
    likelihood: pd.DataFrame,
    likelihood_tolerance: float,
    exclude: str = "",
    lag: int = 5,
    n_std: int = 3,
    mode: str = "or",
    limit: int = 10,
) -> pd.DataFrame:
    """Mark all outliers in experiment and replaces them using a uni-variate linear interpolation approach.

    Note that this approach only works for equally spaced data (constant camera acquisition rates).

    Args:
        experiment (pd.DataFrame): Data frame with time series representing the x, y positions of every body part.
        likelihood (pd.DataFrame): Data frame with likelihood data per body part as extracted from deeplabcut.
        likelihood_tolerance (float): Minimum tolerated likelihood, below which an outlier is called.
        exclude (str): Body part to exclude from the analysis (to concatenate with bpart alignment).
        lag (int): Size of the convolution window used to compute the moving average.
        n_std (int): Number of standard deviations over the moving average to be considered an outlier.
        mode (str): If "and" both x and y have to be marked in order to call an outlier. If "or" (default), one is enough.
        limit (int): Maximum of consecutive outliers to interpolate. Defaults to 10.

    Returns:
        interpolated_exp (pd.DataFrame): Interpolated version of experiment.

    """
    interpolated_exp = experiment.copy()

    # Creates a mask marking all outliers
    mask = full_outlier_mask(
        experiment, likelihood, likelihood_tolerance, exclude, lag, n_std, mode
    )

    interpolated_exp[mask] = np.nan
    interpolated_exp.interpolate(
        method="linear", limit=limit, limit_direction="both", inplace=True
    )
    # Add original frames to what happens before lag
    interpolated_exp = pd.concat(
        [experiment.iloc[:lag, :], interpolated_exp.iloc[lag:, :]]
    )

    return interpolated_exp


def filter_columns(columns: list, selected_id: str) -> list:
    """Given a set of TableDict columns, returns those that correspond to a given animal, specified in selected_id.

    Args:
        columns (list): List of columns to filter.
        selected_id (str): Animal ID to filter for.

    Returns:
        filtered_columns (list): List of filtered columns.

    """
    if selected_id is None:
        return columns

    columns_to_keep = []
    for column in columns:
        # Speed transformed columns
        if selected_id == "supervised" and column in [
            "nose2nose",
            "sidebyside",
            "sidereside",
        ]:
            columns_to_keep.append(column)
        if type(column) == str and column.startswith(selected_id):
            columns_to_keep.append(column)
        # Raw coordinate columns
        if column[0].startswith(selected_id) and column[1] in ["x", "y", "rho", "phi"]:
            columns_to_keep.append(column)
        # Raw distance and angle columns
        elif len(column) in [2, 3] and all([i.startswith(selected_id) for i in column]):
            columns_to_keep.append(column)
        elif column[0].lower().startswith("pheno"):
            columns_to_keep.append(column)

    return columns_to_keep


def load_segmentation_model(path):
    model_url = "https://datashare.mpcdf.mpg.de/s/GccLGXXZmw34f8o/download"

    if path is None:
        installation_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(
            installation_path,
            "trained_models",
            "arena_segmentation",
            "sam_vit_h_4b8939.pth",
        )

    if not os.path.exists(path):
        # Creating directory if it does not exist
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        print("Arena segmentation model not found. Downloading...")

        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        with open(path, "wb") as file:
            total_length = int(response.headers.get("content-length"))
            for chunk in tqdm(
                response.iter_content(chunk_size=1024),
                total=total_length // 1024,
                unit="KB",
            ):
                if chunk:
                    file.write(chunk)

    # Load the model using PyTorch
    sam = sam_model_registry["vit_h"](checkpoint=path)
    sam.to(device="cpu")
    predictor = SamPredictor(sam)

    return predictor


def get_arenas(
    coordinates: coordinates,
    tables: table_dict,
    arena: str,
    arena_dims: int,
    project_path: str,
    project_name: str,
    segmentation_model_path: str,
    videos: list = None,
    debug: bool = False,
    test: bool = False,
):
    """Extract arena parameters from a project or coordinates object.

    Args:
        coordinates (coordinates): Coordinates object.
        tables (table_dict): TableDict object containing tracklets per animal.
        arena (str): Arena type (must be either "polygonal-manual", "circular-manual", "polygonal-autodetect", or "circular-autodetect").
        arena_dims (int): Arena dimensions.
        project_path (str): Path to project.
        project_name (str): Name of project.
        segmentation_model_path (str): Path to segmentation model used for automatic arena detection.
        videos (list): List of videos to extract arena parameters from. Defaults to None (all videos are used).
        debug (bool): If True, a frame per video with the detected arena is saved. Defaults to False.
        test (bool): If True, the function is run in test mode. Defaults to False.

    Returns:
        arena_params (list): List of arena parameters.

    """
    scales = []
    arena_params = []
    video_resolution = []

    def get_first_length(arena_corners):
        return math.dist(arena_corners[0], arena_corners[1])

    if arena in ["polygonal-manual", "circular-manual"]:  # pragma: no cover

        propagate_last = False
        for i, video_path in enumerate(videos):

            if not propagate_last:
                arena_corners, h, w = extract_polygonal_arena_coordinates(
                    os.path.join(project_path, project_name, "Videos", video_path),
                    arena,
                    i,
                    videos,
                )

                if arena_corners is None:
                    propagate_last = True

                else:
                    cur_scales = [
                        *np.mean(arena_corners, axis=0).astype(int),
                        get_first_length(arena_corners),
                        arena_dims,
                    ]

            if propagate_last:
                cur_arena_params = arena_params[-1]
                cur_scales = scales[-1]
            else:
                cur_arena_params = arena_corners

            if arena == "circular-manual":

                if not propagate_last:
                    cur_arena_params = fit_ellipse_to_polygon(cur_arena_params)

                scales.append(
                    list(
                        np.array(
                            [
                                cur_arena_params[0][0],
                                cur_arena_params[0][1],
                                np.mean(
                                    [cur_arena_params[1][0], cur_arena_params[1][1]]
                                )
                                * 2,
                            ]
                        )
                    )
                    + [arena_dims]
                )
            else:
                scales.append(cur_scales)

            arena_params.append(cur_arena_params)
            video_resolution.append((h, w))

    elif arena in ["polygonal-autodetect", "circular-autodetect"]:

        # Open GUI for manual labelling of two scaling points in the first video
        arena_reference = None
        if arena == "polygonal-autodetect":  # pragma: no cover

            if test:
                arena_reference = np.zeros((4, 2))
            else:
                arena_reference = extract_polygonal_arena_coordinates(
                    os.path.join(project_path, project_name, "Videos", videos[0]),
                    arena,
                    0,
                    [videos[0]],
                )[0]

        # Load SAM
        segmentation_model = load_segmentation_model(segmentation_model_path)

        for vid_index, _ in enumerate(videos):
            arena_parameters, h, w = automatically_recognize_arena(
                coordinates=coordinates,
                tables=tables,
                videos=videos,
                vid_index=vid_index,
                path=os.path.join(project_path, project_name, "Videos"),
                arena_type=arena,
                arena_reference=arena_reference,
                segmentation_model=segmentation_model,
                debug=debug,
            )

            if "polygonal" in arena:

                closest_side_points = closest_side(
                    simplify_polygon(arena_parameters), arena_reference[:2]
                )

                scales.append(
                    [
                        *np.mean(arena_parameters, axis=0).astype(int),
                        dist(*closest_side_points),
                        arena_dims,
                    ]
                )

            elif "circular" in arena:
                # scales contains the coordinates of the center of the arena,
                # the absolute diameter measured from the video in pixels, and
                # the provided diameter in mm (1 -default- equals not provided)
                scales.append(
                    list(
                        np.array(
                            [
                                arena_parameters[0][0],
                                arena_parameters[0][1],
                                np.mean(
                                    [arena_parameters[1][0], arena_parameters[1][1]]
                                )
                                * 2,
                            ]
                        )
                    )
                    + [arena_dims]
                )

            arena_params.append(arena_parameters)
            video_resolution.append((h, w))

    elif not arena:
        return None, None, None

    else:  # pragma: no cover
        raise NotImplementedError(
            "arenas must be set to one of: 'polygonal-manual', 'polygonal-autodetect', 'circular-manual', 'circular-autodetect'"
        )

    return np.array(scales), arena_params, video_resolution


def simplify_polygon(polygon: list, relative_tolerance: float = 0.05):
    """Simplify a polygon using the Ramer-Douglas-Peucker algorithm.

    Args:
        polygon (list): List of polygon coordinates.
        relative_tolerance (float): Relative tolerance for simplification. Defaults to 0.05.

    Returns:
        simplified_poly (list): List of simplified polygon coordinates.

    """
    poly = Polygon(polygon)
    perimeter = poly.length
    tolerance = perimeter * relative_tolerance

    simplified_poly = poly.simplify(tolerance, preserve_topology=False)
    return list(simplified_poly.exterior.coords)[
        :-1
    ]  # Exclude last point (same as first)


def closest_side(polygon: list, reference_side: list):
    """Find the closest side in other polygons to a reference side in the first polygon.

    Args:
        polygon (list): List of polygons.
        reference_side (list): List of coordinates of the reference side.

    Returns:
        closest_side_points (list): List of coordinates of the closest side.

    """

    def angle(p1, p2):
        return atan2(p2[1] - p1[1], p2[0] - p1[0])

    ref_length = dist(*reference_side)
    ref_angle = angle(*reference_side)

    min_difference = float("inf")
    closest_side_points = None

    for i in range(len(polygon)):
        side_points = (polygon[i], polygon[(i + 1) % len(polygon)])
        side_length = dist(*side_points)
        side_angle = angle(*side_points)
        total_difference = abs(side_length - ref_length) + abs(side_angle - ref_angle)

        if total_difference < min_difference:
            min_difference = total_difference
            closest_side_points = list(side_points)

    return closest_side_points


# noinspection PyUnboundLocalVariable
def automatically_recognize_arena(
    coordinates: coordinates,
    tables: table_dict,
    videos: list,
    vid_index: int,
    path: str = ".",
    arena_type: str = "circular-autodetect",
    arena_reference: list = None,
    segmentation_model: torch.nn.Module = None,
    debug: bool = False,
) -> Tuple[np.array, int, int]:
    """Return numpy.ndarray with information about the arena recognised from the first frames of the video.

    WARNING: estimates won't be reliable if the camera moves along the video.

    Args:
        coordinates (coordinates): Coordinates object.
        tables (table_dict): Dictionary of tables per experiment.
        videos (list): Relative paths of the videos to analise.
        vid_index (int): Element of videos list to use.
        path (str): Full path of the directory where the videos are.
        potentially more accurate in poor lighting conditions.
        arena_type (string): Arena type; must be one of ['circular-autodetect', 'circular-manual', 'polygon-manual'].
        arena_reference (list): List of coordinates defining the reference arena annotated by the user.
        segmentation_model (torch.nn.Module): Model used for automatic arena detection.
        debug (bool): If True, save a video frame with the arena detected.

    Returns:
        arena (np.ndarray): 1D-array containing information about the arena. If the arena is circular, returns a 3-element-array) -> center, radius, and angle. If arena is polygonal, returns a list with x-y position of each of the n the vertices of the polygon.
        h (int): Height of the video in pixels.
        w (int): Width of the video in pixels.

    """
    # Read video as a 3D array
    current_video = imread(os.path.join(path, videos[vid_index]))
    h, w = current_video[0].shape[:2]

    # Select the corresponding tracklets
    current_tab = tables[
        get_close_matches(
            videos[vid_index].split(".")[0],
            [
                vid
                for vid in tables.keys()
                if (
                    vid.startswith(videos[vid_index].split(".")[0])
                    or videos[vid_index].startswith(vid)
                )
            ],
            cutoff=0.01,
            n=1,
        )[0]
    ]

    # Get distances of all body parts and timepoints to both center and periphery
    distances_to_center = cdist(
        current_tab.values.reshape(-1, 2), np.array([[w // 2, h // 2]])
    ).reshape(current_tab.shape[0], -1)

    possible_frames = np.nanmin(distances_to_center, axis=1) > np.nanpercentile(
        distances_to_center, 5.0
    )
    possible_distances_to_center = distances_to_center[possible_frames]
    current_video = current_video[: possible_frames.shape[0]][possible_frames]

    if arena_reference is not None:
        # If a reference is provided manually, avoid frames where the mouse is too close to the edges, which can
        # hinder segmentation
        min_distance_to_arena = cdist(
            current_tab.values.reshape(-1, 2), arena_reference
        ).reshape([distances_to_center.shape[0], -1, len(arena_reference)])

        min_distance_to_arena = min_distance_to_arena[possible_frames]
        current_frame = np.argmax(
            np.nanmin(np.nanmin(min_distance_to_arena, axis=1), axis=1)
        )

    else:
        # If not, use the maximum distance to the center as a proxy
        current_frame = np.argmin(np.nanmax(possible_distances_to_center, axis=1))

    frame = current_video[current_frame].compute()

    # Get mask using the segmentation model
    segmentation_model.set_image(frame)

    frame_masks, score, logits = segmentation_model.predict(
        point_coords=np.array([[w // 2, h // 2]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    # Get arenas for all retrieved masks, and select that whose area is the closest to the reference
    if arena_reference is not None:
        arenas = [
            arena_parameter_extraction(frame_mask, arena_type)
            for frame_mask in frame_masks
        ]
        arena = arenas[
            np.argmin(
                np.abs(
                    [Polygon(arena_reference).area - Polygon(a).area for a in arenas]
                )
            )
        ]
    else:
        arena = arena_parameter_extraction(frame_masks[np.argmax(score)], arena_type)

    if debug:

        # Save frame with mask and arena detected
        frame_with_arena = np.ascontiguousarray(frame.copy(), dtype=np.uint8)

        if "circular" in arena_type:
            cv2.ellipse(
                img=frame_with_arena,
                center=arena[0],
                axes=arena[1],
                angle=arena[2],
                startAngle=0.0,
                endAngle=360.0,
                color=(40, 86, 236),
                thickness=3,
            )

        elif "polygonal" in arena_type:

            cv2.polylines(
                img=frame_with_arena,
                pts=[arena],
                isClosed=True,
                color=(40, 86, 236),
                thickness=3,
            )

            # Plot scale references
            closest_side_points = closest_side(
                simplify_polygon(arena), arena_reference[:2]
            )

            for point in closest_side_points:
                cv2.circle(
                    frame_with_arena,
                    list(map(int, point)),
                    radius=10,
                    color=(40, 86, 236),
                    thickness=2,
                )

        cv2.imwrite(
            os.path.join(
                coordinates.project_path,
                coordinates.project_name,
                "Arena_detection",
                f"{videos[vid_index][:-4]}_arena_detection.png",
            ),
            frame_with_arena,
        )

    return arena, h, w


def retrieve_corners_from_image(
    frame: np.ndarray, arena_type: str, cur_vid: int, videos: list
):  # pragma: no cover
    """Open a window and waits for the user to click on all corners of the polygonal arena.

    The user should click on the corners in sequential order.

    Args:
        frame (np.ndarray): Frame to display.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygon-manual".
        cur_vid (int): Index of the current video in the list of videos.
        videos (list): List of videos to be processed.

    Returns:
        corners (np.ndarray): nx2 array containing the x-y coordinates of all n corners.

    """
    corners = []

    def click_on_corners(event, x, y, flags, param):
        # Callback function to store the coordinates of the clicked points
        nonlocal corners, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))

    # Resize frame to a standard size
    frame = frame.copy()

    # Create a window and display the image
    cv2.startWindowThread()

    while True:
        frame_copy = frame.copy()

        cv2.imshow(
            "deepof - Select polygonal arena corners - (q: exit / d: delete{}) - {}/{} processed".format(
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            ),
            frame_copy,
        )

        cv2.setMouseCallback(
            "deepof - Select polygonal arena corners - (q: exit / d: delete{}) - {}/{} processed".format(
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            ),
            click_on_corners,
        )

        # Display already selected corners
        if len(corners) > 0:
            for c, corner in enumerate(corners):
                cv2.circle(frame_copy, (corner[0], corner[1]), 4, (40, 86, 236), -1)
                # Display lines between the corners
                if len(corners) > 1 and c > 0:
                    if "polygonal" in arena_type or len(corners) < 5:
                        cv2.line(
                            frame_copy,
                            (corners[c - 1][0], corners[c - 1][1]),
                            (corners[c][0], corners[c][1]),
                            (40, 86, 236),
                            2,
                        )

        # Close the polygon
        if len(corners) > 2:
            if "polygonal" in arena_type or len(corners) < 5:
                cv2.line(
                    frame_copy,
                    (corners[0][0], corners[0][1]),
                    (corners[-1][0], corners[-1][1]),
                    (40, 86, 236),
                    2,
                )
        if len(corners) >= 5 and "circular" in arena_type:
            cv2.ellipse(
                frame_copy,
                *fit_ellipse_to_polygon(corners),
                startAngle=0,
                endAngle=360,
                color=(40, 86, 236),
                thickness=3,
            )

        cv2.imshow(
            "deepof - Select polygonal arena corners - (q: exit / d: delete{}) - {}/{} processed".format(
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            ),
            frame_copy,
        )

        # Remove last added coordinate if user presses 'd'
        if cv2.waitKey(1) & 0xFF == ord("d"):
            corners = corners[:-1]

        # Exit is user presses 'q'
        if len(corners) > 2:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Exit and copy all coordinates if user presses 'c'
        if cur_vid > 0 and cv2.waitKey(1) & 0xFF == ord("p"):
            corners = None
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Return the corners
    return corners


def extract_polygonal_arena_coordinates(
    video_path: str, arena_type: str, video_index: int, videos: list
):  # pragma: no cover
    """Read a random frame from the selected video, and opens an interactive GUI to let the user delineate the arena manually.

    Args:
        video_path (str): Path to the video file.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygonal-manual".
        video_index (int): Index of the current video in the list of videos.
        videos (list): List of videos to be processed.

    Returns:
        np.ndarray: nx2 array containing the x-y coordinates of all n corners of the polygonal arena.
        int: Height of the video.
        int: Width of the video.

    """
    current_video = imread(video_path)
    current_frame = np.random.choice(current_video.shape[0])

    # Get and return the corners of the arena
    try:
        import google.colab

        arena_corners = retrieve_corners_from_colab(
            current_video[current_frame].compute(),
            arena_type,
            video_index,
            videos,
        )

    except ImportError:
        arena_corners = retrieve_corners_from_image(
            current_video[current_frame].compute(),
            arena_type,
            video_index,
            videos,
        )
    return arena_corners, current_video.shape[2], current_video.shape[1]


def fit_ellipse_to_polygon(polygon: list):  # pragma: no cover
    """Fit an ellipse to the provided polygon.

    Args:
        polygon (list): List of (x,y) coordinates of the corners of the polygon.

    Returns:
        tuple: (x,y) coordinates of the center of the ellipse.
        tuple: (a,b) semi-major and semi-minor axes of the ellipse.
        float: Angle of the ellipse.

    """
    # Detect the main ellipse containing the arena
    ellipse_params = cv2.fitEllipse(np.array(polygon))

    # Parameters to return
    center_coordinates = tuple([int(i) for i in ellipse_params[0]])
    axes_length = tuple([int(i) // 2 for i in ellipse_params[1]])
    ellipse_angle = ellipse_params[2]

    return center_coordinates, axes_length, ellipse_angle


def arena_parameter_extraction(
    frame: np.ndarray,
    arena_type: str,
) -> np.array:
    """Return x,y position of the center, the lengths of the major and minor axes, and the angle of the recognised arena.

    Args:
        frame (np.ndarray): numpy.ndarray representing an individual frame of a video
        arena_type (str): Type of arena to be used. Must be either "circular" or "polygonal".

    """
    # Obtain contours from the image, and retain the largest one
    cnts, _ = cv2.findContours(
        frame.astype(np.int64), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    main_cnt = np.argmax([len(c) for c in cnts])

    if "circular" in arena_type:
        center_coordinates, axes_length, ellipse_angle = fit_ellipse_to_polygon(
            cnts[main_cnt]
        )
        return center_coordinates, axes_length, ellipse_angle

    elif "polygonal" in arena_type:
        return np.squeeze(cnts[main_cnt])


def rolling_speed(
    dframe: pd.DatetimeIndex,
    window: int = 3,
    rounds: int = 3,
    deriv: int = 1,
    center: str = None,
    shift: int = 2,
    typ: str = "coords",
) -> pd.DataFrame:
    """Return the average speed over n frames in pixels per frame.

    Args:
        dframe (pandas.DataFrame): Position over time dataframe.
        window (int): Number of frames to average over.
        rounds (int): Float rounding decimals.
        deriv (int): Position derivative order; 1 for speed, 2 for acceleration, 3 for jerk, etc.
        center (str): For internal usage only; solves an issue with pandas.MultiIndex that arises when centering frames to a specific body part.
        shift (int): Window shift for rolling speed calculation.
        typ (str): Type of dataset. Intended for internal usage only.

    Returns:
        speeds (pd.DataFrame): Data frame containing 2D speeds for each body part in the original data or their
        consequent derivatives.

    """
    original_shape = dframe.shape
    try:
        body_parts = dframe.columns.levels[0]
    except AttributeError:
        body_parts = dframe.columns

    speeds = pd.DataFrame

    for der in range(deriv):
        features = 2 if der == 0 and typ == "coords" else 1

        distances = (
            np.concatenate(
                [
                    np.array(dframe).reshape([-1, features], order="C"),
                    np.array(dframe.shift(shift)).reshape([-1, features], order="C"),
                ],
                axis=1,
            )
            / shift
        )

        distances = np.array(compute_dist(distances))
        distances = distances.reshape(
            [
                original_shape[0],
                (original_shape[1] // 2 if typ == "coords" else original_shape[1]),
            ],
            order="C",
        )
        distances = pd.DataFrame(distances, index=dframe.index)
        speeds = np.round(distances.rolling(window).mean(), rounds)
        dframe = speeds

    speeds.columns = body_parts

    return speeds.fillna(0.0)


def filter_short_bouts(
    cluster_assignments: np.ndarray,
    cluster_confidence: np.ndarray,
    confidence_indices: np.ndarray,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
):  # pragma: no cover
    """Filter out cluster assignment bouts shorter than min_bout_duration.

    Args:
        cluster_assignments (np.ndarray): Array of cluster assignments.
        cluster_confidence (np.ndarray): Array of cluster confidence values.
        confidence_indices (np.ndarray): Array of confidence indices.
        min_confidence (float): Minimum confidence value.
        min_bout_duration (int): Minimum bout duration in frames.

    Returns:
        np.ndarray: Mask of confidence indices to keep.

    """
    # Compute bout lengths, and filter out bouts shorter than min_bout_duration
    bout_lengths = np.diff(
        np.where(
            np.diff(np.concatenate([[np.inf], cluster_assignments, [np.inf]])) != 0
        )[0]
    )

    if min_bout_duration is None:
        min_bout_duration = np.mean(bout_lengths)

    confidence_indices[
        np.repeat(bout_lengths, bout_lengths) < min_bout_duration
    ] = False

    # Compute average confidence per bout
    cum_bout_lengths = np.concatenate([[0], np.cumsum(bout_lengths)])
    bout_average_confidence = np.array(
        [
            cluster_confidence[confidence_indices][
                cum_bout_lengths[i] : cum_bout_lengths[i + 1]
            ].mean()
            for i in range(len(bout_lengths))
        ]
    )

    return (np.repeat(bout_average_confidence, bout_lengths) >= min_confidence) & (
        confidence_indices
    )


# MACHINE LEARNING FUNCTIONS #


def gmm_compute(x: np.array, n_components: int, cv_type: str) -> list:
    """Fit a Gaussian Mixture Model to the provided data and returns evaluation metrics.

    Args:
        x (numpy.ndarray): Data matrix to train the model
        n_components (int): Number of Gaussian components to use
        cv_type (str): Covariance matrix type to use. Must be one of "spherical", "tied", "diag", "full".

    Returns:
        - gmm_eval (list): model and associated BIC for downstream selection.

    """
    gmm = mixture.GaussianMixture(
        n_components=n_components,
        covariance_type=cv_type,
        max_iter=100000,
        init_params="kmeans",
    )
    gmm.fit(x)
    gmm_eval = [gmm, gmm.bic(x)]

    return gmm_eval


def gmm_model_selection(
    x: pd.DataFrame,
    n_components_range: range,
    part_size: int,
    n_runs: int = 100,
    n_cores: int = False,
    cv_types: Tuple = ("spherical", "tied", "diag", "full"),
) -> Tuple[List[list], List[np.ndarray], Union[int, Any]]:
    """Run GMM clustering model selection on the specified X dataframe.

    Outputs the bic distribution per model, a vector with the median BICs and an object with the overall best model.

    Args:
        x (pandas.DataFrame): Data matrix to train the models
        n_components_range (range): Generator with numbers of components to evaluate
        n_runs (int): Number of bootstraps for each model
        part_size (int): Size of bootstrap samples for each model
        n_cores (int): Number of cores to use for computation
        cv_types (tuple): Covariance Matrices to try. All four available by default

    Returns:
        - bic (list): All recorded BIC values for all attempted parameter combinations (useful for plotting).
        - m_bic(list): All minimum BIC values recorded throughout the process (useful for plottinh).
        - best_bic_gmm (sklearn.GMM): Unfitted version of the best found model.

    """
    # Set the default of n_cores to the most efficient value
    if not n_cores:
        n_cores = min(multiprocessing.cpu_count(), n_runs)

    bic = []
    m_bic = []
    lowest_bic = np.inf
    best_bic_gmm = 0

    pbar = tqdm(total=len(cv_types) * len(n_components_range))

    for cv_type in cv_types:

        for n_components in n_components_range:

            res = Parallel(n_jobs=n_cores, prefer="threads")(
                delayed(gmm_compute)(
                    x.sample(part_size, replace=True), n_components, cv_type
                )
                for _ in range(n_runs)
            )
            bic.append([i[1] for i in res])

            pbar.update(1)
            m_bic.append(np.median([i[1] for i in res]))
            if m_bic[-1] < lowest_bic:
                lowest_bic = m_bic[-1]
                best_bic_gmm = res[0][0]

    return bic, m_bic, best_bic_gmm


# RESULT ANALYSIS FUNCTIONS #


def cluster_transition_matrix(
    cluster_sequence: np.array,
    nclusts: int,
    autocorrelation: bool = True,
    return_graph: bool = False,
) -> Tuple[Union[nx.Graph, Any], np.ndarray]:
    """Compute the transition matrix between clusters and the autocorrelation in the sequence.

    Args:
        cluster_sequence (numpy.array): Sequence of cluster assignments.
        nclusts (int): Number of clusters in the sequence.
        autocorrelation (bool): Whether to compute the autocorrelation of the sequence.
        return_graph (bool): Whether to return the transition matrix as an networkx.DiGraph object.

    Returns:
        trans_normed (numpy.ndarray / networkx.Graph): Transition matrix as numpy.ndarray or networkx.DiGraph.
        autocorr (numpy.array): If autocorrelation is True, returns a numpy.ndarray with all autocorrelation values on cluster assignment.
    """
    # Stores all possible transitions between clusters
    clusters = [str(i) for i in range(nclusts)]
    cluster_sequence = cluster_sequence.astype(str)

    trans = {t: 0 for t in product(clusters, clusters)}
    k = len(clusters)

    # Stores the cluster sequence as a string
    transtr = "".join(list(cluster_sequence))

    # Assigns to each transition the number of times it occurs in the sequence
    for t in trans.keys():
        trans[t] = len(re.findall("".join(t), transtr, overlapped=True))

    # Normalizes the counts to add up to 1 for each departing cluster
    trans_normed = np.zeros([k, k]) + 1e-5
    for t in trans.keys():
        trans_normed[int(t[0]), int(t[1])] = np.round(
            trans[t]
            / (sum({i: j for i, j in trans.items() if i[0] == t[0]}.values()) + 1e-5),
            3,
        )

    # If specified, returns the transition matrix as an nx.Graph object
    if return_graph:
        trans_normed = nx.Graph(trans_normed)

    if autocorrelation:
        cluster_sequence = list(map(int, cluster_sequence))
        autocorr = np.corrcoef(cluster_sequence[:-1], cluster_sequence[1:])
        return trans_normed, autocorr

    return trans_normed


def time_to_seconds(time_string: str) -> float:
    """Compute seconds as float based on a time string.

    Args:
        time_string (str): time string as input (format HH:MM:SS or HH:MM:SS.SSS...).

    Returns:
        seconds (float): time in seconds
    """
    seconds = None
    if re.match(r"^\b\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?$", time_string) is not None:
        time_array = np.array(re.findall(r"[-+]?\d*\.?\d+", time_string)).astype(float)
        seconds = 3600 * time_array[0] + 60 * time_array[1] + time_array[2]

    return seconds
