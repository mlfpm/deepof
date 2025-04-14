# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Functions and general utilities for the deepof package."""
import argparse
import copy
import math
import multiprocessing
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations, product
from math import atan2, dist
from typing import Any, List, NewType, Tuple, Union


import cv2
import h5py
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
import regex as re
import requests
import sleap_io as sio
import torch
from joblib import Parallel, delayed
from scipy.signal import savgol_filter, medfilt
from scipy.spatial.distance import cdist
from segment_anything import SamPredictor, sam_model_registry
from shapely.geometry import Point, Polygon
from sklearn import mixture
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm

from deepof.config import PROGRESS_BAR_FIXED_WIDTH
import deepof.data
from deepof.data_loading import get_dt, load_dt, save_dt, _suppress_warning



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# CONNECTIVITY AND GRAPH REPRESENTATIONS


@nb.njit
def rts_smoother_numba(measurements, F, H, Q, R):  # pragma: no cover
    """
    Implements the Rauch-Tung-Striebel (RTS) smoother for state estimation.

    This function performs both forward and backward passes to estimate the optimal state
    sequence given a set of noisy measurements. It first applies the Kalman filter in a
    forward pass and then refines the estimates using the RTS smoother in a backward pass.

    Args:
        measurements (np.ndarray): Array of measurements, shape (n_timesteps, n_dim_measurement).
        F (np.ndarray): State transition matrix, shape (n_dim_state, n_dim_state).
        H (np.ndarray): Observation matrix, shape (n_dim_measurement, n_dim_state).
        Q (np.ndarray): Process noise covariance matrix, shape (n_dim_state, n_dim_state).
        R (np.ndarray): Measurement noise covariance matrix, shape (n_dim_measurement, n_dim_measurement).

    Returns:
        smoothed_states (np.ndarray): Smoothed state estimates, shape (n_timesteps, n_dim_state).

    """
    n_timesteps, n_dim_measurement = measurements.shape
    n_dim_state = F.shape[0]

    # Ensure all inputs are float64
    measurements = measurements.astype(np.float64)
    F = F.astype(np.float64)
    H = H.astype(np.float64)
    Q = Q.astype(np.float64)
    R = R.astype(np.float64)

    # Forward pass (Kalman filter)
    filtered_states = np.zeros((n_timesteps, n_dim_state), dtype=np.float64)
    filtered_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state), dtype=np.float64
    )
    predicted_states = np.zeros((n_timesteps, n_dim_state), dtype=np.float64)
    predicted_covariances = np.zeros(
        (n_timesteps, n_dim_state, n_dim_state), dtype=np.float64
    )

    # Initialize
    filtered_states[0] = measurements[0]
    filtered_covariances[0] = (
        np.eye(n_dim_state, dtype=np.float64) * 1000
    )  # Large initial uncertainty

    for t in range(1, n_timesteps):
        # Predict
        predicted_states[t] = F @ filtered_states[t - 1]
        predicted_covariances[t] = F @ filtered_covariances[t - 1] @ F.T + Q

        # Update
        innovation = measurements[t] - H @ predicted_states[t]
        S = H @ predicted_covariances[t] @ H.T + R
        K = predicted_covariances[t] @ H.T @ np.linalg.inv(S)
        filtered_states[t] = predicted_states[t] + K @ innovation
        filtered_covariances[t] = (
            np.eye(n_dim_state, dtype=np.float64) - K @ H
        ) @ predicted_covariances[t]

    # Backward pass (RTS smoother)
    smoothed_states = np.zeros_like(filtered_states)
    smoothed_covariances = np.zeros_like(filtered_covariances)
    smoothed_states[-1] = filtered_states[-1]
    smoothed_covariances[-1] = filtered_covariances[-1]

    for t in range(n_timesteps - 2, -1, -1):
        C = filtered_covariances[t] @ F.T @ np.linalg.inv(predicted_covariances[t + 1])
        smoothed_states[t] = filtered_states[t] + C @ (
            smoothed_states[t + 1] - predicted_states[t + 1]
        )
        smoothed_covariances[t] = (
            filtered_covariances[t]
            + C @ (smoothed_covariances[t + 1] - predicted_covariances[t + 1]) @ C.T
        )

    return smoothed_states


@nb.njit
def enforce_skeleton_constraints_numba(
    data, skeleton_constraints, original_pos, tolerance=0.1, correction_factor=0.5
):  # pragma: no cover
    """
    Adjusts the positions of body parts in each frame to ensure that the distances between connected parts
    adhere to predefined skeleton constraints within a specified tolerance.

    Args:
        data (np.ndarray): Motion capture data, shape (n_frames, n_body_parts, 2).
        skeleton_constraints (list): List of tuples (part1, part2, dist) defining the
                                     constraints between body parts and their expected distances.
        original_pos (np.ndarray): Boolean array indicating original (non-interpolated) positions,
                                   shape (n_frames, n_body_parts, 2).
        tolerance (float): Allowable deviation from the constraint distance (default: 0.1).
        correction_factor (float): Factor to control the strength of position adjustments (default: 0.5).

    Returns:
        np.ndarray: Adjusted motion capture data with enforced skeleton constraints.

    """
    n_frames, _, _ = data.shape
    for frame in range(n_frames):

        if np.all(original_pos[frame, 
                                0]):
            continue  # Skip this frame

        for part1, part2, dist in skeleton_constraints:
            p1, p2 = data[frame, part1], data[frame, part2]
            current_dist = np.sqrt(np.sum((p1 - p2) ** 2))
            if current_dist > dist * (1 + tolerance) or current_dist < dist * (
                1 - tolerance
            ):
                correction = (
                    (current_dist - dist)
                    / (2 * current_dist + 0.00001)
                    * correction_factor
                )
                pm = (data[frame, part1] + data[frame, part2]) / 2
                if original_pos[frame, part1][0]:
                    data[frame, part2] += 2 * correction * (pm - p2)
                elif original_pos[frame, part2][0]:
                    data[frame, part1] += 2 * correction * (pm - p1)
                else:
                    data[frame, part1] += correction * (pm - p1)
                    data[frame, part2] += correction * (pm - p2)
    return data


class MouseTrackingImputer:
    """
    A class for imputing and processing mouse tracking data.

    This class provides methods for interpolating missing data points, enforcing skeleton
    constraints, and smoothing trajectories in mouse tracking experiments.

    Attributes:
        n_iterations (int): Number of iterations for imputation (default: 10).
        connectivity (object): Connectivity information for body parts.
        full_imputation (bool): Whether to perform full imputation or only a partial linear imputation (default: False).
        body_part_indices (OrderedDict): Mapping of body part names to indices.
        skeleton_constraints (list): List of skeleton constraints.
        mouse_body_estimation_samples (int): Number of sample frames with non-nan data to estimate valid mouse shapes (default: 100).
        lin_interp_limit (int): Limit for linear interpolation (default: 3).
    """

    def __init__(self, n_iterations=10, connectivity=None, full_imputation=False):
        self.full_imputation = full_imputation
        self.n_iterations = n_iterations
        self.connectivity = connectivity
        self.body_part_indices = None
        self.skeleton_constraints = None
        self.mouse_body_estimation_samples = 100
        self.lin_interp_limit = 3

    def _initialize_constraints(self, data):
        """
        Initializes the body part constraints based on a sample of frames with complete mouse data

        Args:
            data (pd.DataFrame): Input tracking data.

        Raises:
            ValueError: If no complete frames are found in the data.
        """
        # Map body part names to indices
        self.body_part_indices = OrderedDict()
        for i, col in enumerate(data.columns):
            body_part_name = col[0]
            if body_part_name != "Row":
                self.body_part_indices[body_part_name] = int(
                    (i - 1) / 2
                )  # workaround as "np.unique" changes sorting

        # Find frames that contain no nans
        complete_frames = []
        for i, row in data.iterrows():
            if not row.isna().any():
                complete_frames.append(row)

        if not complete_frames:
            raise ValueError(
                "No complete frames found in the data. Cannot initialize constraints."
            )

        # Sample a subset of complete frames
        total_frames = len(complete_frames)
        step = max(1, total_frames // self.mouse_body_estimation_samples)
        sampled_frames = [complete_frames[i] for i in range(0, total_frames, step)]

        # Generate skeleton constraints from average distance between sample of connected body parts
        self.skeleton_constraints = []
        for part1, connected_parts in self.connectivity.adj.items():
            for part2 in connected_parts:
                if part1 in self.body_part_indices and part2 in self.body_part_indices:
                    idx1, idx2 = (
                        self.body_part_indices[part1],
                        self.body_part_indices[part2],
                    )
                    dists = [
                        np.sqrt(
                            np.sum(
                                (
                                    np.array([row[part1]["x"], row[part1]["y"]])
                                    - np.array([row[part2]["x"], row[part2]["y"]])
                                )
                                ** 2
                            )
                        )
                        for row in sampled_frames
                    ]
                    self.skeleton_constraints.append((idx1, idx2, np.mean(dists)))

        assert len(self.skeleton_constraints) > 0, (
            " None of the table headers and mouse connectivity dict entries did match during constraint initialization.\n"
            " This usually happens if none or incorrect animal ids were given.\n"
            " Please check if you provided the correct animal_ids as input for the Project."
        )

    @_suppress_warning(
        ["A value is trying to be set on a copy of a slice from a DataFrame"]
    )
    def fit_transform(self, data):
        """
        Performs linear interpolation for small gaps and, if full_imputation is True
        applies a multi-step imputation process for larger gaps.

        Args:
            data (pd.DataFrame): Input tracking data.

        Returns:
            np.ndarray: Processed tracking data.
        """

        # interpolate small gaps linearily
        data.interpolate(
            method="linear",
            limit=self.lin_interp_limit,
            limit_direction="both",
            inplace=True,
        )

        # slow multi step imputation to also close larger gaps
        if self.full_imputation and any((np.isnan(data.iloc[:, :])).any()):
            if self.skeleton_constraints is None:
                self._initialize_constraints(data)

            # reshape to 3D numpy array for processing
            reshaped_data = data.values.reshape(len(data), -1, 2)

            # save non-missing position indices
            original_pos = ~np.isnan(reshaped_data)

            # get data rows with nans and neighboring rows as reference
            nan_frames = [
                (~original_pos[k, :]).any() for k in range(0, original_pos.shape[0])
            ]
            nan_frames = np.convolve(nan_frames, np.ones(15), mode="same") > 0
            data_snippets = reshaped_data[nan_frames]
            # print(f"{key} {np.sum(nan_frames)}")

            # complete data with iterative imputation
            completed_data = copy.copy(reshaped_data)
            if data_snippets.shape[0] > 50:
                completed_data[nan_frames] = self._iterative_imputation(data_snippets)
            else:
                completed_data = self._iterative_imputation(reshaped_data)
            completed_data[original_pos] = reshaped_data[original_pos]

            # smooth data
            smoothed_data = self._kalman_smoothing(completed_data)
            # fill back in original positions
            smoothed_data[original_pos] = reshaped_data[original_pos]

            # enforce skeleton constraints
            constrained_data = enforce_skeleton_constraints_numba(
                smoothed_data, self.skeleton_constraints, original_pos
            )

            return constrained_data.reshape(data.shape)
        else:
            return data

    def _kalman_smoothing(self, data):
        """
        Apply Kalman smoothing to the tracking data. Uses a Rauch-Tung-Striebel (RTS) smoother
        to smooth the trajectories of each body part coordinate.

        Args:
            data (np.ndarray): Input tracking data, shape (n_timesteps, n_body_parts, n_coords).

        Returns:
            np.ndarray: Smoothed tracking data.
        """
        _, n_body_parts, n_coords = data.shape

        # Define model parameters (you may need to adjust these)
        dt = 1.0  # time step
        F = np.array([[1, dt], [0, 1]])  # State transition matrix
        H = np.array([[1, 0]])  # Measurement matrix
        Q = (
            np.array([[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]]) * 0.01
        )  # Process noise covariance
        R = np.array([[0.1]])  # Measurement noise covariance

        smoothed_data = np.zeros_like(data)

        for bp in range(n_body_parts):
            for coord in range(n_coords):
                measurements = data[:, bp, coord].reshape(-1, 1)
                smoothed_states = rts_smoother_numba(measurements, F, H, Q, R)
                smoothed_data[:, bp, coord] = smoothed_states[:, 0]

        return smoothed_data

    @_suppress_warning(["Early stopping criterion not reached."])
    def _iterative_imputation(self, data):
        """
        Perform iterative imputation on the tracking data usingses scikit-learn's IterativeImputer
        to fill in missing values in the data.

        Args:
            data (np.ndarray): Input tracking data.

        Returns:
            np.ndarray: Imputed tracking data.
        """

        # reshape data for imputation
        scaler = StandardScaler()
        original_shape = data.shape
        to_impute = data.reshape(*data.shape[:-2], -1)

        # scale and impute
        imputed = IterativeImputer(
            skip_complete=True,
            max_iter=100,
            n_nearest_features=8,
            tol=1e-1,
        ).fit_transform(scaler.fit_transform(to_impute))

        # undo scaling
        imputed = scaler.inverse_transform(imputed)
        data = imputed.reshape(original_shape)
        return data


def connect_mouse(
    animal_ids=None, exclude_bodyparts: list = None, graph_preset: str = "deepof_14"
) -> nx.Graph:
    """Create a nx.Graph object with the connectivity of the bodyparts in the DLC topview model for a single mouse.

    Used later for angle computing, among others.

    Args:
        animal_ids (str): if more than one animal is tagged, specify the animal identyfier as a string.
        exclude_bodyparts (list): Remove the specified nodes from the graph.
        graph_preset (str): Connectivity preset to use. Currently supported: "deepof_14", "deepof_11"  and "deepof_8".

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
                "deepof_11": {
                    "Nose": ["Left_ear", "Right_ear"],
                    "Spine_1": ["Center", "Left_ear", "Right_ear"],
                    "Center": ["Left_fhip", "Right_fhip", "Spine_2"],
                    "Spine_2": ["Left_bhip", "Right_bhip", "Tail_base"],
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
        bool: If conversion is not possible, it raises an error

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
        animal_presence_mask, typ="animal_presence_mask", table_path=quality._table_path, animal_ids=quality._animal_ids
    )


def iterative_imputation(
    project: project, tab_dict: dict, lik_dict: dict, full_imputation: bool = False
):
    """Perform iterative imputation on occluded body parts. Run per animal and experiment.

    Args:
        project (project): Project object.
        tab_dict (dict): Dictionary with the coordinates of the body parts.
        lik_dict (dict): Dictionary with the likelihood of the tracking for each body part and animal.
        full_imputation (bool): Determines if only small gaps get linearily imputed (False) or additionally IterativeImputer and a few other steps are executed to close all gaps (True)

    Returns:
        tab_dict (dict): Dictionary with the coordinates of the body parts after imputation.

    """
    presence_masks = compute_animal_presence_mask(lik_dict)
    table_path=os.path.join(project.project_path, project.project_name, "Tables")
    tab_dict = deepof.data.TableDict(
        tab_dict, typ="coords", table_path=table_path, animal_ids=project.animal_ids
    )
    imputed_tabs = copy.deepcopy(tab_dict)

    for animal_id in project.animal_ids:

        for k, tab in tab_dict.filter_id(animal_id).items():

            try:

                # get table for current animal
                sub_table = tab.iloc[np.where(presence_masks[k][animal_id].values)[0]]
                # add row number info (twice as it makes things easier later when splitting in x and y)
                sub_table.insert(
                    0, ("Row", "x"), np.where(presence_masks[k][animal_id].values)[0]
                )
                sub_table.insert(
                    0, ("Row", "y"), np.where(presence_masks[k][animal_id].values)[0]
                )

                # impute missing values
                imputer = MouseTrackingImputer(
                    n_iterations=5,
                    connectivity=project.connectivity[animal_id],
                    full_imputation=full_imputation,
                )
                imputed = imputer.fit_transform(sub_table)

                # reshape back to original format and update values
                imputed = pd.DataFrame(
                    imputed,
                    index=sub_table.index,
                    columns=sub_table.columns,
                )
                imputed = imputed.drop(("Row", "x"), axis=1)
                imputed = imputed.drop(("Row", "y"), axis=1)
                imputed_tabs[k].update(imputed)

                if tab.shape[1] != imputed.shape[1]: # pragma: no cover
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
    try:
        if animal_ids is None:
            animal_ids = coordinates.animal_ids
        table_path=os.path.join(coordinates.project_path, coordinates.project_name, "Tables")
    except AttributeError:
        if animal_ids is None:
            animal_ids = coordinates._animal_ids
        table_path=os.path.join(coordinates._project_path, coordinates._project_name, "Tables")

    presence_masks = compute_animal_presence_mask(lik_dict)
    tab_dict = deepof.data.TableDict(tab_dict, typ="qc", table_path=table_path, animal_ids=animal_ids)

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
    if len(tab_.shape)==1:
        tab_ = tab_.reshape(1, -1)
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
    original_shape = cartesian_df.shape
    if type(cartesian_df.columns) == pd.MultiIndex:
        #"levels" seems to be bugged and still finds columns that are not included in the datframe anymore
        body_parts = [column[0] for column in cartesian_df.columns]
        body_parts = np.array(body_parts)[np.unique(body_parts, return_index=True)[1]]
    else:
        body_parts = cartesian_df.columns

    result = []
    for df in list(body_parts):
        result.append(bp2polar(cartesian_df[df]))
    result = pd.concat(result, axis=1)
    idx = pd.MultiIndex.from_product(
        [list(body_parts), ["rho", "phi"]]
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

    #calculate euclidean distance fast
    dist = np.sqrt(np.einsum("...i,...i", ab, ab))
    return pd.DataFrame(dist) 


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
    #restrict to valid range
    cosine_angle=np.clip(cosine_angle, -1, 1)

    ang = np.arccos(cosine_angle)

    return ang


def compute_areas(polygon_xy_stack: np.array) -> np.array:
    """Compute polygon areas for the provided stack of sets of data point-xy coordinates.

    Args:
        polygon_xy_stack: 3D numpy array [NPolygons (i.e. NFrames), Npoints, NDim (x,y)]

    Returns:
        areas (np.ndarray): areas for the provided xy coordinates.

    """

    # list of polygon areas, a list entry is set to np.nan if points forming the respective polygon are missing
    polygon_areas = np.array(
        [
            Polygon(polygon_xy_stack[i]).area
            if not np.isnan(polygon_xy_stack[i]).any()
            else np.nan
            for i in range(len(polygon_xy_stack))
        ]
    )

    return polygon_areas


@nb.njit(parallel=True)
def compute_areas_numba(polygon_xy_stack: np.array) -> np.array:  # pragma: no cover
    """
    Compute polygon areas for the provided stack of sets of data point-xy coordinates.

    Args:
        polygon_xy_stack (np.ndarray): 3D numpy array [NPolygons (i.e. NFrames), Npoints, NDim (x,y)]

    Returns:
        areas (np.ndarray): areas for the provided xy coordinates.

    """
    n_polygons, n_vertices, n_dims = polygon_xy_stack.shape
    polygon_areas = np.zeros(n_polygons, dtype=np.float64)

    for i in np.arange(n_polygons):
        polygon_areas[i] = polygon_area_numba(polygon_xy_stack[i])

    return polygon_areas


@nb.njit
def polygon_area_numba(vertices: np.ndarray) -> float:  # pragma: no cover
    """
    Calculate the area of a single polygon given its vertices.

    Args:
        vertices (np.ndarray): Array of shape [Npoints, 2] containing the (x, y) coordinates of the polygon's vertices.

    Returns:
        float: Area of the polygon.
    """
    n = len(vertices)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]

    area = abs(area) / 2

    return area


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


@nb.njit(parallel=True)
def rotate_all_numba(data: np.array, angles: np.array) -> np.array:  # pragma: no cover
    """Rotates Return a 2D numpy.ndarray with the initial values rotated by angles radians.

    Args:
        p (numpy.ndarray): 2D Array containing positions of bodyparts over time.
        angles (numpy.ndarray): Set of angles (in radians) to rotate p with.

    Returns:
        - rotated (numpy.ndarray): rotated positions over time

    """

    # initializations
    aligned_trajs = np.zeros(data.shape)
    new_shape = (data.shape[1] // 2, 2)
    rotated_frame = np.empty(new_shape, dtype=np.float64)
    reshaped_frame = np.empty(new_shape, dtype=np.float64)

    for frame in range(data.shape[0]):

        # reshape [x1,y1,x2,y2,...] to [[x1,y1],[x1,y2],...]
        for i in range(new_shape[0]):
            reshaped_frame[i, 0] = data[frame][2 * i]
            reshaped_frame[i, 1] = data[frame][2 * i + 1]

        # rotate frame
        rotated_frame = rotate_numba(reshaped_frame, angles[frame])

        # undo reshaping
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                aligned_trajs[frame][i * new_shape[1] + j] = rotated_frame[i, j]

    return aligned_trajs


@nb.njit
def rotate_numba(
    p: np.array, angles: np.array, origin: np.array = np.array([0, 0])
) -> np.array:  # pragma: no cover
    """Return a 2D numpy.ndarray with the initial values rotated by angles radians.

    Args:
        p (numpy.ndarray): 2D Array containing positions of bodyparts over time.
        angles (numpy.ndarray): Set of angles (in radians) to rotate p with.
        origin (numpy.ndarray): Rotation axis (zero vector by default).

    Returns:
        - rotated (numpy.ndarray): rotated positions over time

    """
    # initializations
    arr_shape = p.shape
    p_centered = np.zeros(arr_shape)
    rotated = np.empty(arr_shape, dtype=np.float64)

    # define rotation matrix
    R = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]])

    # ensure p is a 2D array
    if p.ndim <= 1:
        p = p.reshape(1, p.size)

    # substract origin
    for i in range(arr_shape[1]):
        for j in range(arr_shape[0]):
            p_centered[j][i] = p[j][i] - origin[i]
    # rotate matrix
    rotated_centered = (R @ p_centered.T).T
    # re-add origin
    for i in range(arr_shape[1]):
        for j in range(arr_shape[0]):
            rotated[j][i] = rotated_centered[j][i] + origin[i]

    return rotated


def point_in_polygon(points: np.array, polygon: Polygon) -> np.array:
    """
    Check if a set of points is inside a polygon.

    Args:
        points (np.ndarray): An array of shape (M, 2) containing the coordinates of the points.
        polygon (shapely.geometry.polygon.Polygon): Shapely polygon.

    Returns:
        np.ndarray: A boolean array of shape (M,) indicating whether each point is inside the polygon.
    """
    if type(polygon != Polygon):
        polygon=Polygon(polygon)
    inside = np.array([polygon.contains(Point(n)) for n in points])
    return inside


@nb.njit(parallel=True)
def point_in_polygon_numba(
    points: np.array, polygon: np.array
) -> np.array:  # pragma: no cover
    """
    This function was generated by Perplexity.ai
    Check if a set of points is inside a polygon.

    Args:
        points (np.ndarray): An array of shape (M, 2) containing the coordinates of the points.
        polygon (np.ndarray): An array of shape (N, 2) containing the coordinates of the polygon vertices.

    Returns:
        np.ndarray: A boolean array of shape (M,) indicating whether each point is inside the polygon.
    """
    M = points.shape[0]
    N = polygon.shape[0]
    inside = np.zeros(M, dtype=np.bool_)

    for i in nb.prange(M):
        x, y = points[i]
        inside[i] = _is_point_inside_numba(x, y, polygon)

    return inside


@nb.njit
def _is_point_inside_numba(
    x: float, y: float, polygon: np.array
) -> bool:  # pragma: no cover
    """
    This function was generated by Perplexity.ai
    Check if a point is inside a polygon using the ray casting algorithm.

    Args:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        polygon (np.ndarray): An array of shape (N, 2) containing the coordinates of the polygon vertices.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    N = polygon.shape[0]
    inside = False

    for i in range(N):
        j = (i + 1) % N
        x1, y1 = polygon[i]
        x2, y2 = polygon[j]

        if y > min(y1, y2) and y <= max(y1, y2) and x <= max(x1, x2):
            if y1 != y2:
                xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x1 == x2 or x <= xinters:
                inside = not inside

    return inside


def mouse_in_roi(tab, aid, in_roi_criterion, roi_polygon, run_numba: bool = False):
    """Checks if a given animal for a given table is in a given roi by given criterion.

    Args:
        tab (dataTable): Datatable containing mouse tracking data.
        aid (str): ainimal id of the mouse to check
        in_roi_criterion (str): Criterion for in roi check, checks by "Center" bodypart being inside or outside of roi by default   
        roi_polygon (np.ndarray): 2D numpy array containing the coordinats of the ROI
        run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)
    """

    points=np.array(tab[aid+"_"+in_roi_criterion])

    if run_numba:
        mouse_in_polygon=deepof.utils.point_in_polygon_numba(points,roi_polygon)
    else:
        mouse_in_polygon=deepof.utils.point_in_polygon(points,roi_polygon)

    return mouse_in_polygon


# noinspection PyArgumentList
def align_trajectories(
    data: np.array, mode: str = "all", run_numba: bool = False
) -> np.array:  # pragma: no cover
    """Remove rotational variance on the trajectories.

    Returns a numpy.array with the positions rotated in a way that the center (0 vector), and body part in the first
    column of data are aligned with the y-axis.

    Args:
        data (numpy.ndarray): 3D array containing positions of body parts over time, where shape is N (sliding window instances) * m (sliding window size) * l (features)
        mode (string): Specifies if *all* instances of each sliding window get aligned, or only the *center*
        run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)

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

    # run numba version for large videos
    if run_numba:
        aligned_trajs = rotate_all_numba(data, angles)
    else:
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
        animal_ids (list): List with the animal ids in case of multiple tracked animals. Is expected to be None if there is only a single animal getting tracked.
        
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
    offsets: list, s: float = 2.0, gamma: float = 1.0, n=None, T=None, k=None
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
        n: used to adjust the fixed cost function (not dependent of the given offsets). Which is needed if you want to compare bursts for different inputs.
        T: used to adjust the fixed cost function (not dependent of the given offsets). Which is needed if you want to compare bursts for different inputs.
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
    gaps = np.diff(offsets).astype(np.float64)

    if not np.all(gaps):
        raise ValueError("Input cannot contain events with zero time between!")

    if T is None:
        T = np.sum(gaps)

    if n is None:
        n = np.size(gaps)

    if k is None:
        # number of hidden states. Changed to be not higher than 3
        k = np.min(
            [
                6,
                int(
                    math.ceil(
                        float(
                            1
                            + (math.log(T) / math.log(s))
                            + (math.log(1.0 / np.amin(gaps)) / math.log(s))
                        )
                    )
                ),
            ]
        )

    # no run numba option here as this function gets called extremely often in the codeand is generally pretty slow
    # slow core part of kleinberg
    q = kleinberg_core_numba(
        gaps, np.float64(s), np.float64(gamma), int(n), np.float64(T), int(k)
    )

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


@nb.njit
def kleinberg_core_numba(
    gaps: np.array, s: np.float64, gamma: np.float64, n: int, T: np.float64, k: int
) -> np.array:  # pragma: no cover
    """Computation intensive core part of Kleinberg's algorithm (described in 'Bursty and Hierarchical Structure in Streams').

        The algorithm models activity bursts in a time series as an
        infinite hidden Markov model.

        Taken from pybursts (https://github.com/romain-fontugne/pybursts/blob/master/pybursts/pybursts.py)
        and rewritten for compatibility with numba.

        Args:
            gaps (np.array): an array of gap sizes between time offsets (numeric)
            s (float): the base of the exponential distribution that is used for modeling the event frequencies
            gamma (float): coefficient for the transition costs between states
            n: used to adjust the fixed cost function (not dependent of the given offsets). Which is needed if you want to compare bursts for different inputs.
            T: used to adjust the fixed cost function (not dependent of the given offsets). Which is needed if you want to compare bursts for different inputs.
            k: maximum burst level / number of hidden states
    :+
    """
    g_hat = T / n
    gamma_log_n = gamma * math.log(n)

    alpha = np.empty(k, dtype=np.float64)
    for x in range(k):
        alpha[x] = s**x / g_hat

    C = np.repeat(np.inf, k)
    C[0] = 0

    q = np.empty((k, 0))
    # iterate over all gap positions
    for t in range(gaps.shape[0]):
        C_prime = np.repeat(np.inf, k)
        q_prime = np.empty((k, t + 1))
        q_prime.fill(np.nan)

        # iterate over all hidden states
        for j in range(k):
            cost = np.empty(k, dtype=np.float64)

            # calculate cost for each new state
            for i in range(k):
                if i >= j:
                    cost[i] = C[i]
                else:
                    cost[i] = C[i] + (j - i) * gamma_log_n

            # state with minimum cost
            el = np.argmin(cost)

            # update Costs
            if (alpha[j] * math.exp(-alpha[j] * gaps[t])) > 0:
                C_prime[j] = cost[el] - math.log(
                    alpha[j] * math.exp(-alpha[j] * gaps[t])
                )

            # update state squence
            if t > 0:
                q_prime[j, :t] = q[el, :]

            # init next iteration of state sequence
            q_prime[j, t] = j + 1

        C = C_prime
        q = q_prime

    j = np.argmin(C)
    q = q[j, :]
    return q


def smooth_boolean_array(
    a: np.array, scale: int = 1, sigma = 2.0, batch_size: int = 50000
) -> np.array:
    """LEGACY FILTER FOR BEHAVIORAL ANALYSIS. REPLACED BY multi_step_paired_smoothing 
    Return a boolean array in which isolated appearances of a feature are smoothed.

        Args:
            a (numpy.ndarray): Boolean instances.
            scale (int): Kleinberg scale parameter. Higher values result in stricter smoothing.
            batch_size (int): Batch size for input processing
    
        Returns:
            a (numpy.ndarray): Smoothened boolean instances.

    """

    n = len(a)
    a_smooth = np.zeros(n, dtype=bool)  # Initialize the output vector

    # Process the input array in batches
    for start in range(0, n, batch_size // 2):
        end = min(start + batch_size, n)
        batch = a[start:end]

        # check if any behavior was detected
        offsets = np.where(batch)[0]
        if len(offsets) == 0:
            continue  # skip batch if tehre was no detected activity

        # Process the current batch
        batch_bursts = kleinberg(offsets, gamma=0.3, s=sigma)

        # Apply calculated smoothing to current batch
        a_smooth_batch = np.zeros(np.size(batch), dtype=bool)
        for i in batch_bursts:
            if i[0] == scale:
                a_smooth_batch[int(i[1]) : int(i[2])] = True

        # Update the output vector with the results of the current batch
        # Overwrite second half of last batch with new values to reduce "leakage"
        a_smooth[start:end] = a_smooth_batch

    return a_smooth


def multi_step_paired_smoothing(
        behavior_in: np.array,
        not_behavior: np.array = None,
        exclude: np.array =None,
        min_length: int =6,
        get_both: bool =False
        ) -> np.array:
    """This filtering approach will first gradually merge together very close behavioral instances (how close is regulated by min_length), 
    then filter out remaining short instances. In this way multiple instances close to each other are kept and united and isolated very 
    short bursts are filtered out. It replaces the kleinberg filtering approach with a similar idea but without the drawback of potentially 
    mashing multiple single and relatively far away behavior instances together into one big single behavior occurence, that is often mostly 
    false positive (as the merging distance is limted by min_length and not only by window length).

        Args:
            behavior_in (numpy.ndarray): Boolean instances of detected raw behavior.
            not_behavior (numpy.ndarray): Boolean instances of raw behavior not occuring.
            exclude (numpy.ndarray): Additional boolean instances that will always be rated as "no behavior".
            min_length (int): Determines the degree of smoothing. The smaller, the more short behavioral instances are kept and the sharper the behavioral edges remain.
            get_both (bool): If True, will also return the not_behavior instances that get smoothed along with the behavior instances.
    
        Returns:
            behavior (numpy.ndarray): Smoothened boolean instances.
            not_behavior (numpy.ndarray): Smoothened boolean not-behavior instances.

    """

    @nb.njit
    def _resolve_conflicts(behavior, not_behavior, behavior_avg, not_behavior_avg): # pragma: no cover
        """Determines if conflicting frames (behavior and not_behavior are True) are either one or the other
        based on the identity of surrounding frames represented by behavior_avg and not_behavior_avg"""
        n = len(behavior)
        for i in range(n):
            if behavior[i] and not_behavior[i]:
                if behavior_avg[i] >= not_behavior_avg[i]:
                    not_behavior[i] = False
                else:
                    behavior[i] = False
        return behavior, not_behavior
    
    if exclude is None:
        exclude=np.ones(len(behavior_in)).astype(np.bool_)
   
    if not_behavior is None:
        behavior = exclude & behavior_in.astype(np.bool_)
        not_behavior = exclude & ~(behavior_in.astype(np.bool_))
    else:
        behavior = behavior_in.astype(np.bool_)

    # Type corrections
    if type(behavior) == pd.core.series.Series:
        behavior = behavior.to_numpy()
    if type(not_behavior) == pd.core.series.Series:
        not_behavior = not_behavior.to_numpy()
    if type(exclude) == pd.core.series.Series:
        exclude = exclude.to_numpy()

    #widens all behavior detections
    behavior = moving_average(behavior, lag=min_length).astype(np.bool_)
    not_behavior = moving_average(not_behavior, lag=min_length).astype(np.bool_)

    # Due to the widening step, it will happen that frames are simutaneously beheavior and not behavior.
    # To resolve tehse conflicts, first run a larger moving average giving float values as a "percentage" of activeness / passiveness
    behavior_avg = moving_average(behavior, lag=min_length*4).astype(float)
    not_behavior_avg = moving_average(not_behavior, lag=min_length*4).astype(float)

    # Then resolve these conflicting frames based on their surrounding frames
    behavior, not_behavior = _resolve_conflicts(behavior, not_behavior, behavior_avg, not_behavior_avg) #merges close blcoks and narrows blocks
    
    # Re-apply exclude mask to re-sharpen edges
    behavior &= exclude
    not_behavior &= exclude
    
    # To ensure that sections are labeled more consistently as either behavior or not_behavior (with less not_behavior blips during behavior), 
    # use a moving median to get rid of very short not_behavior detections in behavior sections, then adjust not_behavior Frames accordingly
    behavior_med = binary_moving_median_numba(behavior.astype(np.float64), lag=min_length * 4 + 1) #widens blocks
    behavior = (behavior_med >= 0.5).astype(np.bool_)
    
    # Remove overlaps
    overlap = not_behavior & behavior
    not_behavior[overlap] = False
    
    # Filter short segments.
    behavior = filter_short_true_segments_numba(behavior, min_length)
    not_behavior = filter_short_true_segments_numba(not_behavior, min_length) #removes short blocks
    
    # Final exclude mask
    behavior &= exclude
    not_behavior &= exclude
    
    if get_both:
        return behavior, not_behavior
    else:
        return behavior
    

def rolling_window(
    a: np.ndarray,
    window_size: int,
    window_step: int,
) -> np.ndarray:
    """Return a 3D numpy.array with a sliding-window extra dimension.

    Args:
        a (np.ndarray): N (instances) * m (features) shape
        window_size (int): Size of the window to apply
        window_step (int): Step of the window to apply

    Returns:
        rolled_a (np.ndarray): N (sliding window instances) * l (sliding window size) * m (features)

    """

    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    rolled_a = np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=True
    )[::window_step]

    return rolled_a


def extract_windows(
    to_window: table_dict,
    window_size: int,
    window_step: int,
    save_as_paths: bool = False,
    shuffle: bool = False,
    windows_desc : str = "Get windows"
) -> np.ndarray:
    """Apply the rupture method independently to each experiment, and concatenate into a single dataset at the end.

    Returns a dataset and the rupture indices, adapted to be used in a concatenated version
    of the labels.

    Args:
        to_window (table_dict): table_dict with all experiments.
        window_size (int): specifies the length of the sliding window.
        window_step (int): specifies the stride of the sliding window.
        save_as_paths (bool): save result as paths in dictionary instead of keeping it in RAM
        shuffle (bool): Whether to shuffle the data for each dataset. Defaults to False.


    Returns:
        to_window (dict): Dictionary containing stacks of windowed data samples for each table. Shape of the stacks: [N_samples, window_size, N_features]
        output_shape (Tuple): shape of the output array (N_samples, window_size, N_features).

    """   
    # Iterate over all experiments and populate them
    out_len=0

    with tqdm(total=len(to_window.keys()), desc=f"{windows_desc:<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
        for key in to_window.keys():
                            
            #load tab from disk if not already loaded
            tab, tab_path = get_dt(to_window, key, True)  
            tab=np.array(tab)

            tab = rolling_window(
                tab,
                window_size,
                window_step,
            )
            if shuffle:
                shuffle_idcs = np.random.choice(
                    tab.shape[0], tab.shape[0], replace=False
                )
                tab = tab[shuffle_idcs]

            out_len=out_len+tab.shape[0]
            to_window[key] = save_dt(tab,tab_path,save_as_paths)
            pbar.update()


    output_shape=(out_len,tab.shape[1],tab.shape[2])
    return to_window, output_shape


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

    # savgol_filter cannot handle NaNs (i.e. it turns vast chuncks of neighboring frames
    # of nans to nans after processing). Hence this workaround.
    # get positions of nans in signal
    # nan_positions = np.isnan(series)

    # interpolate nans
    # interpolated_series = pd.DataFrame(series)
    # interpolated_series.interpolate(
    #    method="linear", limit_direction="both", inplace=True
    # )

    # apply filter
    smoothed_series = savgol_filter(
        series, polyorder=(w_length - alpha), window_length=w_length, axis=0
    )

    # re-add nans
    # smoothed_series[nan_positions]=np.nan

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

@nb.njit
def binary_moving_median_numba(time_series, lag): # pragma: no cover
    """will applay a moving emdian like filter on a binary signal, i.e. if a window of size lag 
    has more 1s than 0s set the frame to 1 for that window, set it to 0 otherwise. 
    Will only work for windows of uneven length N i.e. returns the same for lag=N and lag=N+1"""
    pad = (lag - 1) // 2
    padded = np.zeros(len(time_series), dtype=np.bool_)
    for i in range(pad,len(time_series)-pad):
        s = 0
        for k in time_series[i-pad:i+pad+1]:
            if k:
                s += 1
        padded[i] = s > pad

    return padded


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


def remove_outliers(
    experiment: pd.DataFrame,
    likelihood: pd.DataFrame,
    likelihood_tolerance: float,
    exclude: str = "",
    lag: int = 5,
    n_std: int = 3,
    mode: str = "or",
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

    Returns:
        interpolated_exp (pd.DataFrame): Interpolated version of experiment.

    """
    interpolated_exp = experiment.copy()

    # Creates a mask marking all outliers
    mask = full_outlier_mask(
        experiment, likelihood, likelihood_tolerance, exclude, lag, n_std, mode
    )
    warn_nans=False
    if np.sum(np.sum(mask))/np.prod(mask.shape) > 0.3:
        warn_nans=True


    interpolated_exp[mask] = np.nan

    return interpolated_exp, warn_nans


def filter_animal_id_in_table(table: pd.DataFrame, selected_id: str = None, table_type: str = None):
    """Filter a DataFrame to keep only those columns related to the selected id.

    Leave labels untouched if present.

    Args:
        table (pd.DataFrame): a dataFrame to be filtered
        selected_id (str): select a single animal on multi animal settings. Defaults to None (all animals are processed).
        table_type (str): type of the tableDict

    Returns:
        pd.DataFrame: Filtered dataFrame, keeping only the selected animal.
    """

    #filter columns, only keep the ones having a specific animal id    
    columns_to_keep = filter_columns(table.columns, selected_id, table_type=table_type)
    table = table.loc[
        :, [bpa for bpa in table.columns if bpa in columns_to_keep]
    ]

    return table


def filter_columns(columns: list, selected_id: str, table_type:str = None) -> list:
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
        if selected_id == "supervised" and column in [ #maybe bug but need to confirm
            "nose2nose",
            "sidebyside",
            "sidereside",
        ]:
            columns_to_keep.append(column)
        if type(column) == str and table_type=="supervised" and selected_id in column:
            columns_to_keep.append(column)
        elif type(column) == str and column.startswith(selected_id):
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
    """Loads model for automatic arena segmentation"""

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
    number_of_rois: int,
    segmentation_model_path: str,
    video_path: str,
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
        number_of_rois (int): number of behavior rois,
        segmentation_model_path (str): Path to segmentation model used for automatic arena detection.
        video_path (str): Path to folder with videos.
        videos (dict): Dictionary of videos to extract arena parameters from. Defaults to None (all videos are used).
        debug (bool): If True, a frame per video with the detected arena is saved. Defaults to False.
        test (bool): If True, the function is run in test mode. This means that instead of waiting for user-inputs fixed artifical user-inputs are used. Defaults to False.

    Returns:
        scales (dict): Dictionary of scaling information. Each scales object consists of:
            - x position of the center of arena in mm
            - y position of the center of the arena in mm
            - diameter of the arena (when circular) or length of first edge in pixels
            - diameter of the arena (when circular) or length of first edge in mm
        
        arena_params (dict): Dictionary of arena parameters. Each arena parameter object consists of:
            (when circular)
            - x position of the center of arena in pixel
            - y position of the center of the arena in pixel
            - x axis radii of the arena in pixel
            - y axis radii of the arena in pixel
            - angle of the elipse
            (when polygonal)
            - x and y positions of the polygon vertices in pixel

        video_resolution (dict): Dictionary of video resolutions. Each video resolution object consists of:
            - height of the video in pixel
            - width of teh video in pixel

        

    """
    scales = {}
    arena_params = {}
    roi_dicts = {}
    video_resolution = {}
    list_of_rois=list(range(1,number_of_rois+1))

    get_arena = True
    propagate_rois=[]
    

    def get_first_length(arena_corners):
        return math.dist(arena_corners[0], arena_corners[1])
    
    #set message for user
    if "polygon" in arena:

        multi_line_message = [
        "Note: The first line you draw will be used for scaling.",
        "This means that this line should correspond to the",
        "\"real world\" length of " + str(arena_dims) + " mm you set during",
        "project creation."
        ]
    else:
        multi_line_message = [
        "Note: The diameter of the arena will be used for scaling.",
        "This means that the diameter should correspond to the",
        "\"real world\" length of " + str(arena_dims) + " mm you set during",
        "project creation."
        ]

    if arena in ["polygonal-manual", "circular-manual"]:  # pragma: no cover

        display_message(multi_line_message)
                
        with tqdm(total=len(videos), desc=f"{'Detecting arenas':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="arena") as pbar:
            for vid_idx, key in enumerate(videos.keys()):
                  
                # let user draw arena
                arena_corners, roi_corners, h, w = extract_polygonal_arena_coordinates(
                    os.path.join(video_path, videos[key]),
                    arena,
                    vid_idx,
                    videos,
                    list_of_rois=list_of_rois,
                    get_arena=get_arena,
                    test=test,
                )

                if arena_corners is None:
                    get_arena = False
                else:
                    arena_dist=get_first_length(arena_corners)

                    cur_scales=[
                        *(np.mean(arena_corners, axis=0)*(arena_dims/arena_dist)),
                        arena_dist,
                        arena_dims,
                    ]


                for roi in copy.copy(list_of_rois):
                    if roi_corners[roi] is None:
                        propagate_rois.append(roi)
                        list_of_rois.remove(roi)
                
                cur_roi_corners={}
                for roi in list(range(1,number_of_rois+1)):
                    if roi in propagate_rois:
                        cur_roi_corners[roi]=roi_dicts[list(roi_dicts.keys())[-1]][roi]
                    else:
                        cur_roi_corners[roi]=roi_corners[roi]
                                   

                if get_arena:  
                    cur_arena_params = arena_corners
                else:
                    cur_arena_params = arena_params[list(arena_params.keys())[-1]]
                    cur_scales = scales[list(scales.keys())[-1]]

                if arena == "circular-manual":

                    if get_arena:
                        cur_arena_params = fit_ellipse_to_polygon(cur_arena_params)

                    arena_diameter=np.mean([cur_arena_params[1][0], cur_arena_params[1][1]])* 2

                    scales[key]=list(
                            np.array(
                                [
                                    cur_arena_params[0][0]*(arena_dims/arena_diameter),
                                    cur_arena_params[0][1]*(arena_dims/arena_diameter),
                                    arena_diameter
                                ]
                            )
                        ) + [arena_dims]
                        

                else:
                    scales[key]=cur_scales

                arena_params[key]=cur_arena_params
                roi_dicts[key]=cur_roi_corners
                video_resolution[key]=(h, w)
                pbar.update()

    elif arena in ["polygonal-autodetect", "circular-autodetect"]:

        # Open GUI for manual labelling of two scaling points in the first video
        arena_reference = None
        if not test:  # pragma: no cover

            #skip arena retival if circular-autodetect was selected
            get_arena=True
            if arena == "circular-autodetect" and number_of_rois==0:
                get_arena=False
            else:
                display_message(multi_line_message)

        # Early return in test mode to avoid redundant slow arena detection
        if test:
            get_arena=False
            if "polygonal" in arena:
                scales={'test2': [279.5, 213.5, 420.12, 380], 'test': [279.5, 213.5, 420.12, 380]}
                arena_params={'test2': ((108, 30), (539, 29), (533, 438), (104, 431)), 'test': ((108, 30), (539, 29), (533, 438), (104, 431))}
                video_resolution={'test2': (480, 640), 'test': (480, 640)}
                if test == "detect_arena":
                    arena_reference=np.array([(108, 30), (539, 29), (533, 438), (104, 431)])
                else:
                    return scales, arena_params, None, video_resolution
        
            elif "circular" in arena:
                scales={'test2': [300.0, 38.0, 252.0, 380], 'test': [300.0, 38.0, 252.0, 380]}
                arena_params={'test2': ((200, 195), (167, 169), 14.071887016296387), 'test': ((200, 195), (167, 169), 14.071887016296387)}
                video_resolution={'test2': (404, 416), 'test': (404, 416)}
                if test == "detect_arena":
                    pass
                else:
                    return scales, arena_params, None, video_resolution


        # Load SAM 
        segmentation_model = load_segmentation_model(segmentation_model_path)                           
        with tqdm(total=len(videos), desc=f"{'Detecting arenas':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="arena") as pbar:
            for vid_idx, key in enumerate(videos.keys()):
                
                if not test:
                    arena_corners, roi_corners, _ , _ = extract_polygonal_arena_coordinates(
                        os.path.join(video_path, videos[key]),
                        arena,
                        vid_idx,
                        videos,
                        list_of_rois=list_of_rois,
                        get_arena=get_arena,
                        test=test,
                    )
                    get_arena=False

                    for roi in copy.copy(list_of_rois):
                        if roi_corners[roi] is None:
                            propagate_rois.append(roi)
                            list_of_rois.remove(roi)
                
                    cur_roi_corners={}
                    for roi in list(range(1,number_of_rois+1)):
                        if roi in propagate_rois:
                            cur_roi_corners[roi]=roi_dicts[list(roi_dicts.keys())[-1]][roi]
                        else:
                            cur_roi_corners[roi]=roi_corners[roi]
                        
                    roi_dicts[key]=cur_roi_corners

                    if arena_corners is not None:
                        arena_reference = arena_corners

            for key in videos.keys():    
                
                arena_parameters, h, w = automatically_recognize_arena(
                    coordinates=coordinates,
                    tables=tables,
                    videos=videos,
                    vid_key=key,
                    path=video_path,
                    arena_type=arena,
                    arena_reference=arena_reference,
                    segmentation_model=segmentation_model,
                    debug=debug,
                )

                if "polygonal" in arena:

                    closest_side_points = closest_side(
                        simplify_polygon(arena_parameters), arena_reference[:2]
                    )

                    arena_dist=dist(*closest_side_points)

                    scales[key]=[
                            *(np.mean(arena_parameters, axis=0)*(arena_dims/arena_dist)),
                            arena_dist,
                            arena_dims,
                        ]
                    

                elif "circular" in arena:
                    # scales contains the coordinates of the center of the arena,
                    # the absolute diameter measured from the video in pixels, and
                    # the provided diameter in mm (1 -default- equals not provided)

                    arena_diameter=np.mean([arena_parameters[1][0], arena_parameters[1][1]])* 2

                    scales[key]=list(
                            np.array(
                                [
                                    arena_parameters[0][0]*(arena_dims/arena_diameter),
                                    arena_parameters[0][1]*(arena_dims/arena_diameter),
                                    arena_diameter,
                                ]
                            )
                        )+ [arena_dims]
                            
                arena_params[key]=arena_parameters
                video_resolution[key]=(h, w)
                pbar.update()

    elif not arena:
        return None, None, None, None

    else:  # pragma: no cover
        raise NotImplementedError(
            "arenas must be set to one of: 'polygonal-manual', 'polygonal-autodetect', 'circular-manual', 'circular-autodetect'"
        )
    
    #scale rois to mm
    for key in roi_dicts.keys():
        for k, roi in roi_dicts[key].items():
            scaling_ratio = scales[key][3]/scales[key][2]
            roi_dicts[key][k] = np.array(roi)*scaling_ratio

    return scales, arena_params, roi_dicts, video_resolution


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


@_suppress_warning(warn_messages=["All-NaN slice encountered"])
def automatically_recognize_arena(
    coordinates: coordinates,
    tables: table_dict,
    videos: dict,
    vid_key: str,
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
        vid_key (str): key of video to use.
        path (str): Full path of the directory where the videos are.
        arena_type (string): Arena type; must be one of ['circular-autodetect', 'circular-manual', 'polygon-manual'].
        arena_reference (list): List of coordinates defining the reference arena annotated by the user.
        segmentation_model (torch.nn.Module): Model used for automatic arena detection.
        debug (bool): If True, save a video frame with the arena detected.

    Returns:
        arena (np.ndarray): 1D-array containing information about the arena. If the arena is circular, returns a 3-element-array) -> center, radius, and angle. If arena is polygonal, returns a list with x-y position of each of the n the vertices of the polygon.
        h (int): Height of the video in pixels.
        w (int): Width of the video in pixels.

    """
    # create video capture object and read frame info
    current_video_cap = cv2.VideoCapture(os.path.join(path, videos[vid_key]))
    h = int(current_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(current_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Select the corresponding tracklets
    current_tab = get_dt(tables,vid_key)

    # Get distances of all body parts and timepoints to both center and periphery
    distances_to_center = cdist(
        current_tab.values.reshape(-1, 2), np.array([[w // 2, h // 2]])
    ).reshape(current_tab.shape[0], -1)

    # throws "All-NaN slice encountered" if in at least one frame no body parts could be detected
    possible_frames = np.nanmin(distances_to_center, axis=1) > np.nanpercentile(
        distances_to_center, 5.0
    )

    # save indices of valid frames, shorten distances vector
    possible_indices = np.where(possible_frames)[0]
    possible_distances_to_center = distances_to_center[possible_indices]

    if arena_reference is not None:
        # If a reference is provided manually, avoid frames where the mouse is too close to the edges, which can
        # hinder segmentation
        min_distance_to_arena = cdist(
            current_tab.values.reshape(-1, 2), arena_reference
        ).reshape([distances_to_center.shape[0], -1, len(arena_reference)])

        min_distance_to_arena = min_distance_to_arena[possible_indices]
        frame_index = np.argmax(
            np.nanmin(np.nanmin(min_distance_to_arena, axis=1), axis=1)
        )

    else:
        # If not, use the maximum distance to the center as a proxy
        frame_index = np.argmin(np.nanmax(possible_distances_to_center, axis=1))

    current_frame = possible_indices[frame_index]
    current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    reading_successful, numpy_im = current_video_cap.read()
    current_video_cap.release()

    # Get mask using the segmentation model
    segmentation_model.set_image(numpy_im)

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
        frame_with_arena = np.ascontiguousarray(numpy_im.copy(), dtype=np.uint8)

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
                f"{videos[vid_key][:-4]}_arena_detection.png",
            ),
            frame_with_arena,
        )

    return arena, h, w


def display_message(message: List[str]): # pragma: no cover
    """
    Opens a window that displays a message for the user

    Args:
        message: List of strings containing the message
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White color
    line_type = 2
    
    # Calculate dimensions based on message content
    max_line_length = max(len(line) for line in message)
    line_height = 30  # Height per line of text
    image_height = line_height * len(message) + 20  # Add some padding
    image_width = max(600, max_line_length * 12)  # Minimum width or based on longest line

    # Create a blank image with calculated dimensions
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Initial position for the first line of text
    x, y = 10, line_height
    
    # Loop through each line and put it on the image
    for line in message:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, line_type)
        y += line_height  # Move down for the next line

    window_name = "Arena scaling"
    
    # Display the image in a window
    cv2.imshow(window_name, image)
    
    try:
        # Wait for a key press or until the window is closed
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Exit on 'q' key press
                break
            
            # Check if window is still open
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except Exception as e:
        print(f"An error occurred: {e}")   # Handle window close exception gracefully

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(window_name)


def retrieve_corners_from_image(
    frame: np.ndarray, arena_type: str, cur_vid: int, videos: list, current_roi: int = 0, test: bool = False
):  # pragma: no cover
    """Open a window and waits for the user to click on all corners of the polygonal arena.

    The user should click on the corners in sequential order.

    Args:
        frame (np.ndarray): Frame to display.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygon-manual".
        cur_vid (int): Index of the current video in the list of videos.
        videos (list): List of videos to be processed.
        current_roi (int): Current ROI to be extracted. 0 is the global arena ROI
        test (bool): Runs project in test mode and bypasses manual inputs, defaults to false

    Returns:
        corners (np.ndarray): nx2 array containing the x-y coordinates of all n corners.

    """
    corners = []

    #hard coded as these colors seem to be defined at compilation time, otehrwise open cv crashes
    roi_colors=[(204,  20,  20),
       (204, 131,  20),
       (167, 204,  20),
       ( 57, 204,  20),
       ( 20, 204,  94),
       ( 20, 204, 204),
       ( 20,  94, 204),
       ( 57,  20, 204),
       (167,  20, 204),
       (204,  20, 131),
       (153,  15,  15),
       (153,  98,  15),
       (125, 153,  15),
       (43, 153,  15),
       (15, 153,  70),
       (15, 153, 153),
       (15,  70, 153),
       (43,  15, 153),
       (125,  15, 153),
       (153,  15,  98)]

    #early return of set of square corners
    if test:
        return [(111, 49), (541, 31), (553, 438), (126, 452)]
    
    if current_roi == 0:
        display_text="deepof - Select polygonal arena corners - (q: exit / d: delete{}) - {}/{} processed".format(
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            )
        color = (40, 86, 236)

    # current roi > 0
    elif current_roi<21:
        display_text="deepof - Select polygonal region of interest corners for roi {} - (q: exit / d: delete{}) - {}/{} processed".format(
                current_roi,
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            )
        color = roi_colors[current_roi-1]
    else:
        raise ValueError(
            "only up to 20 ROIs are allowed (what for do you even need so many ROIs?)"
        )


    def click_on_corners(event, x, y, flags, param):
        # Callback function to store the coordinates of the clicked points
        nonlocal corners, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))

    # Resize frame to a standard size
    frame = frame.copy()

    # Create a window and display the image
    cv2.startWindowThread()

    alpha=0.3
    while True:
        frame_copy = frame.copy()

        cv2.imshow(
            display_text,
            frame_copy,
        )

        cv2.setMouseCallback(
            display_text,
            click_on_corners,
        )

        # Display already selected corners
        if len(corners) > 0:
            for c, corner in enumerate(corners):
                cv2.circle(frame_copy, (corner[0], corner[1]), 4, color, -1)
                # Display lines between the corners
                if len(corners) > 1 and c > 0:
                    if "polygonal" in arena_type or len(corners) < 5:
                        cv2.line(
                            frame_copy,
                            (corners[c - 1][0], corners[c - 1][1]),
                            (corners[c][0], corners[c][1]),
                            color,
                            2,
                        )

        # Close the polygon
        if len(corners) > 2:
            if "polygonal" in arena_type or len(corners) < 5:
                cv2.line(
                    frame_copy,
                    (corners[0][0], corners[0][1]),
                    (corners[-1][0], corners[-1][1]),
                    color,
                    2,
                )

        if len(corners) >= 5 and "circular" in arena_type:
            cv2.ellipse(
                frame_copy,
                *fit_ellipse_to_polygon(corners),
                startAngle=0,
                endAngle=360,
                color=color,
                thickness=3,
            )
        
        # Create filled overlay for ROIs (these are always polygonal)
        if current_roi > 0 and len(corners) > 2:         
            overlay = frame_copy.copy()
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)           
            # Blend overlay with original frame
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

        cv2.imshow(
            display_text,
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
    video_path: str, arena_type: str, video_index: int, videos: list, list_of_rois: list = 0, get_arena: bool = True, test: bool = False, 
):  # pragma: no cover
    """Read a random frame from the selected video, and opens an interactive GUI to let the user delineate the arena manually.

    Args:
        video_path (str): Path to the video file.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygonal-manual".
        video_index (int): Index of the current video in the list of videos.
        videos (list): List of videos to be processed.
        list_of_rois (int): list of roi numbers to draw,
        get_arena (bool): retrieve arena or skip step (default is True)
        test (bool): Runs project in test mode and bypasses manual inputs, defaults to false


    Returns:
        arena_corners (np.ndarray): nx2 array containing the x-y coordinates of all n corners of the polygonal arena.
        int: Height of the video.
        int: Width of the video.

    """

    # read random frame from video capture object
    current_video_cap = cv2.VideoCapture(video_path)
    total_frames = int(current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_number = np.random.choice(total_frames)
    current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    _, numpy_im = current_video_cap.read()
    current_video_cap.release()

    arena_corners = None
    roi_corners = None

    # open gui and let user pick corners
    if get_arena:
        arena_corners = retrieve_corners_from_image(
            numpy_im,
            arena_type,
            video_index,
            videos,
            current_roi=0,
            test=test,
        )

    #let user pick corners for rois
    if len(list_of_rois) > 0:

        roi_corners= {}

        for k in list_of_rois:
            cur_roi_corners = retrieve_corners_from_image(
                numpy_im,
                "polygonal-manual",
                video_index,
                videos,
                current_roi=k,
                test=test,
            )
            roi_corners[k] =cur_roi_corners

    return arena_corners, roi_corners, numpy_im.shape[0], numpy_im.shape[1]


def fit_ellipse_to_polygon(polygon: list):  # pragma: no cover
    """Fit an ellipse to the provided polygon.

    Args:
        polygon (list): List of (x,y) coordinates of the corners of the polygon.

    Returns:
        center_coordinates (tuple): (x,y) coordinates of the center of the ellipse.
        axes_length (tuple): (a,b) semi-major and semi-minor axes of the ellipse.
        ellipse_angle (float): Angle of the ellipse.

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

    Returns:
        IF arena_type=="circular":
        center_coordinates (tuple): (x,y) coordinates of the center of the ellipse.
        axes_length (tuple): (a,b) semi-major and semi-minor axes of the ellipse.
        ellipse_angle (float): Angle of the ellipse.
        ELIF arena_type=="polygonal"        
        np.ndarray: (x,y) coordinates of all points of the polygon

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
    frame_rate: int = 1,
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
        frame_rate (int): Number of frames per second.

        window (int): Number of frames to average over.
        rounds (int): Float rounding decimals.
        deriv (int): Position derivative order; 1 for speed, 2 for acceleration, 3 for jerk, etc.
        center (str): For internal usage only; solves an issue with pandas.MultiIndex that arises when centering frames to a specific body part.
        shift (int): Window shift for rolling speed calculation.
        typ (str): Type of dataset. Intended for internal usage only.

    Returns:
        speeds (pd.DataFrame): Data frame containing 2D speeds for each body part in the original data or their consequent derivatives.

    """
    original_shape = dframe.shape
    if type(dframe.columns) == pd.MultiIndex:
        #"levels" seems to be bugged and still finds columns that are not included in the datframe anymore
        body_parts = [column[0] for column in dframe.columns]
        body_parts = np.array(body_parts)[np.unique(body_parts, return_index=True)[1]]
    else:
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
        del dframe
        dframe = speeds

    # Speed is in mm per frame
    speeds.columns = body_parts

    # Convert to mm per second
    speeds *= frame_rate



    return speeds#.fillna(0.0)


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
            cluster_confidence[cum_bout_lengths[i] : cum_bout_lengths[i + 1]].mean()
            if np.any(confidence_indices[cum_bout_lengths[i] : cum_bout_lengths[i + 1]])
            else float("nan")
            for i in range(len(bout_lengths))
        ]
    )

    return (np.repeat(bout_average_confidence, bout_lengths) >= min_confidence) & (
        confidence_indices
    )


def filter_short_true_segments(array: np.ndarray, min_length: int):
    """Filters out sahort "True" sections from boolean array "array"

    Args:
        array (np.ndarray): Boolean array
        min_length (int): Minimum length of "true" sections within array.

    Returns:
        np.ndarray: Mask of confidence indices to keep.

    """
    
    #inits 
    n = len(array)
    output_array = np.zeros(n, dtype=np.bool_)
    count = 0
    in_segment = False
    
    for i in range(n):
        if array[i]:
            count += 1
            in_segment = True
        else:
            if in_segment:
                # Check count if True-segment ends
                if count >= min_length:
                    output_array[i - count:i] = True
                # Reset count and segment flag
                count = 0
                in_segment = False
    
    # Check for a segment that may end at the last element
    if in_segment and count >= min_length:
        output_array[n - count:n] = True
    
    return output_array


@nb.njit
def filter_short_true_segments_numba(array: np.ndarray, min_length: int): # pragma: no cover
    """Filters out sahort "True" sections from boolean array "array"

    Args:
        array (np.ndarray): Boolean array
        min_length (int): Minimum length of "true" sections within array.

    Returns:
        np.ndarray: Mask of confidence indices to keep.

    """
    
    #inits 
    n = len(array)
    output_array = np.zeros(n, dtype=np.bool_)
    count = 0
    in_segment = False
    
    for i in range(n):
        if array[i]:
            count += 1
            in_segment = True
        else:
            if in_segment:
                # Check count if True-segment ends
                if count >= min_length:
                    output_array[i - count:i] = True
                # Reset count and segment flag
                count = 0
                in_segment = False
    
    # Check for a segment that may end at the last element
    if in_segment and count >= min_length:
        output_array[n - count:n] = True
    
    return output_array

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
        part_size (int): Size of bootstrap samples for each model
        n_runs (int): Number of bootstraps for each model
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


def get_total_Frames(video_paths: dict) -> int:
    """Get the number of all frames in all videos listed in the input dictionary

    Args:
        video_paths (dict): Paths to all videos in a dicitonary

    Returns:
        total_frames (int): Total number of all video frames
    """

    total_frames = []
    for _, video_path in video_paths.items():
        current_video_cap = cv2.VideoCapture(video_path)
        total_frames.append(int(current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        current_video_cap.release()
    return total_frames
