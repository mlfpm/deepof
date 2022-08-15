# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for the deepof package.

"""

import argparse
import multiprocessing
import os
from copy import deepcopy
from itertools import combinations, product
from typing import Tuple, Any, List, Union, NewType

import cv2
import math
import networkx as nx
import numpy as np
import pandas as pd
import regex as re
import ruptures as rpt
import tensorflow as tf
from dask_image.imread import imread
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from shapely.geometry import Polygon
from sklearn import mixture
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# CONNECTIVITY FOR DLC MODELS


def connect_mouse_topview(animal_id=None) -> nx.Graph:
    """

    Creates a nx.Graph object with the connectivity of the bodyparts in the
    DLC topview model for a single mouse. Used later for angle computing, among others.

    Args:
        animal_id (str): if more than one animal is tagged, specify the animal identyfier as a string.

    Returns:
        connectivity (nx.Graph)

    """

    connectivity = {
        "Nose": ["Left_ear", "Right_ear", "Spine_1"],
        "Left_ear": ["Right_ear", "Spine_1"],
        "Right_ear": ["Spine_1"],
        "Spine_1": ["Center", "Left_fhip", "Right_fhip"],
        "Center": ["Left_fhip", "Right_fhip", "Spine_2", "Left_bhip", "Right_bhip"],
        "Spine_2": ["Left_bhip", "Right_bhip", "Tail_base"],
        "Tail_base": ["Tail_1", "Left_bhip", "Right_bhip"],
        "Tail_1": ["Tail_2"],
        "Tail_2": ["Tail_tip"],
    }

    connectivity = nx.Graph(connectivity)

    if animal_id:
        mapping = {
            node: "{}_{}".format(animal_id, node) for node in connectivity.nodes()
        }
        nx.relabel_nodes(connectivity, mapping, copy=False)

    return connectivity


# QUALITY CONTROL AND PREPROCESSING #


def str2bool(v: str) -> bool:
    """

    Returns the passed string as a boolean.

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


def likelihood_qc(dframe: pd.DataFrame, threshold: float = 0.9) -> np.array:
    """

    Returns a DataFrame filtered dataframe, keeping only the rows entirely above the threshold.

    Args:
        dframe (pandas.DataFrame): DeepLabCut output, with positions over time and associated likelihood.
        threshold (float): Minimum acceptable confidence.

    Returns:
        filt_mask (np.array): mask on the rows of dframe

    """

    Likes = np.array([dframe[i]["likelihood"] for i in list(dframe.columns.levels[0])])
    Likes = np.nan_to_num(Likes, nan=1.0)
    filt_mask = np.all(Likes > threshold, axis=0)

    return filt_mask


def bp2polar(tab: pd.DataFrame) -> pd.DataFrame:
    """

    Returns the DataFrame in polar coordinates.

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
    """

    Returns a pandas.DataFrame in which all the coordinates are polar.

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
    """

    Returns a pandas.DataFrame with the scaled distances between a pair of body parts.

    Args:
        pair_array (numpy.array): np.array of shape N * 4 containing X, y positions.
        over time for a given pair of body parts.
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
    """

    Returns a pandas.DataFrame with the scaled distances between all pairs of body parts.

    Args:
        dataframe (pandas.DataFrame): pd.DataFrame of shape N*(2*bp) containing X,y positions
        over time for a given set of bp body parts.
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


def angle(a: np.array, b: np.array, c: np.array) -> np.array:
    """

    Returns a numpy.array with the angles between the provided instances.

    Args:
        a (np.array): 2D positions over time for a body part.
        b (np.array): 2D positions over time for a body part.
        c (np.array): 2D positions over time for a body part.

    Returns:
        ang (np.array): 1D angles between the three-point-instances.

    """

    ba = a - b
    bc = c - b

    cosine_angle = np.einsum("...i,...i", ba, bc) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    ang = np.arccos(cosine_angle)

    return ang


def angle_trio(bpart_array: np.array) -> np.array:
    """

    Returns a numpy.array with all three possible angles between the provided instances.

    Args:
        bpart_array (numpy.array): 2D positions over time for a bodypart.

    Returns:
        ang_trio (numpy.array): 2D all-three angles between the three-point-instances.

    """

    a, b, c = bpart_array
    ang_trio = np.array([angle(a, b, c), angle(a, c, b), angle(b, a, c)])

    return ang_trio


def compute_areas(coords, animal_id=None):
    """
    Computes relevant areas (head, torso, back, full) for the provided coordinates.
    Args:
        coords: coordinates of the body parts for a single time point.
        animal_id: animal id for the provided coordinates, if any.

    Returns:
        areas: list including head, torso, back, and full areas for the provided coordinates.

    """

    head = ["Nose", "Left_ear", "Left_fhip", "Spine_1"]

    torso = ["Spine_1", "Right_fhip", "Spine_2", "Left_fhip"]

    back = ["Spine_1", "Right_bhip", "Spine_2", "Left_bhip"]

    full = [
        "Nose",
        "Left_ear",
        "Left_fhip",
        "Left_bhip",
        "Tail_base",
        "Right_bhip",
        "Right_fhip",
        "Right_ear",
    ]

    areas = []

    for bps in [head, torso, back, full]:

        if animal_id is not None:
            bps = ["_".join([animal_id, bp]) for bp in bps]

        x = coords.xs(key="x", level=1)[bps]
        y = coords.xs(key="y", level=1)[bps]

        areas.append(Polygon(zip(x, y)).area)

    return areas


def rotate(
    p: np.array, angles: np.array, origin: np.array = np.array([0, 0])
) -> np.array:
    """

    Returns a 2D numpy.ndarray with the initial values rotated by angles radians.

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
    """

    Returns a numpy.array with the positions rotated in a way that the center (0 vector)
    and the body part in the first column of data are aligned with the y axis.

    Args:
        data (numpy.ndarray): 3D array containing positions of body parts over time, where
        shape is N (sliding window instances) * m (sliding window size) * l (features)
        mode (string): Specifies if *all* instances of each sliding window get
        aligned, or only the *center*

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


def kleinberg(offsets, s=np.e, gamma=1.0, n=None, T=None, k=None):
    """Kleinberg's algorithm (described in 'Bursty and Hierarchical Structure
    in Streams'). The algorithm models activity bursts in a time series as an
    infinite hidden Markov model.

    Taken from pybursts (https://github.com/romain-fontugne/pybursts/blob/master/pybursts/pybursts.py)
    and adapted for dependency compatibility reasons.

    Input:
        offsets: a list of time offsets (numeric)
        s: the base of the exponential distribution that is used for modeling
        the event frequencies
        gamma: coefficient for the transition costs between states
        n, T: to have a fixed cost function (not dependent of the given offsets).
        Which is needed if you want to compare bursts for different inputs.
        k: maximum burst level"""

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

    alpha_function = np.vectorize(lambda x: s ** x / g_hat)
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
    """

    Returns a boolean array in which isolated appearances of a feature are smoothed.

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
    a: np.array,
    window_size: int,
    window_step: int,
    automatic_changepoints: str = False,
    precomputed_breaks: np.ndarray = None,
) -> np.ndarray:
    """

    Returns a 3D numpy.array with a sliding-window extra dimension.

    Args:
        a (np.ndarray): N (instances) * m (features) shape
        window_size (int): Size of the window to apply
        window_step (int): Step of the window to apply
        automatic_changepoints (str): Changepoint detection algorithm to apply.
        If False, applies a fixed sliding window.
        precomputed_breaks (np.ndarray): Precomputed breaks to use, bypassing the changepoint detection algorithm.
        None by default (break points are computed).

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
    """

    Apply the rupture method independently to each experiment, and concatenate into a single dataset
    at the end. Returns a dataset and the rupture indices, adapted to be used in a concatenated version
    of the labels.

    Args:
        table_dict (deepof.data.table_dict): table_dict with all experiments.
        to_rupture (np.ndarray): Array with dataset to rupture.
        rupture_indices (list): Indices of tables to rupture. Useful to select training and test sets.
        automatic_changepoints (str): Rupture method to apply.
        If false, a sliding window of window_length * window_size is obtained.
        If one of "l1", "l2" or "rbf", different automatic change point detection algorithms are applied
        on each independent experiment.
        window_size (int): If automatic_changepoints is False, specifies the length of the sliding window.
        If not, it determines the minimum size of the obtained time series breaks.
        window_step (int): If automatic_changepoints is False, specifies the stride of the sliding window.
        If not, it determines the minimum step size of the obtained time series breaks.
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
                (
                    to_rupture[cumulative_shape : cumulative_shape + current_size]
                    if automatic_changepoints
                    else to_rupture
                ),
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
                if automatic_changepoints:
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
    """

    Returns a smoothed a trajectory using a Savitzky-Golay 1D filter.

    Args:
        series (numpy.ndarray): 1D trajectory array with N (instances)
        alpha (int): 0 <= alpha < w_length; indicates the difference between the degree of the polynomial and the window
        length for the Savitzky-Golay filter used for smoothing. Higher values produce a worse fit, hence more smoothing.
        w_length (int): Length of the sliding window to which the filter fit. Higher values yield a coarser fit,
        hence more smoothing.

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
    """

    Fast implementation of a moving average function.

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
    """

    Returns a mask over the bivariate trajectory of a body part, identifying as True all detected outliers.
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
    """

    Iterates over all body parts of experiment, and outputs a dataframe where all x, y positions are
    replaced by a boolean mask, where True indicates an outlier.

    Args:
        experiment (pd.DataFrame): Data frame with time series representing the x, y positions of a every body part
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
    """

    Marks all outliers in experiment and replaces them using a uni-variate linear interpolation approach.
    Note that this approach only works for equally spaced data (constant camera acquisition rates).

    Args:
        experiment (pd.DataFrame): Data frame with time series representing the x, y positions of a every body part.
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
    """

    Given a set of TableDict columns, returns those that correspond to a given animal, specified in selected_id.

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


# noinspection PyUnboundLocalVariable
def automatically_recognize_arena(
    videos: list,
    vid_index: int,
    path: str = ".",
    tables: dict = None,
    recoglimit: int = 500,
    high_fidelity: bool = False,
    arena_type: str = "circular-autodetect",
    detection_mode: str = "rule-based",
    cnn_model: tf.keras.models.Model = None,
) -> Tuple[np.array, int, int]:
    """

    Returns numpy.array with information about the arena recognised from the first frames
    of the video. WARNING: estimates won't be reliable if the camera moves along the video.

    Args:
        videos (list): Relative paths of the videos to analise.
        vid_index (int): Element of videos list to use.
        path (str): Full path of the directory where the videos are.
        tables (dict): Dictionary with DLC time series in DataFrames as values.
        recoglimit (int): Number of frames to use for position estimates.
        high_fidelity (bool): If True, runs arena recognition on the whole video. Slow, but
        potentially more accurate in poor lighting conditions.
        arena_type (string): Arena type; must be one of ['circular-autodetect', 'circular-manual', 'polygon-manual'].
        detection_mode (str): Algorithm to use to detect the arena. "cnn" uses a
        pretrained model based on ResNet50 to predict the ellipse parameters from
        the image. "rule-based" (default) uses a simpler (and faster) image segmentation approach.
        cnn_model (tf.keras.models.Model): Model to use if detection_mode=="cnn".

    Returns:
        arena (np.ndarray): 1D-array containing information about the arena.
        "circular-autodetect" (3-element-array) -> x-y position of the center and the radius.
        "circular-manual" (3-element-array) -> x-y position of the center and the radius.
        "polygon-manual" (2n-element-array) -> x-y position of each of the n the vertices of the polygon.
        h (int): Height of the video in pixels.
        w (int): Width of the video in pixels.

    """

    cap = cv2.VideoCapture(os.path.join(path, videos[vid_index]))

    if high_fidelity:
        recoglimit = int(1e10)  # set recoglimit to a very big value

    if tables is not None:
        # Select relevant table to check animal positions; if animals are close to the arena, do not take those frames
        # into account
        centers = tables[list(tables.keys())[vid_index]].iloc[:recoglimit, :]

        # Fix the edge case where there are less frames than the minimum specified for recognition
        recoglimit = np.min([recoglimit, centers.shape[0]])

        # Select animal centers
        centers = centers.loc[
            :, [bpart for bpart in centers.columns if "Tail" not in bpart[0]]
        ]
        centers_shape = centers.shape

    # Loop over the first frames in the video to get resolution and center of the arena
    arena, fnum, h, w = None, 0, None, None

    while cap.isOpened() and fnum < recoglimit:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:  # pragma: no cover
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if arena_type == "circular-autodetect":

            # Detect arena and extract positions
            temp_center, temp_axes, temp_angle = circular_arena_recognition(
                frame, detection_mode=detection_mode, cnn_model=cnn_model
            )
            temp_arena = np.array([[*temp_center, *temp_axes, temp_angle]])

            # Set if not assigned, else concat and return the median
            if arena is None:
                arena = temp_arena
            else:
                arena = np.concatenate([arena, temp_arena], axis=0)

            if h is None and w is None:
                w, h = frame.shape[0], frame.shape[1]

        fnum += 1

    cap.release()
    cv2.destroyAllWindows()

    # Compute the distance between animal centers and the center of the video, for
    # the arena to be based on frames which minimize obstruction of its borders
    if tables is not None:
        center_distances = np.nanmax(
            np.linalg.norm(
                centers.to_numpy().reshape(-1, 2) - (w / 2, h / 2), axis=1
            ).reshape(-1, centers_shape[1] // 2),
            axis=1,
        )
        # Within the frame recognition limit, only the 1% less obstructed will contribute to the arena
        # fitting
        center_quantile = np.quantile(center_distances, 0.05)
        arena = arena[center_distances < center_quantile]
        weights = 1 / center_distances[center_distances < center_quantile]

    # Compute the median across frames and return to tuple format for downstream compatibility
    arena = np.average(arena[~np.any(np.isnan(arena), axis=1)], axis=0)
    arena = (tuple(arena[:2].astype(int)), tuple(arena[2:4].astype(int)), arena[4])

    return arena, h, w


def retrieve_corners_from_image(frame: np.ndarray, arena_type: str):  # pragma: no cover
    """

    Opens a window and waits for the user to click on all corners of the polygonal arena.
    The user should click on the corners in sequential order.

    Args:
        frame (np.ndarray): Frame to display.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygon-manual".

    Returns:

        corners (np.ndarray): nx2 array containing the x-y coordinates of all n corners.

    """

    corners = []

    def click_on_corners(event, x, y, flags, param):
        # Callback function to store the coordinates of the clicked points
        nonlocal corners, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))

    # Create a window and display the image
    cv2.startWindowThread()

    while True:
        frame_copy = frame.copy()

        cv2.imshow(
            "deepof - Select polygonal arena corners - (q: exit / d: delete)",
            frame_copy,
        )
        cv2.setMouseCallback(
            "deepof - Select polygonal arena corners - (q: exit / d: delete)",
            click_on_corners,
        )

        # Display already selected corners
        if len(corners) > 0:
            for c, corner in enumerate(corners):
                cv2.circle(frame_copy, (corner[0], corner[1]), 4, (40, 86, 236), -1)
                # Display lines between the corners
                if len(corners) > 1 and c > 0:
                    if arena_type == "polygonal-manual" or len(corners) < 5:
                        cv2.line(
                            frame_copy,
                            (corners[c - 1][0], corners[c - 1][1]),
                            (corners[c][0], corners[c][1]),
                            (40, 86, 236),
                            2,
                        )

        # Close the polygon
        if len(corners) > 2:
            if arena_type == "polygonal-manual" or len(corners) < 5:
                cv2.line(
                    frame_copy,
                    (corners[0][0], corners[0][1]),
                    (corners[-1][0], corners[-1][1]),
                    (40, 86, 236),
                    2,
                )
        if len(corners) >= 5 and arena_type == "circular-manual":
            cv2.ellipse(
                frame_copy,
                *fit_ellipse_to_polygon(corners),
                startAngle=0,
                endAngle=360,
                color=(40, 86, 236),
                thickness=3,
            )

        cv2.imshow(
            "deepof - Select polygonal arena corners - (q: exit / d: delete)",
            frame_copy,
        )

        # Remove last added coordinate if user presses 'd'
        if cv2.waitKey(1) & 0xFF == ord("d"):
            corners = corners[:-1]

        # Exit is user presses q
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Return the corners
    return corners


def extract_polygonal_arena_coordinates(
    video_path: str, arena_type: str
):  # pragma: no cover
    """

    Reads a random frame from the selected video, and opens an interactive GUI to let the user delineate
    the arena manually.

    Args:
        video_path: Path to the video file.
        arena_type: Type of arena to be used. Must be one of the following: "circular-manual", "polygon-manual".

    Returns:
        np.ndarray: nx2 array containing the x-y coordinates of all n corners of the polygonal arena.
        int: Height of the video.
        int: Width of the video.

    """

    current_video = imread(video_path)
    current_frame = np.random.choice(current_video.shape[0])

    # Get and return the corners of the arena
    arena_corners = retrieve_corners_from_image(
        current_video[current_frame].compute(), arena_type
    )
    return arena_corners, current_video.shape[2], current_video.shape[1]


def fit_ellipse_to_polygon(polygon: list):  # pragma: no cover
    """

    Fits an ellipse to the provided polygon.

    Args:
        polygon: List of (x,y) coordinates of the corners of the polygon.

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


def circular_arena_recognition(
    frame: np.ndarray,
    detection_mode: str = "rule-based",
    cnn_model: tf.keras.models.Model = None,
) -> np.array:
    """

    Returns x,y position of the center, the lengths of the major and minor axes,
    and the angle of the recognised arena.

    Args:
        frame (np.ndarray): numpy.ndarray representing an individual frame of a video
        detection_mode (str): Algorithm to use to detect the arena. "cnn" uses a
        pretrained model based on ResNet50 to predict the ellipse parameters from
        the image. "rule-based" uses a simpler (and faster) image segmentation approach.
        cnn_model (tf.keras.models.Model): Model to use if detection_mode=="cnn".

    Returns:
        circles (np.ndarray): 3-element-array containing x,y positions of the center
        of the arena, and a third value indicating the radius.

    """

    if detection_mode == "rule-based":

        # Convert image to grayscale, threshold it and close it with a 5x5 kernel
        kernel = np.ones((5, 5))
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_image, 255 // 4, 255, 0)
        for _ in range(5):
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Obtain contours from the image, and retain the largest one
        cnts, _ = cv2.findContours(
            thresh.astype(np.int64), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_TC89_KCOS
        )
        main_cnt = np.argmax([len(c) for c in cnts])

        center_coordinates, axes_length, ellipse_angle = fit_ellipse_to_polygon(
            cnts[main_cnt]
        )

    elif detection_mode == "cnn":

        input_shape = tuple(cnn_model.input.shape[1:-1])
        image_temp = cv2.resize(frame, input_shape)
        image_temp = image_temp / 255

        # Detect the main ellipse containing the arena
        predicted_arena = cnn_model.predict(image_temp[np.newaxis, :])[0]

        # Parameters to return
        center_coordinates = tuple(
            (predicted_arena[:2] * frame.shape[:2][::-1] / input_shape).astype(int)
        )
        axes_length = tuple(
            (predicted_arena[2:4] * frame.shape[:2][::-1] / input_shape).astype(int)
        )
        ellipse_angle = predicted_arena[4]

    else:
        raise ValueError(
            "Invalid detection mode. Select between 'cnn' and 'rule-based'"
        )

    # noinspection PyUnboundLocalVariable
    return center_coordinates, axes_length, ellipse_angle


def rolling_speed(
    dframe: pd.DatetimeIndex,
    window: int = 3,
    rounds: int = 3,
    deriv: int = 1,
    center: str = None,
    shift: int = 2,
    typ: str = "coords",
) -> pd.DataFrame:
    """

    Returns the average speed over n frames in pixels per frame.

    Args:
        dframe (pandas.DataFrame): Position over time dataframe.
        window (int): Number of frames to average over.
        rounds (int): Float rounding decimals.
        deriv (int): Position derivative order; 1 for speed, 2 for acceleration, 3 for jerk, etc.
        center (str): For internal usage only; solves an issue with pandas.MultiIndex that arises when centering frames
        to a specific body part.
        shift (int): Window shift for rolling speed calculation.
        typ (str): Type of dataset. Intended for internal usage only.

    Returns:
        speeds (pd.DataFrame): Data frame containing 2D speeds for each body part in the original data or their
        consequent derivatives.

    """

    original_shape = dframe.shape
    if center:
        body_parts = [bp for bp in dframe.columns.levels[0] if center not in bp]
    else:
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


# MACHINE LEARNING FUNCTIONS #


def gmm_compute(x: np.array, n_components: int, cv_type: str) -> list:
    """

    Fits a Gaussian Mixture Model to the provided data and returns evaluation metrics.

    Args:
        x (numpy.ndarray): Data matrix to train the model
        n_components (int): Number of Gaussian components to use
        cv_type (str): Covariance matrix type to use.
        Must be one of "spherical", "tied", "diag", "full".

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
    """

    Runs GMM clustering model selection on the specified X dataframe, outputs the bic distribution per model,
    a vector with the median BICs and an object with the overall best model.

    Args:
        x (pandas.DataFrame): Data matrix to train the models
        n_components_range (range): Generator with numbers of components to evaluate
        n_runs (int): Number of bootstraps for each model
        part_size (int): Size of bootstrap samples for each model
        n_cores (int): Number of cores to use for computation
        cv_types (tuple): Covariance Matrices to try. All four available by default

    Returns:
        - bic (list): All recorded BIC values for all attempted parameter combinations
        (useful for plotting)
        - m_bic(list): All minimum BIC values recorded throughout the process
        (useful for plottinh)
        - best_bic_gmm (sklearn.GMM): Unfitted version of the best found model

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
    """Computes the transition matrix between clusters and the autocorrelation in the sequence.

    Args:
        cluster_sequence (numpy.array): Sequence of cluster assignments.
        nclusts (int): Number of clusters in the sequence.
        autocorrelation (bool): Whether to compute the autocorrelation of the sequence.
        return_graph (bool): Whether to return the transition matrix as an networkx.DiGraph object.

    Returns:
        trans_normed (numpy.ndarray / networkx.Graph): Transition matrix as numpy.ndarray or networkx.DiGraph.
        autocorr (numpy.array): If autocorrelation is True, returns a numpy.ndarray with all autocorrelation
        values on cluster assignment.
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
