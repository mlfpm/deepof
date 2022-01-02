# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for the deepof package.

"""

import argparse
import gc
import multiprocessing
import os
from copy import deepcopy
from itertools import combinations, product
from typing import Tuple, Any, List, Union, NewType

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import regex as re
import ruptures as rpt
import tensorflow as tf
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
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
        [list(cartesian_df.columns.levels[0]), ["rho", "phi"]],
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
            data[frame].reshape([-1, 2], order="C"),
            angles[frame],
        ).reshape(data.shape[1:], order="C")

    if mode == "all" or mode == "none":
        aligned_trajs = aligned_trajs.reshape(dshape, order="C")

    return aligned_trajs


def smooth_boolean_array(a: np.array) -> np.array:
    """

    Returns a boolean array in which isolated appearances of a feature are smoothed.

    Args:
        a (numpy.ndarray): Boolean instances.

    Returns:
        a (numpy.ndarray): Smoothened boolean instances.

    """

    for i in range(1, len(a) - 1):
        if a[i - 1] == a[i + 1]:
            a[i] = a[i - 1]
    return a == 1


def split_with_breakpoints(a: np.ndarray, breakpoints: list) -> np.ndarray:
    """

    Split a numpy.ndarray at the given breakpoints.

    Args:
        a (np.ndarray): N (instances) * m (features) shape
        breakpoints (list): list of breakpoints obtained with ruptures

    Returns:
        split_a (np.ndarray): N (instances) * l (maximum break length) * m (features) shape

    """
    rpt_lengths = list(np.array(breakpoints)[1:] - np.array(breakpoints)[:-1])
    max_rpt_length = np.max([breakpoints[0], np.max(rpt_lengths)])

    # Reshape experiment data according to extracted ruptures
    split_a = np.split(np.expand_dims(a, axis=0), breakpoints[:-1], axis=1)

    split_a = [
        np.pad(
            i,
            ((0, 0), (0, max_rpt_length - i.shape[1]), (0, 0)),
            constant_values=0.0,
        )
        for i in split_a
    ]
    split_a = np.concatenate(split_a, axis=0)

    return split_a


def rolling_window(
    a: np.array, window_size: int, window_step: int, automatic_changepoints: str = False
) -> np.ndarray:
    """

    Returns a 3D numpy.array with a sliding-window extra dimension.

    Args:
        a (np.ndarray): N (instances) * m (features) shape
        window_size (int): Size of the window to apply
        window_step (int): Step of the window to apply
        automatic_changepoints (str): Changepoint detection algorithm to apply.
        If False, applies a fixed sliding window.

    Returns:
        rolled_a (np.ndarray): N (sliding window instances) * l (sliding window size) * m (features)

    """

    breakpoints = None

    if automatic_changepoints:
        # Define change point detection model using ruptures
        # Remove dimensions with low variance (occurring when aligning the animals with the y axis)
        rpt_model = rpt.KernelCPD(
            kernel=automatic_changepoints, min_size=window_size, jump=window_step
        ).fit(VarianceThreshold(threshold=1e-3).fit_transform(a))

        # Extract change points from current experiment
        breakpoints = rpt_model.predict(pen=4.0)
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

    Returns:
        ruptured_dataset (np.ndarray): Dataset with all ruptures concatenated across the first axis.
        rupture_indices (list): Indices of ruptures.

    """

    # Generate a base ruptured training set and a set of breaks
    ruptured_dataset, break_indices = None, None
    cumulative_shape = 0
    # Iterate over all experiments and populate them
    for i, tab in enumerate(table_dict.values()):
        if i in rupture_indices:
            current_size = tab.shape[0]
            current_train, current_breaks = rolling_window(
                to_rupture[cumulative_shape : cumulative_shape + current_size],
                window_size,
                window_step,
                automatic_changepoints,
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

    columns_to_keep = []
    for column in columns:
        # Speed transformed columns
        if type(column) == str and column.startswith(selected_id):
            columns_to_keep.append(column)
        # Raw coordinate columns
        if column[0].startswith(selected_id) and column[1] in [
            "x",
            "y",
            "rho",
            "phi",
        ]:
            columns_to_keep.append(column)
        # Raw distance and angle columns
        elif len(column) in [2, 3] and all([i.startswith(selected_id) for i in column]):
            columns_to_keep.append(column)

    return columns_to_keep


# noinspection PyUnboundLocalVariable
def recognize_arena(
    videos: list,
    vid_index: int,
    path: str = ".",
    tables: dict = None,
    recoglimit: int = 500,
    high_fidelity: bool = False,
    arena_type: str = "circular",
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
        arena_type (string): Arena type; must be one of ['circular'].
        detection_mode (str): Algorithm to use to detect the arena. "cnn" uses a
        pretrained model based on ResNet50 to predict the ellipse parameters from
        the image. "rule-based" (default) uses a simpler (and faster) image segmentation approach.
        cnn_model (tf.keras.models.Model): Model to use if detection_mode=="cnn".

    Returns:
        arena (np.ndarray): 1D-array containing information about the arena.
        "circular" (3-element-array) -> x-y position of the center and the radius.
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

        if arena_type == "circular":

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


def circular_arena_recognition(
    frame: np.array,
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

        # Detect the main ellipse containing the arena
        ellipse_params = cv2.fitEllipse(cnts[main_cnt])

        # Parameters to return
        center_coordinates = tuple([int(i) for i in ellipse_params[0]])
        axes_length = tuple([int(i) // 2 for i in ellipse_params[1]])
        ellipse_angle = ellipse_params[2]

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
        body_parts = [bp for bp in dframe.columns.levels[0] if bp != center]
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


# TODO:
#    - Add center / time in zone to supervised_tagging
