# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for the deepof package. See documentation for details

"""

import argparse
import cv2
import multiprocessing
import networkx as nx
import numpy as np
import os
import pandas as pd
import regex as re
from copy import deepcopy
from itertools import combinations, product
from joblib import Parallel, delayed
from sklearn import mixture
from tqdm import tqdm
from typing import Tuple, Any, List, Union, NewType

# DEFINE CUSTOM ANNOTATED TYPES #


Coordinates = NewType("Coordinates", Any)


# CONNECTIVITY FOR DLC MODELS


def connect_mouse_topview(animal_id=None) -> nx.Graph:
    """Creates a nx.Graph object with the connectivity of the bodyparts in the
    DLC topview model for a single mouse. Used later for angle computing, among others

        Parameters:
            - animal_id (str): if more than one animal is tagged,
            specify the animal identyfier as a string

        Returns:
            - connectivity (nx.Graph)"""

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
    """Returns the passed string as a boolean

    Parameters:
        v (str): string to transform to boolean value

    Returns:
        boolean value. If conversion is not possible, it raises an error
    """

    if isinstance(v, bool):
        return v  # pragma: no cover
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:  # pragma: no cover
        raise argparse.ArgumentTypeError("Boolean compatible value expected.")


def likelihood_qc(dframe: pd.DataFrame, threshold: float = 0.9) -> np.array:
    """Returns a DataFrame filtered dataframe, keeping only the rows entirely above the threshold.

    Parameters:
        - dframe (pandas.DataFrame): DeepLabCut output, with positions over time and associated likelihhod
        - threshold (float): minimum acceptable confidence

    Returns:
        - filt_mask (np.array): mask on the rows of dframe"""

    Likes = np.array([dframe[i]["likelihood"] for i in list(dframe.columns.levels[0])])
    Likes = np.nan_to_num(Likes, nan=1.0)
    filt_mask = np.all(Likes > threshold, axis=0)

    return filt_mask


def bp2polar(tab: pd.DataFrame) -> pd.DataFrame:
    """Returns the DataFrame in polar coordinates.

    Parameters:
        - tab (pandas.DataFrame):Table with cartesian coordinates

    Returns:
        - polar (pandas.DataFrame): Equivalent to input, but with values in polar coordinates"""

    tab_ = np.array(tab)
    complex_ = tab_[:, 0] + 1j * tab_[:, 1]
    polar = pd.DataFrame(np.array([abs(complex_), np.angle(complex_)]).T)
    polar.rename(columns={0: "rho", 1: "phi"}, inplace=True)
    return polar


def tab2polar(cartesian_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a pandas.DataFrame in which all the coordinates are polar.

    Parameters:
        - cartesian_df (pandas.DataFrame):DataFrame containing tables with cartesian coordinates

    Returns:
        - result (pandas.DataFrame): Equivalent to input, but with values in polar coordinates"""

    result = []
    for df in list(cartesian_df.columns.levels[0]):
        result.append(bp2polar(cartesian_df[df]))
    result = pd.concat(result, axis=1)
    idx = pd.MultiIndex.from_product(
        [list(cartesian_df.columns.levels[0]), ["rho", "phi"]],
        names=["bodyparts", "coords"],
    )
    result.columns = idx
    return result


def compute_dist(
    pair_array: np.array, arena_abs: int = 1, arena_rel: int = 1
) -> pd.DataFrame:
    """Returns a pandas.DataFrame with the scaled distances between a pair of body parts.

    Parameters:
        - pair_array (numpy.array): np.array of shape N * 4 containing X,y positions
        over time for a given pair of body parts
        - arena_abs (int): diameter of the real arena in cm
        - arena_rel (int): diameter of the captured arena in pixels

    Returns:
        - result (pd.DataFrame): pandas.DataFrame with the
        absolute distances between a pair of body parts"""

    lim = 2 if pair_array.shape[1] == 4 else 1
    a, b = pair_array[:, :lim], pair_array[:, lim:]
    ab = a - b

    dist = np.sqrt(np.einsum("...i,...i", ab, ab))
    return pd.DataFrame(dist * arena_abs / arena_rel)


def bpart_distance(
    dataframe: pd.DataFrame, arena_abs: int = 1, arena_rel: int = 1
) -> pd.DataFrame:
    """Returns a pandas.DataFrame with the scaled distances between all pairs of body parts.

    Parameters:
        - dataframe (pandas.DataFrame): pd.DataFrame of shape N*(2*bp) containing X,y positions
    over time for a given set of bp body parts
        - arena_abs (int): diameter of the real arena in cm
        - arena_rel (int): diameter of the captured arena in pixels

    Returns:
        - result (pd.DataFrame): pandas.DataFrame with the
        absolute distances between all pairs of body parts"""

    indexes = combinations(dataframe.columns.levels[0], 2)
    dists = []
    for idx in indexes:
        dist = compute_dist(np.array(dataframe.loc[:, list(idx)]), arena_abs, arena_rel)
        dist.columns = [idx]
        dists.append(dist)

    return pd.concat(dists, axis=1)


def angle(a: np.array, b: np.array, c: np.array) -> np.array:
    """Returns a numpy.array with the angles between the provided instances.

    Parameters:
        - a (2D np.array): positions over time for a bodypart
        - b (2D np.array): positions over time for a bodypart
        - c (2D np.array): positions over time for a bodypart

    Returns:
        - ang (1D np.array): angles between the three-point-instances"""

    ba = a - b
    bc = c - b

    cosine_angle = np.einsum("...i,...i", ba, bc) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    ang = np.arccos(cosine_angle)

    return ang


def angle_trio(bpart_array: np.array) -> np.array:
    """Returns a numpy.array with all three possible angles between the provided instances.

    Parameters:
        - bpart_array (2D numpy.array): positions over time for a bodypart

    Returns:
        - ang_trio (2D numpy.array): all-three angles between the three-point-instances"""

    a, b, c = bpart_array
    ang_trio = np.array([angle(a, b, c), angle(a, c, b), angle(b, a, c)])

    return ang_trio


def rotate(
    p: np.array, angles: np.array, origin: np.array = np.array([0, 0])
) -> np.array:
    """Returns a numpy.array with the initial values rotated by angles radians

    Parameters:
        - p (2D numpy.array): array containing positions of bodyparts over time
        - angles (2D numpy.array): set of angles (in radians) to rotate p with
        - origin (2D numpy.array): rotation axis (zero vector by default)

    Returns:
        - rotated (2D numpy.array): rotated positions over time"""

    R = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]])

    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)

    rotated = np.squeeze((R @ (p.T - o.T) + o.T).T)

    return rotated


def align_trajectories(data: np.array, mode: str = "all") -> np.array:
    """Returns a numpy.array with the positions rotated in a way that the center (0 vector)
    and the body part in the first column of data are aligned with the y axis.

        Parameters:
            - data (3D numpy.array): array containing positions of body parts over time, where
            shape is N (sliding window instances) * m (sliding window size) * l (features)
            - mode (string): specifies if *all* instances of each sliding window get
            aligned, or only the *center*

        Returns:
            - aligned_trajs (2D np.array): aligned positions over time"""

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
    """Returns a boolean array in which isolated appearances of a feature are smoothened

    Parameters:
        - a (1D numpy.array): boolean instances

    Returns:
        - a (1D numpy.array): smoothened boolean instances"""

    for i in range(1, len(a) - 1):
        if a[i - 1] == a[i + 1]:
            a[i] = a[i - 1]
    return a == 1


def rolling_window(a: np.array, window_size: int, window_step: int) -> np.array:
    """Returns a 3D numpy.array with a sliding-window extra dimension

    Parameters:
        - a (2D np.array): N (instances) * m (features) shape

    Returns:
        - rolled_a (3D np.array):
        N (sliding window instances) * l (sliding window size) * m (features)"""

    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    rolled_a = np.lib.stride_tricks.as_strided(
        a, shape=shape, strides=strides, writeable=True
    )[::window_step]
    return rolled_a


def smooth_mult_trajectory(series: np.array, alpha: float = 0.99) -> np.array:
    """Returns a smooths a trajectory using exponentially weighted averages

    Parameters:
        - series (numpy.array): 1D trajectory array with N (instances) - alpha (float): 0 <= alpha <= 1;
        indicates the inverse weight assigned to previous observations. Higher (alpha~1) indicates less smoothing;
        lower indicates more (alpha~0)

    Returns:
        - smoothed_series (np.array): smoothed version of the input, with equal shape"""

    result = [series[0]]
    for n in range(len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])

    smoothed_series = np.array(result)

    return smoothed_series


def recognize_arena(
    videos: list,
    vid_index: int,
    path: str = ".",
    recoglimit: int = 1,
    arena_type: str = "circular",
) -> Tuple[np.array, int, int]:
    """Returns numpy.array with information about the arena recognised from the first frames
    of the video. WARNING: estimates won't be reliable if the camera moves along the video.

        Parameters:
            - videos (list): relative paths of the videos to analise
            - vid_index (int): element of videos to use
            - path (string): full path of the directory where the videos are
            - recoglimit (int): number of frames to use for position estimates
            - arena_type (string): arena type; must be one of ['circular']

        Returns:
            - arena (np.array): 1D-array containing information about the arena.
            "circular" (3-element-array) -> x-y position of the center and the radius
            - h (int): height of the video in pixels
            - w (int): width of the video in pixels"""

    cap = cv2.VideoCapture(os.path.join(path, videos[vid_index]))

    # Loop over the first frames in the video to get resolution and center of the arena
    arena, fnum, h, w = False, 0, None, None

    while cap.isOpened() and fnum < recoglimit:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:  # pragma: no cover
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if arena_type == "circular":

            # Detect arena and extract positions
            arena = circular_arena_recognition(frame)[0]
            if h is None and w is None:
                w, h = frame.shape[0], frame.shape[1]

        fnum += 1

    cap.release()
    cv2.destroyAllWindows()

    return arena, h, w


def circular_arena_recognition(frame: np.array) -> np.array:
    """Returns x,y position of the center and the radius of the recognised arena

    Parameters:
        - frame (np.array): numpy.array representing an individual frame of a video

    Returns:
        - circles (np.array): 3-element-array containing x,y positions of the center
        of the arena, and a third value indicating the radius"""

    # Convert image to greyscale, threshold it, blur it and detect the biggest best fitting circle
    # using the Hough algorithm
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_image, 50, 255, 0)
    frame = cv2.medianBlur(thresh, 9)
    circle = cv2.HoughCircles(
        frame,
        cv2.HOUGH_GRADIENT,
        1,
        300,
        param1=50,
        param2=10,
        minRadius=0,
        maxRadius=0,
    )

    circles = []

    if circle is not None:
        circle = np.uint16(np.around(circle[0]))
        circles.append(circle)

    return circles[0]


def rolling_speed(
    dframe: pd.DatetimeIndex,
    window: int = 5,
    rounds: int = 10,
    deriv: int = 1,
    center: str = None,
    typ: str = "coords",
) -> pd.DataFrame:
    """Returns the average speed over n frames in pixels per frame

    Parameters:
        - dframe (pandas.DataFrame): position over time dataframe
        - pause (int):  frame-length of the averaging window
        - rounds (int): float rounding decimals
        - deriv (int): position derivative order; 1 for speed,
        2 for acceleration, 3 for jerk, etc
        - center (str): for internal usage only; solves an issue
        with pandas.MultiIndex that arises when centering frames
        to a specific body part

    Returns:
        - speeds (pd.DataFrame): containing 2D speeds for each body part
        in the original data or their consequent derivatives"""

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

        distances = np.concatenate(
            [
                np.array(dframe).reshape([-1, features], order="F"),
                np.array(dframe.shift()).reshape([-1, features], order="F"),
            ],
            axis=1,
        )

        distances = np.array(compute_dist(distances))
        distances = distances.reshape(
            [
                original_shape[0],
                (original_shape[1] // 2 if typ == "coords" else original_shape[1]),
            ],
            order="F",
        )
        distances = pd.DataFrame(distances, index=dframe.index)
        speeds = np.round(distances.rolling(window).mean(), rounds)
        speeds[np.isnan(speeds)] = 0.0

        dframe = speeds

    speeds.columns = body_parts

    return speeds


# MACHINE LEARNING FUNCTIONS #


def gmm_compute(x: np.array, n_components: int, cv_type: str) -> list:
    """Fits a Gaussian Mixture Model to the provided data and returns evaluation metrics.

    Parameters:
        - x (numpy.array): data matrix to train the model
        - n_components (int): number of Gaussian components to use
        - cv_type (str): covariance matrix type to use.
        Must be one of "spherical", "tied", "diag", "full"

    Returns:
        - gmm_eval (list): model and associated BIC for downstream selection
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
    """Runs GMM clustering model selection on the specified X dataframe, outputs the bic distribution per model,
    a vector with the median BICs and an object with the overall best model

     Parameters:
         - x (pandas.DataFrame): data matrix to train the models
         - n_components_range (range): generator with numbers of components to evaluate
         - n_runs (int): number of bootstraps for each model
         - part_size (int): size of bootstrap samples for each model
         - n_cores (int): number of cores to use for computation
         - cv_types (tuple): Covariance Matrices to try. All four available by default

     Returns:
         - bic (list): All recorded BIC values for all attempted parameter combinations
         (useful for plotting)
         - m_bic(list): All minimum BIC values recorded throughout the process
         (useful for plottinh)
         - best_bic_gmm (sklearn.GMM): unfitted version of the best found model
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

    Parameters:
        - cluster_sequence (numpy.array):
        - nclusts (int):
        - autocorrelation (bool):
        - return_graph (bool):

    Returns:
        - trans_normed (numpy.array / networkx.Graph:
        - autocorr (numpy.array):
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
#    - Add sequence plot to single_behaviour_analysis (show how the condition varies across a specified time window)
#    - Add digging to rule_based_tagging
#    - Add center to rule_based_tagging
#    - Check for features requested by Joeri
