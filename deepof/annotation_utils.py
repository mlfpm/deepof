# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Functions and general utilities for supervised pose estimation. See documentation for details."""

import os
import copy
import pickle
import warnings
from itertools import combinations
from typing import Any, List, NewType, Union

import numba as nb
import numpy as np
import pandas as pd
import sklearn.pipeline
from joblib import Parallel, delayed, parallel_backend
from natsort import os_sorted
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import deepof.post_hoc
import deepof.utils
from deepof.utils import _suppress_warning
from deepof.data_loading import get_dt, load_dt, _suppress_warning
import xgboost #as xgb


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def close_single_contact(
    pos_dframe: pd.DataFrame,
    left: str,
    right: str,
    tol: float,
) -> np.array:
    """Return a boolean array that's True if the specified body parts are closer than tol.

    Args:
        pos_dframe (pandas.DataFrame): DLC output as pandas.DataFrame; only applicable to two-animal experiments.
        left (string): First member of the potential contact
        right (string): Second member of the potential contact
        tol (float): maximum distance for which a contact is reported

    Returns:
        contact_array (np.array): True if the distance between the two specified points is less than tol, False otherwise

    """
    close_contact = None

    if isinstance(right, str):
        close_contact = np.linalg.norm(pos_dframe[left] - pos_dframe[right], axis=1) < tol

    elif isinstance(right, list):
        close_contact = np.any(
            [
                np.linalg.norm(pos_dframe[left] - pos_dframe[r], axis=1) < tol
                for r in right
            ],
            axis=0,
        )

    return close_contact


def close_double_contact(
    pos_dframe: pd.DataFrame,
    #left_len: float,
    left1: str,
    left2: str,
    #right_len: float,
    right1: str,
    right2: str,
    rel_tol: float,
    rev: bool = False,
) -> np.array:
    """Return a boolean array that's True if the specified body parts are closer than tol.

    Parameters:
        pos_dframe (pandas.DataFrame): DLC output as pandas.DataFrame; only applicable to two-animal experiments.
        #left_len (float): Length of animal 1
        left1 (string): First contact point of animal 1
        left2 (string): Second contact point of animal 1
        #right_len (float): Length of animal 2
        right1 (string): First contact point of animal 2
        right2 (string): Second contact point of animal 2
        rel_tol (float): relative shar which affects the maximum distance for which a contact is reported
        rev (bool): reverses the default behaviour (nose2tail contact for both mice)

    Returns:
        double_contact (np.array): True if the distance between the two specified points is less than tol, False otherwise

    """
    #calculate absolute tolerance using areas
    tol=rel_tol#(rel_tol*(left_len+right_len))/2
    
    if rev:
        double_contact = (
            np.linalg.norm(pos_dframe[right1] - pos_dframe[left2], axis=1)
            < tol
        ) & (
            np.linalg.norm(pos_dframe[right2] - pos_dframe[left1], axis=1)
            < tol
        )

    else:
        double_contact = (
            np.linalg.norm(pos_dframe[right1] - pos_dframe[left1], axis=1)
            < tol
        ) & (
            np.linalg.norm(pos_dframe[right2] - pos_dframe[left2], axis=1)
            < tol
        )

    return double_contact


def rotate(origin, point, ang):
    """Auxiliar function to climb_wall and sniff_object. Rotates x,y coordinates over a pivot.
    
    Parameters:
        origin (): 
        point (): 
        ang (): 

    Returns:
        qx (): 
        qy (): """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(ang) * (px - ox) - np.sin(ang) * (py - oy)
    qy = oy + np.sin(ang) * (px - ox) + np.cos(ang) * (py - oy)
    return qx, qy


def outside_ellipse(x, y, e_center, e_axes, e_angle, threshold=0.0):
    """Auxiliar function to climb_wall and sniff_object.

    Returns True if the passed x, y coordinates
    are outside the ellipse denoted by e_center, e_axes and e_angle, with a certain threshold

    """
    x, y = rotate(e_center, (x, y), np.radians(e_angle))

    term_x = (x - e_center[0]) ** 2 / (e_axes[0] + threshold) ** 2
    term_y = (y - e_center[1]) ** 2 / (e_axes[1] + threshold) ** 2
    return term_x + term_y > 1


def climb_arena(
    arena_type: str,
    arena: np.array,
    pos_dict: pd.DataFrame,
    rel_tol: float,
    id: str,
    mouse_len: 50,
    centered_data: bool = False,
    run_numba: bool = False,
) -> np.array:
    """Return True if the specified mouse is climbing the wall.

    Args:
        arena_type (str): arena type; must be one of ['polygonal-manual', 'circular-autodetect']
        arena (np.array): contains arena location and shape details
        pos_dict (table_dict): position over time for all videos in a project
        rel_tol (float): relative tolerance (to mouse length) to report a hit
        id (str): indicates the id + subcondition of the animal
        centered_data (bool): indicates whether the input data is centered
        run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)

    Returns:
        climbing (np.array): boolean array. True if selected animal is climbing the walls of the arena

    """
    nose = copy.deepcopy(pos_dict[id+"Nose"])

    #absolute tolerance       
    tol=mouse_len*rel_tol

    #interpolate nans (done only for climbing for reasons explained in the documentation) 
    nose.interpolate(
    method="linear",
    limit_direction="both",
    inplace=True,
    )

    if arena_type.startswith("circular"):
        center = np.zeros(2) if centered_data else np.array(arena[0])
        axes = arena[1]
        angle = arena[2]
        climbing = outside_ellipse(
            x=nose["x"],
            y=nose["y"],
            e_center=center,
            e_axes=axes,
            e_angle=-angle,
            threshold=tol,
        )

    elif arena_type.startswith("polygon"):

        # intermediary for testing, will be replaced with length-based condition
        if run_numba:

            # extract outer arena polygon coordinates
            xp = np.array(Polygon(arena).buffer(tol).exterior.coords.xy[0])
            yp = np.array(Polygon(arena).buffer(tol).exterior.coords.xy[1])
            outer_polygon = np.transpose(np.array([xp, yp]))

            # get nose positions outside of arena polygon
            climbing = np.invert(point_in_polygon_numba(nose.values, outer_polygon))

        else:

            climbing = np.invert(
                point_in_polygon(nose.values, Polygon(arena).buffer(tol))
            )

    else:
        raise NotImplementedError(
            "Supported values for arena_type are ['polygonal-manual', 'circular-manual', 'circular-autodetect']"
        )

    return climbing


def sniff_object(
    speed_dframe: pd.DataFrame,
    arena_type: str,
    arena: np.array,
    pos_dict: pd.DataFrame,
    tol: float,
    tol_speed: float,
    nose: str,
    center_name: str = "Center",
    centered_data: bool = False,
    s_object: str = "arena",
    animal_id: str = "",
    run_numba: bool = False,
):
    """Return True if the specified mouse is sniffing an object.

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time.
        arena_type (str): arena type; must be one of ['polygonal-manual', 'circular-autodetect'].
        arena (np.array): contains arena location and shape details.
        pos_dict (table_dict): position over time for all videos in a project.
        tol (float): minimum tolerance to report a hit.
        tol_speed (float): minimum speed to report a hit.
        center_name (str): Body part to center coordinates on. "Center" by default.
        nose (str): indicates the name of the body part representing the nose of the selected animal.
        centered_data (bool): indicates whether the input data is centered.
        s_object (str): indicates the object to sniff. Must be one of ['arena', 'object'].
        animal_id (str): indicates the animal to sniff. Must be one of animal_ids.
        run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)

    Returns:
        sniffing (np.array): boolean array. True if selected animal is sniffing the selected object

    """
    nose, nosing = pos_dict[nose], True

    if animal_id != "":
        animal_id += "_"

    if s_object == "arena":
        if arena_type.startswith("circular"):
            center = np.zeros(2) if centered_data else np.array(arena[0])
            axes = arena[1]
            angle = arena[2]

            nosing_min = outside_ellipse(
                x=nose["x"],
                y=nose["y"],
                e_center=center,
                e_axes=axes,
                e_angle=-angle,
                threshold=-tol,
            )
            nosing_max = outside_ellipse(
                x=nose["x"],
                y=nose["y"],
                e_center=center,
                e_axes=axes,
                e_angle=-angle,
                threshold=tol,
            )

        elif arena_type.startswith("polygon"):

            # intermediary for testing, will be replaced with length-based condition
            if run_numba:

                # extract outer arena polygon coordinates
                xp = np.array(Polygon(arena).buffer(-tol).exterior.coords.xy[0])
                yp = np.array(Polygon(arena).buffer(-tol).exterior.coords.xy[1])
                inner_polygon = np.transpose(np.array([xp, yp]))

                # extract inner arena polygon coordinates
                xp = np.array(Polygon(arena).buffer(tol).exterior.coords.xy[0])
                yp = np.array(Polygon(arena).buffer(tol).exterior.coords.xy[1])
                outer_polygon = np.transpose(np.array([xp, yp]))

                # get nose positions outside of outer and inner arena polygon
                nosing_min = np.invert(
                    point_in_polygon_numba(nose.values, inner_polygon)
                )
                nosing_max = np.invert(
                    point_in_polygon_numba(nose.values, outer_polygon)
                )

            else:

                nosing_min = np.invert(
                    point_in_polygon(nose.values, Polygon(arena).buffer(-tol))
                )
                nosing_max = np.invert(
                    point_in_polygon(nose.values, Polygon(arena).buffer(tol))
                )

        # noinspection PyUnboundLocalVariable
        # get nose positions that are close to the outer edge of the arena
        # (not in smaller polygon and [not not] in larger polygon)
        nosing = nosing_min & (~nosing_max)

    else:
        raise NotImplementedError

    speed = speed_dframe[animal_id + center_name] < tol_speed
    sniffing = nosing & speed

    return sniffing


def point_in_polygon(points: np.array, polygon: Polygon) -> np.array:
    """
    Check if a set of points is inside a polygon.

    Args:
        points (np.ndarray): An array of shape (M, 2) containing the coordinates of the points.
        polygon (shapely.geometry.polygon.Polygon): Shapely polygon.

    Returns:
        np.ndarray: A boolean array of shape (M,) indicating whether each point is inside the polygon.
    """
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


def cowering(
    X_huddle: np.ndarray,
    huddle_estimator: sklearn.pipeline.Pipeline,
    animal_id: str = "",
    median_filter_width: int = 11,
    min_immobility: int = 25,
    max_immobility: int = 3000,
) -> np.array:
    """Return true when the mouse is huddling a pretrained model.

    Args:
        X_huddle (pandas.DataFrame): mouse features over time.
        huddle_estimator (sklearn.pipeline.Pipeline): pre-trained model to predict feature occurrence.
        animal_id (str): indicates the animal to sniff. Must be one of animal_ids.
        median_filter_width (int): width of median filter for smoothing results
        min_immobility (int): minimum length of behavior to be considered immobility
        max_immobility (int): maximum length of behavior to be considered immobility (longer is labeled as "sleeping")

    Returns:
        y_huddle (np.array): 1 if the animal is huddling, 0 otherwise

    """
    # Keep only body parts that are relevant for huddling
    required_features = [
        "('{}Right_bhip', '{}Spine_2')_raw".format(animal_id, animal_id),
        "('{}Spine_2', '{}Tail_base')_raw".format(animal_id, animal_id),
        "('{}Left_bhip', '{}Spine_2')_raw".format(animal_id, animal_id),
        "('{}Center', '{}Spine_2')_raw".format(animal_id, animal_id),
        "('{}Left_ear', '{}Nose')_raw".format(animal_id, animal_id),
        "('{}Nose', '{}Right_ear')_raw".format(animal_id, animal_id),
        "('{}Center', '{}Right_fhip')_raw".format(animal_id, animal_id),
        "('{}Center', '{}Left_fhip')_raw".format(animal_id, animal_id),
        "('{}Center', '{}Spine_1')_raw".format(animal_id, animal_id),
        "('{}Right_ear', '{}Spine_1')_raw".format(animal_id, animal_id),
        "('{}Left_ear', '{}Spine_1')_raw".format(animal_id, animal_id),
        "{}head_area_raw".format(animal_id),
        "{}torso_area_raw".format(animal_id),
        "{}back_area_raw".format(animal_id),
        "{}full_area_raw".format(animal_id),
        "{}Center_speed".format(animal_id),
        "{}Left_bhip_speed".format(animal_id),
        "{}Left_ear_speed".format(animal_id),
        "{}Left_fhip_speed".format(animal_id),
        "{}Nose_speed".format(animal_id),
        "{}Right_bhip_speed".format(animal_id),
        "{}Right_ear_speed".format(animal_id),
        "{}Right_fhip_speed".format(animal_id),
        "{}Spine_1_speed".format(animal_id),
        "{}Spine_2_speed".format(animal_id),
        "{}Tail_base_speed".format(animal_id),
    ]
    try:
        X_huddle = X_huddle[required_features]
    except KeyError:
        # Return an array of NaNs if the required features are not present, and raise a warning
        warnings.warn(
            "Skipping huddle annotation as not all required body parts are present. At the moment, huddle annotation "
            "requires the deepof_14 labelling scheme. Read the full documentation for further details."
        )
        return np.full(X_huddle.shape[0], np.nan)

    X_huddle=augment_with_neighbors(X_huddle)
    # Concatenate all relevant data frames and predict using the pre-trained estimator
    X_mask = np.isnan(X_huddle).mean(axis=1) > 0.1
    y_huddle = huddle_estimator.predict(
        StandardScaler().fit_transform(np.nan_to_num(X_huddle))
    ).astype(float)
    y_huddle[X_mask] = False#np.nan
    y_huddle = deepof.utils.binary_moving_median_numba(y_huddle, lag=median_filter_width)
    y_huddle = deepof.utils.filter_short_true_segments_numba(y_huddle, min_length=min_immobility)
    y_sleep = y_huddle #deepof.utils.filter_short_true_segments_numba(y_huddle, min_length=max_immobility)
    y_huddle[y_sleep] = False
    return y_huddle, y_sleep


def augment_with_neighbors(X_huddle, window=5, step=1, window_out=11):
    cols = X_huddle.columns.tolist()
    L = 2 * window + 1
    b = L / window_out
    ranges_list = [(round(i * b), round((i + 1) * b)) for i in range(window_out)]
    
    augmented_dfs = []
    
    for col in cols:
        shifted_data = []
        
        # Build future values (leads) in reverse order
        for lead in range(window * step, 0, -step):
            shifted_data.append(X_huddle[col].shift(-lead))
        
        shifted_data.append(X_huddle[col])  # Current value
        
        # Build past values (lags) in forward order
        for lag in range(step, window * step + 1, step):
            shifted_data.append(X_huddle[col].shift(lag))
        
        shifted_df = pd.concat(shifted_data, axis=1)
        
        # Compute all window_out features for the current column
        col_features = {}
        for k in range(window_out):
            start, end = ranges_list[k]
            mean_series = shifted_df.iloc[:, start:end].mean(axis=1, skipna=False)
            col_features[f'{col}_{k}'] = mean_series
        
        # Append the features as a DataFrame to the list
        augmented_dfs.append(pd.DataFrame(col_features))
    
    # Concatenate all DataFrames at once
    X_augmented = pd.concat(augmented_dfs, axis=1)
    
    return X_augmented


def digging(
    speed_dframe: pd.DataFrame,
    dist_dframe: pd.DataFrame,
    mouse_identity: str,
    close_range: np.ndarray,
    likelihood_dframe: pd.DataFrame,
    tol_speed: float,
    tol_likelihood: float,
    min_length: int,
    center_name: str = "Center",
    animal_id: str = "",
):
    """Return true when the mouse is standing still and either moving (active) or not moving (passive).

    Design considerations:
        Detecting immobility and activity is relatively straightforward by mostly just checking speed thresholds on bodyparts.
        The main problem arises from getting a lot of "flickering" out of the detections, as bodyparts from frame to frame may be
        just above or below that threshold. Respectively most of the detect_activity algorithm is a series of filtering steps to
        alternatingly smooth the predictions and sharpening the edges of predicted behavior. 

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        stationary_active (np.array): True if the animal is standing still and is active, False otherwise
        stationary_passive (np.array): True if the animal is standing still and is passive, False otherwise

    """
    if animal_id != "":
        animal_id += "_"

    # Get frames with undefined speed
    nan_pos = speed_dframe[speed_dframe[animal_id + center_name].isnull()].index.tolist()

    # Detect and smooth frames with mouse being immobile    
    immobile = np.array([False]*len(speed_dframe))
    speed_dframe.interpolate(method='linear', inplace=True)
    immobile = deepof.utils.moving_average((speed_dframe[animal_id + center_name] <= tol_speed*2).to_numpy(), lag=min_length).astype(bool)
    immobile = deepof.utils.filter_short_true_segments(
        array=immobile, min_length=min_length,
    )

    # Init stationary active and passive subsets with immobile frames
    stationary_lookaround=copy.copy(immobile)
    stationary_nonlookaround=copy.copy(immobile)

    # Detect activity when speed and likelyhood is above a threshold for any of the available bodyparts from the list
    bodyparts=[animal_id+"Left_fhip",animal_id+"Right_fhip", animal_id+"Left_bhip", animal_id+"Right_bhip"]

    # Remove missing bodyparts from list
    for bp in bodyparts:
        if not bp in speed_dframe.keys():
            bodyparts.remove(bp)

    
    nose_activity = (tol_speed < speed_dframe[animal_id+"Nose"]).to_numpy() & (likelihood_dframe[animal_id+"Nose"] > tol_likelihood).to_numpy()

    bparts=[animal_id+"Left_bhip", animal_id+"Right_bhip"]

    body_inactivity = np.array([
        (tol_speed*2 >= speed_dframe[part]).to_numpy() & 
        (likelihood_dframe[part] > tol_likelihood).to_numpy()
        for part in bparts
    ]).all(axis=0)

    bodyparts = bodyparts + [animal_id+"Nose"]

    #helper function to check if distance exists
    def check_distance(mouse_id, ear_part):
        """Returns the correct tuple (ear, nose) or (nose, ear) if present."""
        col1 = (f"{mouse_id}{ear_part}", f"{mouse_id}Nose")
        col2 = (f"{mouse_id}Nose", f"{mouse_id}{ear_part}")
        return col1 if col1 in dist_dframe.columns else col2 if col2 in dist_dframe.columns else None

    # Left ear logic
    left_dist = check_distance(mouse_identity, 'Left_ear')
    if left_dist:
        left_max_dist = 0.9*np.nanmedian(dist_dframe[left_dist])
        left_dist = dist_dframe[left_dist] < left_max_dist

    # Right ear logic
    right_dist = check_distance(mouse_identity, 'Right_ear')
    if right_dist:
        right_max_dist = 0.9*np.nanmedian(dist_dframe[right_dist])
        right_dist = dist_dframe[right_dist] < right_max_dist

    # Frames are Stationary active when immobile and active, stationary passive when immobile and not active + smoothing
    stationary_lookaround = immobile & nose_activity & right_dist & left_dist & ~close_range
    stationary_nonlookaround = immobile & ~(nose_activity & right_dist & left_dist & ~close_range)

    stationary_lookaround = deepof.utils.multi_step_paired_smoothing(stationary_lookaround, stationary_nonlookaround, immobile, min_length)

     # Set all Frames that had no speed information in the beginning to False
    stationary_lookaround[nan_pos] = False
    
    return stationary_lookaround




def stationary_lookaround(
    speed_dframe: pd.DataFrame,
    dist_dframe: pd.DataFrame,
    mouse_identity: str,
    close_range: np.ndarray,
    likelihood_dframe: pd.DataFrame,
    tol_speed: float,
    tol_likelihood: float,
    min_length: int,
    center_name: str = "Center",
    animal_id: str = "",
):
    """Return true when the mouse is standing still and either moving (active) or not moving (passive).

    Design considerations:
        Detecting immobility and activity is relatively straightforward by mostly just checking speed thresholds on bodyparts.
        The main problem arises from getting a lot of "flickering" out of the detections, as bodyparts from frame to frame may be
        just above or below that threshold. Respectively most of the detect_activity algorithm is a series of filtering steps to
        alternatingly smooth the predictions and sharpening the edges of predicted behavior. 

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        stationary_active (np.array): True if the animal is standing still and is active, False otherwise
        stationary_passive (np.array): True if the animal is standing still and is passive, False otherwise

    """
    if animal_id != "":
        animal_id += "_"

    # Get frames with undefined speed
    nan_pos = speed_dframe[speed_dframe[animal_id + 'Tail_base'].isnull()].index.tolist()

    # Detect and smooth frames with mouse being immobile    
    immobile = np.array([False]*len(speed_dframe))
    speed_dframe.interpolate(method='linear', inplace=True)
    immobile = deepof.utils.moving_average((speed_dframe[animal_id + 'Tail_base'] <= tol_speed*2).to_numpy(), lag=min_length).astype(bool)
    immobile = deepof.utils.filter_short_true_segments(
        array=immobile, min_length=min_length,
    )

    # Init stationary active and passive subsets with immobile frames
    stationary_lookaround=copy.copy(immobile)
    stationary_nonlookaround=copy.copy(immobile)

    # Detect activity when speed and likelyhood is above a threshold for any of the available bodyparts from the list
    bodyparts=[animal_id+"Left_fhip",animal_id+"Right_fhip", animal_id+"Left_bhip", animal_id+"Right_bhip"]

    # Remove missing bodyparts from list
    for bp in bodyparts:
        if not bp in speed_dframe.keys():
            bodyparts.remove(bp)

    
    nose_activity = (tol_speed < speed_dframe[animal_id+"Nose"]).to_numpy() & (likelihood_dframe[animal_id+"Nose"] > tol_likelihood).to_numpy()

    bparts=[animal_id+"Left_bhip", animal_id+"Right_bhip"]

    body_inactivity = np.array([
        (tol_speed*2 >= speed_dframe[part]).to_numpy() & 
        (likelihood_dframe[part] > tol_likelihood).to_numpy()
        for part in bparts
    ]).all(axis=0)

    bodyparts = bodyparts + [animal_id+"Nose"]

    #helper function to check if distance exists
    def check_distance(mouse_id, ear_part):
        """Returns the correct tuple (ear, nose) or (nose, ear) if present."""
        col1 = (f"{mouse_id}{ear_part}", f"{mouse_id}Nose")
        col2 = (f"{mouse_id}Nose", f"{mouse_id}{ear_part}")
        return col1 if col1 in dist_dframe.columns else col2 if col2 in dist_dframe.columns else None

    # Left ear logic
    left_dist = check_distance(mouse_identity, 'Left_ear')
    if left_dist:
        left_min_dist = 0.9*np.nanmedian(dist_dframe[left_dist])
        left_dist = dist_dframe[left_dist] > left_min_dist

    # Right ear logic
    right_dist = check_distance(mouse_identity, 'Right_ear')
    if right_dist:
        right_min_dist = 0.9*np.nanmedian(dist_dframe[right_dist])
        right_dist = dist_dframe[right_dist] > right_min_dist

    # Frames are Stationary active when immobile and active, stationary passive when immobile and not active + smoothing
    stationary_lookaround = immobile & nose_activity & body_inactivity & right_dist & left_dist & ~close_range
    stationary_nonlookaround = immobile & ~(nose_activity & body_inactivity & right_dist & left_dist & ~close_range)

    stationary_lookaround = deepof.utils.multi_step_paired_smoothing(stationary_lookaround, stationary_nonlookaround, immobile, min_length)

     # Set all Frames that had no speed information in the beginning to False
    stationary_lookaround[nan_pos] = False
    
    return stationary_lookaround


def detect_activity(
    speed_dframe: pd.DataFrame,
    likelihood_dframe: pd.DataFrame,
    tol_speed: float,
    tol_likelihood: float,
    min_length: int,
    center_name: str = "Center",
    animal_id: str = "",
):
    """Return true when the mouse is standing still and either moving (active) or not moving (passive).

    Design considerations:
        Detecting immobility and activity is relatively straightforward by mostly just checking speed thresholds on bodyparts.
        The main problem arises from getting a lot of "flickering" out of the detections, as bodyparts from frame to frame may be
        just above or below that threshold. Respectively most of the detect_activity algorithm is a series of filtering steps to
        alternatingly smooth the predictions and sharpening the edges of predicted behavior. 

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        stationary_active (np.array): True if the animal is standing still and is active, False otherwise
        stationary_passive (np.array): True if the animal is standing still and is passive, False otherwise

    """
    if animal_id != "":
        animal_id += "_"

    # Get frames with undefined speed
    nan_pos = speed_dframe[speed_dframe[animal_id + center_name].isnull()].index.tolist()

    # Detect and smooth frames with mouse being immobile    
    immobile = np.array([False]*len(speed_dframe))
    speed_dframe.interpolate(method='linear', inplace=True)
    immobile = deepof.utils.moving_average((speed_dframe[animal_id + center_name] < tol_speed).to_numpy(), lag=min_length).astype(bool)
    immobile = deepof.utils.filter_short_true_segments(
        array=immobile, min_length=min_length,
    )

    # Init stationary active and passive subsets with immobile frames
    stationary_active=copy.copy(immobile)
    stationary_passive=copy.copy(immobile)

    # Detect activity when speed and likelyhood is above a threshold for any of the available bodyparts from the list
    bodyparts=[animal_id+"Nose",animal_id+"Left_fhip",animal_id+"Right_fhip", animal_id+"Left_bhip", animal_id+"Right_bhip"]

    # Remove missing bodyparts from list
    for bp in bodyparts:
        if not bp in speed_dframe.keys():
            bodyparts.remove(bp)

    # Frame is "active" if any bodyprt from the list is reliably detected and faster than the immobility threshold 
    activity = np.array([
        (tol_speed < speed_dframe[part]).to_numpy() & 
        (likelihood_dframe[part] > tol_likelihood).to_numpy()
        for part in bodyparts
    ]).any(axis=0)

    # Frames are Stationary active when immobile and active, stationary passive when immobile and not active + smoothing
    stationary_active = immobile & activity
    stationary_passive = immobile & ~activity
    
    stationary_active, stationary_passive = deepof.utils.multi_step_paired_smoothing(stationary_active, stationary_passive, immobile, min_length, get_both=True)
    mobile=~(stationary_active + stationary_passive).astype(bool)

    # Set all Frames that had no speed information in the beginning to False
    stationary_active[nan_pos] = False
    stationary_passive[nan_pos] = False
    mobile[nan_pos] = False
    
    return stationary_active, stationary_passive, mobile


def sniff_around(
    speed_dframe: pd.DataFrame,
    likelihood_dframe: pd.DataFrame,
    tol_speed: float,
    tol_likelihood: float,
    center_name: str = "Center",
    animal_id: str = "",
):
    """Return true when the mouse is looking around using simple rules.

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        lookaround (np.array): True if the animal is standing still and looking around, False otherwise

    """
    if animal_id != "":
        animal_id += "_"

    speed = speed_dframe[animal_id + center_name] < tol_speed
    nose_speed = (
        tol_speed < speed_dframe[animal_id + "Nose"]
    )
    nose_likelihood = likelihood_dframe[animal_id + "Nose"] > tol_likelihood

    lookaround = speed & nose_likelihood & nose_speed

    return lookaround


def following_path(
    distance_dframe: pd.DataFrame,
    position_dframe: pd.DataFrame,
    speed_dframe: pd.DataFrame,
    follower: str,
    followed: str,
    frames: int = 20,
    tol: float = 0,
    tol_speed: float = 0,
) -> np.array:
    """Return True if 'follower' is closer than tol to the path that followed has walked over the last specified number of frames.

    For multi animal videos only.

        Args:
            distance_dframe (pandas.DataFrame): distances between bodyparts; generated by the preprocess module
            position_dframe (pandas.DataFrame): position of bodyparts; generated by the preprocess module
            speed_dframe (pandas.DataFrame): speed of body parts over time
            follower (str) identifier for the animal who's following
            followed (str) identifier for the animal who's followed
            frames (int) frames in which to track whether the process consistently occurs,
            tol (float) Maximum distance for which True is returned
            tol_speed (float): Minimum speed for the following mouse


        Returns:
            follow (np.array): boolean sequence, True if conditions are fulfilled, False otherwise

    """
    # Check that follower is close enough to the path that followed has passed though in the last frames
    shift_dict = {
        i: position_dframe[followed + "_Tail_base"].shift(i) for i in range(frames)
    }
    dist_df = pd.DataFrame(
        {
            i: np.linalg.norm(
                position_dframe[follower + "_Nose"] - shift_dict[i], axis=1
            )
            for i in range(frames)
        }
    )

    # Check that the animals are oriented follower's nose -> followed's tail
    right_orient1 = (
        distance_dframe[tuple(os_sorted([follower + "_Nose", followed + "_Tail_base"]))]
        < distance_dframe[
            tuple(os_sorted([follower + "_Tail_base", followed + "_Tail_base"]))
        ]
    )

    right_orient2 = (
        distance_dframe[tuple(os_sorted([follower + "_Nose", followed + "_Tail_base"]))]
        < distance_dframe[tuple(os_sorted([follower + "_Nose", followed + "_Nose"]))]
    )

    # noinspection PyArgumentList
    follow = np.all(
        np.array([(dist_df.min(axis=1) < tol), right_orient1, right_orient2]), axis=0
    )
    speed = (speed_dframe[follower + "_Nose"] > tol_speed).to_numpy()


    return follow & speed


def max_behaviour(
    behaviour_dframe: pd.DataFrame, window_size: int = 10, stepped: bool = False
) -> np.array:
    """Return the most frequent behaviour in a window of window_size frames.

    Args:
        behaviour_dframe (pd.DataFrame): boolean matrix containing occurrence of tagged behaviours per frame in the video
        window_size (int): size of the window to use when computing the maximum behaviour per time slot
        stepped (bool): sliding windows don't overlap if True. False by default

    Returns:
        max_array (np.array): string array with the most common behaviour per instance of the sliding window

    """
    speeds = [col for col in behaviour_dframe.columns if "speed" in col.lower()]

    behaviour_dframe = behaviour_dframe.drop(speeds, axis=1).astype(float)
    win_array = behaviour_dframe.rolling(window_size, center=True).sum()
    if stepped:
        win_array = win_array[::window_size]
    max_array = win_array[1:].idxmax(axis=1)

    return np.array(max_array)


# noinspection PyDefaultArgument
def frame_corners(w, h, corners: dict = {}):
    """Return a dictionary with the corner positions of the video frame.

    Args:
        w (int): width of the frame in pixels
        h (int): height of the frame in pixels
        corners (dict): dictionary containing corners to overwrite

    Returns:
        defaults (dict): dictionary with overwriten parameters. Those not specified in the input retain their default values

    """
    defaults = {
        "downleft": (int(w * 0.3 / 10), int(h / 1.05)),
        "downright": (int(w * 6.5 / 10), int(h / 1.05)),
        "upleft": (int(w * 0.3 / 10), int(h / 20)),
        "upright": (int(w * 6.3 / 10), int(h / 20)),
    }

    for k, v in corners.items():
        defaults[k] = v

    return defaults


# noinspection PyDefaultArgument,PyProtectedMember
#from memory_profiler import profile
#@profile
def supervised_tagging(
    coord_object: coordinates,
    raw_coords: table_dict,
    coords: table_dict,
    dists: table_dict,
    angles: table_dict,
    speeds: table_dict,
    full_features: dict,
    key: str,
    trained_model_path: str = None,
    center: str = "Center",
    params: dict = {},
    run_numba: bool = False,
) -> pd.DataFrame:
    """Output a dataframe with the registered motives per frame.

    If specified, produces a labeled video displaying the information in real time

    Args:
        coord_object (deepof.data.coordinates): coordinates object containing the project information
        raw_coords (deepof.data.table_dict): table_dict with raw coordinates
        coords (deepof.data.table_dict): table_dict with already processed (centered and aligned) coordinates
        dists (deepof.data.table_dict): table_dict with already processed distances
        angles (deepof.data.table_dict): table_dict with already processed angles
        speeds (deepof.data.table_dict): table_dict with already processed speeds
        full_features (dict): A dictionary of aligned kinematics, where the keys are the names of the experimental conditions. The values are the aligned kinematics for each condition.
        key (str): key to the experiment to tag and current set of objects (videos, tables, distances etc.)
        trained_model_path (str): path indicating where all pretrained models are located
        center (str): Body part to center coordinates on. "Center" by default.
        params (dict): dictionary to overwrite the default values of the parameters of the functions that the rule-based pose estimation utilizes. See documentation for details.
        run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)

    Returns:
        tag_df (pandas.DataFrame): table with traits as columns and frames as rows. Each value is a boolean indicating trait detection at a given time

    """
    # Load pre-trained models for ML annotated traits

    with open(
        os.path.join(
            trained_model_path,
            "deepof_supervised",
            "deepof_supervised_huddle_estimator.pkl",
        ),
        "rb",
    ) as est:
        huddle_estimator = pickle.load(est)

    # Extract arena information from coordinates object
    arena_params = coord_object._arena_params[key]
    to_mm_scaling = coord_object._scales[key][3]/coord_object._scales[key][2]
    arena_type = coord_object._arena
    if arena_type.startswith("circular"):
        # Multiply ellipse information (except angle) by scaling factor
        arena_params_scaled= tuple([tuple([x * to_mm_scaling for x in inner]) for inner in arena_params[0:2]] + [arena_params[2]])
    elif arena_type.startswith("polygon"):
        # Multiply set of arena points by scaling factor
        arena_params_scaled= tuple([tuple([x * to_mm_scaling for x in inner]) for inner in arena_params])

    animal_ids = coord_object._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""
               
    #extract various data tables from their Table dicts
    raw_coords = get_dt(raw_coords,key).reset_index(drop=True)
    coords = get_dt(coords,key).reset_index(drop=True)
    dists = get_dt(dists,key).reset_index(drop=True)
    angles = get_dt(angles,key).reset_index(drop=True)
    speeds = get_dt(speeds,key).reset_index(drop=True)
    likelihoods = get_dt(coord_object.get_quality(),key).reset_index(drop=True)

    # Dictionary with motives per frame
    tag_dict = {}

    # Bulk body parts
    main_body = [
        "Left_ear",
        "Right_ear",
        "Spine_1",
        "Center",
        "Spine_2",
        "Left_fhip",
        "Right_fhip",
        "Left_bhip",
        "Right_bhip",
    ]
    main_body = [
        body_part
        for body_part in main_body
        if any(body_part in col[0] for col in coords.columns)
    ]

    #extract mouse normalization information from coordinates object
    mouse_lens={}
    mouse_areas={}
    for _id in animal_ids:
        if _id:
         _id=_id+"_"
        
        #calculate mouse lengths
        backbone=[_id+"Nose",_id+"Spine_1",_id+"Center", _id+"Spine_2", _id+"Tail_base"]

        #remove missing bodyparts from backbone
        for bp in backbone:
            if not bp in raw_coords.keys():
                backbone.remove(bp)

        #calculate overall length of bodypart chain i.e. mouse length
        indices=np.random.choice(np.arange(0, len(raw_coords)), size=np.min([5000, len(raw_coords)]), replace=False)
        if len(backbone)>1:
            mouse_lens_raw=0
            for bp_pos in range(0, len(backbone)-1):
                mouse_lens_raw+=np.apply_along_axis(
                        np.linalg.norm, 1, (
                            raw_coords[backbone[bp_pos+1]].iloc[indices]
                            -raw_coords[backbone[bp_pos]].iloc[indices]
                            )
                        )
            mouse_lens[_id]=np.nanpercentile(mouse_lens_raw,80)
                    
        #assume default mouse length if body parts for length estimation are insufficient
        else:
            mouse_lens[_id]=50
        
        if _id+"full_area" in coord_object._areas[key]:
            mouse_areas[_id]=np.nanpercentile(
                coord_object._areas[key][_id+"full_area"]
                ,80)

    def onebyone_contact(interactors: List, bparts: List):
        """Return a smooth boolean array with 1to1 contacts between two mice."""
        nonlocal raw_coords, animal_ids, params

        try:
            left = interactors[0] + bparts[0]
        except TypeError:
            left = [interactors[0] + "_" + suffix for suffix in bparts[0]]

        try:
            right = interactors[1] + bparts[-1]
        except TypeError:
            right = [interactors[1] + "_" + suffix for suffix in bparts[-1]]

        return deepof.utils.binary_moving_median_numba(
            close_single_contact(
                raw_coords,
                (left if not isinstance(left, list) else right),
                (right if not isinstance(left, list) else left),
                params["close_contact_tol"],
            ),
            lag=params["median_filter_width"],
        )

    def twobytwo_contact(interactors: List, rev: bool):
        """Return a smooth boolean array with side by side contacts between two mice."""
        nonlocal raw_coords, animal_ids, params, mouse_lens
        
        return deepof.utils.binary_moving_median_numba(
            close_double_contact(
            raw_coords,
            #mouse_lens[interactors[0]+"_"],
            interactors[0] + "_Nose",
            interactors[0] + "_Tail_base",
            #mouse_lens[interactors[1]+"_"],
            interactors[1] + "_Nose",
            interactors[1] + "_Tail_base",
            params["side_contact_tol"],
            rev=rev,
            ),
            lag=params["median_filter_width"],
        )
        

    @_suppress_warning(warn_messages=["All-NaN slice encountered"])
    def overall_speed(ovr_speeds, _id, ucond):
        """Return the overall speed of a mouse."""
        bparts = [
            "Center",
            "Spine_1",
            "Spine_2",
            "Nose",
            "Left_ear",
            "Right_ear",
            "Left_fhip",
            "Right_fhip",
            "Left_bhip",
            "Right_bhip",
            "Tail_base",
        ]
        bparts = [
            bpart
            for bpart in bparts
            if bpart
            if any(bpart in col for col in ovr_speeds.columns)
        ]
        array = ovr_speeds[[_id + ucond + bpart for bpart in bparts]]
        avg_speed = np.nanmedian(array[1:], axis=1)
        return np.insert(avg_speed, 0, np.nan, axis=0)

    # Get all animal ID combinations
    animal_pairs = list(combinations(animal_ids, 2))

    if len(animal_ids) >= 2:

        for animal_pair in animal_pairs:
            # Define behaviours that can be computed on the fly from the distance matrix
            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_nose2nose"] = onebyone_contact(
                interactors=animal_pair, bparts=["_Nose"]
            )

            tag_dict[
                f"{animal_pair[0]}_{animal_pair[1]}_sidebyside"
            ] = twobytwo_contact(interactors=animal_pair, rev=False)

            tag_dict[
                f"{animal_pair[0]}_{animal_pair[1]}_sidereside"
            ] = twobytwo_contact(interactors=animal_pair, rev=True)

            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_nose2tail"] = onebyone_contact(
                interactors=animal_pair, bparts=["_Nose", "_Tail_base"]
            )
            tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_nose2tail"] = onebyone_contact(
                interactors=animal_pair, bparts=["_Tail_base", "_Nose"]
            )
            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_nose2body"] = onebyone_contact(
                interactors=animal_pair, bparts=["_Nose", main_body]
            )
            tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_nose2body"] = onebyone_contact(
                interactors=animal_pair, bparts=[main_body, "_Nose"]
            )

            try:
                tag_dict[
                    f"{animal_pair[0]}_{animal_pair[1]}_following"
                ] = deepof.utils.binary_moving_median_numba(
                    following_path(
                        dists,
                        raw_coords,
                        speeds,
                        follower=animal_pair[0],
                        followed=animal_pair[1],
                        frames=params["follow_frames"],
                        tol=params["follow_tol"],
                        tol_speed=params["stationary_threshold"]
                    ),
                    lag=params["median_filter_width"],
                )

                tag_dict[
                    f"{animal_pair[1]}_{animal_pair[0]}_following"
                ] = deepof.utils.binary_moving_median_numba(
                    following_path(
                        dists,
                        raw_coords,
                        speeds,
                        follower=animal_pair[1],
                        followed=animal_pair[0],
                        frames=params["follow_frames"],
                        tol=params["follow_tol"],
                        tol_speed=params["stationary_threshold"],
                    ),
                    lag=params["median_filter_width"],
                )

                # Filter out extremely short segments (Always use numba version, since function is called very often 
                # and its regular version otherwise wastes more time than just compiling it once with numba)
                tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_following"]=deepof.utils.filter_short_true_segments_numba(
                    array=tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_following"], min_length=params["min_follow_frames"],
                )
                tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_following"]=deepof.utils.filter_short_true_segments_numba(
                    array=tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_following"], min_length=params["min_follow_frames"],
                )        


            except KeyError:
                pass

    for _id in animal_ids:
        
        if _id:
            current_features=get_dt(full_features[_id],key)
            close_range = calculate_close_range(dists, _id + undercond, 'Nose', params["side_contact_tol"]) 
        else:
            current_features=get_dt(full_features,key)
            close_range = np.zeros(len(dists), dtype=int)
        

        tag_dict[_id + undercond + "climb_arena"] = deepof.utils.binary_moving_median_numba(
            climb_arena(
                arena_type,
                arena_params_scaled,
                raw_coords,
                params["climb_tol"],
                _id + undercond,
                mouse_lens[_id + undercond],
                run_numba=run_numba,
            ).to_numpy(),
            lag=params["median_filter_width"]
        )


        tag_dict[_id + undercond + "sniff_arena"] = deepof.utils.binary_moving_median_numba(
            sniff_object(
                speed_dframe=speeds,
                arena_type=arena_type,
                arena=arena_params_scaled,
                pos_dict=raw_coords,
                tol=params["sniff_arena_tol"],
                tol_speed=params["stationary_threshold"],
                nose=_id + undercond + "Nose",
                center_name=center,
                s_object="arena",
                animal_id=_id,
                run_numba=run_numba,
            ).to_numpy(),
            lag=params["median_filter_width"]
        )

        tag_dict[_id + undercond + "immobility"], _ = cowering(
            current_features,
            huddle_estimator=huddle_estimator,
            animal_id=_id + undercond,
            median_filter_width = params["median_filter_width"],
            min_immobility = params["min_immobility"],
            max_immobility = 0,#params["max_immobility"],
        )
        #detect immobility and active / passive behavior
        tag_dict[_id + undercond + "stat_lookaround"] = stationary_lookaround(
        speeds,
        dists,
        _id + undercond,
        close_range,
        likelihoods,
        params["stationary_threshold"],
        params["nose_likelihood"],
        params["min_follow_frames"],
        center_name=center,
        animal_id=_id,
        )

        #tag_dict[_id + undercond + "digging"] = digging(
        #speeds,
        #dists,
        #_id + undercond,
        #close_range,
        #likelihoods,
        #params["stationary_threshold"],
        #params["nose_likelihood"],
        #params["min_follow_frames"],
        #center_name=center,
        #animal_id=_id,
        #)
        
        tag_dict[_id + undercond + "stat_active"], tag_dict[_id + undercond + "stat_passive"], tag_dict[_id + undercond + "moving"] = detect_activity(
        speeds,
        likelihoods,
        params["stationary_threshold"],
        params["nose_likelihood"],
        params["min_follow_frames"],
        center_name=center,
        animal_id=_id,
        )

        tag_dict[_id + undercond + "sniffing"] = sniff_around(
            speeds,
            likelihoods,
            params["stationary_threshold"],
            params["nose_likelihood"],
            center_name=center,
            animal_id=_id,
        )
        # NOTE: It's important that speeds remain the last columns.
        # Preprocessing for weakly supervised autoencoders relies on this
        tag_dict[_id + undercond + "speed"] = overall_speed(speeds, _id, undercond)




    tag_df = pd.DataFrame(tag_dict).fillna(0).astype(float)

    return tag_df


def calculate_close_range(df, mouse_id, bodypart, threshold):
    target = f"{mouse_id}{bodypart}"
    relevant_cols = []
    
    for col in df.columns:
        part1, part2 = col
        if part1 == target or part2 == target:
            # Determine which part is the other one
            other_part = part2 if part1 == target else part1
            if not (mouse_id in other_part):
                relevant_cols.append(col)
    
    if not relevant_cols:
        return np.zeros(len(df), dtype=int)
    
    # Check rows where any relevant column is below the threshold
    proximity_mask = (df[relevant_cols] < threshold).any(axis=1)
    return proximity_mask.astype(int).to_numpy()
    

def tagged_video_output(
    coordinates: coordinates,
    tag_dict: table_dict,
    video_output: Union[str, List[str]] = "all",
    frame_limit: int = None,
    debug: bool = False,
    n_jobs: int = 1,
    params: dict = None,
):  # pragma: no cover
    """Output annotated videos.

    Args:
        coordinates: Coordinates object.
        tag_dict: Dictionary with supervised annotations to render on the video.
        video_output: List with the names of the videos to render, or 'all' (default) to render all videos.
        frame_limit: Number of frames to render per output video. If None, all frames are rendered.
        debug: If True, debugging information, such as arena fits and processed tracklets, are displayed.
        n_jobs: Number of jobs to run in parallel.
        params (dict): dictionary to overwrite the default values of the hyperparameters of the functions that the supervised pose estimation utilizes.
    """

    def output_video(key):
        """Output a single annotated video. Enclosed in a function to enable parallelization."""
        deepof.visuals_utils.annotate_video(
            coordinates,
            supervised_annotations=tag_dict,
            key=key,
            debug=debug,
            frame_limit=frame_limit,
            params=params,
        )
        pbar.update(1)

    if isinstance(video_output, list):
        vid_idxs = video_output
    elif video_output == "all":
        vid_idxs = list(coordinates._tables.keys())
    else:
        raise AttributeError(
            "Video output must be either 'all' or a list with the names of the videos to render"
        )

    pbar = tqdm(total=len(vid_idxs))
    with parallel_backend("threading", n_jobs=n_jobs):
        Parallel()(delayed(output_video)(key) for key in vid_idxs)
    pbar.close()


if __name__ == "__main__":
    # Ignore warnings with no downstream effect
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
