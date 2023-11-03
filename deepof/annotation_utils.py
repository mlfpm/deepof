# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Functions and general utilities for supervised pose estimation. See documentation for details."""

import os
import pickle
import warnings
from joblib import delayed, Parallel, parallel_backend
from typing import Any, List, NewType, Union
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from itertools import combinations

import numpy as np
import pandas as pd
import regex as re
import sklearn.pipeline

import deepof.utils
import deepof.post_hoc

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def close_single_contact(
    pos_dframe: pd.DataFrame,
    left: str,
    right: str,
    tol: float,
    arena_abs: int,
    arena_rel: int,
) -> np.array:
    """Return a boolean array that's True if the specified body parts are closer than tol.

    Args:
        pos_dframe (pandas.DataFrame): DLC output as pandas.DataFrame; only applicable to two-animal experiments.
        left (string): First member of the potential contact
        right (string): Second member of the potential contact
        tol (float): maximum distance for which a contact is reported
        arena_abs (int): length in mm of the diameter of the real arena
        arena_rel (int): length in pixels of the diameter of the arena in the video

    Returns:
        contact_array (np.array): True if the distance between the two specified points is less than tol, False otherwise

    """
    close_contact = None

    if isinstance(right, str):
        close_contact = (
            np.linalg.norm(pos_dframe[left] - pos_dframe[right], axis=1) * arena_abs
        ) / arena_rel < tol

    elif isinstance(right, list):
        close_contact = np.any(
            [
                (np.linalg.norm(pos_dframe[left] - pos_dframe[r], axis=1) * arena_abs)
                / arena_rel
                < tol
                for r in right
            ],
            axis=0,
        )

    return close_contact


def close_double_contact(
    pos_dframe: pd.DataFrame,
    left1: str,
    left2: str,
    right1: str,
    right2: str,
    tol: float,
    arena_abs: int,
    arena_rel: int,
    rev: bool = False,
) -> np.array:
    """Return a boolean array that's True if the specified body parts are closer than tol.

    Parameters:
        pos_dframe (pandas.DataFrame): DLC output as pandas.DataFrame; only applicable to two-animal experiments.
        left1 (string): First contact point of animal 1
        left2 (string): Second contact point of animal 1
        right1 (string): First contact point of animal 2
        right2 (string): Second contact point of animal 2
        tol (float): maximum distance for which a contact is reported
        arena_abs (int): length in mm of the diameter of the real arena
        arena_rel (int): length in pixels of the diameter of the arena in the video
        rev (bool): reverses the default behaviour (nose2tail contact for both mice)

    Returns:
        double_contact (np.array): True if the distance between the two specified points is less than tol, False otherwise

    """
    if rev:
        double_contact = (
            (np.linalg.norm(pos_dframe[right1] - pos_dframe[left2], axis=1) * arena_abs)
            / arena_rel
            < tol
        ) & (
            (np.linalg.norm(pos_dframe[right2] - pos_dframe[left1], axis=1) * arena_abs)
            / arena_rel
            < tol
        )

    else:
        double_contact = (
            (np.linalg.norm(pos_dframe[right1] - pos_dframe[left1], axis=1) * arena_abs)
            / arena_rel
            < tol
        ) & (
            (np.linalg.norm(pos_dframe[right2] - pos_dframe[left2], axis=1) * arena_abs)
            / arena_rel
            < tol
        )

    return double_contact


def rotate(origin, point, ang):
    """Auxiliar function to climb_wall and sniff_object. Rotates x,y coordinates over a pivot."""
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


def climb_wall(
    arena_type: str,
    arena: np.array,
    pos_dict: pd.DataFrame,
    tol: float,
    nose: str,
    centered_data: bool = False,
) -> np.array:
    """Return True if the specified mouse is climbing the wall.

    Args:
        arena_type (str): arena type; must be one of ['polygonal-manual', 'circular-autodetect']
        arena (np.array): contains arena location and shape details
        pos_dict (table_dict): position over time for all videos in a project
        tol (float): minimum tolerance to report a hit
        nose (str): indicates the name of the body part representing the nose of the selected animal
        centered_data (bool): indicates whether the input data is centered

    Returns:
        climbing (np.array): boolean array. True if selected animal is climbing the walls of the arena

    """
    nose = pos_dict[nose]

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

        climbing = np.array(
            [not Polygon(arena).buffer(tol).contains(Point(n)) for n in nose.values]
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
):
    """Return True if the specified mouse is sniffing an object.

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time.
        arena_type (str): arena type; must be one of ['polygonal-manual', 'circular-autodetect'].
        arena (np.array): contains arena location and shape details.
        pos_dict (table_dict): position over time for all videos in a project.
        tol (float): minimum tolerance to report a hit.
        center_name (str): Body part to center coordinates on. "Center" by default.
        nose (str): indicates the name of the body part representing the nose of the selected animal.
        centered_data (bool): indicates whether the input data is centered.
        s_object (str): indicates the object to sniff. Must be one of ['arena', 'object'].
        animal_id (str): indicates the animal to sniff. Must be one of animal_ids.
        tol_speed (float): minimum speed to report a hit.

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

            nosing_min = np.array(
                [
                    not Polygon(arena).buffer(-tol).contains(Point(n))
                    for n in nose.values
                ]
            )
            nosing_max = np.array(
                [not Polygon(arena).buffer(tol).contains(Point(n)) for n in nose.values]
            )

        # noinspection PyUnboundLocalVariable
        nosing = nosing_min & (~nosing_max)

    else:
        raise NotImplementedError

    speed = speed_dframe[animal_id + center_name] < tol_speed
    sniffing = nosing & speed

    return sniffing


def huddle(
    X_huddle: np.ndarray,
    huddle_estimator: sklearn.pipeline.Pipeline,
    animal_id: str = "",
) -> np.array:
    """Return true when the mouse is huddling a pretrained model.

    Args:
        X_huddle (pandas.DataFrame): mouse features over time.
        huddle_estimator (sklearn.pipeline.Pipeline): pre-trained model to predict feature occurrence.
        animal_id (str): indicates the animal to sniff. Must be one of animal_ids.

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

    # Concatenate all relevant data frames and predict using the pre-trained estimator
    X_mask = np.isnan(X_huddle).mean(axis=1) == 1
    y_huddle = huddle_estimator.predict(
        StandardScaler().fit_transform(np.nan_to_num(X_huddle))
    )
    y_huddle[X_mask] = np.nan
    return y_huddle


def look_around(
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
        speed_dframe[animal_id + center_name] < speed_dframe[animal_id + "Nose"]
    )
    nose_likelihood = likelihood_dframe[animal_id + "Nose"] > tol_likelihood

    lookaround = speed & nose_likelihood & nose_speed

    return lookaround


def following_path(
    distance_dframe: pd.DataFrame,
    position_dframe: pd.DataFrame,
    follower: str,
    followed: str,
    frames: int = 20,
    tol: float = 0,
) -> np.array:
    """Return True if 'follower' is closer than tol to the path that followed has walked over the last specified number of frames.

    For multi animal videos only.

        Args:
            distance_dframe (pandas.DataFrame): distances between bodyparts; generated by the preprocess module
            position_dframe (pandas.DataFrame): position of bodyparts; generated by the preprocess module
            follower (str) identifier for the animal who's following
            followed (str) identifier for the animal who's followed
            frames (int) frames in which to track whether the process consistently occurs,
            tol (float) Maximum distance for which True is returned

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
        distance_dframe[tuple(sorted([follower + "_Nose", followed + "_Tail_base"]))]
        < distance_dframe[
            tuple(sorted([follower + "_Tail_base", followed + "_Tail_base"]))
        ]
    )

    right_orient2 = (
        distance_dframe[tuple(sorted([follower + "_Nose", followed + "_Tail_base"]))]
        < distance_dframe[tuple(sorted([follower + "_Nose", followed + "_Nose"]))]
    )

    # noinspection PyArgumentList
    follow = np.all(
        np.array([(dist_df.min(axis=1) < tol), right_orient1, right_orient2]), axis=0
    )

    return follow


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
def get_hparameters(hparams: dict = {}) -> dict:
    """Return the most frequent behaviour in a window of window_size frames.

    Args:
        hparams (dict): dictionary containing hyperparameters to overwrite

    Returns:
        defaults (dict): dictionary with overwritten parameters. Those not specified in the input retain their default values

    """
    defaults = {
        "speed_pause": 5,
        "climb_tol": 10,
        "close_contact_tol": 25,
        "side_contact_tol": 45,
        "follow_frames": 10,
        "follow_tol": 5,
        "huddle_speed": 2,
        "nose_likelihood": 0.85,
    }

    for k, v in hparams.items():
        defaults[k] = v

    return defaults


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
def supervised_tagging(
    coord_object: coordinates,
    raw_coords: table_dict,
    coords: table_dict,
    dists: table_dict,
    speeds: table_dict,
    full_features: dict,
    video: str,
    trained_model_path: str = None,
    center: str = "Center",
    params: dict = {},
) -> pd.DataFrame:
    """Output a dataframe with the registered motives per frame.

    If specified, produces a labeled video displaying the information in real time

    Args:
        coord_object (deepof.data.coordinates): coordinates object containing the project information
        raw_coords (deepof.data.table_dict): table_dict with raw coordinates
        coords (deepof.data.table_dict): table_dict with already processed (centered and aligned) coordinates
        dists (deepof.data.table_dict): table_dict with already processed distances
        speeds (deepof.data.table_dict): table_dict with already processed speeds
        full_features (dict): dictionary with
        video (str): string name of the experiment to tag
        trained_model_path (str): path indicating where all pretrained models are located
        center (str): Body part to center coordinates on. "Center" by default.
        params (dict): dictionary to overwrite the default values of the parameters of the functions that the rule-based pose estimation utilizes. See documentation for details.

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

    # Extract useful information from coordinates object
    tracks = list(coord_object._tables.keys())
    vid_index = coord_object._videos.index(video)

    arena_params = coord_object._arena_params[vid_index]
    arena_type = coord_object._arena

    animal_ids = coord_object._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    try:
        vid_name = re.findall("(.*?)DLC", tracks[vid_index])[0]
    except IndexError:
        vid_name = tracks[vid_index]

    raw_coords = raw_coords[vid_name].reset_index(drop=True)
    coords = coords[vid_name].reset_index(drop=True)
    dists = dists[vid_name].reset_index(drop=True)

    # angs = angs[vid_name].reset_index(drop=True)
    speeds = speeds[vid_name].reset_index(drop=True)
    likelihoods = coord_object.get_quality()[vid_name].reset_index(drop=True)
    arena_abs = coord_object._scales[vid_index][-1]
    arena_rel = coord_object._scales[vid_index][-2]

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

    def onebyone_contact(interactors: List, bparts: List):
        """Return a smooth boolean array with 1to1 contacts between two mice."""
        nonlocal raw_coords, animal_ids, params, arena_abs, arena_params

        try:
            left = interactors[0] + bparts[0]
        except TypeError:
            left = [interactors[0] + "_" + suffix for suffix in bparts[0]]

        try:
            right = interactors[1] + bparts[-1]
        except TypeError:
            right = [interactors[1] + "_" + suffix for suffix in bparts[-1]]

        return deepof.utils.smooth_boolean_array(
            close_single_contact(
                raw_coords,
                (left if not isinstance(left, list) else right),
                (right if not isinstance(left, list) else left),
                params["close_contact_tol"],
                arena_abs,
                arena_rel,
            )
        )

    def twobytwo_contact(interactors: List, rev: bool):
        """Return a smooth boolean array with side by side contacts between two mice."""
        nonlocal raw_coords, animal_ids, params, arena_abs, arena_params
        return deepof.utils.smooth_boolean_array(
            close_double_contact(
                raw_coords,
                interactors[0] + "_Nose",
                interactors[0] + "_Tail_base",
                interactors[1] + "_Nose",
                interactors[1] + "_Tail_base",
                params["side_contact_tol"],
                rev=rev,
                arena_abs=arena_abs,
                arena_rel=arena_rel,
            )
        )

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
                ] = deepof.utils.smooth_boolean_array(
                    following_path(
                        dists,
                        raw_coords,
                        follower=animal_pair[0],
                        followed=animal_pair[1],
                        frames=params["follow_frames"],
                        tol=params["follow_tol"],
                    )
                )

                tag_dict[
                    f"{animal_pair[1]}_{animal_pair[0]}_following"
                ] = deepof.utils.smooth_boolean_array(
                    following_path(
                        dists,
                        raw_coords,
                        follower=animal_pair[1],
                        followed=animal_pair[0],
                        frames=params["follow_frames"],
                        tol=params["follow_tol"],
                    )
                )
            except KeyError:
                pass

    for _id in animal_ids:
        tag_dict[_id + undercond + "climbing"] = deepof.utils.smooth_boolean_array(
            climb_wall(
                arena_type,
                arena_params,
                raw_coords,
                params["climb_tol"],
                _id + undercond + "Nose",
            )
        )
        tag_dict[_id + undercond + "sniffing"] = deepof.utils.smooth_boolean_array(
            sniff_object(
                speed_dframe=speeds,
                arena_type=arena_type,
                arena=arena_params,
                pos_dict=raw_coords,
                tol=params["climb_tol"],
                tol_speed=params["huddle_speed"],
                nose=_id + undercond + "Nose",
                center_name=center,
                s_object="arena",
                animal_id=_id,
            )
        )

        tag_dict[_id + undercond + "huddle"] = deepof.utils.smooth_boolean_array(
            huddle(
                (full_features[_id][vid_name] if _id else full_features[vid_name]),
                huddle_estimator=huddle_estimator,
                animal_id=_id + undercond,
            )
        )
        tag_dict[_id + undercond + "lookaround"] = look_around(
            speeds,
            likelihoods,
            params["huddle_speed"],
            params["nose_likelihood"],
            center_name=center,
            animal_id=_id,
        )
        # NOTE: It's important that speeds remain the last columns.
        # Preprocessing for weakly supervised autoencoders relies on this
        tag_dict[_id + undercond + "speed"] = overall_speed(speeds, _id, undercond)

    tag_df = pd.DataFrame(tag_dict).fillna(0).astype(float)

    return tag_df


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

    def output_video(idx):
        """Output a single annotated video. Enclosed in a function to enable parallelization."""
        deepof.visuals.annotate_video(
            coordinates,
            tag_dict=tag_dict[idx],
            vid_index=list(coordinates._tables.keys()).index(idx),
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
