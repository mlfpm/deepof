# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for supervised pose estimation. See documentation for details

"""

import os
import pickle
import warnings
from typing import Any, List, NewType

import cv2
import numpy as np
import pandas as pd
import regex as re
import sklearn.pipeline

import deepof.utils

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
    """Returns a boolean array that's True if the specified body parts are closer than tol.

    Parameters:
        - pos_dframe (pandas.DataFrame): DLC output as pandas.DataFrame; only applicable
        to two-animal experiments.
        - left (string): First member of the potential contact
        - right (string): Second member of the potential contact
        - tol (float): maximum distance for which a contact is reported
        - arena_abs (int): length in mm of the diameter of the real arena
        - arena_rel (int): length in pixels of the diameter of the arena in the video

    Returns:
        - contact_array (np.array): True if the distance between the two specified points
        is less than tol, False otherwise"""

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
    """Returns a boolean array that's True if the specified body parts are closer than tol.

    Parameters:
        - pos_dframe (pandas.DataFrame): DLC output as pandas.DataFrame; only applicable
        to two-animal experiments.
        - left1 (string): First contact point of animal 1
        - left2 (string): Second contact point of animal 1
        - right1 (string): First contact point of animal 2
        - right2 (string): Second contact point of animal 2
        - tol (float): maximum distance for which a contact is reported
        - arena_abs (int): length in mm of the diameter of the real arena
        - arena_rel (int): length in pixels of the diameter of the arena in the video
        - rev (bool): reverses the default behaviour (nose2tail contact for both mice)

    Returns:
        - double_contact (np.array): True if the distance between the two specified points
        is less than tol, False otherwise"""

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
    """Auxiliar function to climb_wall and sniff_object. Rotates x,y coordinates over a pivot"""

    ox, oy = origin
    px, py = point

    qx = ox + np.cos(ang) * (px - ox) - np.sin(ang) * (py - oy)
    qy = oy + np.sin(ang) * (px - ox) + np.cos(ang) * (py - oy)
    return qx, qy


def outside_ellipse(x, y, e_center, e_axes, e_angle, threshold=0.0):
    """Auxiliar function to climb_wall and sniff_object. Returns True if the passed x, y coordinates
    are outside the ellipse denoted by e_center, e_axes and e_angle, with a certain threshold"""

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
    """Returns True if the specified mouse is climbing the wall

    Parameters:
        - arena_type (str): arena type; must be one of ['circular']
        - arena (np.array): contains arena location and shape details
        - pos_dict (table_dict): position over time for all videos in a project
        - tol (float): minimum tolerance to report a hit
        - nose (str): indicates the name of the body part representing the nose of
        the selected animal
        - arena_dims (int): indicates radius of the real arena in mm
        - centered_data (bool): indicates whether the input data is centered

    Returns:
        - climbing (np.array): boolean array. True if selected animal
        is climbing the walls of the arena"""

    nose = pos_dict[nose]

    if arena_type == "circular":
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

    else:
        raise NotImplementedError("Supported values for arena_type are ['circular']")

    return climbing


def sniff_object(
    speed_dframe: pd.DataFrame,
    arena_type: str,
    arena: np.array,
    pos_dict: pd.DataFrame,
    tol: float,
    tol_speed: float,
    nose: str,
    centered_data: bool = False,
    s_object: str = "arena",
    animal_id: str = "",
):
    """Returns True if the specified mouse is sniffing an object

    Parameters:
        - speed_dframe (pandas.DataFrame): speed of body parts over time
        - arena_type (str): arena type; must be one of ['circular']
        - arena (np.array): contains arena location and shape details
        - pos_dict (table_dict): position over time for all videos in a project
        - tol (float): minimum tolerance to report a hit
        - nose (str): indicates the name of the body part representing the nose of
        the selected animal
        - arena_dims (int): indicates radius of the real arena in mm
        - centered_data (bool): indicates whether the input data is centered
        - object (str): indicates the object that the animal is sniffing.
        Can be one of ['arena', 'partner']

    Returns:
        - sniffing (np.array): boolean array. True if selected animal
        is sniffing the selected object"""

    nose, nosing = pos_dict[nose], True

    if animal_id != "":
        animal_id += "_"

    if s_object == "arena":
        if arena_type == "circular":
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
            nosing = nosing_min & (~nosing_max)

    else:
        raise NotImplementedError

    speed = speed_dframe[animal_id + "Center"] < tol_speed
    sniffing = nosing & speed

    return sniffing


def huddle(
    pos_dframe: pd.DataFrame,
    speed_dframe: pd.DataFrame,
    huddle_estimator: sklearn.pipeline.Pipeline,
) -> np.array:
    """Returns true when the mouse is huddling a pretrained model.

    Parameters:
        - pos_dframe (pandas.DataFrame): position of body parts over time
        - speed_dframe (pandas.DataFrame): speed of body parts over time
        - huddle_estimator (sklearn.pipeline.Pipeline): pre-trained model to predict feature occurrence

    Returns:
        y_huddle (np.array): 1 if the animal is huddling, 0 otherwise
    """

    # Concatenate all relevant data frames and predict using the pre-trained estimator
    X_huddle = pd.concat([pos_dframe, speed_dframe], axis=1).to_numpy()
    y_huddle = huddle_estimator.predict(X_huddle)

    return y_huddle


def dig(
    pos_dframe: pd.DataFrame,
    speed_dframe: pd.DataFrame,
    dig_estimator: sklearn.pipeline.Pipeline,
):
    """Returns true when the mouse is digging using a pretrained model.

    Parameters:
        - pos_dframe (pandas.DataFrame): position of body parts over time
        - speed_dframe (pandas.DataFrame): speed of body parts over time
        - dig_estimator (sklearn.pipeline.Pipeline): pre-trained model to predict feature occurrence

    Returns:
        dig (np.array): True if the animal is digging, False otherwise
    """

    # Concatenate all relevant data frames and predict using the pre-trained estimator
    X_dig = pd.concat([pos_dframe, speed_dframe], axis=1).to_numpy()
    y_dig = dig_estimator.predict(X_dig)

    return y_dig


def look_around(
    speed_dframe: pd.DataFrame,
    likelihood_dframe: pd.DataFrame,
    tol_speed: float,
    tol_likelihood: float,
    animal_id: str = "",
):
    """Returns true when the mouse is looking around using simple rules.

    Parameters:
        - speed_dframe (pandas.DataFrame): speed of body parts over time
        - likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time,
        as directly obtained from DeepLabCut
        - tol_speed (float): Maximum tolerated speed for the center of the mouse
        - tol_likelihood (float): Maximum tolerated likelihood for the nose (if the animal
        is digging, the nose is momentarily occluded).

    Returns:
        lookaround (np.array): True if the animal is standing still and looking around, False otherwise
    """

    if animal_id != "":
        animal_id += "_"

    speed = speed_dframe[animal_id + "Center"] < tol_speed
    nose_speed = speed_dframe[animal_id + "Center"] < speed_dframe[animal_id + "Nose"]
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
    """For multi animal videos only. Returns True if 'follower' is closer than tol to the path that
    followed has walked over the last specified number of frames

        Parameters:
            - distance_dframe (pandas.DataFrame): distances between bodyparts; generated by the preprocess module
            - position_dframe (pandas.DataFrame): position of bodyparts; generated by the preprocess module
            - follower (str) identifier for the animal who's following
            - followed (str) identifier for the animal who's followed
            - frames (int) frames in which to track whether the process consistently occurs,
            - tol (float) Maximum distance for which True is returned

        Returns:
            - follow (np.array): boolean sequence, True if conditions are fulfilled, False otherwise"""

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
        np.array([(dist_df.min(axis=1) < tol), right_orient1, right_orient2]),
        axis=0,
    )

    return follow


def max_behaviour(
    behaviour_dframe: pd.DataFrame, window_size: int = 10, stepped: bool = False
) -> np.array:
    """Returns the most frequent behaviour in a window of window_size frames

    Parameters:
            - behaviour_dframe (pd.DataFrame): boolean matrix containing occurrence
            of tagged behaviours per frame in the video
            - window_size (int): size of the window to use when computing
            the maximum behaviour per time slot
            - stepped (bool): sliding windows don't overlap if True. False by default

    Returns:
        - max_array (np.array): string array with the most common behaviour per instance
        of the sliding window"""

    speeds = [col for col in behaviour_dframe.columns if "speed" in col.lower()]

    behaviour_dframe = behaviour_dframe.drop(speeds, axis=1).astype(float)
    win_array = behaviour_dframe.rolling(window_size, center=True).sum()
    if stepped:
        win_array = win_array[::window_size]
    max_array = win_array[1:].idxmax(axis=1)

    return np.array(max_array)


# noinspection PyDefaultArgument
def get_hparameters(hparams: dict = {}) -> dict:
    """Returns the most frequent behaviour in a window of window_size frames

    Parameters:
        - hparams (dict): dictionary containing hyperparameters to overwrite

    Returns:
        - defaults (dict): dictionary with overwritten parameters. Those not
        specified in the input retain their default values"""

    defaults = {
        "speed_pause": 5,
        "climb_tol": 10,
        "close_contact_tol": 35,
        "side_contact_tol": 80,
        "follow_frames": 10,
        "follow_tol": 5,
        "huddle_forward": 15,
        "huddle_speed": 2,
        "nose_likelihood": 0.85,
        "fps": 24,
    }

    for k, v in hparams.items():
        defaults[k] = v

    return defaults


# noinspection PyDefaultArgument
def frame_corners(w, h, corners: dict = {}):
    """Returns a dictionary with the corner positions of the video frame

    Parameters:
        - w (int): width of the frame in pixels
        - h (int): height of the frame in pixels
        - corners (dict): dictionary containing corners to overwrite

    Returns:
        - defaults (dict): dictionary with overwriten parameters. Those not
        specified in the input retain their default values"""

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
    angs: table_dict,
    speeds: table_dict,
    video: str,
    trained_model_path: str = None,
    params: dict = {},
) -> pd.DataFrame:
    """Outputs a dataframe with the registered motives per frame. If specified, produces a labeled
    video displaying the information in real time

    Parameters:
        - coord_object (deepof.data.coordinates): coordinates object containing the project information
        - raw_coords (deepof.data.table_dict): table_dict with raw coordinates
        - coords (deepof.data.table_dict): table_dict with already processed (centered and aligned) coordinates
        - dists (deepof.data.table_dict): table_dict with already processed distances
        - angs (deepof.data.table_dict): table_dict with already processed angles
        - speeds (deepof.data.table_dict): table_dict with already processed speeds
        - video (str): string name of the experiment to tag
        - trained_model_path (str): path indicating where all pretrained models are located
        - params (dict): dictionary to overwrite the default values of the parameters of the functions
        that the rule-based pose estimation utilizes. See documentation for details.

    Returns:
        - tag_df (pandas.DataFrame): table with traits as columns and frames as rows. Each
        value is a boolean indicating trait detection at a given time"""

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
    with open(
        os.path.join(
            trained_model_path,
            "deepof_supervised",
            "deepof_supervised_dig_estimator.pkl",
        ),
        "rb",
    ) as est:
        dig_estimator = pickle.load(est)

    # Extract useful information from coordinates object
    tracks = list(coord_object._tables.keys())
    vid_index = coord_object._videos.index(video)

    arena_params = coord_object._arena_params[vid_index]
    arena_type = coord_object._arena

    params = get_hparameters(params)
    animal_ids = coord_object._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    try:
        vid_name = re.findall("(.*)DLC", tracks[vid_index])[0]
    except IndexError:
        vid_name = tracks[vid_index]

    raw_coords = raw_coords[vid_name].reset_index(drop=True)
    coords = coords[vid_name].reset_index(drop=True)
    dists = dists[vid_name].reset_index(drop=True)
    # angs = angs[vid_name].reset_index(drop=True)
    speeds = speeds[vid_name].reset_index(drop=True)
    likelihoods = coord_object.get_quality()[vid_name].reset_index(drop=True)
    arena_abs = coord_object.get_arenas[1][0]

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

    def onebyone_contact(bparts: List):
        """Returns a smooth boolean array with 1to1 contacts between two mice"""
        nonlocal raw_coords, animal_ids, params, arena_abs, arena_params

        try:
            left = animal_ids[0] + bparts[0]
        except TypeError:
            left = [animal_ids[0] + "_" + suffix for suffix in bparts[0]]

        try:
            right = animal_ids[1] + bparts[-1]
        except TypeError:
            right = [animal_ids[1] + "_" + suffix for suffix in bparts[-1]]

        return deepof.utils.smooth_boolean_array(
            close_single_contact(
                raw_coords,
                (left if not isinstance(left, list) else right),
                (right if not isinstance(left, list) else left),
                params["close_contact_tol"],
                arena_abs,
                arena_params[1][1],
            )
        )

    def twobytwo_contact(rev):
        """Returns a smooth boolean array with side by side contacts between two mice"""

        nonlocal raw_coords, animal_ids, params, arena_abs, arena_params
        return deepof.utils.smooth_boolean_array(
            close_double_contact(
                raw_coords,
                animal_ids[0] + "_Nose",
                animal_ids[0] + "_Tail_base",
                animal_ids[1] + "_Nose",
                animal_ids[1] + "_Tail_base",
                params["side_contact_tol"],
                rev=rev,
                arena_abs=arena_abs,
                arena_rel=arena_params[1][1],
            )
        )

    def overall_speed(ovr_speeds, _id, ucond):
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
        array = ovr_speeds[[_id + ucond + bpart for bpart in bparts]]
        avg_speed = np.nanmedian(array[1:], axis=1)
        return np.insert(avg_speed, 0, np.nan, axis=0)

    if len(animal_ids) == 2:
        # Define behaviours that can be computed on the fly from the distance matrix
        tag_dict["nose2nose"] = onebyone_contact(bparts=["_Nose"])

        tag_dict["sidebyside"] = twobytwo_contact(rev=False)

        tag_dict["sidereside"] = twobytwo_contact(rev=True)

        tag_dict[animal_ids[0] + "_nose2tail"] = onebyone_contact(
            bparts=["_Nose", "_Tail_base"]
        )
        tag_dict[animal_ids[1] + "_nose2tail"] = onebyone_contact(
            bparts=["_Tail_base", "_Nose"]
        )
        tag_dict[animal_ids[0] + "_nose2body"] = onebyone_contact(
            bparts=[
                "_Nose",
                main_body,
            ]
        )
        tag_dict[animal_ids[1] + "_nose2body"] = onebyone_contact(
            bparts=[
                main_body,
                "_Nose",
            ]
        )

        for _id in animal_ids:
            tag_dict[_id + "_following"] = deepof.utils.smooth_boolean_array(
                following_path(
                    dists,
                    raw_coords,
                    follower=_id,
                    followed=[i for i in animal_ids if i != _id][0],
                    frames=params["follow_frames"],
                    tol=params["follow_tol"],
                )
            )

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
                speeds,
                arena_type,
                arena_params,
                raw_coords,
                params["climb_tol"],
                params["huddle_speed"],
                _id + undercond + "Nose",
                s_object="arena",
                animal_id=_id,
            )
        )
        tag_dict[_id + undercond + "huddle"] = huddle(
            coords.loc[  # Filter coordinates to keep only the current animal
                :,
                [
                    col
                    for col in coords.columns
                    if col in deepof.utils.filter_columns(coords.columns, _id)
                ],
            ],
            speeds.loc[  # Filter speeds to keep only the current animal
                :,
                [
                    col
                    for col in speeds.columns
                    if col in deepof.utils.filter_columns(speeds.columns, _id)
                ],
            ],
            huddle_estimator,
        )
        tag_dict[_id + undercond + "dig"] = dig(
            coords.loc[  # Filter coordinates to keep only the current animal
                :,
                [
                    col
                    for col in coords.columns
                    if col in deepof.utils.filter_columns(coords.columns, _id)
                ],
            ],
            speeds.loc[  # Filter speeds to keep only the current animal
                :,
                [
                    col
                    for col in speeds.columns
                    if col in deepof.utils.filter_columns(speeds.columns, _id)
                ],
            ],
            dig_estimator,
        )
        tag_dict[_id + undercond + "lookaround"] = deepof.utils.smooth_boolean_array(
            look_around(
                speeds,
                likelihoods,
                params["huddle_speed"],
                params["nose_likelihood"],
                animal_id=_id,
            )
        )
        # NOTE: It's important that speeds remain the last columns.
        # Preprocessing for weakly supervised autoencoders relies on this
        tag_dict[_id + undercond + "speed"] = overall_speed(speeds, _id, undercond)

    tag_df = pd.DataFrame(tag_dict).fillna(0).astype(float)

    return tag_df


def tag_annotated_frames(
    frame,
    font,
    frame_speeds,
    animal_ids,
    corners,
    tag_dict,
    fnum,
    undercond,
    hparams,
    arena,
    debug,
    coords,
):
    """Helper function for annotate_video. Annotates a given frame with on-screen information
    about the recognised patterns"""

    arena, w, h = arena

    def write_on_frame(text, pos, col=(255, 255, 255)):
        """Partial closure over cv2.putText to avoid code repetition"""
        return cv2.putText(frame, text, pos, font, 0.75, col, 2)

    def conditional_flag():
        """Returns a tag depending on a condition"""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return left_flag
        return right_flag

    def conditional_pos():
        """Returns a position depending on a condition"""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return corners["downleft"]
        return corners["downright"]

    def conditional_col(cond=None):
        """Returns a colour depending on a condition"""
        if cond is None:
            cond = frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]
        if cond:
            return 150, 255, 150
        return 150, 150, 255

    # Keep track of space usage in the output video
    # The flags are set to False as soon as the lower
    # corners are occupied with text
    left_flag, right_flag = True, True

    if debug:
        # Print arena for debugging
        cv2.ellipse(
            img=frame,
            center=arena[0],
            axes=arena[1],
            angle=arena[2],
            startAngle=0,
            endAngle=360,
            color=(0, 255, 0),
            thickness=3,
        )
        # Print body parts for debuging
        for bpart in coords.columns.levels[0]:
            if not np.isnan(coords[bpart]["x"][fnum]):
                cv2.circle(
                    frame,
                    (int(coords[bpart]["x"][fnum]), int(coords[bpart]["y"][fnum])),
                    radius=3,
                    color=(
                        (255, 0, 0) if bpart.startswith(animal_ids[0]) else (0, 0, 255)
                    ),
                    thickness=-1,
                )
        # Print frame number
        write_on_frame("Frame " + str(fnum), (int(w * 0.3 / 10), int(h / 1.15)))

    if len(animal_ids) > 1:

        if tag_dict["nose2nose"][fnum]:
            write_on_frame("Nose-Nose", conditional_pos())
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

        if tag_dict[animal_ids[0] + "_nose2body"][fnum] and left_flag:
            write_on_frame("nose2body", corners["downleft"])
            left_flag = False

        if tag_dict[animal_ids[1] + "_nose2body"][fnum] and right_flag:
            write_on_frame("nose2body", corners["downright"])
            right_flag = False

        if tag_dict[animal_ids[0] + "_nose2tail"][fnum] and left_flag:
            write_on_frame("Nose-Tail", corners["downleft"])
            left_flag = False

        if tag_dict[animal_ids[1] + "_nose2tail"][fnum] and right_flag:
            write_on_frame("Nose-Tail", corners["downright"])
            right_flag = False

        if tag_dict["sidebyside"][fnum] and left_flag and conditional_flag():
            write_on_frame(
                "Side-side",
                conditional_pos(),
            )
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

        if tag_dict["sidereside"][fnum] and left_flag and conditional_flag():
            write_on_frame(
                "Side-Rside",
                conditional_pos(),
            )
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

    zipped_pos = list(
        zip(
            animal_ids,
            [corners["downleft"], corners["downright"]],
            [corners["upleft"], corners["upright"]],
            [left_flag, right_flag],
        )
    )

    for _id, down_pos, up_pos, flag in zipped_pos:

        if flag:

            if tag_dict[_id + undercond + "climbing"][fnum]:
                write_on_frame("climbing", down_pos)
            elif tag_dict[_id + undercond + "huddle"][fnum]:
                write_on_frame("huddling", down_pos)
            elif tag_dict[_id + undercond + "sniffing"][fnum]:
                write_on_frame("sniffing", down_pos)
            elif tag_dict[_id + undercond + "dig"][fnum]:
                write_on_frame("digging", down_pos)
            # elif tag_dict[_id + undercond + "lookaround"][fnum]:
            #     write_on_frame("lookaround", down_pos)

        # Define the condition controlling the colour of the speed display
        if len(animal_ids) > 1:
            colcond = frame_speeds[_id] == max(list(frame_speeds.values()))
        else:
            colcond = hparams["huddle_speed"] < frame_speeds

        write_on_frame(
            str(
                np.round(
                    (frame_speeds if len(animal_ids) == 1 else frame_speeds[_id]), 2
                )
            )
            + " mmpf",
            up_pos,
            conditional_col(cond=colcond),
        )


# noinspection PyProtectedMember,PyDefaultArgument
def annotate_video(
    coordinates: coordinates,
    tag_dict: pd.DataFrame,
    vid_index: int,
    frame_limit: int = np.inf,
    debug: bool = False,
    params: dict = {},
) -> True:
    """Renders a version of the input video with all supervised taggings in place.

    Parameters:
        - coordinates (deepof.preprocessing.coordinates): coordinates object containing the project information
        - debug (bool): if True, several debugging attributes (such as used body parts and arena) are plotted in
        the output video
        - vid_index: for internal usage only; index of the video to tag in coordinates._videos
        - frame_limit (float): limit the number of frames to output. Generates all annotated frames by default
        - params (dict): dictionary to overwrite the default values of the hyperparameters of the functions
        that the supervised pose estimation utilizes. Values can be:
            - speed_pause (int): size of the rolling window to use when computing speeds
            - close_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - side_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - follow_frames (int): number of frames during which the following trait is tracked
            - follow_tol (int): maximum distance between follower and followed's path during the last follow_frames,
            in order to report a detection
            - huddle_forward (int): maximum distance between ears and forward limbs to report a huddle detection
            - huddle_speed (int): maximum speed to report a huddle detection
            - fps (float): frames per second of the analysed video. Same as input by default


    Returns:
        True

    """

    # Extract useful information from coordinates object
    tracks = list(coordinates._tables.keys())
    videos = coordinates._videos
    path = os.path.join(coordinates._path, "Videos")

    params = get_hparameters(params)
    animal_ids = coordinates._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    try:
        vid_name = re.findall("(.*)DLC", tracks[vid_index])[0]
    except IndexError:
        vid_name = tracks[vid_index]

    arena_params = coordinates._arena_params[vid_index]
    h, w = coordinates._video_resolution[vid_index]
    corners = frame_corners(h, w)

    cap = cv2.VideoCapture(os.path.join(path, videos[vid_index]))
    # Keep track of the frame number, to align with the tracking data
    fnum = 0
    writer = None
    frame_speeds = (
        {_id: -np.inf for _id in animal_ids} if len(animal_ids) > 1 else -np.inf
    )

    # Loop over the frames in the video
    while cap.isOpened() and fnum < frame_limit:

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:  # pragma: no cover
            print("Can't receive frame (stream end?). Exiting ...")
            break

        font = cv2.FONT_HERSHEY_DUPLEX

        # Capture speeds
        try:
            if (
                list(frame_speeds.values())[0] == -np.inf
                or fnum % params["speed_pause"] == 0
            ):
                for _id in animal_ids:
                    frame_speeds[_id] = tag_dict[_id + undercond + "speed"][fnum]
        except AttributeError:
            if frame_speeds == -np.inf or fnum % params["speed_pause"] == 0:
                frame_speeds = tag_dict["speed"][fnum]

        # Display all annotations in the output video
        tag_annotated_frames(
            frame,
            font,
            frame_speeds,
            animal_ids,
            corners,
            tag_dict,
            fnum,
            undercond,
            params,
            (arena_params, h, w),
            debug,
            coordinates.get_coords(center=False)[vid_name],
        )

        if writer is None:
            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            # Define the FPS. Also frame size is passed.
            writer = cv2.VideoWriter()
            writer.open(
                vid_name + "_tagged.avi",
                cv2.VideoWriter_fourcc(*"MJPG"),
                params["fps"],
                (frame.shape[1], frame.shape[0]),
                True,
            )

        writer.write(frame)
        fnum += 1

    cap.release()
    cv2.destroyAllWindows()

    return True


if __name__ == "__main__":
    # Ignore warnings with no downstream effect
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# TODO:
#    - Scale CD1 features to match those of black6
#    - Use scaled features (not standard, but scaled using some notion of animal size)
