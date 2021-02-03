# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for rule-based pose estimation. See documentation for details

"""

import cv2
import deepof.utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import regex as re
import seaborn as sns
import warnings
from itertools import combinations
from scipy import stats
from typing import Any, List, NewType

# Ignore warning with no downstream effect
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# Create custom string type
Coordinates = NewType("Coordinates", Any)


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

    close_contact = (
        np.linalg.norm(pos_dframe[left] - pos_dframe[right], axis=1) * arena_abs
    ) / arena_rel < tol

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

    def rotate(origin, point, angle):
        ox, oy = origin
        px, py = point

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def outside_ellipse(x, y, e_center, e_axes, e_angle, threshold=0.0):

        x, y = rotate(e_center, (x, y), np.radians(e_angle))

        term_x = (x - e_center[0]) ** 2 / (e_axes[0] + threshold) ** 2
        term_y = (y - e_center[1]) ** 2 / (e_axes[1] + threshold) ** 2
        return term_x + term_y > 1

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


def huddle(
    pos_dframe: pd.DataFrame,
    speed_dframe: pd.DataFrame,
    tol_forward: float,
    tol_speed: float,
    animal_id: str = "",
) -> np.array:
    """Returns true when the mouse is huddling using simple rules. (!!!) Designed to
    work with deepof's default DLC mice models; not guaranteed to work otherwise.

        Parameters:
            - pos_dframe (pandas.DataFrame): position of body parts over time
            - speed_dframe (pandas.DataFrame): speed of body parts over time
            - tol_forward (float): Maximum tolerated distance between ears and
            forward limbs
            - tol_rear (float): Maximum tolerated average distance between spine
            body parts
            - tol_speed (float): Maximum tolerated speed for the center of the mouse

        Returns:
            hudd (np.array): True if the animal is huddling, False otherwise
    """

    if animal_id != "":
        animal_id += "_"

    forward = (
        np.linalg.norm(
            pos_dframe[animal_id + "Left_bhip"] - pos_dframe[animal_id + "Left_fhip"],
            axis=1,
        )
        < tol_forward
    ) | (
        np.linalg.norm(
            pos_dframe[animal_id + "Right_bhip"] - pos_dframe[animal_id + "Right_fhip"],
            axis=1,
        )
        < tol_forward
    )

    speed = speed_dframe[animal_id + "Center"] < tol_speed
    hudd = forward & speed

    return hudd


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

    follow = np.all(
        np.array([(dist_df.min(axis=1) < tol), right_orient1, right_orient2]),
        axis=0,
    )

    return follow


def single_behaviour_analysis(
    behaviour_name: str,
    treatment_dict: dict,
    behavioural_dict: dict,
    plot: int = 0,
    stat_tests: bool = True,
    save: str = None,
    ylim: float = None,
) -> list:
    """Given the name of the behaviour, a dictionary with the names of the groups to compare, and a dictionary
    with the actual tags, outputs a box plot and a series of significance tests amongst the groups

     Parameters:
         - behaviour_name (str): name of the behavioural trait to analize
         - treatment_dict (dict): dictionary containing video names as keys and experimental conditions as values
         - behavioural_dict (dict): tagged dictionary containing video names as keys and annotations as values
         - plot (int): Silent if 0; otherwise, indicates the dpi of the figure to plot
         - stat_tests (bool): performs FDR corrected Mann-U non-parametric tests among all groups if True
         - save (str): Saves the produced figure to the specified file
         - ylim (float): y-limit for the boxplot. Ignored if plot == False

     Returns:
         - beh_dict (dict): dictionary containing experimental conditions as keys and video names as values
         - stat_dict (dict): dictionary containing condition pairs as keys and stat results as values"""

    beh_dict = {condition: [] for condition in treatment_dict.keys()}

    for condition in beh_dict.keys():
        for ind in treatment_dict[condition]:
            beh_dict[condition] += np.sum(behavioural_dict[ind][behaviour_name]) / len(
                behavioural_dict[ind][behaviour_name]
            )

    return_list = [beh_dict]

    if plot > 0:

        fig, ax = plt.subplots(dpi=plot)

        sns.boxplot(
            x=list(beh_dict.keys()),
            y=list(beh_dict.values()),
            orient="vertical",
            ax=ax,
        )

        ax.set_title("{} across groups".format(behaviour_name))
        ax.set_ylabel("Proportion of frames")

        if ylim is not None:
            ax.set_ylim(ylim)

        if save is not None:  # pragma: no cover
            plt.savefig(save)

        return_list.append(fig)

    if stat_tests:
        stat_dict = {}
        for i in combinations(treatment_dict.keys(), 2):
            # Solves issue with automatically generated examples
            if np.any(
                np.array(
                    [
                        beh_dict[i[0]] == beh_dict[i[1]],
                        np.var(beh_dict[i[0]]) == 0,
                        np.var(beh_dict[i[1]]) == 0,
                    ]
                )
            ):
                stat_dict[i] = "Identical sources. Couldn't run"
            else:
                stat_dict[i] = stats.mannwhitneyu(
                    beh_dict[i[0]], beh_dict[i[1]], alternative="two-sided"
                )
        return_list.append(stat_dict)

    return return_list


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

    behaviour_dframe = behaviour_dframe.drop(speeds, axis=1).astype("float")
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
        "huddle_speed": 1,
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
def rule_based_tagging(
    tracks: List,
    videos: List,
    coordinates: Coordinates,
    coords: Any,
    dists: Any,
    speeds: Any,
    vid_index: int,
    arena_type: str,
    recog_limit: int = 1,
    path: str = os.path.join("."),
    params: dict = {},
) -> pd.DataFrame:
    """Outputs a dataframe with the registered motives per frame. If specified, produces a labeled
    video displaying the information in real time

    Parameters:
        - tracks (list): list containing experiment IDs as strings
        - videos (list): list of videos to load, in the same order as tracks
        - coordinates (deepof.preprocessing.coordinates): coordinates object containing the project information
        - coords (deepof.preprocessing.table_dict): table_dict with already processed coordinates
        - dists (deepof.preprocessing.table_dict): table_dict with already processed distances
        - speeds (deepof.preprocessing.table_dict): table_dict with already processed speeds
        - vid_index (int): index in videos of the experiment to annotate
        - path (str): directory in which the experimental data is stored
        - recog_limit (int): number of frames to use for arena recognition (1 by default)
        - params (dict): dictionary to overwrite the default values of the hyperparameters of the functions
        that the rule-based pose estimation utilizes. Values can be:
            - speed_pause (int): size of the rolling window to use when computing speeds
            - close_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - side_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - follow_frames (int): number of frames during which the following trait is tracked
            - follow_tol (int): maximum distance between follower and followed's path during the last follow_frames,
            in order to report a detection
            - huddle_forward (int): maximum distance between ears and forward limbs to report a huddle detection
            - huddle_speed (int): maximum speed to report a huddle detection

    Returns:
        - tag_df (pandas.DataFrame): table with traits as columns and frames as rows. Each
        value is a boolean indicating trait detection at a given time"""

    params = get_hparameters(params)
    animal_ids = coordinates._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    try:
        vid_name = re.findall("(.*)DLC", tracks[vid_index])[0]
    except IndexError:
        vid_name = tracks[vid_index]

    coords = coords[vid_name]
    dists = dists[vid_name]
    speeds = speeds[vid_name]
    arena_abs = coordinates.get_arenas[1][0]
    arena, h, w = deepof.utils.recognize_arena(
        videos, vid_index, path, recog_limit, coordinates._arena
    )

    # Dictionary with motives per frame
    tag_dict = {}

    def onebyone_contact(bparts: List):
        """Returns a smooth boolean array with 1to1 contacts between two mice"""
        nonlocal coords, animal_ids, params, arena_abs, arena

        return deepof.utils.smooth_boolean_array(
            close_single_contact(
                coords,
                animal_ids[0] + bparts[0],
                animal_ids[1] + bparts[-1],
                params["close_contact_tol"],
                arena_abs,
                arena[1][1],
            )
        )

    def twobytwo_contact(rev):
        """Returns a smooth boolean array with side by side contacts between two mice"""

        nonlocal coords, animal_ids, params, arena_abs, arena
        return deepof.utils.smooth_boolean_array(
            close_double_contact(
                coords,
                animal_ids[0] + "_Nose",
                animal_ids[0] + "_Tail_base",
                animal_ids[1] + "_Nose",
                animal_ids[1] + "_Tail_base",
                params["side_contact_tol"],
                rev=rev,
                arena_abs=arena_abs,
                arena_rel=arena[1][1],
            )
        )

    def overall_speed(speeds, _id, undercond):
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
        array = speeds[[_id + undercond + bpart for bpart in bparts]]
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

        for _id in animal_ids:
            tag_dict[_id + "_following"] = deepof.utils.smooth_boolean_array(
                following_path(
                    dists,
                    coords,
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
                arena,
                coords,
                params["climb_tol"],
                _id + undercond + "Nose",
            )
        )
        tag_dict[_id + undercond + "speed"] = overall_speed(speeds, _id, undercond)
        tag_dict[_id + undercond + "huddle"] = deepof.utils.smooth_boolean_array(
            huddle(
                coords,
                speeds,
                params["huddle_forward"],
                params["huddle_speed"],
                animal_id=_id,
            )
        )

    tag_df = pd.DataFrame(tag_dict)

    return tag_df


def tag_rulebased_frames(
    frame,
    font,
    frame_speeds,
    animal_ids,
    corners,
    tag_dict,
    fnum,
    dims,
    undercond,
    hparams,
    arena,
    debug,
    coords,
):
    """Helper function for rule_based_video. Annotates a given frame with on-screen information
    about the recognised patterns"""

    w, h = dims
    arena, h, w = arena

    def write_on_frame(text, pos, col=(255, 255, 255)):
        """Partial closure over cv2.putText to avoid code repetition"""
        return cv2.putText(frame, text, pos, font, 1, col, 2)

    def conditional_pos():
        """Returns a position depending on a condition"""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return corners["downleft"]
        else:
            return corners["downright"]

    def conditional_col(cond=None):
        """Returns a colour depending on a condition"""
        if cond is None:
            cond = frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]
        if cond:
            return 150, 255, 150
        else:
            return 150, 150, 255

    zipped_pos = list(
        zip(
            animal_ids,
            [corners["downleft"], corners["downright"]],
            [corners["upleft"], corners["upright"]],
        )
    )

    if len(animal_ids) > 1:

        if debug:
            # Print arena for debugging
            cv2.ellipse(frame, arena[0], arena[1], arena[2], 0, 360, (0, 255, 0), 3)
            # Print body parts for debuging
            for bpart in coords.columns.levels[0]:
                if not np.isnan(coords[bpart]["x"][fnum]):
                    cv2.circle(
                        frame,
                        (int(coords[bpart]["x"][fnum]), int(coords[bpart]["y"][fnum])),
                        radius=3,
                        color=(
                            (255, 0, 0)
                            if bpart.startswith(animal_ids[0])
                            else (0, 0, 255)
                        ),
                        thickness=-1,
                    )
            # Print frame number
            write_on_frame("Frame " + str(fnum), corners["downleft"])

        if tag_dict["nose2nose"][fnum] and not tag_dict["sidebyside"][fnum]:
            write_on_frame("Nose-Nose", conditional_pos())
        if (
            tag_dict[animal_ids[0] + "_nose2tail"][fnum]
            and not tag_dict["sidereside"][fnum]
        ):
            write_on_frame("Nose-Tail", corners["downleft"])
        if (
            tag_dict[animal_ids[1] + "_nose2tail"][fnum]
            and not tag_dict["sidereside"][fnum]
        ):
            write_on_frame("Nose-Tail", corners["downright"])
        if tag_dict["sidebyside"][fnum]:
            write_on_frame(
                "Side-side",
                conditional_pos(),
            )
        if tag_dict["sidereside"][fnum]:
            write_on_frame(
                "Side-Rside",
                conditional_pos(),
            )
        # for _id, down_pos, up_pos in zipped_pos:
        #     if (
        #         tag_dict[_id + "_following"][fnum]
        #         and not tag_dict[_id + "_climbing"][fnum]
        #     ):
        #         write_on_frame(
        #             "*f",
        #             (int(w * 0.3 / 10), int(h / 10)),
        #             conditional_col(),
        #         )

    for _id, down_pos, up_pos in zipped_pos:

        if tag_dict[_id + undercond + "climbing"][fnum]:
            write_on_frame("Climbing", down_pos)
        if (
            tag_dict[_id + undercond + "huddle"][fnum]
            and not tag_dict[_id + undercond + "climbing"][fnum]
        ):
            write_on_frame("huddle", down_pos)

        # Define the condition controlling the colour of the speed display
        if len(animal_ids) > 1:
            colcond = frame_speeds[_id] == max(list(frame_speeds.values()))
        else:
            colcond = hparams["huddle_speed"] > frame_speeds

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
def rule_based_video(
    coordinates: Coordinates,
    tracks: List,
    videos: List,
    vid_index: int,
    tag_dict: pd.DataFrame,
    frame_limit: int = np.inf,
    recog_limit: int = 1,
    path: str = os.path.join("."),
    params: dict = {},
    debug: bool = False,
) -> True:
    """Renders a version of the input video with all rule-based taggings in place.

    Parameters:
        - tracks (list): list containing experiment IDs as strings
        - videos (list): list of videos to load, in the same order as tracks
        - coordinates (deepof.preprocessing.coordinates): coordinates object containing the project information
        - debug (bool): if True, several debugging attributes (such as used body parts and arena) are plotted in
        the output video
        - vid_index (int): index in videos of the experiment to annotate
        - fps (float): frames per second of the analysed video. Same as input by default
        - path (str): directory in which the experimental data is stored
        - frame_limit (float): limit the number of frames to output. Generates all annotated frames by default
        - recog_limit (int): number of frames to use for arena recognition (1 by default)
        - params (dict): dictionary to overwrite the default values of the hyperparameters of the functions
        that the rule-based pose estimation utilizes. Values can be:
            - speed_pause (int): size of the rolling window to use when computing speeds
            - close_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - side_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - follow_frames (int): number of frames during which the following trait is tracked
            - follow_tol (int): maximum distance between follower and followed's path during the last follow_frames,
            in order to report a detection
            - huddle_forward (int): maximum distance between ears and forward limbs to report a huddle detection
            - huddle_speed (int): maximum speed to report a huddle detection

    Returns:
        True

    """

    # DATA OBTENTION AND PREPARATION
    params = get_hparameters(params)
    animal_ids = coordinates._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    try:
        vid_name = re.findall("(.*)DLC", tracks[vid_index])[0]
    except IndexError:
        vid_name = tracks[vid_index]

    arena, h, w = deepof.utils.recognize_arena(
        videos, vid_index, path, recog_limit, coordinates._arena
    )
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

        font = cv2.FONT_HERSHEY_COMPLEX_SMALL

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
        tag_rulebased_frames(
            frame,
            font,
            frame_speeds,
            animal_ids,
            corners,
            tag_dict,
            fnum,
            (w, h),
            undercond,
            params,
            (arena, h, w),
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


# TODO:
#    - Is border sniffing anything you might consider interesting?
