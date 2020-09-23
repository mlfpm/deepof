# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Functions and general utilities for rule-based pose estimation. See documentation for details

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import regex as re
import seaborn as sns
import deepof.utils
from itertools import combinations
from scipy import spatial
from scipy import stats
from tqdm import tqdm
from typing import Any, List, NewType


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
    arena_type: str, arena: np.array, pos_dict: pd.DataFrame, tol: float, nose: str
) -> np.array:
    """Returns True if the specified mouse is climbing the wall

        Parameters:
            - arena_type (str): arena type; must be one of ['circular']
            - arena (np.array): contains arena location and shape details
            - pos_dict (table_dict): position over time for all videos in a project
            - tol (float): minimum tolerance to report a hit
            - nose (str): indicates the name of the body part representing the nose of
            the selected animal

        Returns:
            - climbing (np.array): boolean array. True if selected animal
            is climbing the walls of the arena"""

    nose = pos_dict[nose]

    if arena_type == "circular":
        center = np.array(arena[:2])
        climbing = np.linalg.norm(nose - center, axis=1) > (arena[2] + tol)

    else:
        raise NotImplementedError("Supported values for arena_type are ['circular']")

    return climbing


def huddle(
    pos_dframe: pd.DataFrame,
    speed_dframe: pd.DataFrame,
    tol_forward: float,
    tol_spine: float,
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
            pos_dframe[animal_id + "Left_ear"] - pos_dframe[animal_id + "Left_fhip"],
            axis=1,
        )
        < tol_forward
    ) & (
        np.linalg.norm(
            pos_dframe[animal_id + "Right_ear"] - pos_dframe[animal_id + "Right_fhip"],
            axis=1,
        )
        < tol_forward
    )

    spine = [
        animal_id + "Spine_1",
        animal_id + "Center",
        animal_id + "Spine_2",
        animal_id + "Tail_base",
    ]
    spine_dists = []
    for comb in range(2):
        spine_dists.append(
            np.linalg.norm(
                pos_dframe[spine[comb]] - pos_dframe[spine[comb + 1]], axis=1
            )
        )
    spine = np.mean(spine_dists) < tol_spine
    speed = speed_dframe[animal_id + "Center"] < tol_speed
    hudd = forward & spine & speed

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
        np.array([(dist_df.min(axis=1) < tol), right_orient1, right_orient2]), axis=0,
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
            beh_dict[condition].append(
                np.sum(behavioural_dict[ind][behaviour_name])
                / len(behavioural_dict[ind][behaviour_name])
            )

    return_list = [beh_dict]

    if plot > 0:

        fig, ax = plt.subplots(dpi=plot)

        sns.boxplot(
            list(beh_dict.keys()), list(beh_dict.values()), orient="vertical", ax=ax
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
            if (
                beh_dict[i[0]] == beh_dict[i[1]]
                or np.var(beh_dict[i[0]]) == 0
                or np.var(beh_dict[i[1]]) == 0
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


def get_hparameters(hparams: dict = {}) -> dict:
    """Returns the most frequent behaviour in a window of window_size frames

            Parameters:
                    - hparams (dict): dictionary containing hyperparameters to overwrite

            Returns:
                - defaults (dict): dictionary with overwriten parameters. Those not
                specified in the input retain their default values"""

    defaults = {
        "speed_pause": 10,
        "close_contact_tol": 15,
        "side_contact_tol": 15,
        "follow_frames": 20,
        "follow_tol": 20,
        "huddle_forward": 15,
        "huddle_spine": 10,
        "huddle_speed": 1,
    }

    for k, v in hparams.items():
        defaults[k] = v

    return defaults


# noinspection PyDefaultArgument
def rule_based_tagging(
    tracks: List,
    videos: List,
    coordinates: Coordinates,
    vid_index: int,
    animal_ids: List = None,
    show: bool = False,
    save: bool = False,
    fps: float = 0.0,
    path: str = os.path.join("./"),
    hparams: dict = {},
    arena_type: str = "circular",
    frame_limit: float = np.inf,
    recog_limit: int = 1,
) -> pd.DataFrame:
    """Outputs a dataframe with the registered motives per frame. If specified, produces a labeled
    video displaying the information in real time

    Parameters:
        - tracks (list): list containing experiment IDs as strings
        - videos (list): list of videos to load, in the same order as tracks
        - coordinates (deepof.preprocessing.coordinates): coordinates object containing the project information
        - vid_index (int): index in videos of the experiment to annotate
        - animal_ids (list): IDs identifying multiple animals on the arena. None if there's only one
        - show (bool): if True, enables the display of the annotated video in a separate window
        - save (bool): if True, saves the annotated video to an mp4 file
        - fps (float): frames per second of the analysed video. Same as input by default
        - path (str): directory in which the experimental data is stored
        - frame_limit (float): limit the number of frames to output. Generates all annotated frames by default
        - recog_limit (int): number of frames to use for arena recognition (1 by default)
        - arena_type (str): type of the arena used in the experiments. Must be one of 'circular'"
        - hparams (dict): dictionary to overwrite the default values of the hyperparameters of the functions
        that the rule-based pose estimation utilizes. Values can be:
            - speed_pause (int): size of the rolling window to use when computing speeds
            - close_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - side_contact_tol (int): maximum distance between single bodyparts that can be used to report the trait
            - follow_frames (int): number of frames during which the following trait is tracked
            - follow_tol (int): maximum distance between follower and followed's path during the last follow_frames,
            in order to report a detection
            - huddle_forward (int): maximum distance between ears and forward limbs to report a huddle detection
            - huddle_spine (int): maximum average distance between spine body parts to report a huddle detection
            - huddle_speed (int): maximum speed to report a huddle detection

    Returns:
        - tag_df (pandas.DataFrame): table with traits as columns and frames as rows. Each
        value is a boolean indicating trait detection at a given time"""

    hparams = get_hparameters(hparams)

    vid_name = re.findall("(.*?)_", tracks[vid_index])[0]

    coords = coordinates.get_coords()[vid_name]
    speeds = coordinates.get_coords(speed=1)[vid_name]
    arena_abs = coordinates.get_arenas[1][0]
    arena, h, w = deepof.utils.recognize_arena(
        videos, vid_index, path, recog_limit, arena_type
    )

    # Dictionary with motives per frame
    tag_dict = {}

    if animal_ids:
        # Define behaviours that can be computed on the fly from the distance matrix
        tag_dict["nose2nose"] = deepof.utils.smooth_boolean_array(
            close_single_contact(
                coords,
                animal_ids[0] + "_Nose",
                animal_ids[1] + "_Nose",
                hparams["close_contact_tol"],
                arena_abs,
                arena[2],
            )
        )
        tag_dict[animal_ids[0] + "_nose2tail"] = deepof.utils.smooth_boolean_array(
            close_single_contact(
                coords,
                animal_ids[0] + "_Nose",
                animal_ids[1] + "_Tail_base",
                hparams["close_contact_tol"],
                arena_abs,
                arena[2],
            )
        )
        tag_dict[animal_ids[1] + "_nose2tail"] = deepof.utils.smooth_boolean_array(
            close_single_contact(
                coords,
                animal_ids[1] + "_Nose",
                animal_ids[0] + "_Tail_base",
                hparams["close_contact_tol"],
                arena_abs,
                arena[2],
            )
        )
        tag_dict["sidebyside"] = deepof.utils.smooth_boolean_array(
            close_double_contact(
                coords,
                animal_ids[0] + "_Nose",
                animal_ids[0] + "_Tail_base",
                animal_ids[1] + "_Nose",
                animal_ids[1] + "_Tail_base",
                hparams["side_contact_tol"],
                rev=False,
                arena_abs=arena_abs,
                arena_rel=arena[2],
            )
        )
        tag_dict["sidereside"] = deepof.utils.smooth_boolean_array(
            close_double_contact(
                coords,
                animal_ids[0] + "_Nose",
                animal_ids[0] + "_Tail_base",
                animal_ids[1] + "_Nose",
                animal_ids[1] + "_Tail_base",
                hparams["side_contact_tol"],
                rev=True,
                arena_abs=arena_abs,
                arena_rel=arena[2],
            )
        )
        for _id in animal_ids:
            tag_dict[_id + "_following"] = deepof.utils.smooth_boolean_array(
                following_path(
                    coords[vid_name],
                    coords,
                    follower=_id,
                    followed=[i for i in animal_ids if i != _id][0],
                    frames=hparams["follow_frames"],
                    tol=hparams["follow_tol"],
                )
            )
            tag_dict[_id + "_climbing"] = deepof.utils.smooth_boolean_array(
                pd.Series(
                    (
                        spatial.distance.cdist(
                            np.array(coords[_id + "_Nose"]), np.zeros([1, 2])
                        )
                        > (w / 200 + arena[2])
                    ).reshape(coords.shape[0]),
                    index=coords.index,
                ).astype(bool)
            )
            tag_dict[_id + "_speed"] = speeds[_id + "_speed"]
            tag_dict[_id + "_huddle"] = deepof.utils.smooth_boolean_array(
                huddle(
                    coords,
                    speeds,
                    hparams["huddle_forward"],
                    hparams["huddle_spine"],
                    hparams["huddle_speed"],
                )
            )

    else:
        tag_dict["climbing"] = deepof.utils.smooth_boolean_array(
            pd.Series(
                (
                    spatial.distance.cdist(np.array(coords["Nose"]), np.zeros([1, 2]))
                    > (w / 200 + arena[2])
                ).reshape(coords.shape[0]),
                index=coords.index,
            ).astype(bool)
        )
        tag_dict["speed"] = speeds["Center"]
        tag_dict["huddle"] = deepof.utils.smooth_boolean_array(
            huddle(
                coords,
                speeds,
                hparams["huddle_forward"],
                hparams["huddle_spine"],
                hparams["huddle_speed"],
            )
        )

    if any([show, save]):

        cap = cv2.VideoCapture(os.path.join(path, videos[vid_index]))
        # Keep track of the frame number, to align with the tracking data
        fnum = 0
        writer = None
        frame_speeds = {_id: -np.inf for _id in animal_ids} if animal_ids else -np.inf

        # Loop over the frames in the video
        pbar = tqdm(total=min(coords.shape[0] - recog_limit, frame_limit))
        while cap.isOpened() and fnum < frame_limit:

            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:  # pragma: no cover
                print("Can't receive frame (stream end?). Exiting ...")
                break

            font = cv2.FONT_HERSHEY_COMPLEX_SMALL

            # Label positions
            downleft = (int(w * 0.3 / 10), int(h / 1.05))
            downright = (int(w * 6.5 / 10), int(h / 1.05))
            upleft = (int(w * 0.3 / 10), int(h / 20))
            upright = (int(w * 6.3 / 10), int(h / 20))

            # Capture speeds
            try:
                if (
                    list(frame_speeds.values())[0] == -np.inf
                    or fnum % hparams["speed_pause"] == 0
                ):
                    for _id in animal_ids:
                        frame_speeds[_id] = speeds[_id + "_Center"][fnum]
            except AttributeError:
                if frame_speeds == -np.inf or fnum % hparams["speed_pause"] == 0:
                    frame_speeds = speeds["Center"][fnum]

            # Display all annotations in the output video
            if animal_ids:
                if tag_dict["nose2nose"][fnum] and not tag_dict["sidebyside"][fnum]:
                    cv2.putText(
                        frame,
                        "Nose-Nose",
                        (
                            downleft
                            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]
                            else downright
                        ),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                if (
                    tag_dict[animal_ids[0] + "_nose2tail"][fnum]
                    and not tag_dict["sidereside"][fnum]
                ):
                    cv2.putText(
                        frame, "Nose-Tail", downleft, font, 1, (255, 255, 255), 2
                    )
                if (
                    tag_dict[animal_ids[1] + "_nose2tail"][fnum]
                    and not tag_dict["sidereside"][fnum]
                ):
                    cv2.putText(
                        frame, "Nose-Tail", downright, font, 1, (255, 255, 255), 2
                    )
                if tag_dict["sidebyside"][fnum]:
                    cv2.putText(
                        frame,
                        "Side-side",
                        (
                            downleft
                            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]
                            else downright
                        ),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                if tag_dict["sidereside"][fnum]:
                    cv2.putText(
                        frame,
                        "Side-Rside",
                        (
                            downleft
                            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]
                            else downright
                        ),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )
                for _id, down_pos, up_pos in zip(
                    animal_ids, [downleft, downright], [upleft, upright]
                ):
                    if tag_dict[_id + "_climbing"][fnum]:
                        cv2.putText(
                            frame, "Climbing", down_pos, font, 1, (255, 255, 255), 2
                        )
                    if (
                        tag_dict[_id + "_huddle"][fnum]
                        and not tag_dict[_id + "_climbing"][fnum]
                    ):
                        cv2.putText(
                            frame, "Huddling", down_pos, font, 1, (255, 255, 255), 2
                        )
                    if (
                        tag_dict[_id + "_following"][fnum]
                        and not tag_dict[_id + "_climbing"][fnum]
                    ):
                        cv2.putText(
                            frame,
                            "*f",
                            (int(w * 0.3 / 10), int(h / 10)),
                            font,
                            1,
                            (
                                (150, 150, 255)
                                if frame_speeds[animal_ids[0]]
                                > frame_speeds[animal_ids[1]]
                                else (150, 255, 150)
                            ),
                            2,
                        )
                    cv2.putText(
                        frame,
                        _id + ": " + str(np.round(frame_speeds[_id], 2)) + " mmpf",
                        (up_pos[0] - 20, up_pos[1]),
                        font,
                        1,
                        (
                            (150, 150, 255)
                            if frame_speeds[_id] == max(list(frame_speeds.values()))
                            else (150, 255, 150)
                        ),
                        2,
                    )

            else:
                if tag_dict["climbing"][fnum]:
                    cv2.putText(
                        frame, "Climbing", downleft, font, 1, (255, 255, 255), 2
                    )
                if tag_dict["huddle"][fnum] and not tag_dict["climbing"][fnum]:
                    cv2.putText(frame, "huddle", downleft, font, 1, (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    str(np.round(frame_speeds, 2)) + " mmpf",
                    upleft,
                    font,
                    1,
                    (
                        (150, 150, 255)
                        if hparams["huddle_speed"] > frame_speeds
                        else (150, 255, 150)
                    ),
                    2,
                )

            if show:  # pragma: no cover
                cv2.imshow("frame", frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            if save:

                if writer is None:
                    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                    # Define the FPS. Also frame size is passed.
                    writer = cv2.VideoWriter()
                    writer.open(
                        re.findall("(.*?)_", tracks[vid_index])[0] + "_tagged.avi",
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        (fps if fps != 0 else cv2.CAP_PROP_FPS),
                        (frame.shape[1], frame.shape[0]),
                        True,
                    )

                print(cv2.CAP_PROP_FPS)

                writer.write(frame)

            pbar.update(1)
            fnum += 1

        cap.release()
        cv2.destroyAllWindows()

    tag_df = pd.DataFrame(tag_dict)

    return tag_df
