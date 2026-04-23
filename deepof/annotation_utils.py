# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Functions and general utilities for supervised pose estimation. See documentation for details."""

import os
import copy
import pickle
import warnings
from itertools import combinations, cycle
from typing import Any, List, NewType, Union, Tuple, Callable, Optional, Mapping
from enum import Enum, auto
from dataclasses import dataclass, field, replace

import numba as nb
import numpy as np
import pandas as pd
import sklearn.pipeline
from joblib import Parallel, delayed, parallel_backend
from natsort import os_sorted
from shapely.geometry import Polygon
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import re

import deepof.post_hoc
import deepof.utils
from deepof.utils import _suppress_warning
from deepof.data_loading import get_dt, _suppress_warning
import xgboost #as xgb
from deepof.config import SINGLE_BEHAVIORS, SYMMETRIC_BEHAVIORS,ASYMMETRIC_BEHAVIORS,CONTINUOUS_BEHAVIORS,CUSTOM_BEHAVIOR_COLOR_MAP


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


#######################
# BEHAVIOR BASE CLASSES
#######################

class Behavior_scope(Enum):
    INDIVIDUAL = auto()
    PAIR = auto()
    PAIR_NONDIRECTIONAL = auto()
    #GLOBAL = auto() #may be added later on


class Behavior_output(Enum):
    BINARY = auto()
    CONTINUOUS = auto()

BehaviorResult = Union[np.ndarray, pd.Series, Mapping[str, Union[np.ndarray, pd.Series]]]
animal_ids = Union[str, Tuple[str, str], None]
BehaviorFn = Callable[["BehaviorContext", animal_ids], BehaviorResult]
PostprocessFn = Callable[[np.ndarray, "BehaviorContext", animal_ids], np.ndarray]


@dataclass
class BehaviorContext:
    # identifiers / metadata
    key: str
    animal_ids: list[str]
    frame_rate: float
    arena_type: Any
    arena_params: Any
    roi_dict: dict

    # core tables
    raw_coords: pd.DataFrame
    coords: pd.DataFrame
    dists: pd.DataFrame
    angles: pd.DataFrame
    speeds: pd.DataFrame
    likelihoods: pd.DataFrame

    # features (optionally per-animal; keep flexible)
    full_features: Any

    # parameters + execution options
    params: dict[str, Any]
    run_numba: bool = False

    # optional extras (user extensions)
    extra: dict[str, Any] = field(default_factory=dict)

    def prefix(self, animal_id: str) -> str:
        """For multi-animal, columns are e.g. 'A_Nose', 'B_Nose' etc. for single-animal, just 'Nose'."""
        return f"{animal_id}_" if animal_id else ""

    def bp(self, animal_id: str, bodypart: str) -> str:
        """Convenience: ctx.bp("A","Nose")->"A_Nose"; ctx.bp("","Nose")->"Nose"."""
        return f"{animal_id}_{bodypart}" if animal_id else bodypart

    def iter_subjects(self, scope: Behavior_scope):
        if scope is Behavior_scope.INDIVIDUAL:
            yield from self.animal_ids
        elif scope is Behavior_scope.PAIR:
            # only makes sense if >=2 animals; combinations([]) is empty anyway
            yield from combinations(self.animal_ids, 2)
        elif scope is Behavior_scope.GLOBAL:
            yield None
        else:
            raise ValueError(f"Unknown behavior_scope: {scope}")


def postprocess_median_filtering(y: np.ndarray, ctx: BehaviorContext, behavior_output: Behavior_output) -> np.ndarray:
    """Default postprocessing for most binary behaviors.
    """
    y = np.asarray(y)

    # Apply binary moving median
    y_bool = np.nan_to_num(y, nan=0.0).astype(bool)
    y_bool = deepof.utils.binary_moving_median_numba(
        y_bool, lag=int(ctx.params["median_filter_width"])
    )
    return y_bool.astype(float)


def postprocess_following(y: np.ndarray, ctx: BehaviorContext, animal_ids: animal_ids) -> np.ndarray:
    """Standard postprocessing, then removal of short segments"""
    # First default binary smoothing
    y = postprocess_median_filtering(y, ctx, Behavior_output.BINARY).astype(bool)

    # Then filter our short segments
    y = deepof.utils.filter_short_true_segments_numba(
        array=y,
        min_length=int(ctx.params["min_follow_frames"]),
    )
    return y.astype(float)


def postprocess_identity(y: np.ndarray, ctx: BehaviorContext, animal_ids: animal_ids) -> np.ndarray:
    """Does not apply any postprocessing, used for e.g. continuous behaviors"""
    return np.asarray(y).astype(float, copy=False)


@dataclass(frozen=True)
class DeepOF_behavior:
    """Class for different types of behaviors that The supervised annotations of DeepOF can process.

    """
    name: str
    scope: Behavior_scope
    output_kind: Behavior_output
    compute: BehaviorFn
    unit: Optional[str] = "a.u."
    

    # Optional: assign a user defined hex color. If None, a color from deepof.config.CUSTOM_BEHAVIOR_COLOR_MAP will be assigned
    color: Optional[str] = None

    # Optional: override postprocess; if None, use default_postprocess(...)
    postprocess: Optional[PostprocessFn] = None

    # Optional: simple dependency documentation/validation
    requires: Tuple[str, ...] = ()

    # Optional: ordering control (e.g., keep speed measures last)
    order: int = 0

    def set_color(self, color: Optional[str]) -> "DeepOF_behavior":
        return replace(self, color=color)

    def column_name(self, ctx: BehaviorContext, animal_ids: animal_ids) -> str:
        if self.scope is Behavior_scope.INDIVIDUAL:
            animal_id = animal_ids  # type: ignore[assignment]
            return f"{ctx.prefix(animal_id)}{self.name}"
        if self.scope is Behavior_scope.PAIR:
            a, b = animal_ids  # type: ignore[misc]
            return f"{a}_{b}_{self.name}"
        if self.scope is Behavior_scope.GLOBAL:
            return self.name
        raise ValueError(f"Unsupported scope: {self.scope}")

    def annotate_behavior(self, ctx: BehaviorContext, animal_ids: animal_ids) -> np.ndarray:
        # optional dependency checks
        for attr in self.requires:
            if not hasattr(ctx, attr):
                raise AttributeError(f"Behavior '{self.name}' requires ctx.{attr} to exist")

        res = self.compute(ctx, animal_ids)

        if isinstance(res, Mapping):
            out: dict[str, np.ndarray] = {}
            for subkey, arr in res.items():
                y = np.asarray(arr)
                y = self.postprocess(y, ctx, animal_ids)
                out[subkey] = y
            return out
        else:
            
            y = np.asarray(res)

            if self.postprocess is not None:
                y = np.asarray(self.postprocess(y, ctx, animal_ids))
            else:
                y = postprocess_median_filtering(y, ctx, self.output_kind)

            return y


###########################
# PAIRED BEHAVIOR INSTANCES
###########################


def compute_nose2nose(ctx: BehaviorContext, mice_pair: animal_ids) -> np.ndarray:
    """nondirectional, noses of both mice are close"""
    a, b = mice_pair  
    tol = float(ctx.params["close_contact_tol"])
    return close_single_contact(
        ctx.raw_coords,
        ctx.bp(a, "Nose"),
        ctx.bp(b, "Nose"),
        tol,
    )

def compute_sidebyside(ctx: BehaviorContext, mice_pair: animal_ids) -> np.ndarray:
    """nondirectional, mice are next to each other nose by nose"""
    a, b = mice_pair 
    return close_double_contact(
        ctx.raw_coords,
        ctx.bp(a, "Nose"),
        ctx.bp(a, "Tail_base"),
        ctx.bp(b, "Nose"),
        ctx.bp(b, "Tail_base"),
        rel_tol=float(ctx.params["side_contact_tol"]),
        rev=False,
    )

def compute_sidereside(ctx: BehaviorContext, mice_pair: animal_ids) -> np.ndarray:
    """nondirectional, mice are next to each other nose by tail"""
    a, b = mice_pair  
    return close_double_contact(
        ctx.raw_coords,
        ctx.bp(a, "Nose"),
        ctx.bp(a, "Tail_base"),
        ctx.bp(b, "Nose"),
        ctx.bp(b, "Tail_base"),
        rel_tol=float(ctx.params["side_contact_tol"]),
        rev=True,
    )

def compute_nose2tail(ctx: BehaviorContext, mice_pair: animal_ids) -> np.ndarray:
    """Directional: (a,b) means a_nose close to b_tailbase"""
    a, b = mice_pair  
    tol = float(ctx.params["close_contact_tol"])
    return close_single_contact(
        ctx.raw_coords,
        ctx.bp(a, "Nose"),
        ctx.bp(b, "Tail_base"),
        tol,
    )

def compute_nose2body(ctx: BehaviorContext, mice_pair: animal_ids) -> np.ndarray:
    """Directional: (a,b) means a_nose close to any of b main_body parts."""
    a, b = mice_pair 
    tol = float(ctx.params["close_contact_tol"])
    main_body = ctx.extra["main_body"]  # list like ["Left_ear", "Right_ear", ...]
    body_cols = [ctx.bp(b, bp) for bp in main_body]
    return close_single_contact(
        ctx.raw_coords,
        ctx.bp(a, "Nose"),
        body_cols,
        tol,
    )

def compute_following(ctx: BehaviorContext, mice_pair: animal_ids) -> np.ndarray:
    """Directional: (a,b) means a follows b."""
    a, b = mice_pair  
    return following_path(
        distance_dframe=ctx.dists,
        position_dframe=ctx.raw_coords,
        speed_dframe=ctx.speeds,
        follower=a,
        followed=b,
        frames=int(ctx.params["follow_frames"]),
        tol=float(ctx.params["follow_tol"]),
        tol_speed=float(ctx.params["stationary_threshold"]),
    )


###########################
# SINGLE BEHAVIOR INSTANCES
###########################


def compute_climb_arena(ctx: BehaviorContext, animal_id: animal_ids) -> np.ndarray:
    aid = animal_id  # type: ignore[assignment]
    prefix = ctx.prefix(aid)
    mouse_len = ctx.extra.get("mouse_lens", {}).get(prefix, 50)

    return climb_arena(
        arena_type=ctx.arena_type,
        arena=ctx.arena_params,
        pos_dict=ctx.raw_coords,
        rel_tol=float(ctx.params["climb_tol"]),
        id=prefix,
        mouse_len=mouse_len,
        centered_data=False,
        run_numba=ctx.run_numba,
    )


def compute_sniff_arena(ctx: BehaviorContext, animal_id: animal_ids) -> np.ndarray:
    aid = animal_id  # type: ignore[assignment]
    center = ctx.extra.get("center", "Center")

    return sniff_object(
        speed_dframe=ctx.speeds,
        arena=ctx.arena_params,
        pos_dict=ctx.raw_coords,
        tol=float(ctx.params["sniff_arena_tol"]),
        tol_speed=float(ctx.params["stationary_threshold"]),
        nose=ctx.bp(aid, "Nose"),
        center_name=center,
        centered_data=False,
        s_object="arena",
        animal_id=aid,  # IMPORTANT: without underscore, matches old call
        run_numba=ctx.run_numba,
    ).to_numpy()


def compute_immobility(ctx: BehaviorContext, animal_id: animal_ids) -> np.ndarray:
    aid = animal_id  # type: ignore[assignment]
    est = ctx.extra["immobility_estimator"]

    # match old feature selection logic
    if aid:
        X = get_dt(ctx.full_features[aid], ctx.key)
    else:
        X = get_dt(ctx.full_features, ctx.key)

    y_imm, _ = immobility(
        X_huddle=X,
        huddle_estimator=est,
        animal_id=ctx.prefix(aid),  # NOTE: with underscore as in old code
        median_filter_width=int(ctx.params["median_filter_width"]),
        min_immobility=int(ctx.params["min_immobility"]),
        max_immobility=0,  # keep as in your current call
    )
    return y_imm


def compute_stat_lookaround(ctx: BehaviorContext, animal_id: animal_ids) -> np.ndarray:
    aid = animal_id  # type: ignore[assignment]

    # close_range matches your current behavior (int array)
    if len(ctx.animal_ids) > 1:
        close_range = calculate_close_range(
            ctx.dists,
            mouse_id=ctx.prefix(aid),       # e.g. "A_"
            bodypart="Nose",
            threshold=float(ctx.params["side_contact_tol"]),
        )
    else:
        close_range = np.zeros(len(ctx.dists), dtype=int)

    return stationary_lookaround(
        speed_dframe=ctx.speeds,
        dist_dframe=ctx.dists,
        likelihood_dframe=ctx.likelihoods,
        mouse_identity=ctx.prefix(aid),  # e.g. "A_"
        close_range=close_range,
        tol_speed=float(ctx.params["stationary_threshold"]),
        tol_likelihood=float(ctx.params["nose_likelihood"]),
        min_length=int(ctx.params["min_follow_frames"]),
        animal_id=aid,  # without underscore, as in old call
    )


def compute_detect_activity(ctx: BehaviorContext, animal_id: animal_ids) -> dict[str, np.ndarray]:
    aid = animal_id  # type: ignore[assignment]
    center = ctx.extra.get("center", "Center")

    stat_a, stat_p, mov = detect_activity(
        speed_dframe=ctx.speeds,
        likelihood_dframe=ctx.likelihoods,
        tol_speed=float(ctx.params["stationary_threshold"]),
        tol_likelihood=float(ctx.params["nose_likelihood"]),
        min_length=int(ctx.params["min_follow_frames"]),
        center_name=center,
        animal_id=aid,
    )
    return {
        "stat-active": stat_a,
        "stat-passive": stat_p,
        "moving": mov,
    }


def compute_sniffing(ctx: BehaviorContext, animal_id: animal_ids) -> np.ndarray:
    aid = animal_id  # type: ignore[assignment]
    center = ctx.extra.get("center", "Center")
    return np.asarray(sniff_around(ctx.speeds, ctx.likelihoods,
                                  float(ctx.params["stationary_threshold"]),
                                  float(ctx.params["nose_likelihood"]),
                                  center_name=center, animal_id=aid))


###############################
# CONTINUOUS BEHAVIOR INSTANCES
###############################


@_suppress_warning(warn_messages=["All-NaN slice encountered"])
def compute_continuous_measures(ctx: BehaviorContext, animal_id: animal_ids) -> dict[str, np.ndarray]:
    aid = animal_id  # type: ignore[assignment]

    bparts = ["Center","Spine_1","Spine_2","Nose","Left_ear","Right_ear",
              "Left_fhip","Right_fhip","Left_bhip","Right_bhip","Tail_base"]
    cols = [ctx.bp(aid, bp) for bp in bparts if ctx.bp(aid, bp) in ctx.speeds.columns]

    if len(cols) == 0:
        n = len(ctx.speeds)
        nan = np.full(n, np.nan)
        return {"distance": nan, "cum-distance": nan, "speed": nan}

    array = ctx.speeds[cols]
    avg_speed = np.nanmedian(array.iloc[1:].to_numpy(), axis=1)
    avg_speed = np.insert(avg_speed, 0, np.nan, axis=0)

    avg_distance = avg_speed * (1.0 / float(ctx.frame_rate))
    cum_distance = np.cumsum(np.nan_to_num(avg_distance, copy=True))

    return {
        "distance": avg_distance,
        "cum-distance": cum_distance,
        "speed": avg_speed,
    }


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

    term_x = (x - e_center[0]) ** 2 / (np.max([e_axes[0] + threshold,1e-12])) ** 2
    term_y = (y - e_center[1]) ** 2 / (np.max([e_axes[1] + threshold,1e-12])) ** 2
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

    if isinstance(arena, Tuple): # Circular (legacy) arena_type.startswith("circular"):
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
        ).to_numpy()

    elif isinstance(arena, np.ndarray): #polygonal arena_type.startswith("polygon"):

        # intermediary for testing, will be replaced with length-based condition
        if run_numba:

            # extract outer arena polygon coordinates
            xp = np.array(Polygon(arena).buffer(tol).exterior.coords.xy[0])
            yp = np.array(Polygon(arena).buffer(tol).exterior.coords.xy[1])
            outer_polygon = np.transpose(np.array([xp, yp]))

            # get nose positions outside of arena polygon
            climbing = np.invert(deepof.utils.point_in_polygon_numba(nose.values, outer_polygon))

        else:

            climbing = np.invert(
                deepof.utils.point_in_polygon(nose.values, Polygon(arena).buffer(tol))
            )

    else:
        raise NotImplementedError(
            "Supported values for arena_type are ['polygonal-manual', 'polygonal-autodetect', 'circular-manual', 'circular-autodetect']"
        )

    return climbing


def sniff_object(
    speed_dframe: pd.DataFrame,
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
        if isinstance(arena, Tuple): # Circular (legacy)   arena_type.startswith("circular"):
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

        elif isinstance(arena, np.ndarray): # Polygonal   arena_type.startswith("polygon"):

            # intermediary for testing, will be replaced with length-based condition
            if run_numba:

                # extract outer arena polygon coordinates
                xy = np.array(Polygon(arena).buffer(-tol).exterior.coords.xy)
                xp = xy[0]
                yp = xy[1]
                inner_polygon = np.transpose(np.array([xp, yp]))

                # extract inner arena polygon coordinates
                xy = np.array(Polygon(arena).buffer(tol).exterior.coords.xy)
                xp = xy[0]
                yp = xy[1]
                outer_polygon = np.transpose(np.array([xp, yp]))

                # get nose positions outside of outer and inner arena polygon
                nosing_min = np.invert(
                    deepof.utils.point_in_polygon_numba(nose.values, inner_polygon)
                )
                nosing_max = np.invert(
                    deepof.utils.point_in_polygon_numba(nose.values, outer_polygon)
                )

            else:

                nosing_min = np.invert(
                    deepof.utils.point_in_polygon(nose.values, Polygon(arena).buffer(-tol))
                )
                nosing_max = np.invert(
                    deepof.utils.point_in_polygon(nose.values, Polygon(arena).buffer(tol))
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


def immobility(
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
            "\033[38;5;208m"
            "Skipping huddle annotation as not all required body parts are present. At the moment, huddle annotation "
            "requires the deepof_11 or deepof_14 labelling scheme. Read the full documentation for further details."
            "\033[0m"
        )
        return np.full(X_huddle.shape[0], np.nan), np.full(X_huddle.shape[0], np.nan)

    X_huddle=augment_with_neighbors(X_huddle)
    # Concatenate all relevant data frames and predict using the pre-trained estimator
    X_mask = np.isnan(X_huddle).mean(axis=1) > 0.1
    y_huddle = huddle_estimator.predict(
        StandardScaler().fit_transform(np.nan_to_num(X_huddle))
    ).astype(float)
    #y_huddle = np.zeros(len(X_mask))
    y_huddle[X_mask] = False#np.nan
    y_huddle = deepof.utils.binary_moving_median_numba(y_huddle, lag=median_filter_width)
    y_huddle = deepof.utils.filter_short_true_segments_numba(y_huddle, min_length=min_immobility)
    y_sleep = y_huddle #deepof.utils.filter_short_true_segments_numba(y_huddle, min_length=max_immobility)
    #y_huddle[y_sleep] = False
    return y_huddle, y_sleep


def augment_with_neighbors(X_huddle, window=5, step=1, window_out=11):
    """Expands a given set of features with leading and lagging features on the time axis. Will only return speed based features.

    Args:
        X_huddle (pandas.DataFrame): mouse features over time.
        window (int): steps to go forward and backward in time for each feature
        step (int): step size for the window
        window_out (int): total length of the output window

    Returns:
        X_augmented (pandas.DataFrame): mouse features over time including leading and lagging features (only speed features) for each frame

    """    
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
            col_features[f'{col}_{k-int(window_out/2)}'] = mean_series
        
        # Append the features as a DataFrame to the list
        augmented_dfs.append(pd.DataFrame(col_features))
    
    # Concatenate all DataFrames at once
    X_augmented = pd.concat(augmented_dfs, axis=1)

    # Filter columns that contain 'speed'
    filtered_columns = [col for col in X_augmented.columns if 'speed' in col] # or '0' in col]

    # Select only the filtered columns
    X_augmented = X_augmented[filtered_columns]
    
    return X_augmented


def digging(
    speed_dframe: pd.DataFrame,
    dist_dframe: pd.DataFrame,
    likelihood_dframe: pd.DataFrame,
    mouse_identity: str,
    close_range: np.ndarray,
    tol_speed: float,
    tol_likelihood: float,
    min_length: int,
    center_name: str = "Center",
    animal_id: str = "",
): # pragma: no cover
    """Return true when the mouse is digging. Experimental and currently not included.

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        dist_dframe (pandas.DataFrame): distance between body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        mouse_identity (str): animal id without the _
        close_range (np.ndarray): boolean array that denotes if the nose of the current mouse is close to any other mouse for each frame.
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        min_length (int): minimum length that True segments need to have to not get filtered out.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        stationary_active (np.array): True if the animal is standing still and is active, False otherwise
        stationary_passive (np.array): True if the animal is standing still and is passive, False otherwise

    """

    # Experimental and too unspecific, called via
    #tag_dict[_id + undercond + "digging"] = digging(
    #speeds,
    #dists,
    #likelihoods,
    #_id + undercond,
    #close_range,
    #params["stationary_threshold"],
    #params["nose_likelihood"],
    #params["min_follow_frames"],
    #center_name=center,
    #animal_id=_id,
    #)

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
    for bp in bodyparts.copy():
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
    likelihood_dframe: pd.DataFrame,
    mouse_identity: str,
    close_range: np.ndarray,
    tol_speed: float,
    tol_likelihood: float,
    min_length: int,
    animal_id: str = "",
):
    """Return true when the mouse is standing still and looking around (moving nose without head being tilted too much).

    Design considerations:
        Detecting immobility and activity is relatively straightforward by mostly just checking speed thresholds on bodyparts.
        The main problem arises from getting a lot of "flickering" out of the detections, as bodyparts from frame to frame may be
        just above or below that threshold. Respectively most of the detect_activity algorithm is a series of filtering steps to
        alternatingly smooth the predictions and sharpening the edges of predicted behavior. 

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        dist_dframe (pandas.DataFrame): distance between body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        mouse_identity (str): animal id without the _
        close_range (np.ndarray): boolean array that denotes if the nose of the current mouse is close to any other mouse for each frame.
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        min_length (int): minimum length that True segments need to have to not get filtered out.
        animal_id (str): ID of the current animal.

    Returns:
        stationary_lookaround (np.array): True if the animal is standing still and looking around (moving nose without head being tilted too much), False otherwise

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
    for bp in bodyparts.copy():
        if not bp in speed_dframe.keys():
            bodyparts.remove(bp)

    
    nose_activity = (tol_speed < speed_dframe[animal_id+"Nose"]).to_numpy() & (likelihood_dframe[animal_id+"Nose"] > tol_likelihood).to_numpy()

    bparts=[animal_id+"Left_bhip", animal_id+"Right_bhip"]

    if bparts[0] in speed_dframe.columns and bparts[1] in speed_dframe.columns:
        body_inactivity = np.array([
            (tol_speed*2 >= speed_dframe[part]).to_numpy() & 
            (likelihood_dframe[part] > tol_likelihood).to_numpy()
            for part in bparts
        ]).all(axis=0)
    else: 
        body_inactivity = np.full(speed_dframe.shape[0], True)

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
    """Return true when the mouse is either moving (moving), standing still and either moving (active) or not moving (passive).

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
        min_length (int): minimum length that True segments need to have to not get filtered out.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        stationary_active (np.array): True if the animal is standing still and is active, False otherwise
        stationary_passive (np.array): True if the animal is standing still and is passive, False otherwise
        mobile (np.array): True if the animal is not standing still, False otherwise

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
    for bp in bodyparts.copy():
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
    """Return true when the mouse is sniffing around using simple rules.

    Args:
        speed_dframe (pandas.DataFrame): speed of body parts over time
        likelihood_dframe (pandas.DataFrame): likelihood of body part tracker over time, as directly obtained from DeepLabCut
        tol_speed (float): Maximum tolerated speed for the center of the mouse
        tol_likelihood (float): Maximum tolerated likelihood for the nose.
        center_name (str): Body part to center coordinates on. "Center" by default.
        animal_id (str): ID of the current animal.

    Returns:
        lookaround (np.array): True if the animal is standing still and sniffing around, False otherwise

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
    immobility_estimator: str = None,
    center: str = "Center",
    params: dict = {},
    run_numba: bool = False,
    custom_behaviors: list[DeepOF_behavior] = None,
    custom_behavior_inputs: dict = {}
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
        immobility_estimator (str): classifier to determine if a mouse is immobile or not.
        center (str): Body part to center coordinates on. "Center" by default.
        params (dict): dictionary to overwrite the default values of the parameters of the functions that the rule-based pose estimation utilizes. See documentation for details.
        run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)
        custom_behaviors (list[DeepOF_behavior]): a list of custom DeepOF_behavior objects. Added at the beginning of supervised behaviors if provided
        custom_behavior_inputs (dict): a dictionary containing additional information you need for your custom behaviors
        
    Returns:
        tag_df (pandas.DataFrame): table with traits as columns and frames as rows. Each value is a boolean indicating trait detection at a given time

    """

    animal_ids = coord_object._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    #extract various data tables from their Table dicts
    raw_coords = get_dt(raw_coords,key).reset_index(drop=True)
    coords = get_dt(coords,key).reset_index(drop=True)
    dists = get_dt(dists,key).reset_index(drop=True)
    angles = get_dt(angles,key).reset_index(drop=True)
    speeds = get_dt(speeds,key).reset_index(drop=True)
    likelihoods = get_dt(coord_object.get_quality(),key).reset_index(drop=True)

    # Initialize context + behavior class instances               
    behavior_ctx = BehaviorContext(
        key=key,
        animal_ids=coord_object._animal_ids,
        frame_rate=coord_object._frame_rate,
        arena_type=coord_object._arena,
        arena_params=coord_object._arena_params[key],
        roi_dict=coord_object._roi_dicts[key],

        raw_coords = raw_coords,
        coords = coords,
        dists = dists,
        angles = angles,
        speeds = speeds,
        likelihoods = likelihoods,

        # features (optionally per-animal; keep flexible)
        full_features = full_features,
        params = params,
        run_numba = run_numba,       
    )

    behavior_nose2nose = DeepOF_behavior(
        name="nose2nose",
        scope=Behavior_scope.PAIR,
        output_kind=Behavior_output.BINARY,
        compute=compute_nose2nose,
        requires=("raw_coords",),
    )

    behavior_sidebyside = DeepOF_behavior(
        name="sidebyside",
        scope=Behavior_scope.PAIR,
        output_kind=Behavior_output.BINARY,
        compute=compute_sidebyside,
        requires=("raw_coords",),
    )

    behavior_sidereside = DeepOF_behavior(
        name="sidereside",
        scope=Behavior_scope.PAIR,
        output_kind=Behavior_output.BINARY,
        compute=compute_sidereside,
        requires=("raw_coords",),
    )

    behavior_nose2tail = DeepOF_behavior(
        name="nose2tail",
        scope=Behavior_scope.PAIR,
        output_kind=Behavior_output.BINARY,
        compute=compute_nose2tail,
        requires=("raw_coords",),
    )

    behavior_nose2body = DeepOF_behavior(
        name="nose2body",
        scope=Behavior_scope.PAIR,
        output_kind=Behavior_output.BINARY,
        compute=compute_nose2body,
        requires=("raw_coords",),
    )

    behavior_following = DeepOF_behavior(
        name="following",
        scope=Behavior_scope.PAIR,
        output_kind=Behavior_output.BINARY,
        compute=compute_following,
        postprocess=postprocess_following,
        requires=("dists", "raw_coords", "speeds"),
    )

    behavior_climb_arena = DeepOF_behavior(
        name="climb-arena",
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.BINARY,
        compute=compute_climb_arena,
        requires=("raw_coords",),
    )

    behavior_sniff_arena = DeepOF_behavior(
        name="sniff-arena",
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.BINARY,
        compute=compute_sniff_arena,
        requires=("raw_coords", "speeds"),
    )

    behavior_immobility = DeepOF_behavior(
        name="immobility",
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.BINARY,
        compute=compute_immobility,
        postprocess=postprocess_identity,  
    )

    behavior_stat_lookaround = DeepOF_behavior(
        name="stat-lookaround",
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.BINARY,
        compute=compute_stat_lookaround,
        postprocess=postprocess_identity,  
    )

    behavior_detect_activity = DeepOF_behavior(
        name="detect_activity",  # mutlti-behavior, name is not used in column naming for dict outputs
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.BINARY,
        compute=compute_detect_activity,
        postprocess=postprocess_identity, 
    )

    behavior_sniffing = DeepOF_behavior(
        name="sniffing",
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.BINARY,
        compute=compute_sniffing,
        postprocess=postprocess_identity,  
    )

    behavior_continuous = DeepOF_behavior(
        name="continuous",  # mutlti-behavior, name is not used in column naming for dict outputs
        scope=Behavior_scope.INDIVIDUAL,
        output_kind=Behavior_output.CONTINUOUS,
        compute=compute_continuous_measures,
        postprocess=postprocess_identity, 
    )


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
    for aid in animal_ids:
        if aid:
         aid=aid+"_"
        
        #calculate mouse lengths
        backbone=[aid+"Nose",aid+"Spine_1",aid+"Center", aid+"Spine_2", aid+"Tail_base"]

        #remove missing bodyparts from backbone
        for bp in backbone:
            if not bp in raw_coords.keys():
                backbone.remove(bp)

        #calculate overall length of bodypart chain i.e. mouse length
        subset_cols = [col for col in raw_coords.columns if col[0] in backbone]
        if len(backbone)>1 and len(raw_coords.dropna(subset=subset_cols))>=400: 
            indices=np.random.choice(raw_coords.dropna(subset=subset_cols).index, size=np.min([5000, len(raw_coords.dropna(subset=subset_cols))]), replace=False)
            mouse_lens_raw=0
            for bp_pos in range(0, len(backbone)-1):
                mouse_lens_raw+=np.apply_along_axis(
                        np.linalg.norm, 1, (
                            raw_coords[backbone[bp_pos+1]].iloc[indices]
                            -raw_coords[backbone[bp_pos]].iloc[indices]
                            )
                        )
            mouse_lens[aid]=np.nanpercentile(mouse_lens_raw,80)
                    
        #assume default mouse length if body parts for length estimation are insufficient
        else:
            mouse_lens[aid]=50
        
        if aid+"full_area" in coord_object._areas[key]:
            mouse_areas[aid]=np.nanpercentile(
                coord_object._areas[key][aid+"full_area"]
                ,80)
            
    behavior_ctx.extra["main_body"] = main_body
    behavior_ctx.extra["immobility_estimator"]=immobility_estimator
    behavior_ctx.extra["mouse_lens"] = mouse_lens
    behavior_ctx.extra.update(custom_behavior_inputs)
        
    # Get all animal ID combinations
    animal_pairs = list(combinations(animal_ids, 2))


    # Paired behaviors
    if len(animal_ids) >= 2:

        for animal_pair in animal_pairs:

            if custom_behaviors is not None:
                for custom_behavior in custom_behaviors:

                    if custom_behavior.scope is Behavior_scope.PAIR_NONDIRECTIONAL:

                        # Pairs of directional behaviors (inverted order for other behavior direction)
                        tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_" + custom_behavior.name] = custom_behavior.annotate_behavior(behavior_ctx, animal_pair)
                    
                    elif custom_behavior.scope is Behavior_scope.PAIR:

                        # Pairs of directional behaviors (inverted order for other behavior direction)
                        tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_" + custom_behavior.name] = custom_behavior.annotate_behavior(behavior_ctx, animal_pair)
                        tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_" + custom_behavior.name] = custom_behavior.annotate_behavior(behavior_ctx, (animal_pair[1],animal_pair[0]))


            # Nondirectional behaviors
            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_nose2nose"] = behavior_nose2nose.annotate_behavior(behavior_ctx, animal_pair)           
            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_sidebyside"] = behavior_sidebyside.annotate_behavior(behavior_ctx, animal_pair)
            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_sidereside"] = behavior_sidereside.annotate_behavior(behavior_ctx, animal_pair)

            # Pairs of directional behaviors (inverted order for other behavior direction)
            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_nose2tail"] = behavior_nose2tail.annotate_behavior(behavior_ctx, animal_pair)
            tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_nose2tail"] = behavior_nose2tail.annotate_behavior(behavior_ctx, (animal_pair[1],animal_pair[0])) 

            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_nose2body"] = behavior_nose2body.annotate_behavior(behavior_ctx, animal_pair)
            tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_nose2body"] = behavior_nose2body.annotate_behavior(behavior_ctx, (animal_pair[1],animal_pair[0]))

            tag_dict[f"{animal_pair[0]}_{animal_pair[1]}_following"] = behavior_following.annotate_behavior(behavior_ctx, animal_pair)
            tag_dict[f"{animal_pair[1]}_{animal_pair[0]}_following"] = behavior_following.annotate_behavior(behavior_ctx, (animal_pair[1],animal_pair[0]))
   

    # Single behaviors
    for aid in animal_ids:    

        if custom_behaviors is not None:
            for custom_behavior in custom_behaviors:

                if custom_behavior.scope is Behavior_scope.INDIVIDUAL:

                    # Pairs of directional behaviors (inverted order for other behavior direction)
                    tag_dict[aid + undercond + custom_behavior.name] = custom_behavior.annotate_behavior(behavior_ctx, aid) 
      
        tag_dict[aid + undercond + "climb-arena"] = behavior_climb_arena.annotate_behavior(behavior_ctx, aid) 

        tag_dict[aid + undercond + "sniff-arena"] = behavior_sniff_arena.annotate_behavior(behavior_ctx, aid)

        tag_dict[aid + undercond + "immobility"] = behavior_immobility.annotate_behavior(behavior_ctx, aid)
    
        tag_dict[aid + undercond + "stat-lookaround"] = behavior_stat_lookaround.annotate_behavior(behavior_ctx, aid)
    
        # Multi-behavior activity
        activity_dict = behavior_detect_activity.annotate_behavior(behavior_ctx, aid)
    
        tag_dict[aid + undercond + "stat-active"] = activity_dict["stat-active"]
        tag_dict[aid + undercond + "stat-passive"] = activity_dict["stat-passive"]
        tag_dict[aid + undercond + "moving"] = activity_dict["moving"]
    
        tag_dict[aid + undercond + "sniffing"] = behavior_sniffing.annotate_behavior(behavior_ctx, aid)
    
        # Multi-behavior for continuous behaviors
        continuous_meaures = behavior_continuous.annotate_behavior(behavior_ctx, aid)
    
        # NOTE: It's important that speeds remain the last columns.
        # Preprocessing for weakly supervised autoencoders relies on this (or at least did rely on it at some point)
        tag_dict[aid + undercond + "distance"] = continuous_meaures["distance"] 
        tag_dict[aid + undercond + "cum-distance"] = continuous_meaures["cum-distance"]  
        tag_dict[aid + undercond + "speed"] = continuous_meaures["speed"]   
        
    tag_df = pd.DataFrame(tag_dict).fillna(0).astype(float)

    return tag_df


def calculate_close_range(df: pd.DataFrame, mouse_id: str, bodypart: str, threshold: float):
    """Detects for a given set of mouse coordinates if the selected bodypart of the selected mouse is close to any bodypart of any other mouse for each frame.

    Args:
        df (pd.DataFrame): Dataframe containing coordinates of multiple mice
        mouse_id (str): Id of the target mouse
        bodypart (str): Bodypart of the target mouse that should be used for distance calculation
        threshold (float): Maximum distance that triggers "closeness"

    Returns:
        proximity_mask (np.array): Boolean numpy array set to True for each frame in which the lected bodypart of the selected mosue was closer than threshold to any other mouse, False otherwise.

    """    
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


def validate_custom_behaviors(custom_behaviors: list[DeepOF_behavior] = None, custom_behavior_inputs: dict = {}): 

    if custom_behaviors is None:
        return None
    if custom_behaviors is not None and (not isinstance(custom_behaviors,list) or not isinstance(custom_behaviors[0], DeepOF_behavior)): # pragma: no cover
        raise ValueError("\"custom_behaviors\" need to be a list of DeepOF_behavior objects or None!")
    if not isinstance(custom_behavior_inputs,dict): # pragma: no cover
        raise ValueError("\"custom_behavior_inputs\" needs to be a dictionary!")
    CUSTOM_BEHAVIORS=[]
    for custom_behavior in custom_behaviors:
        if "_" in custom_behavior.name: # pragma: no cover
            raise ValueError("No \"_\" allowed in behavior names. Use \"-\" instead")
        if not custom_behavior.scope==Behavior_scope.INDIVIDUAL and custom_behavior.output_kind==Behavior_output.CONTINUOUS: # pragma: no cover
            raise NotImplementedError("Currently continuous behaviors are only supported for individuals!")
        if (custom_behavior.name in SINGLE_BEHAVIORS or custom_behavior.name in SYMMETRIC_BEHAVIORS or 
        custom_behavior.name in ASYMMETRIC_BEHAVIORS or custom_behavior.name in CONTINUOUS_BEHAVIORS): # pragma: no cover
            raise ValueError(f"The behavior name {custom_behavior.name} is already in use!")
        if custom_behavior.name in CUSTOM_BEHAVIORS: # pragma: no cover
            raise ValueError(f"All your custom behaviors need unique names. The name {custom_behavior.name} occurs at least twice!")
        CUSTOM_BEHAVIORS.append(custom_behavior.name)


def assign_custom_behavior_colors(custom_behaviors: list[DeepOF_behavior] = None):
    """Returns a list of hex colors (same order as custom_behaviors), uses user defined colors if available"""

    if custom_behaviors is None:
        return None

    pal = cycle(list(CUSTOM_BEHAVIOR_COLOR_MAP.values()))
    for idx, custom_behavior in enumerate(custom_behaviors): 

        if custom_behavior.color is not None and isinstance(custom_behavior.color, str) and re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', custom_behavior.color):
            continue
        else:
            custom_behaviors[idx] = custom_behavior.set_color(next(pal))
    
    return custom_behaviors

    
        

    
    

if __name__ == "__main__":
    # Ignore warnings with no downstream effect
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
