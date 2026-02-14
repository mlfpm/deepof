# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof

"""Plotting utility functions for the deepof package."""
import calendar
import copy
import os
import re
import time
import warnings
import itertools
from typing import Any, List, NewType, Tuple, Union, Optional, NamedTuple
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from matplotlib.patches import Ellipse
from natsort import os_sorted
from enum import Enum
from scipy.interpolate import interp1d


import deepof.annotation_utils
import deepof.post_hoc
import deepof.utils
from deepof.data_loading import get_dt
from deepof.config import (
    PROGRESS_BAR_FIXED_WIDTH,
    ONE_ANIMAL_COLOR_MAP,
    TWO_ANIMALS_COLOR_MAP,
    DEEPOF_8_BODYPARTS,
    DEEPOF_11_BODYPARTS,
    DEEPOF_14_BODYPARTS,
    BODYPART_COLORS,
    SINGLE_BEHAVIORS,
    SYMMETRIC_BEHAVIORS,
    ASYMMETRIC_BEHAVIORS,
    CONTINUOUS_BEHAVIORS,
    ARENA_COLOR,
    ROI_COLORS,
)


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# ENUMS # 

# DeepOF saves all distances internally in mm, correspondingly thsi enum contains appropriate conversion factors
class DistanceUnit(Enum):
    pixel = 0.0 
    px = 0.0
    mm = 1.0 # identity, measures are saved in mm per default
    millimeter = 1.0
    cm = 10
    centimeter = 10
    m = 1000
    meter = 1000
    km = 1000000
    kilometer = 1000000

    def factor(self, mm_to_pix):
        """Multiplier to convert mm -> this unit. mm_to_pix can be scalar or array-like."""
        if self in (DistanceUnit.px, DistanceUnit.pixel):
            return np.asarray(mm_to_pix, dtype=float)
        return 1.0 / self.value

    @classmethod
    def parse(cls, unit: str) -> "DistanceUnit":
        try:
            return cls[unit]
        except KeyError as e:
            opts = ", ".join(cls.__members__.keys())
            raise ValueError(f'Unknown distance unit "{unit}". Valid options are: {opts}') from e

# DeepOF saves all distances internally in frames, correspondingly this enum calculates appropriate conversion factors
class TimeUnit(Enum):
    fr = 0.0 # identity (frames -> frames)
    frames = 0.0   
    s     = 1.0   # seconds per unit
    seconds = 1.0
    min   = 60.0
    minutes = 60.0
    h     = 3600.0
    hours = 3600.0

    def factor(self, fps: float) -> float:
        """Multiplier to convert frames -> this unit."""
        if self is TimeUnit.frames or fps is None:
            return 1.0
        return 1.0 / (fps * self.value)
    
    @classmethod
    def parse(cls, unit: str) -> "TimeUnit":
        try:
            return cls[unit]
        except KeyError as e:
            opts = ", ".join(cls.__members__.keys())
            raise ValueError(f'Unknown time unit "{unit}". Valid options are: {opts}') from e


# Native time unit in DeepOF is 
class Speed_Unit(Enum):
    mm_s = 1 
    m_s = 0.001
    m_h = 3.6


def time_to_seconds(time_string: str) -> float:
    """Compute seconds as float based on a time string.

    Args:
        time_string (str): time string as input (format HH:MM:SS or HH:MM:SS.SSS...).

    Returns:
        seconds (float): time in seconds
    """
    seconds = None
    if re.match(r"^\b\d{1,6}:\d{1,6}:\d{1,6}(?:\.\d{1,9})?$", time_string) is not None:
        time_array = np.array(re.findall(r"[-+]?\d*\.?\d+", time_string)).astype(float)
        seconds = 3600 * time_array[0] + 60 * time_array[1] + time_array[2]
        seconds=np.round(seconds * 10**9) / 10**9

    return seconds


def seconds_to_time(seconds: float, cut_milliseconds: bool = True) -> str:
    """Compute a time string based on seconds as float.

    Args:
        seconds (float): time in seconds
        cut_milliseconds (bool): decides if milliseconds should be part of the output, defaults to True

    Returns:
        time_string (str): time string (format HH:MM:SS or HH:MM:SS.SSS...)
    """
    time_string = None
    _hours = np.floor(seconds / 3600)
    _minutes = np.floor((seconds - _hours * 3600) / 60)
    _seconds = np.floor((seconds - _hours * 3600 - _minutes * 60))
    _milli_seconds = seconds - np.floor(seconds)

    if cut_milliseconds:
        time_string = f"{int(_hours):02d}:{int(_minutes):02d}:{int(_seconds):02d}"
    else:
        time_string = f"{int(_hours):02d}:{int(_minutes):02d}:{int(_seconds):02d}.{int(np.round(_milli_seconds*10**9)):09d}"
        l_max = time_string.find(".") + 10
        time_string = time_string[0:l_max]

    return time_string


def hex_to_BGR(hex_color):
    color = hex_color.lstrip('#')
    return tuple(int(color[i:i+2], 16) for i in (4, 2, 0))

def BGR_to_hex(bgr_color):
    r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def RGB_to_hex(bgr_color):
    r, g, b = bgr_color[0], bgr_color[1], bgr_color[2]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

def get_behavior_colors(behaviors: list, animal_ids: Union[list, pd.DataFrame]=None):
    """
    Gets corresponding colors for all supervised behaviors or clusters within behaviors list.

    Args:
        behaviors (list): List of strings containing behaviors
        animal_ids Union[list,pd.DataFrame]: Either list of strings representing animal ids or supervised dataframe from which said list can be automatically extracted.

    Returns:
        list: A list of strings that contain hex color codes for each behavior. Will return None and display a warning for unknown behaviors.
    """    


    # Organize input
    if type(behaviors)==str:
        behaviors=[behaviors]
    if animal_ids is None:
        pass
    elif type(animal_ids) == str:
        animal_ids=[animal_ids]
    # extract aids from data frame columns    
    elif type(animal_ids)==pd.DataFrame:
        animal_ids_raw=animal_ids.columns
        # Get list of all aids in each behavior
        animal_ids_raw=[s.split('_')[:-1] for s in animal_ids_raw]
        # Flatten list
        flat_aid_list = [aid for aid_list in animal_ids_raw for aid in aid_list]
        animal_ids=list(np.sort(np.unique(flat_aid_list)))
        if len(animal_ids) == 0:
            animal_ids = ['']
    else:
        animal_ids=list(np.sort(animal_ids))
        
    #######
    # Set cluster colors
    #######

    # Find all cluster behaviors if any
    clusters=[re.search(r'Cluster(_| )\d+', behavior)[0] 
                for behavior 
                in behaviors
                if (re.search(r'Cluster(_| )\d+', behavior)) is not None
    ]
    # Find maximum Cluster
    Cluster_max=1
    if len(clusters) > 0:
        Cluster_max=np.max([int(re.search(r'\d+', cluster)[0]) for cluster in clusters])+1
    # Generate color map of appropriate length
    cluster_colors = np.tile(
        list(sns.color_palette("tab20").as_hex()),
        int(np.ceil(Cluster_max / 20)),
    )

    #######
    # Set supervised colors
    #######

    # Behavior name lists. Should ideally be imported from elsewhere in the future
    single_behaviors=SINGLE_BEHAVIORS
    symmetric_behaviors=SYMMETRIC_BEHAVIORS
    asymmetric_behaviors=ASYMMETRIC_BEHAVIORS

    # create names of supervised behaviors from animal ids and raw behavior names in correct order
    if animal_ids is None or animal_ids[0]=='':
        supervised = single_behaviors
        color_map = ONE_ANIMAL_COLOR_MAP
    elif len(animal_ids)==1:
        supervised = single_behaviors
        supervised =  [animal_ids[0] + "_" + behavior for behavior in single_behaviors]
        color_map = ONE_ANIMAL_COLOR_MAP
    else:
        supervised = generate_behavior_combinations(animal_ids,symmetric_behaviors,asymmetric_behaviors,single_behaviors, False)
        color_map = TWO_ANIMALS_COLOR_MAP

    supervised_max = 1
    if len(supervised) > 0:
        supervised_max = len(supervised)
    # Generate color map of appropriate length
    supervised_colors = np.tile(
        color_map,
        int(np.ceil(supervised_max / len(color_map))),
    )

    # Select appropriate color for all given behaviors
    colors=[]
    for behavior in behaviors:
        if behavior in clusters:
            colors.append(cluster_colors[int(re.search(r'\d+', behavior)[0])])
        elif behavior in supervised:
            colors.append(supervised_colors[supervised.index(behavior)])
        else:
            colors.append(None)

    return colors


def generate_behavior_combinations(animal_ids, symmetric_behaviors=True, asymmetric_behaviors=True, single_behaviors=True, continuous_behaviors=True):
    """
    Generates combinations of animal IDs with different types of behaviors exactly as in supervised annotations.

    Args:
        animal_ids (list): List of strings representing animal IDs.
        symmetric_behaviors (list): List of symmetric paired behaviors.
        asymmetric_behaviors (list): List of asymmetric paired behaviors.
        single_behaviors (list): List of single mouse behaviors.

    Returns:
        list: A list of strings with the combined animal IDs and behaviors.
    """
    result = []
    # Defaults for boolean true false inputs if no list of names is given
    if symmetric_behaviors==True:
        symmetric_behaviors = SYMMETRIC_BEHAVIORS
    elif symmetric_behaviors==False:
        symmetric_behaviors=[]
    if asymmetric_behaviors==True:
        asymmetric_behaviors = ASYMMETRIC_BEHAVIORS
    elif asymmetric_behaviors==False:
        asymmetric_behaviors=[]
    if single_behaviors==True:
        single_behaviors = SINGLE_BEHAVIORS
    elif single_behaviors==False:
        single_behaviors=[]
    if continuous_behaviors==True:   
        continuous_behaviors=CONTINUOUS_BEHAVIORS
    elif continuous_behaviors==False:
        continuous_behaviors=[]

    if animal_ids is None:
        animal_ids=[""]
    else:
        animal_ids=[id + "_" for id in animal_ids]

    
    # Process symmetric paired behaviors
    for behavior in symmetric_behaviors:
        for pair in itertools.combinations(animal_ids, 2):
            # Sort the pair to ensure consistent order and avoid duplicates
            sorted_pair = sorted(pair)
            combined = f"{sorted_pair[0]}{sorted_pair[1]}{behavior}"
            result.append(combined)
    
    # Process asymmetric paired behaviors
    for behavior in asymmetric_behaviors:
        for pair in itertools.permutations(animal_ids, 2):
            combined = f"{pair[0]}{pair[1]}{behavior}"
            result.append(combined)
    
    # Process single mouse behaviors
    for animal_id in animal_ids:
        for behavior in single_behaviors:
            if behavior != "missing" and behavior not in CONTINUOUS_BEHAVIORS:
                combined = f"{animal_id}{behavior}"
                result.append(combined)
    
    # Add missing
    if "missing" in single_behaviors:            
        result = result + [id + "missing" for id in animal_ids] 
    # Add continuous behaviors
    for cont_behavior in continuous_behaviors:
        result = result + [id + cont_behavior for id in animal_ids]           
    
    return result


def calculate_average_arena(
    all_vertices: dict[List[Tuple[float, float]]], num_points: int = 10000
) -> np.array:
    """
    Calculates the average arena based on a list of polynomial vertices
    lists representing arenas. Polynomial vertices can have different lengths and start at different positions

    Args:
        all_vertices (dict[List[Tuple[float, float]]]): A dictionary of lists of 2D tuples representing the vertices of the arenas.
        num_points (int): number of points in the averaged arena.

    Returns:
        numpy.ndarray: A 2D NumPy array containing the averaged arena.
    """

    # ensure that enough points are available for interpolation
    max_length = max(len(lst) for lst in all_vertices.values()) + 1
    assert (
        num_points > max_length
    ), "The num_points variable needs to be larger than the longest list of vertices!"

    # initialize averaged arena polynomial
    avg_points = np.empty([num_points, 2])
    avg_points.fill(0.0)

    # iterate over all arenas
    for key in all_vertices.keys():
        # calculate relative segment lengths between vertices
        vertices = np.stack(all_vertices[key]).astype(float)
        vertices = np.insert(vertices, 0, vertices[-1, :]).reshape(
            -1, 2
        )  # close polynomial
        xy1 = vertices[:-1, :]
        xy2 = vertices[1:, :]
        segment_lengths = np.sqrt(((xy1 - xy2) ** 2).sum(1))
        segment_lengths = segment_lengths / (np.sum(segment_lengths) + 0.00001)

        # Calculate the number of additional points for each segment
        N_new_points = np.round(segment_lengths * (num_points)).astype(int)

        # ensure that the sum of all lengths after discretization is the chosen interpolated length
        if np.sum(N_new_points) != (num_points):
            N_new_points[np.argmax(N_new_points)] += num_points - np.sum(N_new_points)

        # cumulative sum for indexing and new empty arrays
        Cumsum_points = np.insert(np.cumsum(N_new_points), 0, 0)
        intp_points = np.empty([num_points, 2])
        intp_points.fill(np.nan)

        # Fill interpolated arena with new values from interpolation for all edges
        for j in range(len(vertices) - 1):

            start_point = vertices[j, :]
            end_point = vertices[j + 1, :]
            interp_points = N_new_points[j]

            intp_points[Cumsum_points[j] : Cumsum_points[j + 1], 0] = np.linspace(
                start_point[0], end_point[0], interp_points
            )
            intp_points[Cumsum_points[j] : Cumsum_points[j + 1], 1] = np.linspace(
                start_point[1], end_point[1], interp_points
            )

        # reorganize points so that array starts at top left corner and sum to average
        min_pos = np.argmin(np.sum(intp_points, 1))
        avg_points[0 : (num_points - min_pos)] += intp_points[min_pos:]
        avg_points[(num_points - min_pos) :] += intp_points[:min_pos]

    avg_points = avg_points / len(all_vertices)

    return avg_points


def _filter_embeddings(
    coordinates,
    embeddings,
    soft_counts,
    supervised_annotations,
    exp_condition,
):
    """Auxiliary function to plot_embeddings. Filters all available data based on the provided keys and experimental condition."""
    # Get experimental conditions per video
    if embeddings is None and supervised_annotations is None:
        raise ValueError(
            "Either embeddings and soft_counts or supervised_annotations must be provided."
        )

    try:
        if exp_condition is None:
            exp_condition = list(coordinates.get_exp_conditions.values())[0].columns[0]

        concat_hue = [
            str(coordinates.get_exp_conditions[i][exp_condition].values[0])
            for i in list(embeddings.keys())
        ]
        soft_counts = soft_counts.filter_videos(embeddings.keys())

    except AttributeError:
        if exp_condition is None:
            exp_condition = list(supervised_annotations._exp_conditions.values())[
                0
            ].columns[0]

        concat_hue = [
            str(coordinates.get_exp_conditions[i][exp_condition].values[0])
            for i in list(supervised_annotations.keys())
        ]

    # Keep only those experiments for which we have an experimental condition assigned
    if embeddings is not None:
        embeddings = {
            key: val
            for key, val in embeddings.items()
            if key in coordinates.get_exp_conditions.keys()
        }
        soft_counts = {
            key: val
            for key, val in soft_counts.items()
            if key in coordinates.get_exp_conditions.keys()
        }
    if supervised_annotations is not None:
        supervised_annotations = {
            key: val
            for key, val in supervised_annotations.items()
            if key in coordinates.get_exp_conditions.keys()
        }

    return embeddings, soft_counts, supervised_annotations, concat_hue


def _get_polygon_coords(data, animal_id=""):
    """Generate polygons to animate for the indicated animal in the provided dataframe."""
    if animal_id:
        animal_id += "_"

    elif animal_id is None:
        animal_id = ""

    bodypart_list=list(data.columns.levels[0])
    # Remove animal ids from bodyparts to compare with different types of raw bodypart lists
    if any([bp.startswith(animal_id) for bp in bodypart_list]):
        id_start = len(animal_id)
        bodypart_list=list(set([bp[id_start:] for bp in bodypart_list]))
    bodypart_list.sort()
    if bodypart_list == DEEPOF_11_BODYPARTS:
        head_names = [f"{animal_id}Nose", f"{animal_id}Left_ear", 
                      f"{animal_id}Spine_1", f"{animal_id}Right_ear"]
        body_names = [f"{animal_id}Spine_1", f"{animal_id}Left_fhip", 
                      f"{animal_id}Left_bhip", f"{animal_id}Spine_2", 
                      f"{animal_id}Right_bhip", f"{animal_id}Right_fhip"]
        tail_names = [f"{animal_id}Spine_2", f"{animal_id}Tail_base"]
    elif bodypart_list == DEEPOF_14_BODYPARTS:
        head_names = [f"{animal_id}Nose", f"{animal_id}Left_ear", 
                      f"{animal_id}Spine_1", f"{animal_id}Right_ear"]
        body_names = [f"{animal_id}Spine_1", f"{animal_id}Left_fhip", 
                      f"{animal_id}Left_bhip", f"{animal_id}Tail_base", 
                      f"{animal_id}Right_bhip", f"{animal_id}Right_fhip"]
        tail_names = [f"{animal_id}Tail_base", f"{animal_id}Tail_1",
                      f"{animal_id}Tail_2", f"{animal_id}Tail_tip"]
    elif bodypart_list == DEEPOF_8_BODYPARTS:
        head_names = [f"{animal_id}Nose", f"{animal_id}Left_ear", 
                      f"{animal_id}Right_ear"]
        body_names = [f"{animal_id}Left_fhip", f"{animal_id}Right_fhip", 
                      f"{animal_id}Tail_base"]
        tail_names = [f"{animal_id}Tail_base", f"{animal_id}Tail_tip"]
    else:
        raise ValueError(f"Invalid configuration: {list(data.columns.levels[0]).sort()}")

    # Helper function to safely extract body parts
    def extract_parts(names):
        parts = []
        for name in names:
            try:
                parts.append(data.xs(name, axis=1).values)
            except KeyError:
                continue
        return np.concatenate(parts, axis=1) if parts else np.empty((data.shape[0], 0))
    
    # Extract all segments
    head = extract_parts(head_names)
    body = extract_parts(body_names)
    tail = extract_parts(tail_names)

    return [head, body, tail]


def _process_animation_data(
    coords: table_dict,
    cur_embeddings: table_dict,
    cur_soft_counts: table_dict,
    min_confidence: float,
    min_bout_duration: int,
    selected_cluster: np.ndarray,
):
    """Auxiliary function to process data for animation outputs.

        Args:
        coords (table_dict): position data to prepare for displaying.
        cur_embeddings (table_dict): embedding data to prepare for displaying
        cur_soft_counts (table_dict): soft_counts to prepare for displaying
        min_confidence (float): Minimum confidence threshold to render a cluster assignment bout.
        min_bout_duration (int): Minimum number of frames to render a cluster assignment bout.
        selected_cluster (int): cluster to filter. If provided together with cluster_assignments,

        Returns:
        coords (table_dict): position data afetr preprocessing
        twoDim_embeddings (list(np.ndarray)): 2D UMAP representation of embeddings
        cluster_embedding (list(np.ndarray)): 2D UMAP representation of embeddings for specific cluster with sufficient confidence
        concat_embedding (np.ndarray): 2D UMAP representation of embeddings with sufficient confidence
        hard_counts (np.ndarray): 1D array of active cluster number for each frame

    """
    
    cluster_embedding, concat_embedding = None, None

    # Filter assignments and embeddings
    cluster_confidence = cur_soft_counts.max(axis=1)
    hard_counts = cur_soft_counts.argmax(axis=1)
    confidence_indices = np.ones(hard_counts.shape[0], dtype=bool)

    # Compute bout lengths, and filter out bouts shorter than min_bout_duration
    full_confidence_indices = deepof.utils.filter_short_bouts(
        hard_counts,
        cluster_confidence,
        confidence_indices,
        min_confidence,
        min_bout_duration,
    )
    confidence_indices = full_confidence_indices.copy()

    # Reduce full embeddings to 2D UMAP
    reducers = deepof.post_hoc.compute_UMAP(cur_embeddings, hard_counts)
    twoDim_embeddings = reducers[1].transform(reducers[0].transform(cur_embeddings))


    # Ensure that shapes are matching
    assert (
        twoDim_embeddings.shape[0] == coords.shape[0]
    ), "there should be one embedding per row in data"
    assert (
        len(cur_soft_counts) == coords.shape[0]
    ), "there should be one cluster assignment per row in data"

    concat_embedding = twoDim_embeddings

    # Only keep information for specific cluster if specific cluster is chosen
    if selected_cluster is not None:

        assert selected_cluster in set(
            hard_counts
        ), "The cluster you selected did not occur in the data range given!"

        cluster_embedding = twoDim_embeddings[hard_counts == selected_cluster]
        confidence_indices = confidence_indices[
            hard_counts == selected_cluster
        ]

        coords = coords.loc[hard_counts == selected_cluster, :]
        coords = coords.loc[confidence_indices, :]
        cluster_embedding = cluster_embedding[confidence_indices]
        concat_embedding = concat_embedding[full_confidence_indices]
        hard_counts = hard_counts[full_confidence_indices]

        assert coords.shape[0]>0, (
            "In the given range the selected cluster did occur, but was only predicted with low confidence or in very short sections!\n"
            "Either increase bin_size, increase min_confidence or lower min_bout_duration!"
        )

    else:
        cluster_embedding = twoDim_embeddings

    return (
        coords,
        [twoDim_embeddings],
        [cluster_embedding],
        concat_embedding,
        hard_counts,
    )



def create_bin_pairs(L_array: int, N_time_bins: int):
    """
    Creates a List of bin_index and bin_size pairs when splitting a list in N_time_bins

    Args:
        L_array (int): Length of the array to index.
        N_time_bins (int): number of time bins to create.

    Returns:
        bin_pairs (list(tuple)): A 2D list containing start and end positions of each bin.
    """

    if L_array < N_time_bins:
        L_array = N_time_bins
        print(
            "Number of bins needs to be smaller or equal array length! Set L_array=N_time_bins!"
        )

    # Calculate the base bin size and the number of bins that need an extra element
    base_bin_size = L_array // N_time_bins
    extra_elements = L_array % N_time_bins

    bin_pairs = []
    current_index = 0

    for i in range(N_time_bins):
        # Determine the size of the current bin
        if i < extra_elements:
            bin_size = base_bin_size + 1
        else:
            bin_size = base_bin_size

        # Add the pair (bin_size, bin_index) to the result
        bin_pairs.append([current_index, current_index + bin_size - 1])

        # Update the current_index for the next iteration
        current_index += bin_size

    return bin_pairs


def validate_custom_bins(N_time_bins, L_shortest, custom_time_bins = None, hide_time_bins = None, min_bins_required = 4):

    # Init bin ranges if not given
    if not custom_time_bins:
        custom_time_bins = create_bin_pairs(
            L_shortest, N_time_bins
        )
    
    # Init hidden bins if not given
    if not hide_time_bins:
        hide_time_bins = np.array([False] * len(custom_time_bins))
    elif not len(hide_time_bins) == len(custom_time_bins):
        raise ValueError(
            f'The variables "hide_time_bins" and "custom_time_bins" need to have the same length!'
        )
    else:
       hide_time_bins= np.array(hide_time_bins)

    # Check custom_time_bin validity
    if len(
        custom_time_bins
    ) >= min_bins_required and all(  # list has at least 4 bins (less lead to failing of the interpol. function later)
        isinstance(sublist, list) and len(sublist) == 2 for sublist in custom_time_bins
    ):  # List has shape Nx2

        # Convert time string elements to integers
        custom_time_bins = [
            [
                int(np.round(time_to_seconds(sublist[k]) * coordinates._frame_rate))
                if type(sublist[k]) == str
                else sublist[k]
                for k in range(len(sublist))
            ]
            for sublist in custom_time_bins
        ]

        # Further checks
        if not all(
            all(isinstance(x, int) and x >= 0 for x in sublist)
            for sublist in custom_time_bins
        ) or not all(  # Lists consist of positive integers
            sublist[0] <= sublist[1] for sublist in custom_time_bins
        ):  # List elements increase
            raise ValueError(
                f'Each element of "custom_time_bins" needs to contain either two integers > 0 and int2 > int1\n'
                "or the corresponding time strings given as HH:MM:SS.SS... with t_str2 > t_str1!"
            )
        elif np.max(custom_time_bins) >= L_shortest:
            raise ValueError(
                f'"custom_time_bins" contains at least one element that exceeds the length of your shortest data set!'
            )
        # Warn in case of overlapping elements
        elif not (
            list(itertools.chain(*custom_time_bins)) == sorted(list(itertools.chain(*custom_time_bins)))
        ):
            warning_message = (
                "\033[38;5;208m\n"
                'Warning! Your "custom_time_bins" list contains overlapping elements!\n'
                f"Ignore this warning if providing overlapping or repeating bins was your intention.\n"
                "\033[0m"
            )
            warnings.warn(warning_message)
    else:
        raise ValueError(
            f'At least {min_bins_required} bins are required! If "custom_time_bins" is used, it needs to be a list of at least 4 elments with each element being a list!'
        )
    
    return custom_time_bins, hide_time_bins


def postprocess_df_bins(df: pd.DataFrame, bin_lengths, hide_time_bins):

    #Remove missing values (less than 20% of values remaining per group) 
    min_frac = 0.05 #20
    num_bins = len(bin_lengths)
    condition_values = sorted(df["exp_condition"].astype(str).unique().tolist())
    behavior_to_plot = df.columns[2]

    # Add bin length column
    time_bin_idx = df.columns.get_loc('time_bin')
    df.insert(time_bin_idx + 1, 'bin_length', np.array(bin_lengths)[df['time_bin'].astype(int)])
    
    # Create table denoting the nan percentage for each group and bin
    coverage = df.pivot_table(
        index="time_bin", columns="exp_condition", values=behavior_to_plot,
        aggfunc=lambda s: s.notna().mean()
    ).reindex(index=range(num_bins), columns=list(condition_values)).fillna(0.0)

    enough_data_per_bin = coverage.ge(min_frac).all(axis=1).to_numpy()
    hide_time_bins = hide_time_bins | ~enough_data_per_bin # hide additonal bins if groups in these bins only have little data
    if not all(enough_data_per_bin):
        warning_message = (
            "\033[38;5;208m\n"
            f'Warning! The time bins {np.where(~enough_data_per_bin)[0]+1}\n'
            f"are empty in more than {100-min_frac*100}% of your tables and hence were excluded!\n"
            "\033[0m"
        )
        print(warning_message)    

    # exclude time bins that are always NaN (which will also exclude them from statistics)
    mask = df.groupby(['time_bin'])[behavior_to_plot].transform(lambda x: x.notna().any())
    df = df[mask].copy()

    assert np.sum(df[behavior_to_plot])>0.000001, "None of the selected behavior was measured within the given time bins and ROI!" 

    return df



def cohend(array_a: np.array, array_b: np.array):
    """
    calculate Cohen's d effect size. Does not assume equal population standard deviations, and can still be used for unequal sample sizes

    Args:
        array_a (np.array): First array of values to compare.
        array_b (np.array): Second array of values to compare.

    Returns:
        Cohens d (int):

    Cohen's d can be used to calculate the standardized difference between two categories, e.g. difference between means
    The value of Cohen's d varies from 0 to infinity. Sign indicates directionality?
    show both hypothesis test (likelihood of observing the data given an assumption (null hypothesis) w p-value) and effect size (quantify the size of the effect assuming that the effect is present)
    Cohen's d measures the difference between the mean from two Gaussian-distributed variables.
    It is a standard score that summarizes the difference in terms of the number of standard deviations.
    Because the score is standardized, there is a table for the interpretation of the result, summarized as:

        Small Effect Size: d=0.20
        Medium Effect Size: d=0.50
        Large Effect Size: d=0.80.
    """
    if not hasattr(cohend, '_warning_issued'):
        cohend._warning_issued = False

    if len(array_a)<2 or len(array_b) < 2:
        warnings.warn(
            '\033[33mInfo! At least one of the selected groups has only one element!\n Setting cohens D to 0!\033[0m'
            ) 
        return 0

    # Calculate the size of samples
    n1, n2 = len(array_a), len(array_b)
    # Calculate the means of the samples
    u1, u2 = np.mean(array_a), np.mean(array_b)

    # Calculate the pooled standard deviation, unbiased estimate of the variance (with ddof=1), and it adjusts for the degrees of freedom in the calculation.
    s = np.sqrt(
        ((n1 - 1) * np.var(array_a, ddof=1) + (n2 - 1) * np.var(array_b, ddof=1))
        / (n1 + n2 - 2)
    )
    # Check if the pooled standard deviation is 0
    if s < 1e-10:
        # Handle the case when the standard deviation is 0 by setting effect size to 0
        if not cohend._warning_issued:
            print("Standard deviation is close to 0 (std < 1e-10). Setting Cohen's d to 0.")
            cohend._warning_issued = True
        return 0
    else:
        # Calculate the effect size (Cohen's d)
        return (u1 - u2) / s


def cohend_effect_size(d: float):  # pragma: no cover
    """
    categorizes Cohen's d effect size.

    Args:
        d (float): Cohens d

    Returns:
        int: Categorized effect size 
    """

    if abs(d) >= 0.8:
        return 3  # Large effect
    elif abs(d) >= 0.5:
        return 2  # Medium effect
    elif abs(d) < 0.5:
        return 1  # Small effect
    else:
        return 0

class _BinningResult(NamedTuple):
    """A structured result from a binning strategy."""
    bin_info: Any
    start_too_late: dict[str, bool]
    end_too_late: dict[str, bool]
    pre_start_warnings: dict[str, str]
    bin_size_frames: Optional[int] = None


def _get_bins_from_precomputed(
    precomputed_bins: np.ndarray, table_lengths: dict[str, int]
) -> _BinningResult:
    """Strategy for when precomputed bins are provided."""
    bin_info = {}
    end_too_late = {key: False for key in table_lengths}

    for key, length in table_lengths.items():
        arr = np.full(length, False, dtype=bool)
        effective_len = min(length, len(precomputed_bins))
        arr[:effective_len] = precomputed_bins[:effective_len]
        bin_info[key] = np.where(arr)[0]

        if len(precomputed_bins) > length:
            end_too_late[key] = True

    return _BinningResult(bin_info, {}, end_too_late, {})


def _get_bins_from_frames(
    bin_size: int, bin_index: int, table_lengths: dict[str, int], frame_rate: float
) -> _BinningResult:
    """Strategy for when bin size/index are given as integers."""
    bin_size_frames = bin_size
    if bin_size_frames <= 0:
        raise ValueError("bin_size must result in a frame count greater than 0.")

    bin_info = {}
    start_too_late = {key: False for key in table_lengths}
    end_too_late = {key: False for key in table_lengths}

    for key, length in table_lengths.items():
        start_frame = bin_index
        end_frame = start_frame + bin_size_frames

        if start_frame >= length:
            start_too_late[key] = True
        if end_frame > length:
            end_too_late[key] = True

        bin_start = min(length, start_frame)
        bin_end = min(length, end_frame)
        bin_info[key] = np.arange(bin_start, bin_end)

    return _BinningResult(bin_info, start_too_late, end_too_late, {}, bin_size_frames)


def _get_bins_from_integers(
    bin_size: int, bin_index: int, table_lengths: dict[str, int], frame_rate: float
) -> _BinningResult:
    """Strategy for when bin size/index are given as integers."""
    bin_size_frames = int(round(bin_size * frame_rate))
    if bin_size_frames <= 0:
        raise ValueError("bin_size must result in a frame count greater than 0.")

    bin_info = {}
    start_too_late = {key: False for key in table_lengths}
    end_too_late = {key: False for key in table_lengths}

    for key, length in table_lengths.items():
        start_frame = bin_size_frames * bin_index
        end_frame = start_frame + bin_size_frames

        if start_frame >= length:
            start_too_late[key] = True
        if end_frame > length:
            end_too_late[key] = True

        bin_start = min(length, start_frame)
        bin_end = min(length, end_frame)
        bin_info[key] = np.arange(bin_start, bin_end)

    return _BinningResult(bin_info, start_too_late, end_too_late, {}, bin_size_frames)


def _get_bins_from_strings(
    bin_size_str: str, bin_index_str: str, table_lengths: dict[str, int],
    start_times: dict[str, str], frame_rate: float
) -> _BinningResult:
    """Strategy for when bin size/index are given as time strings."""
    bin_size_frames = int(round(time_to_seconds(bin_size_str) * frame_rate))
    if bin_size_frames <= 0:
        raise ValueError("bin_size string must represent a duration > 0.")

    bin_info = {}
    start_too_late = {key: False for key in table_lengths}
    end_too_late = {key: False for key in table_lengths}
    pre_start_warnings = {}

    bin_index_sec = time_to_seconds(bin_index_str)

    for key, length in table_lengths.items():
        exp_start_sec = time_to_seconds(start_times[key])
        start_offset_frames = int(round((bin_index_sec - exp_start_sec) * frame_rate))

        if start_offset_frames < 0:
            truncated_len_sec = (start_offset_frames + bin_size_frames) / frame_rate
            pre_start_warnings[key] = seconds_to_time(max(0, truncated_len_sec))
        
        if start_offset_frames >= length:
            start_too_late[key] = True
        
        bin_start = np.clip(start_offset_frames, 0, length)
        bin_end = np.clip(start_offset_frames + bin_size_frames, 0, length)
        
        if start_offset_frames + bin_size_frames > length:
            end_too_late[key] = True

        bin_info[key] = np.arange(bin_start, bin_end)

    return _BinningResult(bin_info, start_too_late, end_too_late, pre_start_warnings, bin_size_frames)


def _get_full_range_bins(table_lengths: dict[str, int]) -> _BinningResult:
    """Strategy to use the full time range for each experiment."""
    bin_info = {key: np.arange(length) for key, length in table_lengths.items()}
    return _BinningResult(bin_info, {}, {}, {})


def _downsample_bins(bin_info: Any, samples_max: int, down_sample: bool) -> Any:
    """Downsamples bin indices if they exceed the maximum allowed samples."""
    downsampled_info = {}
    downsampled_at_all = False
    
    for key, indices in bin_info.items():
        full_length = len(indices)
        if full_length > samples_max:
            downsampled_at_all = True
            if down_sample:
                selected_indices = np.linspace(0, full_length - 1, samples_max, dtype=int)
            else:
                selected_indices = np.arange(samples_max)
            downsampled_info[key] = indices[selected_indices]
        else:
            downsampled_info[key] = indices

    if downsampled_at_all:
        print(
            "\033[33m\n"
            f"Info! Selected range exceeds {samples_max} samples and has been "
            f"{'downsampled' if down_sample else 'cut'} by a factor of approx. {int((full_length-1)/samples_max)}\n"
            "To avoid this, increase 'samples_max'. This will also result in increased computation time"
            "\033[0m"
        )

    return downsampled_info


def _validate_and_warn(
    result: Any,
    table_lengths: dict[str, int],
    frame_rate: float,
    bin_size_orig: Union[int, str],
):
    """Handles all final validation checks and user warnings."""
    if result.pre_start_warnings:
        warn_str = ", ".join(f"{k}: {v}" for k, v in result.pre_start_warnings.items())
        warnings.warn(
            "\033[38;5;208m\n"
            "Chosen time range starts before the data time axis begins in at least "
            f"one experiment. Truncated bin lengths are: {warn_str}"
            "\033[0m"
        )

    for key, is_late in result.start_too_late.items():
        if is_late:
            max_time = seconds_to_time(table_lengths[key] / frame_rate, False)
            max_index = int(np.ceil(table_lengths[key] / result.bin_size_frames)) -1
            raise ValueError(
                f"[Error in {key}]: bin_index is out of range. "
                f"It must be less than {max_time} or index < {max_index} for a "
                f"bin_size of {bin_size_orig}."
            )

    warned_once = False
    for key, is_truncated in result.end_too_late.items():
        if is_truncated and not warned_once:
            truncated_len = seconds_to_time(len(result.bin_info[key]) / frame_rate, False)
            message = (
                "\033[38;5;208m\n"
                f"[For {key} and possibly others]: Chosen time range exceeds signal length. "
                f"Bin size was truncated to {truncated_len}."
                "\033[0m"
            )
            # Add helpful suggestion only if applicable
            if result.bin_size_frames and table_lengths[key] > result.bin_size_frames:
                max_start_time = seconds_to_time((table_lengths[key] - result.bin_size_frames) / frame_rate, False)
                max_index = int(np.ceil(table_lengths[key] / result.bin_size_frames)) - 2
                message += (
                    "\033[38;5;208m\n"
                    f"\nFor full bins, choose start time <= {max_start_time} or "
                    f"index <= {max_index} for a bin_size of {bin_size_orig}."
                    "\033[0m"
                )
            warnings.warn(message)
            warned_once = True


def _preprocess_time_bins(
    coordinates: coordinates,
    bin_size: Optional[Union[int, str]] = None,
    bin_index: Optional[Union[int, str]] = None,
    precomputed_bins: Optional[np.ndarray] = None,
    tab_dict_for_binning: Optional[table_dict] = None,
    experiment_id: Optional[str] = None,
    samples_max: int = 20000,
    down_sample: bool = True,
    given_in_frames=False,
):
    """
    Preprocesses various time-bin formats into a consistent dictionary of indices.

    This function determines the correct indices to use for time-based analysis
    based on one of several possible user inputs (e.g., precomputed arrays,
    integer-based bins, or time-string based bins).

    Args:
        coordinates: deepOF project object containing data and metadata.
        bin_size: Bin size for time filtering. Can be an integer (seconds) or a
                  time string ('HH:MM:SS.sss').
        bin_index: Start of the bin. Can be an integer index or a time string
                   ('HH:MM:SS.sss') for the absolute start time.
        precomputed_bins: A pre-calculated boolean or index array. If provided,
                          `bin_size` and `bin_index` are ignored.
        tab_dict_for_binning: Optional table dictionary to use as a reference for
                              video lengths. Defaults to `coordinates`.
        experiment_id: If specified, processing is limited to this single experiment.
        samples_max: Maximum number of samples to return per experiment. Data is
                     downsampled or cut if the selection is larger.
        down_sample: If True, use uniform downsampling. If False, cut the data
                     at `samples_max`.
        given_in_frames: bin_index and size are directly given in frames with no conversions being necessary

    Returns:
        A dictionary mapping each experiment ID to a numpy array of frame indices.

    Raises:
        ValueError: If inputs are invalid or logically inconsistent (e.g.,
                    bin_size=0, or bin_index is out of bounds).
    """
    # --- 1. Initial Setup and Data Preparation ---
    if precomputed_bins is not None and (bin_size is not None or bin_index is not None):
        warnings.warn(
            "\033[38;5;208m\n"
            "precomputed_bins is provided. Ignoring bin_size and bin_index."
            "\033[0m"
        )

    start_times = coordinates.get_start_times()
    if tab_dict_for_binning:
        table_lengths = {
            k: int(get_dt(tab_dict_for_binning, k, only_metainfo=True)['shape'][0])
            for k in tab_dict_for_binning
        }
    else:
        table_lengths = coordinates.get_table_lengths()

    if experiment_id:
        if experiment_id not in table_lengths:
            raise KeyError(f"Experiment ID '{experiment_id}' not found.")
        start_times = {experiment_id: start_times[experiment_id]}
        table_lengths = {experiment_id: table_lengths[experiment_id]}

    # --- 2. Strategy Selection and Execution ---
    TIME_STR_PATTERN = r"^\d{1,6}:\d{1,6}:\d{1,6}(?:\.\d{1,12})?$"
    result = None

    if precomputed_bins is not None:
        result = _get_bins_from_precomputed(precomputed_bins, table_lengths)
    
    elif isinstance(bin_size, int) and isinstance(bin_index, int) and given_in_frames:
        result = _get_bins_from_frames(
            bin_size, bin_index, table_lengths, coordinates._frame_rate
        )

    elif isinstance(bin_size, int) and isinstance(bin_index, int):
        result = _get_bins_from_integers(
            bin_size, bin_index, table_lengths, coordinates._frame_rate
        )

    elif (isinstance(bin_size, str) and re.match(TIME_STR_PATTERN, bin_size) and
          isinstance(bin_index, str) and re.match(TIME_STR_PATTERN, bin_index)):
        result = _get_bins_from_strings(
            bin_size, bin_index, table_lengths, start_times, coordinates._frame_rate
        )

    elif bin_size is None and bin_index is None:
        result = _get_full_range_bins(table_lengths)

    else:
        warnings.warn(
            "\033[38;5;208m\n"
            "Invalid or mismatched bin_size/bin_index format. "
            "Expected two integers, or two 'HH:MM:SS' strings. "
            "Defaulting to a 60-second bin starting at 0.\033[0m"
        )
        # Recurse with default values for simplicity.
        return _preprocess_time_bins(
            coordinates=coordinates, bin_size=60, bin_index=0, 
            tab_dict_for_binning=tab_dict_for_binning, experiment_id=experiment_id,
            samples_max=samples_max, down_sample=down_sample
        )

    # --- 3. Post-processing: Validation and Downsampling ---
    _validate_and_warn(result, table_lengths, coordinates._frame_rate, bin_size)

    final_bins = _downsample_bins(result.bin_info, samples_max, down_sample)

    return final_bins


def _apply_rois_to_bin_info(
    coordinates: coordinates,
    roi_number: int,
    bin_info_time: dict = None,
    in_roi_criterion: str = "Center",
):
    """Retrieve annotated behaviors that occured within a given roi.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        roi_number (int): number of the roi 
        bin_info_time (dict): A dictionary containing start and end positions or indices for plotting 
        in_roi_criterion (str): Criterion for in roi check, can be a single bodypart, a list of bodyparts or "all" bodyparts of a mouse   
    """

    animal_ids=coordinates._animal_ids
    if animal_ids is None:
        animal_ids=[""]

    # if no time bin info object was given, create one
    if bin_info_time is None:
        bin_info_time={}
        for key in coordinates._tables.keys():
            bin_info_time[key] = np.array(range(0,len(coordinates._tables[key])), dtype=int)

    #unify bin info format
    for key in bin_info_time.keys():
        if len(bin_info_time[key])==2 and bin_info_time[key][0]+1 < bin_info_time[key][1]:
            bin_info_time[key] = np.array(range(bin_info_time[key][0],bin_info_time[key][1]+1), dtype=int)

    bin_info = {}
    for key in bin_info_time.keys():
        bin_info[key] = {}
        bin_info[key]["time"]=bin_info_time[key]
        if roi_number is not None:
            for aid in animal_ids:

                tab = get_dt(coordinates._tables,key)
                roi_polygon=coordinates._roi_dicts[key][roi_number]
                mouse_in_roi = deepof.utils.mouse_in_roi(tab, aid, in_roi_criterion, roi_polygon, coordinates._run_numba)

                # only keep boolean indices that were within the time bins for this mouse
                bin_info[key][aid]=mouse_in_roi[bin_info_time[key]]
            
    return bin_info


def _get_mousevise_behaviors_in_roi(
    cur_supervised: pd.DataFrame,
    local_bin_info: dict,
    animal_ids: Union[str, list], 
):
    """Filter out all frames in which the requested animals are not inside of the ROI"""
    
    # get list of masks for all animals
    masks = [local_bin_info[aid] for aid in animal_ids]
    if not masks:
        return cur_supervised # No animals to filter by, return as is
    
    # Fancy numpy operation
    combined_mask = np.logical_and.reduce(masks)
    
    # Apply the combined mask to the entire DataFrame at once.
    cur_supervised.loc[~combined_mask, :] = np.nan
    return cur_supervised  



def _get_behaviorwise_behaviors_in_roi(
    cur_supervised: pd.DataFrame,
    local_bin_info: dict,
    animal_ids: Union[str, list], 
):
    """Filter out all frames in which the requested animals that take part in each individual behavior are not inside of the ROI"""

    def _get_col_base_name(col: Any) -> str:
        """Safely gets the first level of a column name, handling MultiIndex."""
        return col[0] if isinstance(col, tuple) else col

    # 1. Determine which columns are relevant (involve at least one target_id).
    # This list comprehension is more direct than the original nested loop.
    valid_cols = {
        col for col in cur_supervised.columns 
        if any(_get_col_base_name(col).startswith(animal_id) for animal_id in animal_ids)
    }

    # 2. Invalidate all columns that do not involve any of the target animals.
    invalid_cols = cur_supervised.columns.difference(list(valid_cols))
    if not invalid_cols.empty:
        cur_supervised[invalid_cols] = np.nan

    if not valid_cols:
        return cur_supervised # No relevant columns to process further.

    # 3. Apply ROI masks animal by animal, but only to their relevant columns.
    for animal_id, roi_mask in local_bin_info.items():
        if animal_id == "time":
            continue
        # if there is more than one mosue, add underscores
        animal_id_suffix = animal_id
        if len(local_bin_info)>2:
            animal_id_suffix = animal_id + "_"
            
        # Find which of the valid_cols are associated with the current animal_id
        cols_for_this_animal = [
            col for col in valid_cols 
            if (animal_id_suffix) in _get_col_base_name(col)
        ]
        
        if cols_for_this_animal:
            # Apply the specific ROI mask for this animal to its columns.
            cur_supervised.loc[~roi_mask, cols_for_this_animal] = np.nan
            
    return cur_supervised
    

def get_supervised_behaviors_in_roi(
    cur_supervised: pd.DataFrame,
    local_bin_info: dict,
    animal_ids: Union[str, list], 
    roi_mode: str = "mousewise",
):
    """Filter supervised behaviors based on rois given by animal_ids.

    Args:
        cur_supervised (pd.DataFrame): data frame with supervised behaviors.
        local_bin_info (dict): bin_info dictionary for one experiment, containing field "time" with array of included frames and fields "animal_id" with boolean arrays that denote which mace were within the selcted roi for these frames
        animal_ids (Union[str, list]): single or multiple animal ids
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
 
    Returns:
        cur_supervised (pd.DataFrame): data frame with supervised behaviors with detections outside of the ROI set to NaN
    """
    
    # Check and reformat input
    if not animal_ids:
        return cur_supervised  
    animal_ids = [animal_ids] if isinstance(animal_ids, str) else list(animal_ids)

    cur_supervised=copy.copy(cur_supervised)

    # Filter 
    if roi_mode=="mousewise":
        cur_supervised = _get_mousevise_behaviors_in_roi(cur_supervised,local_bin_info,animal_ids)
    elif roi_mode == "behaviorwise":
        cur_supervised = _get_behaviorwise_behaviors_in_roi(cur_supervised,local_bin_info,animal_ids)
    else:
        raise NotImplementedError("Currently only \"mousewise\" and \"behaviorwise\" are valid roi modes.")
            
    return cur_supervised


def get_unsupervised_behaviors_in_roi(
        cur_unsupervised: np.array,
        local_bin_info: dict,
        animal_ids: str, 
):
    """Filter unsupervised behaviors based on rois given by animal_ids.

    Args:
        cur_unsupervised (np.array): 1D or 2D array with unsupervised behaviors (can be soft or hard counts).
        local_bin_info (dict): bin_info dictionary for one experiment, containing field "time" with array of included frames and fields "animal_id" with boolean arrays that denote which mace were within the selcted roi for these frames
        animal_ids (Union[str, list]): single or multiple animal ids
    
    Returns:
        cur_unsupervised (np.array): 1D or 2D array with unsupervised behaviors with detections outside of the ROI set to NaN (2D) or -1 (1D)
    """

    cur_unsupervised=copy.copy(cur_unsupervised)
    if type(animal_ids)==str:
        animal_ids=[animal_ids]
    elif animal_ids is None:
        animal_ids=[""] 

    if len(cur_unsupervised.shape)==1:
        for aid in animal_ids:
            cur_unsupervised[~local_bin_info[aid]]=-1    
    else:
        for aid in animal_ids: 
            cur_unsupervised[~local_bin_info[aid]]=np.NaN   

    return cur_unsupervised


def get_behavior_frames_in_roi(
    behavior: str,
    local_bin_info: dict,
    animal_ids: Union[str, list],        
):
    """Filter unsupervised behaviors based on rois given by animal_ids.

    Args:
        behavior (str): Behavior for which frames in ROi get determined.
        local_bin_info (dict): bin_info dictionary for one experiment, containing field "time" with array of included frames and fields "animal_id" with boolean arrays that denote which mace were within the selcted roi for these frames
        animal_ids (Union[str, list]): single or multiple animal ids
    
    Returns:
        frames (np.array): 1D array containing all frames for which the animal is (animals are) within the ROI
    """

    if isinstance(animal_ids, str):
        animal_ids=[animal_ids]
    elif animal_ids is None:
        animal_ids=[""]   

    local_bin_info = copy.copy(local_bin_info)
    frames = copy.copy(local_bin_info["time"])

    is_supervised_behavior = False
    if behavior is not None:
        is_supervised_behavior = any([aid+"_" in behavior for aid in animal_ids])
       
    if is_supervised_behavior:
        for aid in local_bin_info.keys():
            if aid == "time":
                continue
            if aid + "_" in behavior:
                frames[~local_bin_info[aid]]=-1
    else:
        for aid in animal_ids:
            frames[~local_bin_info[aid]]=-1
    
    frames=frames[frames >= 0]
    return frames
    

def calculate_FSTTC(preceding_behavior: pd.Series, proximate_behavior: pd.Series, frame_rate: float, delta_T: float=2.0):
    """Calculates the association measure FSTTC between two behaviors given as boolean series"""
    
    # calculate delta T in frames
    delta_T_frames=int(frame_rate*delta_T)
    
    # Get total length of interval in which behaviors can occur
    L = len(preceding_behavior)+1
    # Inits
    preceding_onsets=np.zeros(L)
    proximate_onsets=np.zeros(L)
    preceding_active=np.concatenate(( [0], copy.copy(preceding_behavior.astype(int)), [0] ))
    proximate_active=np.concatenate(( [0], copy.copy(proximate_behavior.astype(int)), [0] )) 

    # Calculate positions of both behavior onsets
    preceding_onsets=np.diff(preceding_active)
    proximate_onsets=np.diff(proximate_active)
    pre_offset_pos = np.where(preceding_onsets == -1)[0]   
    prox_offset_pos = np.where(proximate_onsets == -1)[0]
    prox_onset_pos = np.where(proximate_onsets == 1)[0]


    # Calculate relevant interval of frames close to any behavior starts 
    for pre_stop in pre_offset_pos:
        preceding_active[pre_stop:np.min([pre_stop+delta_T_frames,L])]=1
    for prox_stop in prox_offset_pos:
        proximate_active[prox_stop:np.min([prox_stop+delta_T_frames,L])]=1

    t_A = np.sum(preceding_active)/L #proportion of frames following preceding behavior onset of all frames
    t_B = np.sum(proximate_active)/L #proportion of frames following proximate behavior onset of all frames
    
    if t_A==0 or t_B==0:
        return 0
    else:
        # Calculate FSTTC
        if len(prox_onset_pos) > 0:
            p = np.sum(preceding_active[prox_onset_pos])/len(prox_onset_pos) #proportion of proximate behavior onsets that fall within preceding behavior onset interval
            fsttc = 0.5*((p-t_B)/(1-p*t_B)+(p-t_A)/(1-p*t_A))
        else:
            fsttc = 0
    return fsttc


#@nb.njit(error_model='python')
def calculate_simple_association(
    preceding_behavior: np.ndarray,
    proximate_behavior: np.ndarray,
    frame_rate: float,
    min_T: float = 10.0,
):
    """Calculates Yule's coefficient Q between two behaviors given as boolean arrays"""

    # Early exit in case one of the behaviors is too rare to be meaningful. Returns 0 for no association
    min_T_frames = int(frame_rate * min_T)
    if np.sum(preceding_behavior) < min_T_frames or np.sum(proximate_behavior) < min_T_frames:
        return 0.0
    
    #chi2, p, dof, expected = chi2_contingency(cont_table)
    a = np.sum(preceding_behavior & proximate_behavior) # Both behaviors present
    b = np.sum(preceding_behavior & ~proximate_behavior)  # A present, B absent
    c = np.sum(~preceding_behavior & proximate_behavior)  # A absent, B present
    d = np.sum(~preceding_behavior & ~proximate_behavior)  # Both absent

    # For identical only True or only False arrays
    if ((a * d) + (b * c)) == 0 and (a>0 or d > 0) and (b==0 and c == 0):
        Q = 1
    # For inverse arrays
    elif ((a * d) + (b * c)) == 0 and (b>0 or c > 0) and (a==0 and d == 0):
        Q = -1 
    # Other badly defined cases 
    elif ((a * d) + (b * c)) == 0:
        Q = 0 
    else:
        Q = ((a * d) - (b * c)) / ((a * d) + (b * c))

    return Q


def contiguous_segments(mask: np.ndarray):
    # yields slices for contiguous True blocks
    if mask.ndim != 1:
        mask = np.asarray(mask).ravel()
    if not mask.any():
        return []
    edges = np.where(np.diff(np.r_[False, mask, False]))[0].reshape(-1, 2)
    return [slice(s, e) for s, e in edges]


def scale_units(coordinates, key, data, unit: str, target_distance: str = None, target_time: str = None):
    """
    Scale `data` from `unit` to requested target units and return (scaled, new_unit).
    `unit` can be "<u>" or "<u_num>/<u_den>", where each u is in TimeUnit or DistanceUnit.
    """
    if unit is None:
        return data, None

    fps = float(coordinates._frame_rate)
    mm_to_px = coordinates._scales[key][2] / coordinates._scales[key][3]  # px per mm (per exp_id)

    def sec_per(u: str) -> float:
        tu = TimeUnit.parse(u)
        return (1.0 / fps) if tu.name == "frames" or tu.name == "fr" else float(tu.value)

    def dist_factor(u_from: str, u_to: str) -> float:
        return DistanceUnit.parse(u_to).factor(mm_to_px) / DistanceUnit.parse(u_from).factor(mm_to_px)

    def time_factor(u_from: str, u_to: str) -> float:
        return sec_per(u_from) / sec_per(u_to)

    def convert_component(u: str, invert: bool):
        # distance?
        try:
            DistanceUnit.parse(u)
            u2 = u if target_distance is None else target_distance
            f = 1.0 if u2 == u else dist_factor(u, u2)
            if invert: f = 1.0 / f
            return f, u2
        except ValueError:
            pass

        # time?
        try:
            TimeUnit.parse(u)
            u2 = u if target_time is None else target_time
            f = 1.0 if u2 == u else time_factor(u, u2)
            if invert: f = 1.0 / f
            return f, u2
        except ValueError as e:
            raise ValueError(f'Invalid unit component "{u}". Must be in TimeUnit or DistanceUnit.') from e

    # remove white space and brackets
    u = unit.strip().strip("[]")
    parts = u.split("/", 1)
    num = parts[0]
    den = parts[1] if len(parts) == 2 else None

    f_num, num_out = convert_component(num, invert=False)
    factor = f_num
    unit_out = num_out

    if den is not None:
        f_den, den_out = convert_component(den, invert=True)
        factor *= f_den
        unit_out = f"{num_out}/{den_out}"

    return data * factor, unit_out


######
#Functions not included in property based testing for not having a clean return
######


def _validate_parameter(
    param_name: str,
    param_value: Any,
    valid_options: List[Any],
    is_list: bool = False,
    custom_error_if_empty: Optional[str] = None,
    only_one_of_many: Optional[bool] = True,
    can_be_dict: Optional[bool] = False,
): # pragma: no cover
    """
    A generic helper to validate a single parameter against a list of valid options.

    Args:
        param_name (str): The name of the parameter being checked (for error messages).
        param_value (Any): The value of the parameter provided by the user.
        valid_options (List[Any]): The list of allowed values.
        is_list (bool): If True, checks if param_value is a subset of valid_options.
                        Otherwise, checks if it is a member of valid_options.
        custom_error_if_empty (Optional[str]): A specific error to raise if the
                                               parameter is provided but the list
                                               of valid options is empty.
    """
    if param_value is None:
        return  # Parameter not provided, no validation needed.

    # If the param is provided but there are no valid options to check against
    if not valid_options and custom_error_if_empty:
        raise ValueError(custom_error_if_empty)

    valid_set = set(valid_options)
    is_valid = False

    if isinstance(param_value, dict) and can_be_dict:
        value_set = set(
            [x for lst in param_value.values() for x in lst]          
        )
        if value_set.issubset(valid_set):
            is_valid = True
    
    elif not isinstance(param_value, dict) and is_list:
        # Ensure param_value is a list-like object for set operations
        value_set = set(
            [param_value] if isinstance(param_value, str) else param_value
        )
        if value_set.issubset(valid_set):
            is_valid = True
    else:
        if param_value in valid_set:
            is_valid = True

    if not is_valid:
        # Truncate for readability
        options_preview = str(valid_options[:5])[1:-1]
        if len(valid_options) > 5:
            options_preview += ", ..."

        if only_one_of_many:    
            raise ValueError(
                f'Invalid value for "{param_name}". Must be one of: [{options_preview}]'
            )
        else:
            raise ValueError(
                f'Invalid value for "{param_name}". Must be a subset of: [{options_preview}]'
            )


#not covered by testing as the only purpose of this function is to throw specific exceptions
def _check_enum_inputs(
    coordinates: coordinates,
    supervised_annotations: Optional[table_dict] = None,
    soft_counts: Optional[table_dict] = None,
    origin: Optional[str] = None,
    experiment_ids: Optional[Union[List[str],dict[List[str]]]] = None,
    exp_condition: Optional[str] = None,
    exp_condition_order: Optional[List[str]] = None,
    condition_values: Optional[List[str]] = None,
    behaviors: Optional[List[str]] = None,
    bodyparts: Optional[List[str]] = None,
    in_roi_bodyparts: Optional[List[str]] = None,
    animal_id: Optional[str] = None,
    center: Optional[str] = None,
    visualization: Optional[str] = None,
    normative_model: Optional[str] = None,
    aggregate_experiments: Optional[str] = None,
    colour_by: Optional[str] = None,
    roi_number: Optional[int] = None,
    animals_in_roi: Optional[List[str]] = None,
    roi_mode: str = "mousewise",
    distance_unit: str = None,
):  # pragma: no cover
    """
    Checks and validates enum-like input parameters for various plot functions.

    This function acts as a centralized guard to ensure that all categorical
    and list-based inputs are valid before being used in downstream logic.

    Args:
        coordinates (Coordinates): deepof Coordinates object.
        supervised_annotations (Optional[TableDict]): Info on supervised annotations.
        soft_counts (Optional[TableDict]): Info on unsupervised annotations.
        origin (Optional[str]): Name of the calling function for context-specific checks.
        experiment_ids (Optional[List[str]]): List of experiment IDs to plot.
        exp_condition (Optional[str]): Experimental condition to plot.
        exp_condition_order (Optional[List[str]]): Order for plotting conditions.
        condition_values (Optional[List[str]]): Specific condition values to plot.
        behaviors (Optional[List[str]]): List of animal behaviors to analyze.
        bodyparts (Optional[List[str]]): List of body parts to plot.
        in_roi_bodyparts (Optional[List[str]]): List of body parts to plot, excluding animal ids, including "all".
        animal_id (Optional[str]): ID of a specific animal.
        center (Optional[str]): Center point for position normalization (e.g., 'arena').
        visualization (Optional[str]): Visualization mode (e.g., 'networks', 'heatmaps').
        normative_model (Optional[str]): Cohort to use as a control group.
        aggregate_experiments (Optional[str]): Method to aggregate embeddings.
        colour_by (Optional[str]): Hue for coloring embeddings.
        roi_number (Optional[int]): ROI number to use for filtering.
        animals_in_roi (Optional[List[str]]): Animals that must be inside the ROI.
        roi_mode (str): Mode for ROI filtering ('mousewise' or 'behaviorwise').
        distance_unit (str): Unit for measuring distance
    """
    # Activate warnings for immediate user feedback
    # warnings.simplefilter("always", UserWarning)

    # =========================================================================
    # 1. NORMALIZE INPUTS
    # Ensure that parameters expecting a list are lists, even if a single string was passed.
    # =========================================================================
    def _to_list_if_str(value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    experiment_ids = _to_list_if_str(experiment_ids)
    exp_condition_order = _to_list_if_str(exp_condition_order)
    condition_values = _to_list_if_str(condition_values)
    behaviors = _to_list_if_str(behaviors)
    bodyparts = _to_list_if_str(bodyparts)
    in_roi_bodyparts = _to_list_if_str(in_roi_bodyparts)
    animals_in_roi = _to_list_if_str(animals_in_roi)
    

    # =========================================================================
    # 2. GENERATE LISTS OF VALID OPTIONS
    # =========================================================================
    
    # --- Dynamically generated options from data ---
    exp_id_opts = (["average"] if origin == "plot_heatmaps" else []) + \
                   os_sorted(list(coordinates._tables.keys()))

    behavior_opts = []
    if supervised_annotations:
        first_key = list(supervised_annotations.keys())[0]
        behavior_opts.extend(get_dt(supervised_annotations, first_key, only_metainfo=True)['columns'])
        behavior_opts.extend(coordinates._animal_ids)
    if soft_counts:
        first_key = list(soft_counts.keys())[0]
        n_clusters = get_dt(soft_counts, first_key, only_metainfo=True)['num_cols']
        behavior_opts.extend([f"Cluster_{i}" for i in range(n_clusters)])

    exp_cond_opts, cond_val_opts = [], []
    if coordinates.get_exp_conditions:
        all_conditions = [cond.columns.values for cond in coordinates.get_exp_conditions.values()]
        exp_cond_opts = np.unique(np.concatenate(all_conditions)).tolist()
        if exp_condition in exp_cond_opts:
            all_values = [c[exp_condition].values.astype(str) for c in coordinates.get_exp_conditions.values()]
            cond_val_opts = np.unique(np.concatenate(all_values)).tolist()

    all_bps = []
    for key, table in coordinates._tables.items():
        cols = get_dt(coordinates._tables, key, only_metainfo=True)['columns']
        all_bps.extend([c[0] for c in cols])
    bodypart_opts = [bp for bp in np.unique(all_bps) if bp not in coordinates._excluded]
    # remove ids for in roi version
    if(len(coordinates._animal_ids)>1):
        in_roi_bodypart_opts=[bp.partition("_")[2] for bp in bodypart_opts]
    else:
        in_roi_bodypart_opts=copy.copy(bodypart_opts)
    in_roi_bodypart_opts = in_roi_bodypart_opts + ["all"]

    animal_id_opts = coordinates._animal_ids
    
    roi_num_opts = []
    if coordinates._roi_dicts:
        first_key = list(coordinates._roi_dicts.keys())[0]
        roi_num_opts = list(coordinates._roi_dicts[first_key].keys())

    # --- Statically defined options ---
    center_opts = ["arena"]
    vis_opts = ["networks", "heatmaps"] if origin == "plot_transitions" else ["confusion_matrix", "balanced_accuracy"]
    agg_exp_opts = ["time on cluster", "mean", "median"]
    roi_mode_opts = ["mousewise", "behaviorwise"]
    color_by_opts = ["cluster", "exp_condition", "exp_id"]
    colour_by_is_behaviors=False
    if colour_by is not None and isinstance(colour_by,list):
        colour_by = _to_list_if_str(colour_by)
        color_by_opts=behavior_opts
        colour_by_is_behaviors=True

    # =========================================================================
    # 3. CONFIGURE AND RUN VALIDATION CHECKS
    # Format: (param_name, param_value, valid_options, is_list, custom_error)
    # =========================================================================
    validation_checks = [
        ("experiment_ids", experiment_ids, exp_id_opts, True, None, True, True),
        ("exp_condition", exp_condition, exp_cond_opts, False, "No experiment conditions loaded!", True, False),
        ("exp_condition_order", exp_condition_order, cond_val_opts, True, "No conditions to order; check 'exp_condition'.", False, False),
        ("condition_values", condition_values, cond_val_opts, True, "No condition values available; check 'exp_condition'.", True, False),
        ("normative_model", normative_model, cond_val_opts, False, "No condition values available to select a normative model.", True, False),
        ("behaviors", behaviors, behavior_opts, True, "No supervised annotations or soft counts loaded!", False, False),
        ("bodyparts", bodyparts, bodypart_opts, True, None, False, False),
        ("bodyparts", in_roi_bodyparts, in_roi_bodypart_opts, True, None, False, False),
        ("animals_in_roi", animals_in_roi, animal_id_opts, True, None, True, False),
        ("animal_id", animal_id, animal_id_opts, False, None, True, False),
        ("center", center, center_opts, False, None, True, False),
        ("visualization", visualization, vis_opts, False, None, True, False),
        ("aggregate_experiments", aggregate_experiments, agg_exp_opts, False, None, True, False),
        ("colour_by", colour_by, color_by_opts, colour_by_is_behaviors, "color_by can either be \"cluster\", \"exp_condition\", \"exp_id\" or a list of behaviors!", False, False),
        ("roi_number", roi_number, roi_num_opts, False, "No ROIs were defined for this project.", True, False),
        ("roi_mode", roi_mode, roi_mode_opts, False, None, True, False),
        ("distance_unit", distance_unit, DistanceUnit._member_names_, False, None, False, False)
    ]

    for name, value, options, is_list, error_msg, only_one_of_many, can_be_dict in validation_checks:
        _validate_parameter(name, value, options, is_list, error_msg, only_one_of_many, can_be_dict)

    # =========================================================================
    # 4. HANDLE SPECIAL CASES AND WARNINGS
    # =========================================================================
    if roi_mode != "mousewise" and roi_number is None:
        print(
            '\033[33mInfo! The input "roi_mode" only has an effect if an ROI is '
            'selected via "roi_number"!\033[0m'
        )


def plot_arena(
    coordinates: coordinates, center: str, color: str, ax: Any, key: str, roi_number: int = None,
): # pragma: no cover
    """Plot the arena in the given canvas.

    Args:
        coordinates (coordinates): deepof Coordinates object.
        center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        color (str): color of the displayed arena.
        ax (Any): axes where to plot the arena.
        key str: key of the animal to plot with optional "all of them" (if key=="average").
        roi_number int: number of a roi, if given
    """
    if key != "average" and roi_number is None:
        arena = copy.copy(coordinates._arena_params[key])
    elif key != "average":
        arena = copy.copy(coordinates._roi_dicts[key][roi_number])

    if isinstance(coordinates._arena_params[list(coordinates._arena_params.keys())[0]], Tuple) and roi_number is None:

        if key == "average":
            arena = [
                np.mean(np.array([i[0] for i in coordinates._arena_params.values()]), axis=0),
                np.mean(np.array([i[1] for i in coordinates._arena_params.values()]), axis=0),
                np.mean(np.array([i[2] for i in coordinates._arena_params.values()]), axis=0),
            ]

        ax.add_patch(
            Ellipse(
                xy=((0, 0) if center == "arena" else arena[0]),
                width=arena[1][0] * 2,
                height=arena[1][1] * 2,
                angle=arena[2],
                edgecolor=color,
                fc="None",
                lw=3,
                ls="--",
            )
        )

    elif isinstance(coordinates._arena_params[list(coordinates._arena_params.keys())[0]], np.ndarray) or roi_number is not None:

        if center == "arena" and key == "average":

            if roi_number is None:
                polygon_dictionary = copy.copy(coordinates._arena_params)
            else:
                polygon_dictionary = {
                    exp: copy.copy(roi_data[roi_number]) 
                    for exp, roi_data in coordinates._roi_dicts.items()
                }

            arena = calculate_average_arena(polygon_dictionary)
            avg_scaling = np.mean(np.array(list(coordinates._scales.values()))[:, :2], 0)
            arena -= avg_scaling

        elif center == "arena":
            arena -= np.expand_dims(
                np.array(coordinates._scales[key][:2]).astype(int), axis=1
            ).T

        # Repeat first element for the drawn polygon to be closed
        arena_corners = np.array(list(arena) + [arena[0]])

        ax.plot(
            *arena_corners.T,
            color=color,
            lw=3,
            ls="--",
        )
        
def get_square_shape_for_gridlike_plot(N):
    """get best number of rows and columns for grid like plots"""
    assert N > 0
    assert isinstance(N, int)
    
    sqrt_n = np.sqrt(N)
    # Find divisor closest to sqrt(N)
    n_cols = min(
        (d for d in range(int(sqrt_n), 0, -1) if N % d == 0),
        key=lambda d: abs(d - sqrt_n)
    )
    n_rows = N // n_cols
    return n_rows, n_cols

def heatmap(
    dframe: pd.DataFrame,
    bodyparts: List,
    xlim: tuple = None,
    ylim: tuple = None,
    title: str = None,
    mask: np.ndarray = None,
    extrapolate_heatmap: bool = True,
    save: str = False,
    dpi: int = 200,
    ax: Any = None,
    **kwargs,
) -> plt.figure: # pragma: no cover
    """Return a heatmap of the movement of a specific bodypart in the arena.

    If more than one bodypart is passed, it returns one subplot for each.

    Args:
        dframe (pandas.DataFrame): table_dict value with info to plot bodyparts (List): bodyparts to represent (at least 1).
        bodyparts (list): list of body parts to plot.
        xlim (float): limits of the x-axis.
        ylim (float): limits of the y-axis.
        title (str): title of the figure.
        mask (np.ndarray): mask to apply to the heatmap across time.
        extrapolate_heatmap (bool): Show full heatmap including extrapolated parts (default = True)
        save (str): if provided, saves the figure to the specified file.
        dpi (int): dots per inch of the figure to create.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        kwargs: additional arguments to pass to the seaborn kdeplot function.

    Returns:
        heatmaps (plt.figure): figure with the specified characteristics
    """
    # noinspection PyTypeChecker
    if ax is None:

        n_rows,n_cols=get_square_shape_for_gridlike_plot(len(bodyparts))

        heatmaps, ax = plt.subplots(
            n_rows, 
            n_cols,
            sharex=True,
            sharey=True,
            dpi=dpi,
            figsize=(8 * n_cols, 8 * n_rows),
        )
    # Turn into array for streamlined processing
    if not isinstance(ax,np.ndarray):    
        ax = np.array([ax])

    if isinstance(dframe, dict):

        if mask is not None:
            assert isinstance(
                mask, dict
            ), "If dframe is a dictionary, mask must be one as well."

            # Pad each mask in the dictionary with False values to match the length of each dataframe
            mask = {
                k: np.pad(
                    v, (0, len(dframe[k]) - len(v)), "constant", constant_values=False
                )
                for k, v in mask.items()
            }
            mask = np.concatenate(list(mask.values()), axis=0)

        # Concatenate all dataframes which are values of the dictionary into a single one
        dframe = pd.concat(dframe.values(), axis=0).reset_index(drop=True)

    if mask is None:
        mask = np.ones(len(dframe), dtype=bool)

    else:
        # Pad the mask with False values to match the length of the dataframe
        mask = np.pad(
            mask, (0, len(dframe) - len(mask)), "constant", constant_values=False
        )

    for x, bpart in zip(ax.ravel(),bodyparts):
        heatmap = dframe[bpart].loc[mask].dropna()

        cut=0
        if extrapolate_heatmap:
            cut=3
        sns.kdeplot(
            x=heatmap.x,
            y=heatmap.y,
            cmap="magma",
            fill=True,
            cut=cut,
            #bw_adjust=0.5,
            alpha=1,
            ax=x,
            **kwargs,
        )
        if len(bodyparts) <= 1:
            ax = np.array([ax])

    for x, bp in zip(ax.ravel(), bodyparts):
        if xlim is not None:
            x.set_xlim(xlim)
        if ylim is not None:
            x.set_ylim(ylim)
        if title is not None:
            x.set_title(f"{bp} - {title}", fontsize=10)

    if save:  # pragma: no cover
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_heatmaps{}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    calendar.timegm(time.gmtime()),
                ),
            )
        )


    return ax


def _scatter_embeddings(
    embeddings: np.ndarray,
    soft_counts: np.ndarray = None,
    ax: Any = None,
    save: str = False,
    show: bool = True,
    dpi: int = 200,
) -> plt.figure: # pragma: no cover
    """Return a scatter plot of the passed projection. Each dot represents the trajectory of an entire animal.

    If labels are propagated, it automatically colours all data points with their respective condition.

    Args:
        embeddings (np.ndarray): sequence embeddings obtained with the unsupervised pipeline within deepof
        soft_counts (np.ndarray): labels of the clusters. If None, aggregation method should be provided.
        ax: axes where to plot the arena.
        save (str): if provided, saves the figure to the specified file.
        show (bool): if True, displays the current figure. If not, returns the given axes.
        dpi (int): dots per inch of the figure to create.

    Returns:
        projection_scatter (plt.figure): figure with the specified characteristics
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, dpi=dpi)

    # Plot entire UMAP
    ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=soft_counts,
        cmap=("tab20" if soft_counts is not None else None),
        edgecolor="black",
        linewidths=0.25,
    )

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if not show:
        return ax

    plt.show()


def _preprocess_transitions(
    coordinates: coordinates,
    supervised_annotations: table_dict = None,
    soft_counts: table_dict = None,
    # Time selection parameters
    bin_size: Union[int, str] = None,
    bin_index: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max: int=20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    # Selection parameters
    exp_condition: str = None,
    delta_T: float = 0.5,
    silence_diagonal: bool = False,
    diagonal_behavior_counting: str = "Events",
    normalize:bool = True,
    # Visualization parameters
    visualization="networks",
): # pragma: no cover
    """Compute and plots transition matrices for all data or per condition. Plots can be heatmaps or networks.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        supervised_annotations (table_dict): table dict with supervised annotations.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                      
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.
        delta_T: Time after the offset of one behavior during which the onset of the next behavior counts as a transition      
        silence_diagonal (bool): If True, diagonals are set to zero.
        diagonal_behavior_counting (str): How to count diagonals (self-transitions). Options: 
            - "Frames": Total frames where behavior is active (after extension)
            - "Time": Total time where behavior is active
            - "Events": number of instances of the behavior occuring 
            - "Transitions": number of frame-wise internal behavior transitions e.g. A behavior of 4 frames in length would have 3 transitions.      
        normalize (bool): Row-normalizes transition probabilities if True. Default=True.
        visualization (str): visualization mode. Can be either 'networks', or 'heatmaps'.
        kwargs: additional arguments to pass to the seaborn kdeplot function.
    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        origin="plot_transitions",
        exp_condition=exp_condition,
        visualization=visualization,
        supervised_annotations=supervised_annotations,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
    )
    diagonal_behavior_counting_options=["Frames","Time","Events","Transitions"]  
    if diagonal_behavior_counting not in diagonal_behavior_counting_options:
        raise ValueError(
            '"diagonal_behavior_counting" needs to be one of the following: {}'.format(
                str(diagonal_behavior_counting_options)[1:-1]
            )
        )
    if (supervised_annotations is None and soft_counts is None) or (supervised_annotations is not None and soft_counts is not None):
        raise ValueError(
            "Eet either supervised_annotations or soft_counts, not both or neither!"
        )
    elif supervised_annotations is not None:
        tab_dict=supervised_annotations
    else:
        tab_dict=soft_counts
    if visualization == "networks" and normalize ==False:
        normalize=True
        print(
        '\033[33mInfo! Cannot use networks visulization without normalization!\033[0m'
        )
    if delta_T is None:
        delta_T=0.0
    if animals_in_roi is None:
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animal_id is only relevant if a ROI was selected!\033[0m'
        )

    exp_conditions=None
    if exp_condition is not None:
        exp_conditions = {
            key: str(val.loc[:, exp_condition].values[0])
            for key, val in coordinates.get_exp_conditions.items()
        }

    # preprocess information given for time binning
    bin_info_time = _preprocess_time_bins(
        coordinates, bin_size, bin_index, precomputed_bins, tab_dict_for_binning=soft_counts, samples_max=samples_max, down_sample=False,
    )
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    grouped_transitions, columns, combined_columns = deepof.utils.count_transitions(
        tab_dict=tab_dict,
        exp_conditions=exp_conditions,
        bin_info=bin_info,
        animals_in_roi=animals_in_roi,
        delta_T = delta_T,
        frame_rate=coordinates._frame_rate,
        silence_diagonal=silence_diagonal,
        aggregate=(exp_conditions is not None), 
        normalize=normalize,
        diagonal_behavior_counting=diagonal_behavior_counting
    )

    return grouped_transitions, columns, combined_columns, exp_conditions, normalize


# Thrown for slice averageing if only one value is present
@deepof.data_loading._suppress_warning(
    warn_messages=[
        "Mean of empty slice",
        "Degrees of freedom <= 0 for slice.",
        "All-NaN slice encountered",
    ]
)
def _preprocess_mouse_roi_interaction(
    coordinates: coordinates,
    bodyparts: list,  
    animal_id: str,      
    # Time selection parameters
    N_time_bins: int = 24,
    custom_time_bins: List[List[Union[int, str]]] = None,
    samples_max=20000,
    # ROI functionality
    roi_number: int = None,
    # Visualization parameters
    hide_time_bins: list[bool] = None,
    experiment_ids: list = None,  
    exp_condition: str = None, 
    condition_values: str = None,
    smoothing_factor: float = 0,
    mode: str = "distance",
    add_stats: str = "Mann-Whitney",
    error_bars: str = "sem",
    unit_distance: str = "m",
):
    def _smooth_signal(signal, smoothing_factor):
        
        assert smoothing_factor >= 0 and smoothing_factor <= 1, "Smoothing factor has to be in range 0<=x<=1!"

        l_s = len(signal)
        lag=int(l_s*smoothing_factor)

        # Early exit 
        if lag == 0:
            return signal
    
        signal_padded=np.zeros([2*lag+l_s])
        signal_padded[0:lag]=signal[lag-1::-1]
        signal_padded[lag:l_s+lag]=signal
        signal_padded[l_s+lag:]=signal[l_s-lag:]

        signal_padded=deepof.utils.moving_average(signal_padded,lag, ignore_nans=True)
        return(signal_padded[lag:l_s+lag])

    if roi_number==0:
        roi_number=None
    # Checks and init preprocessing
    _check_enum_inputs(
        coordinates,
        roi_number=roi_number,
        bodyparts=bodyparts,
        experiment_ids=experiment_ids,
        exp_condition=exp_condition,
        condition_values=condition_values,
        animal_id=animal_id,
    )
    # Check if correct inputs for modes are present
    if mode == "fov" and animal_id is not None:
        bodyparts=["Left_ear","Nose","Right_ear"]
        if animal_id is not None and animal_id !="":
            bodyparts=[animal_id + "_" + bp for bp in bodyparts]
    elif mode == "distance" and bodyparts is not None:
        if isinstance(bodyparts, str):
            bodyparts=[bodyparts]  
    else:
        raise ValueError("Error! This function requires either bodyparts for distance mode or an animal_id for foc mode!")
   
    exp_ids_given=True
    if experiment_ids is None:
        exp_ids_given=False
        experiment_ids={'all':list(coordinates._tables.keys())}
    elif isinstance(experiment_ids, str):
        experiment_ids={'selection':[experiment_ids]}
    if isinstance(condition_values, str):
        condition_values=[condition_values]
    if exp_condition is not None and condition_values is None:
        condition_values=coordinates.get_condition_values(exp_condition)

    # Select experiment ids by conditions
    if exp_condition is not None and condition_values is not None:
        experiment_ids={}
        for condition_value in condition_values:
            experiment_ids[condition_value]=[
                    k
                    for k, v in coordinates.get_exp_conditions.items()
                    if v[exp_condition].values.astype(str) == condition_value
                ]
        if exp_ids_given:
            warning_message = (
                "\033[38;5;208m\n"  # Set text color to orange
                "Warning! Since a valid exp_condition / condition_value combination was selected, the experiment_ids will be ignored!"
                "\033[0m"  # Reset text color
            )
            warnings.warn(warning_message)

    L_shortest = min(
        get_dt(coordinates._tables,key,only_metainfo=True)['num_rows'] for key in coordinates._tables.keys()
    )

    # preprocess time bin info
    custom_time_bins, hide_time_bins = validate_custom_bins(N_time_bins, L_shortest, custom_time_bins, hide_time_bins)
    bin_lengths = [sublist[1] - sublist[0] for sublist in custom_time_bins]

    multi_bin_info={}
    # Create bin_info objects for each custom time bin
    for j, (bin_start, bin_end) in enumerate(custom_time_bins):

        #create full time bins covering entire signal
        bin_info_time = _preprocess_time_bins(
        coordinates, bin_index=bin_start, bin_size=bin_end-bin_start+1, samples_max=int(samples_max/len(custom_time_bins)),
        given_in_frames=True,
        )
        
        multi_bin_info[j]=bin_info_time


    # Prepare dict of rois (one per video) to measure distance from
    roi_dict = {}
    # if no roi number is given, take the arena as default "roi"
    if roi_number is None:
        for exp_cond in experiment_ids:
            roi_dict[exp_cond]={}
            for exp_id in experiment_ids[exp_cond]:
            
                params = coordinates._arena_params[exp_id]
                # Legacy, transform cicular arenas to polygons 
                if isinstance(params,tuple):
                    polygon = deepof.arena_utils.extract_corners_from_arena(params)
                else:
                    polygon = params
                roi_dict[exp_cond][exp_id] = polygon
    else:
        for exp_cond in experiment_ids:
            roi_dict[exp_cond]={}
            for exp_id in experiment_ids[exp_cond]:
        
                rois= coordinates._roi_dicts[exp_id]
                polygon = rois[roi_number]
                roi_dict[exp_cond][exp_id] = polygon

    # Collect minimum interaction measure for all sets of experiments and all bins
    interaction_dict = {}
    rows = []  # accumulate for final df

    for exp_cond, exp_polys in roi_dict.items():
        for exp_id, polygon in exp_polys.items():

            bps = coordinates.get_coords_at_key(
                key=exp_id, scale=coordinates._scales[exp_id]
            )[bodyparts]

            # ---------------------------------------------------------------------
            # Compute per-frame interaction signal ONCE for the whole experiment
            # ---------------------------------------------------------------------
            polygon = np.asarray(polygon, dtype=np.float64)
            # Remove possible double points at beginning / end
            if polygon.shape[0] >= 2 and np.allclose(polygon[0], polygon[-1]):
                polygon = polygon[:-1]
            if mode == "fov":
                # bps: (T, 3*2) -> (T, 3, 2)
                pts = bps.to_numpy().reshape(-1, 3, 2)
                polygon = np.asarray(polygon, dtype=np.float64)
                # Remove possible double points at beginning / end
                if polygon.shape[0] >= 2 and np.allclose(polygon[0], polygon[-1]):
                    polygon = polygon[:-1]
                interaction_full = deepof.utils.in_field_of_view_numba(
                    np.asarray(pts, dtype=np.float64), float(90), polygon
                )  # shape (T,)

            elif mode == "distance":
                T = bps.shape[0]
                B = len(bodyparts)

                inside = np.empty((T, B), dtype=bool)
                dists  = np.empty((T, B), dtype=float)

                for k, bp in enumerate(bodyparts):
                    pts = bps[bp].to_numpy().astype(np.float64)  # shape (T, 2) as in your current code
                    inside[:, k] = deepof.utils.point_in_polygon_numba(pts, polygon)
                    dists[:, k]  = deepof.utils.get_point_polygon_distance_numba(pts, polygon)

                # Match old semantics:
                # - arena (roi_number is None): invalidate frames where ANY bp is outside arena
                # - ROI (roi_number not None): invalidate frames where ANY bp is inside ROI
                valid = inside.all(axis=1) if roi_number is None else ~inside.any(axis=1)

                min_dist = np.nanmin(dists, axis=1)
                min_dist[~valid] = np.nan
                min_dist[min_dist == 0] = np.nan  # keep your original behavior

                interaction_full = min_dist * DistanceUnit.parse(unit_distance).factor(coordinates._scales[exp_id][2]/coordinates._scales[exp_id][3])  # shape (T,)

            else:
                raise NotImplementedError(
                    'The only currently available modes are "distance" and "fov" (field of view)'
                )

            # ---------------------------------------------------------------------
            # Bin by slicing precomputed per-frame result
            # ---------------------------------------------------------------------
            for bin_id, bin_info in multi_bin_info.items():
                frames = bin_info[exp_id]              # frame indices for this exp_id and bin
                value = np.nanmean(interaction_full[frames])
                rows.append({"time_bin": bin_id, "exp_condition": str(exp_cond), mode: value})

    df = pd.DataFrame.from_records(rows, columns=["time_bin", "exp_condition", mode])

  
    df = postprocess_df_bins(df, bin_lengths, hide_time_bins)  

    mean_values, error_values, binned_effect_sizes_df = process_df(df, error_bars=error_bars)

    return mean_values, error_values, df, binned_effect_sizes_df, hide_time_bins


def process_df(df: pd.DataFrame, error_bars: str = "sem"):
    """
    Process binned behavioral DF independent of number of exp conditions.

    Returns
    -------
    mean_values : dict[str, np.ndarray]
        Mapping condition -> array of mean values per time_bin (sorted by time_bin).
    error_values : dict[str, np.ndarray]
        Mapping condition -> array of error values per time_bin (sorted by time_bin).
    binned_effect_sizes_df : pd.DataFrame
        Pairwise effect sizes (Cohen's d) for all condition pairs per time_bin.
        Columns: ["time_bin","cond_a","cond_b","Absolute_Cohens_d","Effect_Size_Category"]
        Empty if <2 conditions.
    time_bins : np.ndarray
        Sorted unique time_bin values used for the arrays.
    conditions : list[str]
        Sorted unique exp_condition values (keys of the dicts).
    """
    if df.shape[1] < 4:
        raise ValueError("df is expected to have at least 3 columns: time_bin, exp_condition, <value>")

    value_col = df.columns[3]

    # Stable, explicit ordering (important if bins are missing / not contiguous)
    time_bins = np.sort(df["time_bin"].unique())
    conditions = sorted(df["exp_condition"].astype(str).unique().tolist())

    # Group once
    g = df.groupby(["time_bin", "exp_condition"])[value_col]

    means = (
        g.mean()
        .unstack("exp_condition")
        .reindex(index=time_bins, columns=conditions)
    )

    if error_bars == "sem":
        errs = (
            g.sem()
            .unstack("exp_condition")
            .reindex(index=time_bins, columns=conditions)
        )
    elif error_bars == "std":
        errs = (
            g.std()
            .unstack("exp_condition")
            .reindex(index=time_bins, columns=conditions)
        )
    else:
        raise NotImplementedError(
            'error_bars currently only supports standard deviation ("std") '
            'and standard error of the mean ("sem")!'
        )

    # Return as dicts keyed by condition (more robust than positional lists)
    mean_values = {cond: means[cond].to_numpy() for cond in conditions}
    error_values = {cond: errs[cond].to_numpy() for cond in conditions}

    # Pairwise Cohen's d for all condition pairs per bin
    effect_rows = []
    if len(conditions) >= 2:
        for tb in time_bins:
            for cond_a, cond_b in itertools.combinations(conditions, 2):
                array_a = df.loc[
                    (df["exp_condition"].astype(str) == cond_a) & (df["time_bin"] == tb),
                    value_col
                ].to_numpy()
                array_b = df.loc[
                    (df["exp_condition"].astype(str) == cond_b) & (df["time_bin"] == tb),
                    value_col
                ].to_numpy()

                array_a = array_a[~np.isnan(array_a)]
                array_b = array_b[~np.isnan(array_b)]

                if array_a.size == 0 or array_b.size == 0:
                    d = np.nan
                    d_effect_size = np.nan
                else:
                    d = abs(deepof.visuals_utils.cohend(array_a, array_b))
                    d_effect_size = deepof.visuals_utils.cohend_effect_size(d)

                effect_rows.append(
                    {
                        "time_bin": tb,
                        "cond_a": cond_a,
                        "cond_b": cond_b,
                        "Absolute_Cohens_d": d,
                        "Effect_Size_Category": d_effect_size,
                    }
                )

    binned_effect_sizes_df = pd.DataFrame(
        effect_rows,
        columns=["time_bin", "cond_a", "cond_b", "Absolute_Cohens_d", "Effect_Size_Category"],
    )

    return mean_values, error_values, binned_effect_sizes_df



def plot_binned_line(
    ax,
    x,
    y,
    yerr=None,
    hide_time_bins=None,
    color="C0",
    label=None,
    smooth_points_per_interval: int = 10,
    mean_linewidth: float = 3.0,
    mean_alpha: float = 0.8,
    err_linewidth: float = 1.0,
    err_alpha: float = 0.15,
    marker: str = "o",
):
    """
    Plot a binned mean line with interpolation + markers + error band, leaving gaps
    for hidden bins and NaNs.

    Parameters
    ----------
    ax : matplotlib axis
    x : array-like, shape (n_bins,)
        X positions (must be strictly increasing).
    y : array-like, shape (n_bins,)
        Mean values per bin.
    yerr : array-like or None, shape (n_bins,)
        Error values per bin (sem/std). If None, no error band is drawn.
    hide_time_bins : array-like of bool or None, shape (n_bins,)
        True bins will be hidden (gaps in line, no marker/error there).
    color : str
    label : str
    smooth_points_per_interval : int
        Number of points per bin-to-bin interval for mean interpolation (>=2).
    """

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float).ravel()

    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    if yerr is not None and len(yerr) != n:
        raise ValueError("yerr must have the same length as x and y")
    if hide_time_bins is None:
        hide = np.zeros(n, dtype=bool)
    else:
        hide = np.asarray(hide_time_bins, dtype=bool).ravel()
        if len(hide) != n:
            raise ValueError("hide_time_bins must have the same length as x and y")

    if smooth_points_per_interval < 2:
        raise ValueError("smooth_points_per_interval must be >= 2")

    # Visible points for means/markers (hidden bins and NaNs are excluded)
    visible_mean = (~hide) & (~np.isnan(y)) & (~np.isnan(x))

    # --- 1) Interpolated mean lines per contiguous visible segment
    first_segment = True
    for sl in contiguous_segments(visible_mean):
        x_seg = x[sl]
        y_seg = y[sl]
        m = len(x_seg)

        # Can't interpolate a single point; marker will handle it.
        if m < 2:
            continue

        kind = "cubic" if m >= 4 else "linear"
        f = interp1d(x_seg, y_seg, kind=kind, assume_sorted=True)

        # smooth grid within segment (no duplicates between intervals)
        n_smooth = (m - 1) * (smooth_points_per_interval - 1) + 1
        x_smooth = np.linspace(x_seg[0], x_seg[-1], n_smooth)
        y_smooth = f(x_smooth)

        ax.plot(
            x_smooth,
            y_smooth,
            color=color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            linestyle="-",
            label=label if first_segment else None,  # avoid duplicate legend entries
        )
        first_segment = False

    # --- 2) Markers at original bin centers (hidden bins + NaNs masked)
    point_mask = hide | np.isnan(y) | np.isnan(x)
    marker_handle = ax.plot(
        np.ma.masked_array(x, point_mask),
        np.ma.masked_array(y, point_mask),
        marker=marker,
        linestyle="",
        color=color,
        linewidth=2,
    )[0]

    # --- 3) Errors on original grid (optional), with gaps
    if yerr is not None:
        err_mask = point_mask | np.isnan(yerr)
        x_err = np.ma.masked_array(x, err_mask)
        upper = np.ma.masked_array(y + yerr, err_mask)
        lower = np.ma.masked_array(y - yerr, err_mask)

        ax.plot(x_err, upper, linestyle="--", color=color, alpha=mean_alpha, linewidth=err_linewidth)
        ax.plot(x_err, lower, linestyle="--", color=color, alpha=mean_alpha, linewidth=err_linewidth)
        ax.fill_between(x_err, lower, upper, color=color, alpha=err_alpha)

    return marker_handle



def get_bin_centers(bin_lengths, as_radians: bool = False):
    """Compute centers of consecutive bins given their lengths.

    Args:
        bin_lengths : array-like of positive numbers
        as_radians : bool, default False, if True return result as fractions of 2 pi
    """
    L = np.asarray(bin_lengths, float).ravel()
    if L.size == 0:
        return L, L
    tot = L.sum()
    if tot <= 0:
        raise ValueError("sum(bin_lengths) must be > 0")

    starts = np.r_[0.0, np.cumsum(L)[:-1]] / tot
    centers = starts + 0.5 * (L / tot)

    if as_radians:
        s = 2 * np.pi
        return centers * s, starts * s
    return centers, starts

