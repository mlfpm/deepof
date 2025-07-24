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

import deepof.post_hoc
import deepof.utils
from deepof.data_loading import get_dt
from deepof.config import PROGRESS_BAR_FIXED_WIDTH, ONE_ANIMAL_COLOR_MAP, TWO_ANIMALS_COLOR_MAP, DEEPOF_8_BODYPARTS, DEEPOF_11_BODYPARTS, DEEPOF_14_BODYPARTS, BODYPART_COLORS



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)

 
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
    elif type(animal_ids)==pd.DataFrame:
        animal_ids_raw=animal_ids.columns
        animal_ids_raw=[re.search(r'^[^_]+', string)[0] for string in animal_ids_raw]
        # in case of only one animal what is found is only behavior names
        if "speed" in animal_ids_raw:
            animal_ids=None
        else:
            animal_ids=list(np.sort(np.unique(animal_ids_raw)))
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
    single_behaviors=["climb-arena", "sniff-arena", "immobility", "stat-lookaround", "stat-active", "stat-passive", "moving", "sniffing", "missing", "speed"]
    symmetric_behaviors=["nose2nose","sidebyside","sidereside"]
    asymmetric_behaviors=["nose2tail","nose2body","following"]

    # create names of supervised behaviors from animal ids and raw behavior names in correct order
    if animal_ids is None or len(animal_ids)==1:
        supervised = single_behaviors
        color_map = ONE_ANIMAL_COLOR_MAP
    else:
        supervised = generate_behavior_combinations(animal_ids,symmetric_behaviors,asymmetric_behaviors,single_behaviors)
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


def generate_behavior_combinations(animal_ids, symmetric_behaviors, asymmetric_behaviors, single_behaviors):
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
    
    # Process symmetric paired behaviors
    for behavior in symmetric_behaviors:
        for pair in itertools.combinations(animal_ids, 2):
            # Sort the pair to ensure consistent order and avoid duplicates
            sorted_pair = sorted(pair)
            combined = f"{sorted_pair[0]}_{sorted_pair[1]}_{behavior}"
            result.append(combined)
    
    # Process asymmetric paired behaviors
    for behavior in asymmetric_behaviors:
        for pair in itertools.permutations(animal_ids, 2):
            combined = f"{pair[0]}_{pair[1]}_{behavior}"
            result.append(combined)
    
    # Process single mouse behaviors
    for animal_id in animal_ids:
        for behavior in single_behaviors:
            if behavior != "missing" and behavior != "speed":
                combined = f"{animal_id}_{behavior}"
                result.append(combined)
    
    # Add missing
    if "missing" in single_behaviors:            
        result = result + [id + "_missing" for id in animal_ids] 
    # Add speed
    if "speed" in single_behaviors:            
        result = result + [id + "_speed" for id in animal_ids]           
    
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
    elif supervised_annotations is not None:
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


    # Center sliding window instances
    try:
        win_size = coords.shape[0] - twoDim_embeddings.shape[0]
    except AttributeError:
        win_size = coords.shape[0] - twoDim_embeddings[0].shape[1]
    coords = coords[win_size // 2 : -win_size // 2]

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
        ), "selected cluster should be in the clusters provided"

        cluster_embedding = [twoDim_embeddings[hard_counts == selected_cluster]]
        confidence_indices = confidence_indices[
            hard_counts == selected_cluster
        ]

        coords = coords.loc[hard_counts == selected_cluster, :]
        coords = coords.loc[confidence_indices, :]
        cluster_embedding = cluster_embedding[0][confidence_indices]
        concat_embedding = concat_embedding[full_confidence_indices]
        hard_counts = hard_counts[full_confidence_indices]

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
    if s == 0:
        # Handle the case when the standard deviation is 0 by setting effect size to 0
        print("Standard deviation is 0. Setting Cohen's d to 0.")
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
            f"Selected range exceeds {samples_max} samples and has been "
            f"{'downsampled' if down_sample else 'cut'}. "
            "To disable this, increase 'samples_max' or set 'down_sample=False'."
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
        in_roi_criterion (str): Criterion for in roi check, checks by "Center" bodypart being inside or outside of roi by default   
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
    # We must iterate through all animals in bin_info, not just target_ids.
    for animal_id, roi_mask in local_bin_info.items():
        if animal_id == "time":
            continue
            
        # Find which of the valid_cols are associated with the current animal_id
        cols_for_this_animal = [
            col for col in valid_cols 
            if _get_col_base_name(col).startswith(animal_id)
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


######
#Functions not included in property based testing for not having a clean return
######


def _validate_parameter(
    param_name: str,
    param_value: Any,
    valid_options: List[Any],
    is_list: bool = False,
    custom_error_if_empty: Optional[str] = None,
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
    
    if is_list:
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
            
        raise ValueError(
            f'Invalid value for "{param_name}". Must be one of: [{options_preview}]'
        )


#not covered by testing as the only purpose of this function is to throw specific exceptions
def _check_enum_inputs(
    coordinates: coordinates,
    supervised_annotations: Optional[table_dict] = None,
    soft_counts: Optional[table_dict] = None,
    origin: Optional[str] = None,
    experiment_ids: Optional[List[str]] = None,
    exp_condition: Optional[str] = None,
    exp_condition_order: Optional[List[str]] = None,
    condition_values: Optional[List[str]] = None,
    behaviors: Optional[List[str]] = None,
    bodyparts: Optional[List[str]] = None,
    animal_id: Optional[str] = None,
    center: Optional[str] = None,
    visualization: Optional[str] = None,
    normative_model: Optional[str] = None,
    aggregate_experiments: Optional[str] = None,
    colour_by: Optional[str] = None,
    roi_number: Optional[int] = None,
    animals_in_roi: Optional[List[str]] = None,
    roi_mode: str = "mousewise",
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
        animal_id (Optional[str]): ID of a specific animal.
        center (Optional[str]): Center point for position normalization (e.g., 'arena').
        visualization (Optional[str]): Visualization mode (e.g., 'networks', 'heatmaps').
        normative_model (Optional[str]): Cohort to use as a control group.
        aggregate_experiments (Optional[str]): Method to aggregate embeddings.
        colour_by (Optional[str]): Hue for coloring embeddings.
        roi_number (Optional[int]): ROI number to use for filtering.
        animals_in_roi (Optional[List[str]]): Animals that must be inside the ROI.
        roi_mode (str): Mode for ROI filtering ('mousewise' or 'behaviorwise').
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

    animal_id_opts = coordinates._animal_ids
    
    roi_num_opts = []
    if coordinates._roi_dicts:
        first_key = list(coordinates._roi_dicts.keys())[0]
        roi_num_opts = list(coordinates._roi_dicts[first_key].keys())

    # --- Statically defined options ---
    center_opts = ["arena"]
    vis_opts = ["networks", "heatmaps"] if origin == "plot_transitions" else ["confusion_matrix", "balanced_accuracy"]
    agg_exp_opts = ["time on cluster", "mean", "median"]
    color_by_opts = ["cluster", "exp_condition", "exp_id"]
    roi_mode_opts = ["mousewise", "behaviorwise"]

    # =========================================================================
    # 3. CONFIGURE AND RUN VALIDATION CHECKS
    # Format: (param_name, param_value, valid_options, is_list, custom_error)
    # =========================================================================
    validation_checks = [
        ("experiment_ids", experiment_ids, exp_id_opts, True, None),
        ("exp_condition", exp_condition, exp_cond_opts, False, "No experiment conditions loaded!"),
        ("exp_condition_order", exp_condition_order, cond_val_opts, True, "No conditions to order; check 'exp_condition'."),
        ("condition_values", condition_values, cond_val_opts, True, "No condition values available; check 'exp_condition'."),
        ("normative_model", normative_model, cond_val_opts, False, "No condition values available to select a normative model."),
        ("behaviors", behaviors, behavior_opts, True, "No supervised annotations or soft counts loaded!"),
        ("bodyparts", bodyparts, bodypart_opts, True, None),
        ("animals_in_roi", animals_in_roi, animal_id_opts, True, None),
        ("animal_id", animal_id, animal_id_opts, False, None),
        ("center", center, center_opts, False, None),
        ("visualization", visualization, vis_opts, False, None),
        ("aggregate_experiments", aggregate_experiments, agg_exp_opts, False, None),
        ("colour_by", colour_by, color_by_opts, False, None),
        ("roi_number", roi_number, roi_num_opts, False, "No ROIs were defined for this project."),
        ("roi_mode", roi_mode, roi_mode_opts, False, None),
    ]

    for name, value, options, is_list, error_msg in validation_checks:
        _validate_parameter(name, value, options, is_list, error_msg)

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

    if "circular" in coordinates._arena and roi_number is None:

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

    elif "polygonal" in coordinates._arena or roi_number is not None:

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
     

def heatmap(
    dframe: pd.DataFrame,
    bodyparts: List,
    xlim: tuple = None,
    ylim: tuple = None,
    title: str = None,
    mask: np.ndarray = None,
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
        save (str): if provided, saves the figure to the specified file.
        dpi (int): dots per inch of the figure to create.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        kwargs: additional arguments to pass to the seaborn kdeplot function.

    Returns:
        heatmaps (plt.figure): figure with the specified characteristics
    """
    # noinspection PyTypeChecker
    if ax is None:
        heatmaps, ax = plt.subplots(
            1,
            len(bodyparts),
            sharex=True,
            sharey=True,
            dpi=dpi,
            figsize=(8 * len(bodyparts), 8),
        )

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

    for i, bpart in enumerate(bodyparts):
        heatmap = dframe[bpart].loc[mask]

        if len(bodyparts) > 1:
            sns.kdeplot(
                x=heatmap.x,
                y=heatmap.y,
                cmap="magma",
                fill=True,
                alpha=1,
                ax=ax[i],
                **kwargs,
            )
        else:
            sns.kdeplot(
                x=heatmap.x,
                y=heatmap.y,
                cmap="magma",
                fill=True,
                alpha=1,
                ax=ax,
                **kwargs,
            )
            ax = np.array([ax])

    for x, bp in zip(ax, bodyparts):
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
