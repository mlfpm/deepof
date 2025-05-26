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
from typing import Any, List, NewType, Tuple, Union
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
from deepof.config import PROGRESS_BAR_FIXED_WIDTH, ONE_ANIMAL_COLOR_MAP, TWO_ANIMALS_COLOR_MAP



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
        Cluster_max=np.max([int(re.search(r'\d+', cluster)[0]) for cluster in clusters])
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

    head = np.concatenate(
        [
            data.xs(f"{animal_id}Nose", 1).values,
            data.xs(f"{animal_id}Left_ear", 1).values,
            data.xs(f"{animal_id}Spine_1", 1).values,
            data.xs(f"{animal_id}Right_ear", 1).values,
        ],
        axis=1,
    )

    body = np.concatenate(
        [
            data.xs(f"{animal_id}Spine_1", 1).values,
            data.xs(f"{animal_id}Left_fhip", 1).values,
            data.xs(f"{animal_id}Left_bhip", 1).values,
            data.xs(f"{animal_id}Spine_2", 1).values,
            data.xs(f"{animal_id}Right_bhip", 1).values,
            data.xs(f"{animal_id}Right_fhip", 1).values,
        ],
        axis=1,
    )

    tail = np.concatenate(
        [
            data.xs(f"{animal_id}Spine_2", 1).values,
            data.xs(f"{animal_id}Tail_base", 1).values,
        ],
        axis=1,
    )

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


def _preprocess_time_bins(
    coordinates: coordinates,
    bin_size: Union[int, str],
    bin_index: Union[int, str],
    precomputed_bins: np.ndarray = None,
    tab_dict_for_binning: table_dict = None,
    experiment_id: str = None,
    samples_max: str = 20000,
    down_sample: bool = True,
):
    """Return a heatmap of the movement of a specific bodypart in the arena.

    If more than one bodypart is passed, it returns one subplot for each.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        tab_dict_for_binning (table_dict): table_dict that will be used as reference for maximum allowed table lengths. if None, basic table lengths from coordinates are used. 
        experiment_id (str): id of the experiment of time bins should
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        down_sample (bool): Use downsampling to get samples_max samples (if True). Uses cutting until sample of number samples_max if False.

    Returns:
        bin_info (dict): dictionary containing indices to plot for all experiments
    """

    # warn in case of conflicting inputs
    if precomputed_bins is not None and any(
        [bin_index is not None, bin_size is not None]
    ):
        warning_message = (
            "\033[38;5;208m\n"
            "Warning! If precomputed_bins is given, inputs bin_index and bin_size get ignored!"
            "\033[0m"
        )
        warnings.warn(warning_message)
    # init outputs
    bin_size_int = None
    bin_info = {}
    # dictionary to contain warnings for start time truncations (yes, I'll refactor this when I have some spare time)
  

    # get start and end times for each table
    start_times = coordinates.get_start_times()
    table_lengths = {}
    if tab_dict_for_binning is None:
        table_lengths = coordinates.get_table_lengths()
    else:
        for key in tab_dict_for_binning.keys():
            table_lengths[key]=int(get_dt(tab_dict_for_binning,key,only_metainfo=True)['shape'][0])

    #init specific warnings
    warn_start_time = {}
    start_too_late_flag = dict.fromkeys(table_lengths,False)
    end_too_late_flag = dict.fromkeys(table_lengths,False)

    # if a specific experiment is given, calculate time bin info only for this experiment
    if experiment_id is not None:
        start_times = {experiment_id: start_times[experiment_id]}
        table_lengths = {experiment_id: table_lengths[experiment_id]}

    pattern = r"^\b\d{1,6}:\d{1,6}:\d{1,6}(?:\.\d{1,12})?$"


    # Case 1: Precomputed bins were given
    if precomputed_bins is not None:
        
        for key in table_lengths.keys():
            arr=np.full(table_lengths[key], False, dtype=bool)                     # Create max len boolean array
            arr[:len(precomputed_bins)] = precomputed_bins[:table_lengths[key]]    # Fill array to max with precomputed bins
            bin_info[key] = np.where(arr)[0]                                       # Extract position info of True entries

            if len(precomputed_bins) > len(arr):
                end_too_late_flag[key]=True


    # Case 2: Integer bins were given
    elif type(bin_size) is int and type(bin_index) is int:

        bin_size_int = int(np.round(bin_size * coordinates._frame_rate))           # Get integer bin size based on frame rate
        for key in table_lengths.keys():
            bin_start = np.min([table_lengths[key],bin_size_int * bin_index])      # Get start and end positions that cannot exceed the table lengths
            bin_end = np.min([table_lengths[key],bin_size_int * (bin_index + 1)])
            bin_info[key] = np.arange(bin_start,bin_end,1)                         # Get range between starts and ends

            if bin_size_int * bin_index > table_lengths[key]:
                start_too_late_flag[key]=True
            if bin_size_int * (bin_index + 1) > table_lengths[key]:
                end_too_late_flag[key]=True


    # Case 3: Bins given as valid time-string ranges 
    elif (
        type(bin_size) is str
        and type(bin_index) is str
        and re.match(pattern, bin_size) is not None
        and re.match(pattern, bin_index) is not None
    ):

        # calculate bin size as int
        bin_size_int = int(
            np.round(time_to_seconds(bin_size) * coordinates._frame_rate)
        )

        # find start and end positions with sampling rate
        for key in table_lengths:
            start_time = time_to_seconds(start_times[key])                         # Get start of time vector for specific experiment 
            bin_index_time = time_to_seconds(bin_index)                            # Convert time string to float representing seconds
            start_time_adjusted=int(                                               # Get True start sample number
                np.round((bin_index_time - start_time) * coordinates._frame_rate)
            )
            bin_start = np.max([0,start_time_adjusted])                           # Ensure that samples stay within possible range  
            if bin_start > table_lengths[key]:
                start_too_late_flag[key]=True
            bin_start = np.min([table_lengths[key],bin_start])                         
            bin_end = np.max([0,bin_size_int + start_time_adjusted])
            if bin_end > table_lengths[key]:
                end_too_late_flag[key]=True
            bin_end = np.min([table_lengths[key],bin_end])
            bin_info[key] = np.arange(bin_start,bin_end,1) 

            if start_time_adjusted < 0:                                            # Warn user, if the start-time entered by the user is
                warn_start_time[key]=seconds_to_time(bin_end-bin_start)            # less than the table start time



    # Case 4: If nonsensical input was given, return warning and default bins
    elif bin_size is not None:
        # plot short default bin, if user entered bins incorrectly

        warning_message = (
            "\033[38;5;208m\n"
            "Warning! bin_index or bin_size were given in an incorrect format!\n"
            "Please use either integers or strings with format HH:MM:SS or HH:MM:SS.SSS ...\n"
            "Proceed to plot default binning (bin_index = 0, bin_size = 60)!"
            "\033[0m"
        )
        warnings.warn(warning_message)

        bin_info = dict.fromkeys(table_lengths,np.arange(0, int(np.round(60 * coordinates._frame_rate)),1)) 

    # Case 5: No bins are given, so bins are set to the signal length
    elif precomputed_bins is None and bin_size is None and bin_index is None:

        for key in table_lengths.keys():
            bin_info[key] = np.arange(0,table_lengths[key],1)

    #Downsample bin_info if necessary
    info_message=None
    for key in bin_info.keys():
        full_length=len(bin_info[key])
        if full_length>samples_max:
            if down_sample:
                selected_indices=np.linspace(0, full_length-1, samples_max).astype(int)
            else:
                selected_indices=np.arrange(0,samples_max,1)
            bin_info[key]=bin_info[key][selected_indices]
            
            #I know the info message just needs to be set once, but this does not cause any performance issues and is more readable
            info_message=(
                "\033[33m\n"
                f"Info! The selected range for plotting exceeds the maximum number of {samples_max} samples allowed for plotting!\n"
                f"To plot the selected full range the plot will be downsampled accordingy by a factor of approx. {int((full_length-1)/samples_max)}.\n"
                "To avoid this, you can increase the input parameter \"samples_max\", but this also increases computation time."
                "\033[0m"
            )

    if info_message:
        print(info_message)


    # Validity checks and warnings for created bins
    if bin_size is not None and bin_index is not None:
        # warning messages in case of weird indexing
        bin_warning = False
        if warn_start_time:
            warning_message = (
                "\033[38;5;208m\n"
                "Warning! The chosen time range starts before the data time axis starts in at least one data set!\n"
                f"Therefore, the resulting lengths in the truncated bins are: {warn_start_time}"
                "\033[0m"
            )
            warnings.warn(warning_message)
        
        for key in table_lengths:
            if bin_size_int == 0:
                raise ValueError("Please make sure bin_size is > 0")
            elif start_too_late_flag[key]:
                raise ValueError(
                    "[Error in {}]: Please make sure bin_index is within the time range. i.e < {} or < {} for a bin_size of {}".format(
                        key,
                        seconds_to_time(
                            table_lengths[key] / coordinates._frame_rate, False
                        ),
                        int(np.ceil(table_lengths[key] / bin_size_int)),
                        bin_size,
                    )
                )
            elif end_too_late_flag[key]:
                if not bin_warning:
                    truncated_length = seconds_to_time(
                        (len(bin_info[key])) / coordinates._frame_rate,
                        False,
                    )
                    warning_message = (
                        "\033[38;5;208m\n"
                        "[Warning for {}]: The chosen time range exceeds the signal length for at least one data set!\n"
                        f"Therefore, the chosen bin was truncated to a length of {truncated_length}"
                        "\033[0m".format(key)
                    )
                    if precomputed_bins is None and table_lengths[key] - bin_size_int > 0:
                        warning_message= (warning_message +
                            "\n\033[38;5;208mFor full range bins, choose a start time <= {} or a bin index <= {} for a bin_size of {}\033[0m".format(
                                seconds_to_time(
                                    (table_lengths[key] - bin_size_int)
                                    / coordinates._frame_rate,
                                    False,
                                ),
                                int(np.ceil(table_lengths[key] / bin_size_int)) - 2,
                                bin_size,
                                )
                        )
                            
                        
                    warnings.warn(warning_message)
                    bin_warning = True

    return bin_info


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


def get_supervised_behaviors_in_roi(
    cur_supervised: pd.DataFrame,
    local_bin_info: dict,
    animal_ids: Union[str, list], 
    special_case: bool =True,
):
    """Filter supervised behaviors based on rois given by animal_ids.

    Args:
        cur_supervised (pd.DataFrame): data frame with supervised behaviors.
        local_bin_info (dict): bin_info dictionary for one experiment, containing field "time" with array of included frames and fields "animal_id" with boolean arrays that denote which mace were within the selcted roi for these frames
        animal_ids (Union[str, list]): single or multiple animal ids
    
    Returns:
        cur_supervised (pd.DataFrame): data frame with supervised behaviors with detections outside of the ROI set to NaN
    """
    cur_supervised=copy.copy(cur_supervised)

    if animal_ids is None or animal_ids=="" or animal_ids==[""]:
        animal_ids=[""]
        animal_ids_edited=[""]
    elif type(animal_ids)==str:
        animal_ids=[animal_ids]
        animal_ids_edited=[animal_ids+"_"]
    elif type(animal_ids)==list:
        animal_ids_edited=[aid+"_" for aid in animal_ids]

    # Create set of valid columns that contain any animal id
    valid_cols = set()
    for col in cur_supervised.columns:
        level0 = col[0] if isinstance(col, tuple) else col
        for aid in animal_ids_edited:
            if not special_case or f"{aid}" in level0:
                valid_cols.add(col)
                break  # skip checking for more ids in column

    # Apply ROIs to each behavior for each mouse. Multiple animal behaviors require all involved animals to be in ROI
    for aid_2 in local_bin_info.keys():
        if aid_2 == "time":
            continue #skip "time" array that contains time binning info

        aid_2_cols = []
        if special_case:
            for col in valid_cols:
                level0 = col[0] if isinstance(col, tuple) else col
                if aid_2 in level0:
                    aid_2_cols.append(col)
        else:
            if aid_2 in animal_ids:
               aid_2_cols=list(valid_cols) 
        # Apply ROI filter if there are columns to process
        if aid_2_cols:
            cur_supervised.loc[~local_bin_info[aid_2], aid_2_cols] = np.nan

    # Set all behavior columns to NaN in which none of the requested animals was involved
    invalid_cols = cur_supervised.columns.difference(valid_cols)
    cur_supervised[invalid_cols] = np.nan
            
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

    if len(cur_unsupervised.shape)==1:
        for aid in animal_ids:
            cur_unsupervised[~local_bin_info[aid]]=-1    
    else:
        for aid in animal_ids: 
            cur_unsupervised[~local_bin_info[aid]]=np.NaN   

    return cur_unsupervised


def get_beheavior_frames_in_roi(
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
   

######
#Functions not included in property based testing for not having a clean return
######


#not covered by testing as the only purpose of this function is to throw specific exceptions
def _check_enum_inputs(
    coordinates: coordinates,
    supervised_annotations: table_dict = None,
    soft_counts: table_dict = None,
    origin: str = None,
    experiment_ids: list = None,
    exp_condition: str = None,
    exp_condition_order: list = None,
    condition_values: list = None,
    behaviors: list = None,
    bodyparts: list = None,
    animal_id: str = None,
    center: str = None,
    visualization: str = None,
    normative_model: str = None,
    aggregate_experiments: str = None,
    colour_by: str = None,
    roi_number: int = None,
    animals_in_roi: list = None,
): # pragma: no cover
    """
    Checks and validates enum-like input parameters for the different plot functions.

    Args:
    coordinates (coordinates): deepof Coordinates object.
    supervised_annotations (table_dict): Contains all informations regarding supervised annotations. 
    soft_counts (table_dict): Contains all informations regarding unsupervised annotations. 
    origin (str): name of the function this function was called from (only applicable in specific cases)
    experiment_ids (list): list of data set name of the animal to plot.
    exp_condition (str): Experimental condition to plot.
    exp_condition_order (list): Order in which to plot experimental conditions.
    condition_values (list): Experimental condition value to plot.
    behaviors (list): list of entered animal behaviors.
    bodyparts (list): list of body parts to plot.
    animal_id (str): Id of the animal.
    center (str): Name of the visual marker (i.e. currently only the arena) to which the positions will be centered.
    visualization (str): visualization mode. Can be either 'networks', or 'heatmaps'.
    normative_model (str): Name of the cohort to use as controls.
    aggregate_experiments (str): Whether to aggregate embeddings by experiment (by time on cluster, mean, or median).
    colour_by (str): hue by which to colour the embeddings. Can be one of 'cluster', 'exp_condition', or 'exp_id'.
    roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
    animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of teh ROI get excluded 
       

    """
    # activate warnings (again, because just putting it at the beginning of the skript
    # appears to yield inconsitent results)
    warnings.simplefilter("always", UserWarning)

    #fix types
    if isinstance(experiment_ids, str):
        experiment_ids=[experiment_ids]
    if isinstance(exp_condition_order, str):
        exp_condition_order=[exp_condition_order]
    if isinstance(condition_values, str):
        condition_values=[condition_values]
    if isinstance(behaviors, str):
        behaviors=[behaviors]
    if isinstance(bodyparts, str):
        bodyparts=[bodyparts]
    if isinstance(animals_in_roi, str):
        animals_in_roi=[animals_in_roi]
     

    # Generate lists of possible options for all enum-likes (solution will be improved in the future)
    if origin == "plot_heatmaps":
        experiment_id_options_list = ["average"] + os_sorted(
            list(coordinates._tables.keys())
        )
    else:
        experiment_id_options_list = os_sorted(list(coordinates._tables.keys()))

    #get all possible behaviors from supervised annotations and soft counts
    behaviors_options_list=[]
    if supervised_annotations is not None:
        first_key=list(supervised_annotations.keys())[0]
        behaviors_options_list=get_dt(supervised_annotations,first_key,only_metainfo=True)['columns']
    if soft_counts is not None:
        first_key=list(soft_counts.keys())[0]
        N_clusters=get_dt(soft_counts,first_key,only_metainfo=True)['num_cols']
        behaviors_options_list+=["Cluster_"+str(i) for i in range(N_clusters)]


    if coordinates.get_exp_conditions is not None:
        exp_condition_options_list = np.unique(
            np.concatenate(
                [
                    condition.columns.values[:]
                    for condition in coordinates.get_exp_conditions.values()
                ]
            )
        )
    else:
        exp_condition_options_list = []
    if exp_condition is not None and exp_condition in exp_condition_options_list:
        condition_value_options_list = np.unique(
            np.concatenate(
                [
                    condition[exp_condition].values.astype(str)
                    for condition in coordinates.get_exp_conditions.values()
                ]
            )
        )
    else:
        condition_value_options_list = []
    if roi_number is not None and coordinates._roi_dicts is not None:    
        first_key=list(coordinates._roi_dicts.keys())[0]
        roi_number_options_list=list(coordinates._roi_dicts[first_key].keys())
    else:
        roi_number_options_list=[]

    #get lists of all body parts     
    bodyparts_options_list = np.unique(
        np.concatenate(
            [
            coordinates._tables[key].columns.levels[0] #read first elements from column headers from table
            if type(coordinates._tables[key]) != dict   #if table is not a save path
            else [t[0] for t in get_dt(coordinates._tables,key,only_metainfo=True)['columns']] #otherwise read in saved column headers and then extract first elements
            for key 
            in coordinates._tables.keys()
            ]
        )
    )
    bodyparts_options_list = [
        item for item in bodyparts_options_list if item not in coordinates._excluded
    ]
    animal_id_options_list = coordinates._animal_ids
    # fixed option lists
    center_options_list = ["arena"]
    if origin == "plot_transitions":
        visualization_options_list = ["networks", "heatmaps"]
    else:
        visualization_options_list = ["confusion_matrix", "balanced_accuracy"]
    aggregate_experiments_options_list = ["time on cluster", "mean", "median"]
    colour_by_options_list = ["cluster", "exp_condition", "exp_id"]

    # check if given values are valid. Throw exception and suggest correct values if not
    if experiment_ids is not None and not experiment_ids == [None] and not set(
        experiment_ids
    ).issubset(set(experiment_id_options_list)): 
        raise ValueError(
            'Included experiments need to be a subset of the following: {} ... '.format(
                str(experiment_id_options_list[0:4])[1:-1]
            )
        )
    
    if exp_condition is not None and exp_condition not in exp_condition_options_list:
        if len(exp_condition_options_list) > 0:
            raise ValueError(
                '"exp_condition" needs to be one of the following: {}'.format(
                    str(exp_condition_options_list)[1:-1]
                )
            )
        else:
            raise ValueError("No experiment conditions loaded!")
        
    if exp_condition_order is not None and not exp_condition_order == [None] and not set(
        exp_condition_order
    ).issubset(set(condition_value_options_list)):
        if len(condition_value_options_list) > 0:
            raise ValueError(
                'One or more conditions in "exp_condition_order" are not part of: {}'.format(
                    str(condition_value_options_list)[1:-1]
                )
            )
        else:
            raise ValueError("No experiment conditions loaded!")
        
    if condition_values is not None and not condition_values == [None] and not set(condition_values).issubset(
        set(condition_value_options_list)
    ):
        if len(condition_value_options_list) > 0:
            raise ValueError(
                'One or more condition values in "condition_value(s)" are not part of {}'.format(
                    str(condition_value_options_list)[1:-1]
                )
            )
        else:
            raise ValueError("No experiment conditions loaded!")
        
    if behaviors is not None and not behaviors == [None] and not set(
        behaviors
    ).issubset(set(behaviors_options_list)):
        if len(behaviors_options_list) > 0:
            raise ValueError(
                'One or more behaviors are not part of: {}'.format(
                    str(behaviors_options_list)[1:-1]
                )
            )
        else:
            raise ValueError("No supervised annotations or soft counts loaded!")
        
    if (
        normative_model is not None
        and normative_model not in condition_value_options_list
    ):
        if len(condition_value_options_list) > 0:
            raise ValueError(
                '"normative_model" needs to be one of the following: {}'.format(
                    str(condition_value_options_list)[1:-1]
                )
            )
        else:
            raise ValueError("No experiment conditions loaded!")
        
    if bodyparts is not None and not bodyparts == [None] and not set(bodyparts).issubset(
        set(bodyparts_options_list)
    ):
        raise ValueError(
            'One or more bodyparts in "bodyparts" are not part of: {}'.format(
                str(bodyparts_options_list)[1:-1]
            )
        )
    
    if animals_in_roi is not None and not animals_in_roi == [None] and not set(animals_in_roi).issubset(
        set(animal_id_options_list)
    ):
        raise ValueError(
            'One or more animal_ids in "animal_in_roi" are not part of: {}'.format(
                str(animal_id_options_list)[1:-1]
            )
        )
    
    if animal_id is not None and animal_id not in animal_id_options_list:
        raise ValueError(
            '"animal_id" needs to be one of the following: {}'.format(
                str(animal_id_options_list)
            )
        )
    
    if center is not None and center not in center_options_list:
        raise ValueError(
            'For input "center" currently only {} is supported'.format(
                str(center_options_list)
            )
        )
    
    if visualization is not None and visualization not in visualization_options_list:
        raise ValueError(
            '"visualization" needs to be one of the following: {}'.format(
                str(visualization_options_list)
            )
        )
    
    if (
        aggregate_experiments is not None
        and aggregate_experiments not in aggregate_experiments_options_list
    ):
        raise ValueError(
            '"aggregate_experiments" needs to be one of the following: {}'.format(
                str(aggregate_experiments_options_list)
            )
        )
    
    if colour_by is not None and colour_by not in colour_by_options_list:
        raise ValueError(
            '"colour_by" needs to be one of the following: {}'.format(
                str(colour_by_options_list)
            )
        )
    
    if roi_number is not None and roi_number not in roi_number_options_list:
        if len(roi_number_options_list)>0:
            raise ValueError(
                'If you want to apply ROIs, "roi_number" needs to be one of the following: {}'.format(
                    str(roi_number_options_list)
                )
            )
        else:
            raise ValueError("No regions of interest (ROI)s were defined for this project!\n You can define ROIs during project creation if you have set number_of_rois\n to a number between 1 and 20 during project definition before")
    

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


def _tag_annotated_frames(
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
    arena_type,
    debug,
    coords,
): # pragma: no cover
    """Annotate a given frame with on-screen information about the recognised patterns.

    Helper function for annotate_video. No public use intended.

    """
    arena, w, h = arena

    def write_on_frame(text, pos, col=(150, 255, 150)):
        """Partial closure over cv2.putText to avoid code repetition."""
        return cv2.putText(frame, text, pos, font, 0.75, col, 2)

    def conditional_flag():
        """Return a tag depending on a condition."""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return left_flag
        return right_flag

    def conditional_pos():
        """Return a position depending on a condition."""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return corners["downleft"]
        return corners["downright"]

    def conditional_col(cond=None):
        """Return a colour depending on a condition."""
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

        if arena_type.startswith("circular"):
            # Print arena for debugging
            cv2.ellipse(
                img=frame,
                center=np.round(arena[0]).astype(int),
                axes=np.round(arena[1]).astype(int),
                angle=arena[2],
                startAngle=0,
                endAngle=360,
                color=(40, 86, 236),
                thickness=3,
            )

        elif arena_type.startswith("polygonal"):

            # Draw polygon
            cv2.polylines(
                img=frame,
                pts=[np.array(arena, dtype=np.int32)],
                isClosed=True,
                color=(40, 86, 236),
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

        if tag_dict[animal_ids[0] + "_" + animal_ids[1] + "_nose2nose"][fnum]:
            write_on_frame("Nose-Nose", conditional_pos())
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

        if tag_dict[animal_ids[0] + "_" + animal_ids[1] +  "_nose2body"][fnum] and left_flag:
            write_on_frame("nose2body", corners["downleft"])
            left_flag = False

        if tag_dict[animal_ids[1] + "_" + animal_ids[0] +  "_nose2body"][fnum] and right_flag:
            write_on_frame("nose2body", corners["downright"])
            right_flag = False

        if tag_dict[animal_ids[0] + "_" + animal_ids[1] +  "_nose2tail"][fnum] and left_flag:
            write_on_frame("Nose-Tail", corners["downleft"])
            left_flag = False

        if tag_dict[animal_ids[1] + "_" + animal_ids[0] +  "_nose2tail"][fnum] and right_flag:
            write_on_frame("Nose-Tail", corners["downright"])
            right_flag = False

        if tag_dict[animal_ids[0] + "_" + animal_ids[1] + "_sidebyside"][fnum] and left_flag and conditional_flag():
            write_on_frame("Side-side", conditional_pos())
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

        if tag_dict[animal_ids[0] + "_" + animal_ids[1] + "_sidereside"][fnum] and left_flag and conditional_flag():
            write_on_frame("Side-Rside", conditional_pos())
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

            if tag_dict[_id + undercond + "climb-arena"][fnum]:
                write_on_frame("climb-arena", down_pos)
            elif tag_dict[_id + undercond + "immobility"][fnum]:
                write_on_frame("immobility", down_pos)
            elif tag_dict[_id + undercond + "sniff-arena"][fnum]:
                write_on_frame("sniff-arena", down_pos)

        # Define the condition controlling the colour of the speed display
        if len(animal_ids) > 1:
            colcond = frame_speeds[_id] == max(list(frame_speeds.values()))
        else:
            colcond = hparams["stationary_threshold"] < frame_speeds

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
    supervised_annotations: Union[str, pd.DataFrame],
    key: int,
    frame_limit: int = np.inf,
    debug: bool = False,
    params: dict = {},
) -> True: # pragma: no cover
    """Render a version of the input video with all supervised taggings in place.

    Args:
        coordinates (deepof.preprocessing.coordinates): coordinates object containing the project information.
        tag_dict (Union[str, pd.DataFrame]): Either path to the saving location of teh dataset or the dataste itself
        vid_key: for internal usage only; key of the video to tag in coordinates._videos.
        frame_limit (float): limit the number of frames to output. Generates all annotated frames by default.
        debug (bool): if True, several debugging attributes (such as used body parts and arena) are plotted in the output video.
        params (dict): dictionary to overwrite the default values of the hyperparameters of the functions that the supervised pose estimation utilizes.

    """

    tag_dict=get_dt(supervised_annotations, key)

    

    # Extract useful information from coordinates object
    vid_path=coordinates.get_videos(full_paths=True)[key]

    animal_ids = coordinates._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    arena_params = coordinates._arena_params[key]
    h, w = coordinates._video_resolution[key]
    corners = deepof.annotation_utils.frame_corners(w, h)

    cap = cv2.VideoCapture(vid_path)
    # Keep track of the frame number, to align with the tracking data
    fnum = 0
    writer = None
    frame_speeds = (
        {_id: -np.inf for _id in animal_ids} if len(animal_ids) > 1 else -np.inf
    )


    # scale arena_params back o video res
    scaling_ratio = coordinates._scales[key][2]/coordinates._scales[key][3]
    if "polygonal" in coordinates._arena:
        arena_params=np.array(arena_params)*scaling_ratio
    elif "circular" in coordinates._arena:
        arena_params=(tuple(np.array(arena_params[0])*scaling_ratio),tuple(np.array(arena_params[1])*scaling_ratio),arena_params[2])
                              
    # Loop over the frames in the video
    with tqdm(total=frame_limit, desc="annotating Video", unit="frame") as pbar:
        while cap.isOpened() and fnum < frame_limit:

            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:  # pragma: no cover
                print("Can't receive frame (stream end?). Exiting ...")
                break

            font = cv2.FONT_HERSHEY_DUPLEX

            # Capture speeds
            if len(animal_ids) == 1: #or fnum % params["speed_pause"] == 0:
                frame_speeds = tag_dict["speed"][fnum]
            else:
                for _id in animal_ids:
                    frame_speeds[_id] = tag_dict[_id + undercond + "speed"][fnum]

            # Display all annotations in the output video
            _tag_annotated_frames(
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
                coordinates._arena,
                debug,
                coordinates.get_coords_at_key(center=None, to_video=True, scale=coordinates._scales[key], key=key), #coordinates.get_coords(center="arena")[key],  
            )

            if writer is None:
                # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                # Define the FPS. Also frame size is passed.
                out_folder=os.path.join(coordinates._project_path, coordinates._project_name, "Out_videos")
                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)   
                
                writer = cv2.VideoWriter()
                writer.open(
                    os.path.join(
                        out_folder,
                        key + "_supervised_tagged.avi",
                    ),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    coordinates._frame_rate,
                    (frame.shape[1], frame.shape[0]),
                    True,
                )

            writer.write(frame)
            fnum += 1
            pbar.update()

    cap.release()
    cv2.destroyAllWindows()

    return True


def output_cluster_video(
    cap: Any,
    out: Any,
    frame_mask: list,
    v_width: int,
    v_height: int,
    path: str,
    frame_limit: int = np.inf,
    frames: np.array = None,
    display_time: bool = False,

): # pragma: no cover
    """Output a video with the frames corresponding to the cluster.

    Args:
        cap: video capture object
        out: video writer object
        frame_mask: list of booleans indicating whether a frame should be written
        v_width: video width
        v_height: video height
        path: path to the video file
        frame_limit: maximum number of frames to render
        frames: frames that can be selected from for export.
        display_time (bool): Displays current time in top left corner of the video frame

    """
    # if no frames are specified, take all of them
    if frames is None:
        frames = np.array(range(0,len(frame_mask)))

    valid_frames = frames[frame_mask[frames]]
    diff_frames=np.diff(valid_frames)

    #ensure that no frames are requested that are outside of the provided data
    if len(valid_frames) >= frame_limit:
        valid_frames = valid_frames[0:frame_limit]

    # Prepare text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    thickness = 2
    (text_width_time, text_height_time), baseline = cv2.getTextSize("time: 00:00:00", font, font_scale, thickness)
    x = 10  # 10 pixels from left
    y = 10 + text_height_time  # 10 pixels from top (accounting for text height)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    for i in range(len(valid_frames)):
        if i == 0 or diff_frames[i-1] != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, valid_frames[i])
        ret, frame = cap.read()
        if not cap.isOpened() or ret == False:
            break

        try:

            res_frame = cv2.resize(frame, [v_width, v_height])
            re_path = re.findall(r".+[/\\]([^/.]+?)(?=\.|DLC)", path)[0]

            if path is not None:
                cv2.putText(
                    res_frame,
                    re_path,
                    (int(v_width * 0.3 / 10), int(v_height / 1.05)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

            if display_time:

                disp_time = "time: "  + seconds_to_time(valid_frames[i]/frame_rate)
                # Draw black outline
                cv2.putText(res_frame, disp_time, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
                # Draw white main text
                cv2.putText(res_frame, disp_time, (x, y), font, font_scale, (255, 255, 255), thickness)

            out.write(res_frame)

        except IndexError:
            ret = False

    cap.release()
    cv2.destroyAllWindows()


def output_videos_per_cluster(
    video_paths: dict,
    behavior_dict: table_dict,
    behaviors: Union[str,list],
    frame_rate: float = 25,
    frame_limit_per_video: int = np.inf,
    bin_info: dict = None,
    roi_number: int = None,
    animals_in_roi: list = None,
    single_output_resolution: tuple = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    display_time: bool = False,
    out_path: str = ".",
    special_case: bool = False,
): # pragma: no cover
    """Given a list of videos, and a list of soft counts per video, outputs a video for each cluster.

    Args:
        video_paths: dict of paths to the videos
        behavior_dict: table_dict containing data tables with behavior information (presence or absence of behaviors (columns) for each frame (rows))
        behaviors (Union[str,list]): list of behaviors to annotate
        frame_rate: frame rate of the videos
        frame_limit_per_video: number of frames to render per video.
        bin_info (dict): dictionary containing indices to plot for all experiments
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of teh ROI get excluded 
        single_output_resolution: if single_output is provided, this is the resolution of the output video.
        min_confidence: minimum confidence threshold for a frame to be considered part of a cluster.
        min_bout_duration: minimum duration of a bout to be considered.
        display_time (bool): Displays current time in top left corner of the video frame
        out_path: path to the output directory.
    """

    #manual laoding bar for inner loop
    def _loading_basic(current, total, bar_length=68):
        progress = (current + 1) / total
        filled_length = int(bar_length * progress)
        arrow = '>' if filled_length < bar_length else ''
        bar = '[' + '=' * filled_length + arrow + ' ' * (bar_length - filled_length - len(arrow)) + ']'
        percent = f' {progress:.0%}'
        print(bar + percent, end='\r')
        if current == total - 1:
            print()  # Newline when complete

    meta_info=get_dt(behavior_dict,list(behavior_dict.keys())[0],only_metainfo=True)
    if isinstance(behaviors, str):
        behaviors = [behaviors]
    elif meta_info.get('columns') is not None:
        behaviors=meta_info['columns']
    else:
        behaviors = np.array(range(meta_info['num_cols']))


    # Iterate over all clusters, and output a masked video for each
    for cur_behavior in tqdm(behaviors, desc=f"{'Exporting behavior videos':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="video"):
    
        #creates a new line to ensure that the outer loading bar does not get overwritten by the inner one
        print("")

        out = cv2.VideoWriter(
            os.path.join(
                out_path,
                "Behavior={}_threshold={}_{}.mp4".format(
                    cur_behavior, min_confidence, calendar.timegm(time.gmtime())
                ),
            ),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            single_output_resolution,
        )
      
        bar_len=len(behavior_dict.keys())
        #this loop uses a manual loading bar as tqdm does not work here for some reason
        for i, key in enumerate(behavior_dict.keys()):

            _loading_basic(i, bar_len)
            
            cur_soft_counts = get_dt(behavior_dict,key)

            # If a specific behavior is requested, annotate that behavior
            if type(cur_soft_counts)==np.ndarray:

                # Get Cluster number from behavior input
                if type(cur_behavior) == str:
                    cur_behavior_idx=int(re.search(r'\d+', cur_behavior)[0])
                else:
                    cur_behavior_idx = cur_behavior

                hard_counts = pd.Series(cur_soft_counts[:, cur_behavior_idx]>0.1)
                idx = pd.Series(cur_soft_counts[:, cur_behavior_idx]>0.1)
                confidence = pd.Series(cur_soft_counts[:, cur_behavior_idx])
            else:
                hard_counts = cur_soft_counts[cur_behavior]>0.1
                idx = cur_soft_counts[cur_behavior]>0.1
                confidence = cur_soft_counts[cur_behavior]
            
            hard_counts = hard_counts.astype(str)
            hard_counts[idx]=str(cur_behavior)
            hard_counts[~idx]=""

            
            # Get hard counts and confidence estimates per cluster
            confidence_indices = np.ones(hard_counts.shape[0], dtype=bool)

            # Given a frame mask, output a subset of the given video to disk, corresponding to a particular cluster
            cap = cv2.VideoCapture(video_paths[key])
            v_width, v_height = single_output_resolution

            # Compute confidence mask, filtering out also bouts that are too short
            confidence_indices = deepof.utils.filter_short_bouts(
                idx.astype(float),
                confidence,
                confidence_indices,
                min_confidence,
                min_bout_duration,
            )
            confidence_mask = (hard_counts == str(cur_behavior)) & confidence_indices

            # get frames for current video
            frames = None
            if bin_info is not None:
                if roi_number is not None:
                    if special_case:
                        behavior_in=behaviors[0]
                    else:
                        behavior_in=None
                    frames=get_beheavior_frames_in_roi(behavior=behavior_in, local_bin_info=bin_info[key], animal_ids=animals_in_roi)
                else:
                    frames=bin_info[key]["time"]

            output_cluster_video(
                cap,
                out,
                confidence_mask,
                v_width,
                v_height,
                video_paths[key],
                frame_limit_per_video,
                frames,
                display_time,
            )
        

        out.release()
        #to not flood the output with loading bars
        clear_output()
    get_beheavior_frames_in_roi._warning_issued = False


def output_annotated_video(
    video_path: str,
    tab: np.ndarray,
    behaviors: list,
    frame_rate: float = 25,
    frames: np.array = None,
    display_time: bool = False,
    out_path: str = ".",
): # pragma: no cover
    """Given a video, and soft_counts per frame, outputs a video with the frames annotated with the cluster they belong to.

    Args:
        video_path: full path to the video
        soft_counts: soft cluster assignments for a specific video
        behavior (str): Behavior or Cluster to that gets exported. If none is given, all Clusters get exported for softcounts and only nose2nose gets exported for supervised annotations.
        frame_rate: frame rate of the video
        frames: frames that should be exported.
        display_time (bool): Displays current time in top left corner of the video frame
        out_path: out_path: path to the output directory.

    """
    # if every frame has only one distinct behavior assigned to it, plot all behaviors
    shift_name_box=True
    if behaviors is None and not (np.sum(tab, 1)>1.9).any():
        behaviors = list(tab.columns)
        shift_name_box=False
    elif behaviors is None:
        raise ValueError(
            "Cannot accept no behavior for supervised annotations!"
        )
    # create behavior_df that lists in which frames each behavior occurs
    behavior_df = tab[behaviors]>0.1
    idx = tab[behaviors]>0.1 #OK, I know this looks weird, but it is actually not a bug and works
    behavior_df = behavior_df.astype(str)
    behavior_df[idx]=behaviors
    behavior_df[~idx]=""
    # Else if every frame has only one distinct behavior assigned to it, annotate all behaviors

    
    # Ensure that no frames are requested that are outside of the provided data
    if np.max(frames) >= len(behavior_df):
        frames = np.where(frames<len(behavior_df))[0]

    # Given a frame mask, output a subset of the given video to disk, corresponding to a particular cluster
    cap = cv2.VideoCapture(video_path)

    # Get width and height of current video
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = os.path.join(
        out_path,
        os.path.split(video_path)[-1].split(".")[0] 
        + "_annotated_{}.mp4".format(calendar.timegm(time.gmtime())),
    )

    out = cv2.VideoWriter(
        video_out, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (v_width, v_height)
    )

    # Prepare text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    thickness = 2
    text_widths=[cv2.getTextSize(behavior, font, font_scale, thickness)[0][0] for behavior in behaviors]
    (text_width, text_height), baseline = cv2.getTextSize(behaviors[np.argmax(text_widths)], font, font_scale, thickness)
    (text_width_time, text_height_time), baseline = cv2.getTextSize("time: 00:00:00", font, font_scale, thickness)
    x = 10  # 10 pixels from left
    y = 10 + text_height_time  # 10 pixels from top (accounting for text height)
    padding = 5
    bg_color = get_behavior_colors(behaviors, tab)

    diff_frames = np.diff(frames)
    for i in tqdm(range(len(frames)), desc=f"{'Exporting behavior video':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="Frame"):

        if i == 0 or diff_frames[i-1] != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()

        if ret == False or cap.isOpened() == False:
            break

        try:
            ystep=0
            for z, behavior in enumerate(behaviors):
                if len(behavior_df[behavior][frames[i]])>0:
                    cv2.rectangle(frame, 
                        (v_width - text_width - x , y - text_height - padding +ystep),  # Top-left corner
                        (v_width - padding, y + baseline +ystep),  # Bottom-right corner
                        hex_to_BGR(bg_color[z]),  # Blue color (BGR format)
                        -1)  # Filled rectangle

                    # Draw black outline
                    cv2.putText(frame, str(behavior_df[behavior][frames[i]]), (v_width - text_width - padding, y+ystep), font, font_scale, (0, 0, 0), thickness + 2)
                    # Draw white main text
                    cv2.putText(frame, str(behavior_df[behavior][frames[i]]), (v_width - text_width - padding, y+ystep), font, font_scale, (255, 255, 255), thickness)
                if shift_name_box:
                    ystep=ystep+50
            
            if display_time:

                disp_time = "time: "  + seconds_to_time(frames[i]/frame_rate)
                # Draw black outline
                cv2.putText(frame, disp_time, (x, y), font, font_scale, (0, 0, 0), thickness + 2)
                # Draw white main text
                cv2.putText(frame, disp_time, (x, y), font, font_scale, (255, 255, 255), thickness)

            out.write(frame)
        except IndexError:
            ret = False

    cap.release()
    cv2.destroyAllWindows()

    #writevideo = FFMpegWriter(fps=frame_rate)
    #animation.save(save, writer=writevideo)

    return None


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
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of teh ROI get excluded                      
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.
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
