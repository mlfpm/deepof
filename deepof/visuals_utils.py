# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof

"""Plotting utility functions for the deepof package."""
import calendar
import os
import re
import time
import warnings
from typing import Any, List, NewType, Tuple, Union
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from matplotlib.patches import Ellipse
from natsort import os_sorted

import deepof.post_hoc
import deepof.utils
from deepof.data_loading import get_dt, load_dt
from deepof.config import PROGRESS_BAR_FIXED_WIDTH



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
        behaviors_options_list+=["Cluster "+str(i) for i in range(N_clusters)]


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

    #get lists of all body parts     
    bodyparts_options_list = np.unique(
        np.concatenate(
            [
            coordinates._tables[key].columns.levels[0] #read first elements from column headers from table
            if type(coordinates._tables[key]) != str   #if table is not a save path
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
    

def plot_arena(
    coordinates: coordinates, center: str, color: str, ax: Any, key: str
): # pragma: no cover
    """Plot the arena in the given canvas.

    Args:
        coordinates (coordinates): deepof Coordinates object.
        center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        color (str): color of the displayed arena.
        ax (Any): axes where to plot the arena.
        key str: key of the animal to plot with optional "all of them" (if key=="average").
    """
    if key != "average":
        arena = coordinates._arena_params[key]

    if "circular" in coordinates._arena:

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

    elif "polygonal" in coordinates._arena:

        if center == "arena" and key == "average":

            arena = calculate_average_arena(coordinates._arena_params)
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
                center=arena[0],
                axes=arena[1],
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

            if tag_dict[_id + undercond + "climb_arena"][fnum]:
                write_on_frame("climb_arena", down_pos)
            elif tag_dict[_id + undercond + "immobility"][fnum]:
                write_on_frame("immobility", down_pos)
            elif tag_dict[_id + undercond + "sniff_arena"][fnum]:
                write_on_frame("sniff_arena", down_pos)

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


    """
    # if no frames are specified, take all of them
    if frames is None:
        frames = np.array(range(0,len(frame_mask)))

    valid_frames = frames[frame_mask[frames]]
    diff_frames=np.diff(valid_frames)

    #ensure that no frames are requested that are outside of the provided data
    if len(valid_frames) >= frame_limit:
        valid_frames = valid_frames[0:frame_limit]

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

            out.write(res_frame)

        except IndexError:
            ret = False

    cap.release()
    cv2.destroyAllWindows()


def output_videos_per_cluster(
    video_paths: dict,
    behavior_dict: table_dict,
    behavior: str,
    frame_rate: float = 25,
    frame_limit_per_video: int = np.inf,
    bin_info: dict = None,
    single_output_resolution: tuple = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    out_path: str = ".",
): # pragma: no cover
    """Given a list of videos, and a list of soft counts per video, outputs a video for each cluster.

    Args:
        video_paths: dict of paths to the videos
        soft_counts: table_dict of soft counts per video
        frame_rate: frame rate of the videos
        frame_limit_per_video: number of frames to render per video.
        bin_info (dict): dictionary containing indices to plot for all experiments
        single_output_resolution: if single_output is provided, this is the resolution of the output video.
        min_confidence: minimum confidence threshold for a frame to be considered part of a cluster.
        min_bout_duration: minimum duration of a bout to be considered.
        out_path: path to the output directory.

    """

    meta_info=get_dt(behavior_dict,list(behavior_dict.keys())[0],only_metainfo=True)
    if behavior is not None:
        behaviors = [behavior]
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
      
        #with tqdm(total=len(behavior_dict.keys()), desc=f"{'Collecting experiments':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="experiment", leave=False) as pbar:
        for key in behavior_dict.keys():
            
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
                frames = bin_info[key]

            output_cluster_video(
                cap,
                out,
                confidence_mask,
                v_width,
                v_height,
                video_paths[key],
                frame_limit_per_video,
                frames,
            )
        

        out.release()
        #to not flood the output with loading bars
        clear_output()



def output_annotated_video(
    video_path: str,
    soft_counts: np.ndarray,
    behavior: str,
    frame_rate: float = 25,
    frames: np.array = None,
    out_path: str = ".",
): # pragma: no cover
    """Given a video, and soft_counts per frame, outputs a video with the frames annotated with the cluster they belong to.

    Args:
        video_path: full path to the video
        soft_counts: soft cluster assignments for a specific video
        behavior (str): Behavior or Cluster to that gets exported. If none is given, all are exported for softcounts and only nose2nose is exported for supervised annotations.
        frame_rate: frame rate of the video
        frames: frames that should be exported.
        cluster_names: dictionary with user-defined names for each cluster (useful to output interpretation).
        out_path: out_path: path to the output directory.

    """
    #If a specific behavior is requested, annotate that behavior
    if behavior is not None:
        hard_counts = soft_counts[behavior]>0.1
        idx = soft_counts[behavior]>0.1 #OK, I know this looks weird, but it is actually not a bug and works
        hard_counts = hard_counts.astype(str)
        hard_counts[idx]=str(behavior)
        hard_counts[~idx]=""
    # Else if every frame has only one distinct behavior assigned to it, annotate all behaviors
    elif not (np.sum(soft_counts, 1)>1.9).any():
        hard_counts = soft_counts.idxmax(axis=1)
    else:
        raise ValueError(
            "Cannot accept no behavior for supervised annotations!"
        )
    
    #ensure that no frames are requested that are outside of the provided data
    if np.max(frames) >= len(hard_counts):
        frames = np.where(frames<len(hard_counts))[0]

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

    first_run=True
    for i in tqdm(frames, desc=f"{'Exporting behavior video':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="Frame"):

        if first_run:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            first_run=False

        ret, frame = cap.read()
        if ret == False or cap.isOpened() == False:
            break

        try:
            cv2.putText(
                frame,
                str(hard_counts[i]),
                (int(v_width * 0.3 / 10), int(v_height / 1.05)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            out.write(frame)

        except IndexError:
            ret = False

    cap.release()
    cv2.destroyAllWindows()