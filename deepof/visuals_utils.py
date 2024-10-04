# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof

"""Plotting utility functions for the deepof package."""
import copy
import numpy as np
import re
from typing import Any, List, NewType, Tuple, Union
import warnings
from deepof.data_loading import get_dt, load_dt


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

    return seconds


def seconds_to_time(seconds: float, cut_milliseconds: bool = True) -> str:
    """Compute a time string based on seconds as float.

    Args:
        seconds (float): time in seconds

    Returns:
        time_string (str): time string as input (format HH:MM:SS or HH:MM:SS.SSS...)
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
        l_max=time_string.find('.')+10
        time_string=time_string[0:l_max]

    return time_string


def calculate_average_arena(
    all_vertices: dict[List[Tuple[float, float]]], num_points: int = 10000
) -> np.array:
    """
    Calculates the average arena based on a list of polynomial vertices
    lists representing arenas. Polynomial vertices can have different lengths and start at different positions

    Args:
        vertices (dict[List[Tuple[float, float]]]): A dictionary of lists of 2D tuples representing the vertices of the arenas.
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
        Categorized effect size (int):
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
        max_bin_size (int): Maximum size that is accepted for any bins
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
    #dictionary to contain warnings for start time truncations (yes, I'll refactor this when I have some spare time)
  

    # get start and end times for each table
    start_times = coordinates.get_start_times()
    table_lengths = {}
    if tab_dict_for_binning is None:
        table_lengths = coordinates.get_table_lengths()
    else:
        for key in tab_dict_for_binning.keys():
            table_lengths[key]=int(get_dt(tab_dict_for_binning,key,only_metainfo=True)['num_rows'])

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