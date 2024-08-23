"""General plotting functions for the deepof package."""
# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof


from typing import Any, List, NewType, Tuple, Union
import numpy as np
import re
import warnings

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
    if re.match(r"^\b\d{1,4}:\d{1,4}:\d{1,4}(?:\.\d{1,9})?$", time_string) is not None:
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
    all_vertices: List[List[Tuple[float, float]]], num_points: int = 10000
) -> np.array:
    """
    Calculates the average arena based on a list of polynomial vertices
    lists representing arenas. Polynomial vertices can have different lengths and start at different positions

    Args:
        vertices (list): A list of 2D tuples representing the vertices of the arenas.
        num_points (int): number of points in the averaged arena.

    Returns:
        numpy.ndarray: A 2D NumPy array containing the averaged arena.
    """

    # ensure that enough points are available for interpolation
    max_length = max(len(lst) for lst in all_vertices) + 1
    assert (
        num_points > max_length
    ), "The num_points variable needs to be larger than the longest list of vertices!"

    # initialize averaged arena polynomial
    avg_points = np.empty([num_points, 2])
    avg_points.fill(0.0)

    # iterate over all arenas
    for i in range(len(all_vertices)):
        # calculate relative segment lengths between vertices
        vertices = np.stack(all_vertices[i]).astype(float)
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
    experiment_id: str = None,
):
    """Return a heatmap of the movement of a specific bodypart in the arena.

    If more than one bodypart is passed, it returns one subplot for each.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        experiment_id (str): id of the experiment of time bins should

    Returns:
        bin_size_int (int): preprocessed bin size for time filtering
        bin_index_int (int): preprocessed bin index for time filtering
        bin_starts (dict): dictionary of start position for each bin in each condition
        bin_ends (dict): dictionary of end position for each bin in each condition
        precomputed_bins (np.ndarray): precomputed time bins as alternative to bin_index_int and bin_size_int
        error (boolean): True if unusable bins were selected
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
    bin_index_int = None
    bin_starts = None
    bin_ends = None
    #dictionary to contain warnings for start time truncations (yes, I'll refactor this when I have some spare time)
    warn_start_time = {}

    # skip preprocessing if exact bins are already provided by the user
    if precomputed_bins is None:
        # get start and end times for each table
        start_times = coordinates.get_start_times()
        table_lengths = coordinates.get_table_lengths()
        # if a specific experiment is given, calculate time bin info only for this experiment
        if experiment_id is not None:
            start_times = {experiment_id: start_times[experiment_id]}
            table_lengths = {experiment_id: table_lengths[experiment_id]}

        pattern = r"^\b\d{1,4}:\d{1,4}:\d{1,4}(?:\.\d{1,12})?$"
        # Case 1: Integer bins are only adjusted using the frame rate
        if type(bin_size) is int and type(bin_index) is int:

            bin_size_int = int(np.round(bin_size * coordinates._frame_rate))
            bin_index_int = bin_index
            bin_starts = dict.fromkeys(table_lengths, bin_size_int * bin_index)
            bin_ends = dict.fromkeys(table_lengths, bin_size_int * (bin_index + 1))

        # Case 2: Bins given as valid time ranges are used to fill precomputed_bins
        # to reflect the time ranges given
        elif (
            type(bin_size) is str
            and type(bin_index) is str
            and re.match(pattern, bin_size) is not None
            and re.match(pattern, bin_index) is not None
        ):

            # set starts and ends for all coord items
            bin_starts = {key: 0 for key in table_lengths}
            bin_ends = {key: 0 for key in table_lengths}

            # precumputed bins is based on the longest table
            key_to_longest = max(table_lengths.items(), key=lambda x: x[1])[0]
            precomputed_bins = np.full(table_lengths[key_to_longest], False, dtype=bool)

            # calculate bin size as int
            bin_size_int = int(
                np.round(time_to_seconds(bin_size) * coordinates._frame_rate)
            )

            # find start and end positions with sampling rate
            for key in table_lengths:
                start_time = time_to_seconds(start_times[key])
                bin_index_time = time_to_seconds(bin_index)
                start_time_adjusted=int(
                    np.round((bin_index_time - start_time) * coordinates._frame_rate)
                )
                bin_starts[key] = np.max([0,start_time_adjusted])
                bin_ends[key] = np.max([0,bin_size_int + start_time_adjusted])
                if start_time_adjusted < 0:
                    warn_start_time[key]=seconds_to_time(bin_ends[key]-bin_starts[key])

            precomputed_bins[
                bin_starts[key_to_longest] : (bin_ends[key_to_longest])
            ] = True
        # Case 3: If nonsensical input was given, return error and default bins
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

            bin_size_int = int(np.round(60 * coordinates._frame_rate))
            bin_index_int = 0
            bin_starts = dict.fromkeys(table_lengths, bin_size_int * bin_index_int)
            bin_ends = dict.fromkeys(table_lengths, bin_size_int * (bin_index_int + 1))

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
                elif bin_starts[key] > table_lengths[key]:
                    raise ValueError(
                        "Please make sure bin_index is within the time range. i.e < {} or < {} for a bin_size of {}".format(
                            seconds_to_time(
                                table_lengths[key] / coordinates._frame_rate, False
                            ),
                            int(np.ceil(table_lengths[key] / bin_size_int)),
                            bin_size,
                        )
                    )
                elif bin_ends[key] > table_lengths[key]:
                    bin_ends[key] = table_lengths[key]
                    if not bin_warning:
                        truncated_length = seconds_to_time(
                            (bin_ends[key] - bin_starts[key]) / coordinates._frame_rate,
                            False,
                        )
                        warning_message = (
                            "\033[38;5;208m\n"
                            "Warning! The chosen time range exceeds the signal length for at least one data set!\n"
                            f"Therefore, the chosen bin was truncated to a length of {truncated_length}"
                            "\033[0m"
                        )
                        if table_lengths[key] - bin_size_int > 0:
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

    return bin_size_int, bin_index_int, precomputed_bins, bin_starts, bin_ends