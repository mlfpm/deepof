"""General plotting functions for the deepof package."""
# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof


import re
from typing import Any, List, NewType, Tuple, Union

import numpy as np

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
