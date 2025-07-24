"""General plotting functions for the deepof package."""
# @author lucasmiranda42
# encoding: utf-8
# module deepof

import calendar
import copy
import os
import re
import time
import warnings
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain, combinations, product
from typing import Any, List, NewType, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from natsort import os_sorted
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
from statannotations.Annotator import Annotator

import deepof.post_hoc
import deepof.utils
from deepof.data_loading import get_dt, _suppress_warning
from deepof.config import ROI_COLORS
from deepof.export_video import (
    VideoExportConfig,   
    output_annotated_video,
    output_videos_per_cluster, 
)
from deepof.visuals_utils import (
    _check_enum_inputs,
    plot_arena,
    heatmap,
    seconds_to_time,
    time_to_seconds,
    _preprocess_time_bins,
    _filter_embeddings,
    _process_animation_data,
    _get_polygon_coords,
    _scatter_embeddings,
    get_behavior_colors,
    get_supervised_behaviors_in_roi,
    get_unsupervised_behaviors_in_roi,
    get_behavior_frames_in_roi,
    _apply_rois_to_bin_info,
    BGR_to_hex,
    _preprocess_transitions,
    calculate_FSTTC,
    calculate_simple_association,
)

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)

# activate warnings
warnings.simplefilter("always", UserWarning)

# PLOTTING FUNCTIONS #


# noinspection PyTypeChecker
def plot_heatmaps(
    coordinates: coordinates,
    bodyparts: list,
    center: str = "arena",
    align: str = None,
    exp_condition: str = None,
    condition_value: str = None,
    experiment_id: int = "average",
    # Time selection paramaters
    bin_size: Union[int, str] = None,
    bin_index: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max: int = 20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    display_rois: bool = True,
    # Others
    display_arena: bool = True,
    xlim: float = None,
    ylim: float = None,
    save: bool = False,
    dpi: int = 100,
    ax: Any = None,
    show: bool = True,
    **kwargs,
) -> plt.figure:  # pragma: no cover
    """Plot heatmaps of the specified body parts (bodyparts) of the specified animal (i).

    Args:
        coordinates (coordinates): deepof Coordinates object.
        bodyparts (list): list of body parts to plot.
        center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).
        exp_condition (str): Experimental condition to plot base filters on.
        condition_value (str): Experimental condition value to plot. If available, it filters the experiments to keep only those whose condition value matches the given string in the provided exp_condition.
        experiment_id (str): Name of the experiment to display. When given as "average" positiosn of all animals are averaged.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored. Note: providing precomputed bins with gaps will result in an incorrect time vector depiction.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
        display_rois (bool): Display the active ROI, if a ROI was selected. Defaults to True.              
        display_arena (bool): whether to plot a dashed line with an overlying arena perimeter. Defaults to True.
        xlim (float): x-axis limits.
        ylim (float): y-axis limits.
        save (str):  if provided, the figure is saved to the specified path.
        dpi (int): resolution of the figure.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, a new figure will be created.
        show (bool): whether to show the created figure. If False, returns al axes.
        kwargs: additional arguments to pass to the seaborn kdeplot function.

    Returns:
        heatmaps (plt.figure): figure with the specified characteristics

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        origin="plot_heatmaps",
        bodyparts=bodyparts,
        center=center,
        animals_in_roi=animals_in_roi,
        experiment_ids=experiment_id,
        exp_condition=exp_condition,
        condition_values=[condition_value],
        roi_number=roi_number,
    )

    coords = coordinates.get_coords(center=center, align=align, return_path=False, roi_number=roi_number, animals_in_roi=animals_in_roi)

    #only keep requested experiment conditions
    if exp_condition is not None and condition_value is not None:
        coords = coords.filter_videos(
            [
                k
                for k, v in coordinates.get_exp_conditions.items()
                if v[exp_condition].values.astype(str) == condition_value
            ]
        )

    # preprocess information given for time binning
    e_id=None
    if not experiment_id=="average":
        e_id=experiment_id
    bin_info_time = _preprocess_time_bins(
        coordinates, bin_size, bin_index, experiment_id=e_id, precomputed_bins=precomputed_bins, samples_max=samples_max
    )

    if not center:  # pragma: no cover
        warnings.warn(
            "\033[38;5;208mWarning! Heatmaps look better if you center the data\033[0m"
        )

    # Add experimental conditions to title, if provided
    title_suffix = experiment_id
    if coordinates.get_exp_conditions is not None and exp_condition is None:
        title_suffix += f" - all"

    elif exp_condition is not None:
        title_suffix += f" - {condition_value}"

    if experiment_id != "average":
        coords = coords.filter_videos([experiment_id])

    # for all tables in coords:
    # read tables one by one
    # and cut them in shape according to time bins
    sampled_tabs = []
    for key in coords.keys():

        #load table if not already loaded
        tab = get_dt(coords, key)  
        
        #cut slice from table
        tab=tab.iloc[bin_info_time[key]]

        #append table
        sampled_tabs.append(tab)

    #concatenate table samples into one table for processing
    coords = pd.concat([val for val in sampled_tabs], axis=0).reset_index(
        drop=True
    )

    heatmaps = heatmap(
        coords,
        bodyparts,
        xlim=xlim,
        ylim=ylim,
        title=title_suffix,
        save=save,
        dpi=dpi,
        ax=ax,
        **kwargs,
    )
    
    if display_arena:
        for hmap in heatmaps:
            plot_arena(coordinates, center, "#ec5628", hmap, experiment_id)

    if coordinates._roi_dicts is not None and roi_number is not None and display_rois:
        for hmap in heatmaps:
            color = BGR_to_hex(ROI_COLORS[roi_number-1])
            plot_arena(coordinates, center, color, hmap, experiment_id, roi_number)

    if not ax:
        plt.gca().invert_yaxis()
    else:
        ax.invert_yaxis()

    if show:
        plt.show()
    else:
        return heatmaps


def plot_gantt(
    coordinates: project,
    instance_id: str,
    supervised_annotations: table_dict = None,
    soft_counts: table_dict = None,
    # Time selection parameters
    bin_index: Union[int, str] = None,
    bin_size: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max=20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    # Visualization parameters
    additional_checkpoints: pd.DataFrame = None,
    signal_overlay: pd.Series = None,
    instances_to_plot: list = None,
    ax: Any = None,
    save: bool = False,
):
    """Return a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation, and changepoint detection size visualization.

    Args:
        coordinates (project): deepOF project where the data is stored.
        instance_id (str): Name of the instance to display (can either be an experiment or a behavior).
        supervised_annotations (table_dict): table dict with supervised annotations per video. new figure will be created.      
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        bin_size (Union[int,str]): bin size for time filtering.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored. Note: providing precomputed bins with gaps will result in an incorrect time vector depiction.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI)
        additional_checkpoints (pd.DataFrame): table with additional checkpoints to plot.
        signal_overlay (pd.Series): overlays a continuous signal with all selected behaviors. None by default.
        instances_to_plot (list): list of either behaviors or experiments to plot. If instance_id is an experiment this needs to be a list of behaviors and vice versa. If None, all options are plotted.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """

    #Check if inputs correspond to behavior or experiment subtypes, then call respective gantt function
    if instance_id in list(coordinates._tables.keys()):

        _plot_experiment_gantt(
        coordinates=coordinates,
        experiment_id=instance_id,
        # Time selection parameters
        bin_index=bin_index,
        bin_size=bin_size,
        precomputed_bins=precomputed_bins,
        samples_max=samples_max,
        # Visualization parameters
        soft_counts=soft_counts,
        supervised_annotations=supervised_annotations,
        roi_number=roi_number,
        additional_checkpoints=additional_checkpoints,
        signal_overlay=signal_overlay,
        behaviors_to_plot=instances_to_plot,
        animals_in_roi = animals_in_roi,
        roi_mode = roi_mode,
        ax=ax,
        save=save)

    else:

        _plot_behavior_gantt(
        coordinates=coordinates,
        behavior_id=instance_id,
        # Time selection parameters
        bin_index=bin_index,
        bin_size=bin_size,
        precomputed_bins=precomputed_bins,
        samples_max=samples_max,
        # Visualization parameters
        soft_counts=soft_counts,
        supervised_annotations=supervised_annotations,
        roi_number=roi_number,
        additional_checkpoints=additional_checkpoints,
        signal_overlay=signal_overlay,
        experiments_to_plot=instances_to_plot,
        animals_in_roi = animals_in_roi,
        roi_mode = roi_mode,
        ax=ax,
        save=save)



def _plot_experiment_gantt(
    coordinates: project,
    experiment_id: str,
    # Time selection parameters
    bin_index: Union[int, str] = None,
    bin_size: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max=20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    # Visualization parameters
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    additional_checkpoints: pd.DataFrame = None,
    signal_overlay: pd.Series = None,
    behaviors_to_plot: list = None,
    ax: Any = None,
    save: bool = False,
):
    """Return a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation, and changepoint detection size visualization.

    Args:
        coordinates (project): deepOF project where the data is stored.
        experiment_id (str): Either name of the experiment to display or of a behavior.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        bin_size (Union[int,str]): bin size for time filtering.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored. Note: providing precomputed bins with gaps will result in an incorrect time vector depiction.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI)
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        supervised_annotations (table_dict): table dict with supervised annotations per video. new figure will be created.
        additional_checkpoints (pd.DataFrame): table with additional checkpoints to plot.
        signal_overlay (pd.Series): overlays a continuous signal with all selected behaviors. None by default.
        behaviors_to_plot (list): list of behaviors to plot.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        experiment_ids=experiment_id,
        behaviors=behaviors_to_plot,
        animals_in_roi = animals_in_roi,
        roi_number=roi_number,
        roi_mode = roi_mode,
    )

    if animals_in_roi is None or roi_mode == "behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animals_in_roi is only relevant if a ROI was selected!\033[0m'
        )


    # set active axes if provided
    if ax:
        plt.sca(ax)

    # Determine plot type and length of the whole dataset
    if soft_counts is None and supervised_annotations is not None:
        plot_type = "supervised"
        #get entire supervised behavior data or only the data from a specific ROI
        data_frame=get_dt(supervised_annotations,experiment_id)
       
        # preprocess information given for time binning
        bin_info_time = _preprocess_time_bins(
        coordinates, 
        bin_size, 
        bin_index, 
        precomputed_bins=precomputed_bins, 
        experiment_id=experiment_id, 
        tab_dict_for_binning=supervised_annotations, 
        samples_max=samples_max,
        )
    elif soft_counts is not None and supervised_annotations is None:
        plot_type = "unsupervised"

        data_frame=get_dt(soft_counts,experiment_id)

        # preprocess information given for time binning
        bin_info_time = _preprocess_time_bins(
        coordinates, 
        bin_size, 
        bin_index, 
        precomputed_bins=precomputed_bins, 
        experiment_id=experiment_id, 
        tab_dict_for_binning=soft_counts, 
        samples_max=samples_max,
        )
    else:
        plot_type = "mixed"
        raise NotImplementedError(
            "This function currently only accepts either supervised or unsupervised annotations as inputs, not both at the same time!"
        )
    
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    # get indices to be plotted
    bin_indices=bin_info[experiment_id]["time"]


    # set behavior ids
    if plot_type == "unsupervised":
        hard_counts = np.array([row.argmax() if not np.isnan(row).any() else -1 for row in data_frame])
        behavior_ids = [f"Cluster {str(k)}" for k in range(0, hard_counts.max() + 1)]
    elif plot_type == "supervised":
        behavior_ids = [
            col
            for col in data_frame.columns
            if "speed" not in col
        ]

    # only keep valid ids
    if behaviors_to_plot is not None:
        behaviors_to_plot = np.unique(behaviors_to_plot)
        behaviors_to_plot = [
            behaviors_to_plot[k]
            for k in range(0, len(behaviors_to_plot))
            if behaviors_to_plot[k] in behavior_ids
        ]
        # sort behaviors_to_plot to occur in the same order as in behavior_indices
        behavior_indices = {b: i for i, b in enumerate(behavior_ids)}
        behaviors_to_plot = sorted(
            behaviors_to_plot, key=lambda b: behavior_indices.get(b, float("inf"))
        )
    else:
        behaviors_to_plot = behavior_ids

    # set gantt matrix
    n_available_features = len(behavior_ids)
    n_features = len(behaviors_to_plot)
    gantt = np.zeros([len(behaviors_to_plot), len(bin_indices)])

    # If available, add additional checkpoints to the Gantt matrix
    if additional_checkpoints is not None:
        additional_checkpoints = additional_checkpoints.iloc[:, bin_indices]
        if behaviors_to_plot is not None:
            gantt = np.concatenate([gantt, additional_checkpoints], axis=0)

    # set colors with number of available features to keep color consitent if only a subset is selected
    colors = get_behavior_colors(behaviors_to_plot, coordinates._animal_ids)

    # apply time and roi bins to data 
    if plot_type == "unsupervised":
        time_binned = hard_counts[bin_indices]
        if roi_number is not None:
            time_binned = get_unsupervised_behaviors_in_roi(time_binned, bin_info[experiment_id], animals_in_roi)
            #reset function warning
            get_unsupervised_behaviors_in_roi._warning_issued = False

    
    elif plot_type == "supervised":
        supervised_binned = data_frame.iloc[bin_indices]
        if roi_number is not None:
            supervised_binned=get_supervised_behaviors_in_roi(supervised_binned, bin_info[experiment_id], animals_in_roi, roi_mode)


    # Iterate over features and plot
    rows = 0
    for feature in range(n_available_features):

        # skip if feature is not selected for plotting
        if behavior_ids[feature] not in behaviors_to_plot:
            continue

        # fill gantt row
        if plot_type == "unsupervised":
            gantt[rows] = time_binned == feature
        elif plot_type == "supervised":
            gantt[rows] = supervised_binned[behavior_ids[feature]]

        gantt[rows][gantt[rows]>0]+=rows

        rows+=1

    gantt_plotter(
        coordinates=coordinates,
        gantt_matrix=gantt,
        plot_type=plot_type,
        instance_id=experiment_id,
        n_available_instances=n_available_features,
        instances_to_plot=behaviors_to_plot,
        colors=colors,
        bin_indices=bin_indices,
        additional_checkpoints=additional_checkpoints,
        signal_overlay=signal_overlay,
        ax=ax,
        save=save,
    )


def _plot_behavior_gantt(
    coordinates: project,
    behavior_id: str,
    # Time selection parameters
    bin_index: Union[int, str] = None,
    bin_size: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max=20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    # Visualization parameters
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    additional_checkpoints: pd.DataFrame = None,
    signal_overlay: pd.Series = None,
    experiments_to_plot: list = None,
    ax: Any = None,
    save: bool = False,
):
    """Return a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation, and changepoint detection size visualization.

    Args:
        coordinates (project): deepOF project where the data is stored.
        behavior_id (str): Name of the behavior to display.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        bin_size (Union[int,str]): bin size for time filtering.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored. Note: providing precomputed bins with gaps will result in an incorrect time vector depiction.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)        
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        supervised_annotations (table_dict): table dict with supervised annotations per video. new figure will be created.
        additional_checkpoints (pd.DataFrame): table with additional checkpoints to plot.
        signal_overlay (pd.Series): overlays a continuous signal with all selected behaviors. None by default.
        experiments_to_plot (list): list of experiments to plot. If None, all experiments are plotted.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """

    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        behaviors=behavior_id,
        experiment_ids=experiments_to_plot,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
        roi_mode = roi_mode,
    )

    if animals_in_roi is None or roi_mode == "behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animals_in_roi is only relevant if a ROI was selected!\033[0m'
        )

    # set active axes if provided
    if ax:
        plt.sca(ax)

    # Determine plot type and length of the whole dataset
    if soft_counts is None and supervised_annotations is not None:
        plot_type = "supervised"
        all_experiments=list(supervised_annotations.keys())
        
        # preprocess information given for time binning
        bin_info_time = _preprocess_time_bins(
        coordinates, 
        bin_size, 
        bin_index, 
        precomputed_bins=precomputed_bins, 
        tab_dict_for_binning=supervised_annotations, 
        samples_max=samples_max,
        )
    elif soft_counts is not None and supervised_annotations is None:
        plot_type = "unsupervised"
        all_experiments=list(soft_counts.keys())

        # preprocess information given for time binning
        bin_info_time = _preprocess_time_bins(
        coordinates, 
        bin_size, 
        bin_index, 
        precomputed_bins=precomputed_bins, 
        tab_dict_for_binning=soft_counts, 
        samples_max=samples_max,
        )
    else:
        plot_type = "mixed"
        raise NotImplementedError(
            "This function currently only accepts either supervised or unsupervised annotations as inputs, not both at the same time!"
        )

    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    # only keep valid experiments
    if experiments_to_plot is not None:
        experiments_to_plot = np.unique(experiments_to_plot)
        experiments_to_plot = [
            experiments_to_plot[k]
            for k in range(0, len(experiments_to_plot))
            if experiments_to_plot[k] in all_experiments
        ]
    else:
        experiments_to_plot = all_experiments

    # get common indices between all selected experiments
    bin_indices=bin_info[list(bin_info.keys())[0]]["time"]
    max_index=np.max(bin_indices)
    for exp_id in experiments_to_plot:
        if max_index > np.max(bin_info[exp_id]["time"]):
            max_index=np.max(bin_info[exp_id]["time"])
            bin_indices=bin_info[exp_id]["time"][bin_info[exp_id]["time"]<max_index]
    
    for exp_id in bin_info.keys():
        for id in bin_info[exp_id].keys():
            bin_info[exp_id][id]=bin_info[exp_id][id][0:len(bin_indices)]

    # set gantt matrix
    n_available_experiments = len(all_experiments)
    gantt = np.zeros([len(experiments_to_plot), len(bin_indices)])

    # If available, add additional checkpoints to the Gantt matrix
    if additional_checkpoints is not None:
        additional_checkpoints = additional_checkpoints.iloc[:, bin_indices]
        if experiments_to_plot is not None:
            gantt = np.concatenate([gantt, additional_checkpoints], axis=0)

    # set colors with number of available features to keep color consitent if only a subset is selected
    colors = list(
        np.tile(
            list(sns.color_palette("tab20").as_hex()),
            int(np.ceil(n_available_experiments / 20)),
        )
    )

    # Iterate over experiments and plot
    rows = 0
    for exp_id in range(n_available_experiments):

        # skip if feature is not selected for plotting
        if all_experiments[exp_id] not in experiments_to_plot:
            continue

        # fill gantt row
        if plot_type == "unsupervised":

            hard_counts = get_dt(soft_counts,all_experiments[exp_id]).argmax(axis=1)
            cluster_no = int(re.search(r'\d+', behavior_id).group()) if re.search(r'\d+', behavior_id) else None
            time_binned = hard_counts[bin_indices]
            if roi_number is not None:
                time_binned = get_unsupervised_behaviors_in_roi(time_binned, bin_info[all_experiments[exp_id]], animals_in_roi)
            gantt[rows] = time_binned == cluster_no

        elif plot_type == "supervised":

            supervised_binned = pd.DataFrame(get_dt(supervised_annotations,all_experiments[exp_id])[behavior_id].iloc[bin_indices])
            if roi_number is not None:
                supervised_binned=get_supervised_behaviors_in_roi(supervised_binned, bin_info[all_experiments[exp_id]], animals_in_roi, roi_mode)
            gantt[rows] = supervised_binned[behavior_id]

        gantt[rows][gantt[rows]>0]+=rows
        
        rows += 1
    
    #reset function warning
    get_unsupervised_behaviors_in_roi._warning_issued = False

    gantt_plotter(
        coordinates=coordinates,
        gantt_matrix=gantt,
        plot_type=plot_type,
        instance_id=behavior_id,
        n_available_instances=n_available_experiments,
        instances_to_plot=experiments_to_plot,
        colors=colors,
        bin_indices=bin_indices,
        additional_checkpoints=additional_checkpoints,
        signal_overlay=signal_overlay,
        ax=ax,
        save=save,
    )

    
def gantt_plotter(
    coordinates: project,
    gantt_matrix: np.ndarray,
    plot_type: str,
    instance_id: str,
    n_available_instances: int, 
    instances_to_plot: list,
    colors: list,
    # Time selection parameters
    bin_indices: np.ndarray,
    additional_checkpoints: pd.DataFrame = None,
    signal_overlay: pd.Series = None,
    ax: Any = None,
    save: bool = False,
):
    """Return a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation, and changepoint detection size visualization.

    Args:
        coordinates (project): deepOF project where the data is stored.
        gantt_matrix (np.ndarray): 2D integer matrix denoting time sections with present or absent behavior
        plot_type (str): type of plot, either "supervised" or "unsupervised"
        instance_id (str): Name of the experiment or behavior to display.
        n_available_instances (int): number of all possibly available instances (may be behaviors or experiments)
        instances_to_plot (list): selected instances for plotting as a list (may be behaviors or experiments)
        colors (list): list of color hexcodes for plotting
        bin_indices (np.ndarray): indices to plot
        additional_checkpoints (pd.DataFrame): table with additional checkpoints to plot.
        signal_overlay (pd.Series): overlays a continuous signal with all selected behaviors. None by default.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
  
    #only add "white" as base color if there are frames with no behaviors
    if (gantt_matrix==0).any():
        colors=colors=['#FFFFFF'] + colors
    
    colors = [color for color in colors if color is not None]

    N_colors=int(np.nanmax(gantt_matrix))
    #col_indices=col_indices[np.invert(np.isnan(col_indices))].astype(int)
    #N_colors=len(col_indices)
    sns.heatmap(
        data=gantt_matrix,
        cbar=False,
        cmap=ListedColormap(colors[0:N_colors+1], name="deepof", N=N_colors+1),
        ax=ax,
    )

    n_instances=len(instances_to_plot)

    rows = 0
    for exp_id, color in zip(range(n_available_instances), colors):

        # overlay lineplot with normalized signal
        if signal_overlay is not None:
            standard_signal = (signal_overlay - signal_overlay.min()) / (
                signal_overlay.max() - signal_overlay.min()
            )
            sns.lineplot(
                x=signal_overlay.index[0 : len(bin_indices)],
                y=standard_signal[bin_indices] + rows,
                color="black",
            )



        rows += 1
    
    for k in range(gantt_matrix.shape[0]):
        # plot line for axis to separate between features
        plt.axhline(y=k, color="k", linewidth=0.5)


    # Iterate over additional checkpoints and plot
    if additional_checkpoints is not None:
        for checkpoint in range(additional_checkpoints.shape[0]):
            gantt_cp = gantt_matrix.copy()
            gantt_cp[
                [i for i in range(gantt_matrix.shape[0]) if i != n_instances + checkpoint]
            ] = np.nan
            plt.axhline(y=n_instances + checkpoint, color="k", linewidth=0.5)

            sns.heatmap(
                data=gantt_cp,
                cbar=False,
                cmap=LinearSegmentedColormap.from_list(
                    "deepof", ["white", "black"], N=2
                ),
                ax=ax,
            )

    # set x-ticks
    plt.xticks([])
    if coordinates._frame_rate is not None:
        N_x_ticks = int(plt.gcf().get_size_inches()[1] * 1.25)
        if ax:
            bbox = ax.get_window_extent().transformed(
                plt.gcf().dpi_scale_trans.inverted()
            )
            N_x_ticks = int(bbox.width * 1.25)
        plt.xticks(
            np.linspace(0, len(bin_indices), N_x_ticks),
            [
                seconds_to_time(t)
                for t in np.round(
                    np.linspace(
                        np.min(bin_indices) / coordinates._frame_rate,
                        np.max(bin_indices) / coordinates._frame_rate,
                        N_x_ticks,
                    )
                )
            ],
            rotation=0,
        )
        if np.max(np.diff(bin_indices))>1:
            warning_message = (
                "\033[38;5;208m\n"  # Set text color to orange
                "Warning! Since the provided time bins contain gaps, the time range below may be incorrectly displayed!"
                "\033[0m"  # Reset text color
            )
            warnings.warn(warning_message)

    # set y-ticks
    # set y-ticks
    plt.yticks(
        np.array(range(gantt_matrix.shape[0])) + 0.5,
        # Concatenate cluster IDs and checkpoint names if they exist
        np.concatenate(
            [
                instances_to_plot,
                np.array(additional_checkpoints.index)
                if additional_checkpoints is not None
                else [],
            ]
        ),
        rotation=0,
        fontsize=10,
    )

    # plot stuff
    plt.axhline(y=0, color="k", linewidth=1)
    plt.axhline(y=gantt_matrix.shape[0], color="k", linewidth=2)
    plt.axvline(x=0, color="k", linewidth=1)
    plt.axvline(x=gantt_matrix.shape[1], color="k", linewidth=2)
    plt.xlabel("Time", fontsize=10)
    if coordinates._frame_rate is not None:
        plt.xlabel("Time in HH:MM:SS", fontsize=10)
    plt.ylabel(("Cluster" if plot_type == "unsupervised" else ""), fontsize=10)

    # save figure
    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_gantt{}_type={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    plot_type,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    title = "deepOF - Gantt chart of {} behaviors - {}".format(plot_type, instance_id)
    if ax is not None:
        ax.set_title(title, fontsize=8)
    else:
        plt.title(title, fontsize=8)
        plt.tight_layout()
        plt.show()


def plot_enrichment(
    coordinates: coordinates,
    embeddings: table_dict = None,
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    # Time selection parameters
    bin_index: Union[int, str] = None,
    bin_size: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max: int =100000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise", 
    # Visualization parameters
    polar_depiction: bool = False,
    plot_speed: bool = False,
    add_stats: str = "Mann-Whitney",
    exp_condition: str = None,
    exp_condition_order: list = None,
    normalize: bool = False,
    verbose: bool = False,
    ax: Any = None,
    save: bool = False,
):
    """Violin plots per cluster per condition.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        supervised_annotations (table_dict): table dict with supervised annotations per animal experiment across time.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        bin_size (Union[int,str]): bin size for time filtering.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.     
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded        
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
        polar_depiction (bool): if True, display as polar plot.
        plot_speed (bool): if supervised annotations are provided, display only speed. Useful to visualize speed.
        add_stats (str): test to use. Mann-Whitney (non-parametric) by default. See statsannotations documentation for details.        
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.
        exp_condition_order (list): Order in which to plot experimental conditions. If None (default), the order is determined by the order of the keys in the table dict.
        normalize (bool): whether to represent time fractions or actual time in seconds on the y axis.
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        exp_condition=exp_condition,
        exp_condition_order=exp_condition_order,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
        roi_mode = roi_mode,
    )
    if animals_in_roi is None or roi_mode == "behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animal_id is only relevant if a ROI was selected!\033[0m'
        )
    if normalize and plot_speed:
        print(
            '\033[33mInfo! When plotting speed the normalization option "normalize" is ignored!\033[0m'
        )
    # Checks to throw errors or warn about conflicting inputs
    if supervised_annotations is not None and any(
        [embeddings is not None, 
         soft_counts is not None]
    ):
        raise ValueError(
            "This function only accepts either supervised or unsupervised annotations as inputs, not both at the same time!"
        )

    # Get requested experimental condition. If none is provided, default to the first one available.
    if exp_condition is None:
        exp_conditions = {
            key: str(val.iloc[:, 0].values[0])
            for key, val in coordinates.get_exp_conditions.items()
        }
    else:
        exp_conditions = {
            key: str(val.loc[:, exp_condition].values[0])
            for key, val in coordinates.get_exp_conditions.items()
        }

    # Set default exp_condition_order if none isprovided
    if exp_condition_order is None:
        exp_condition_order = np.unique(list(exp_conditions.values())).astype(str)


    # Preprocess information given for time binning
    if supervised_annotations is not None:
        bin_info_time = _preprocess_time_bins(
            coordinates, bin_size, bin_index, precomputed_bins, 
            tab_dict_for_binning=supervised_annotations, samples_max=samples_max,
        )
    else:
        tab_dict_for_binning = soft_counts
        if soft_counts is None:
            tab_dict_for_binning = embeddings
        
        bin_info_time = _preprocess_time_bins(
            coordinates, bin_size, bin_index, precomputed_bins, 
            tab_dict_for_binning=tab_dict_for_binning, samples_max=samples_max,
        )
    
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    # Get cluster enrichment across conditions for the desired settings
    enrichment = deepof.post_hoc.enrichment_across_conditions(
        soft_counts = soft_counts,
        supervised_annotations = supervised_annotations,
        exp_conditions = exp_conditions,
        plot_speed = plot_speed,
        bin_info = bin_info,
        roi_number = roi_number,
        animals_in_roi = animals_in_roi,
        normalize = normalize,
        roi_mode=roi_mode,
    )
    #extract unique behavior names
    indices=np.unique(enrichment["cluster"], return_index=True)[1]
    behavior_names = [enrichment["cluster"][idx] for idx in sorted(indices)]
    # Sort experiment conditions
    enrichment["exp condition"] = pd.Categorical(
        enrichment["exp condition"], exp_condition_order
    )
    if supervised_annotations is not None and not plot_speed:
        # this assumes that all entries in supervised_annotations always have the same keys
        enrichment["cluster"] = pd.Categorical(
            enrichment["cluster"], categories=behavior_names
        )
    enrichment.sort_values(by=["exp condition", "cluster"], inplace=True)
    enrichment["cluster"] = enrichment["cluster"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Adjust label and y-axis scaling to meaningful units
    if plot_speed and supervised_annotations is not None:
        y_axis_label = "average speed [mm/s]"
    elif normalize:
        y_axis_label = "time on cluster in %"
        enrichment["time on cluster"] = enrichment["time on cluster"] * 100
    elif coordinates._frame_rate is not None:
        y_axis_label = "time on cluster [s]"
        enrichment["time on cluster"] = (
            enrichment["time on cluster"] / coordinates._frame_rate
        )
    else:
        y_axis_label = "time on cluster in frames"

    
    # More inits
    all_exp_conditions = np.unique(enrichment["exp condition"])
    num_exp_conds = len(all_exp_conditions)
    # Additional plot modifications for polar depiction
    if polar_depiction:

        # Yes, all of this is necessary to switch out the input axes object with a polar axis without actually deleting
        # the axis as it is later used outside of the function in the tutorial. Low hanging fruit my ass.
        fig = ax.figure
        position = ax.get_position()
        # Remove the existing axis
        fig.delaxes(ax)
        # Create a new polar axis
        new_ax = fig.add_axes(position, projection="polar")
        # Update the original ax reference to point to the new axis
        ax.__dict__.clear()
        ax.__dict__.update(new_ax.__dict__)
        ax.__class__ = new_ax.__class__
        # Replace the new_ax with ax in the figure's axes list
        fig.axes[fig.axes.index(new_ax)] = ax
        del new_ax

        # Get x labels from cluster names
        unique_indices = np.unique(enrichment["cluster"], return_index=True)
        x_bin_labels = enrichment["cluster"].values[np.sort(unique_indices[1])]
        num_bins = len(x_bin_labels)
        # Get means and std for error
        rich_bin_means = (
            enrichment.groupby(["cluster", "exp condition"])
            .mean(numeric_only=True)
            .reset_index()
        )
        rich_bin_err = (
            enrichment.groupby(["cluster", "exp condition"])
            .std(numeric_only=True)
            .reset_index()
        )

        # Define the angles and mid_angles (angle in the middle of two angles) for each bin
        angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        # Ensure that no angle exceeds 2 pi
        angles = np.mod(angles, 2 * np.pi)
        mid_angles = np.mod(
            angles + np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi]))) / 2,
            2 * np.pi,
        )
        # add first value to end of array for full closed circle plot
        angles = np.concatenate([angles, [angles[0]]])
        mid_angles = np.concatenate([mid_angles, [mid_angles[0]]])

        # Collect means and errors for all experiment conditions
        mean_values, err_values, plot_means, plot_errs = {}, {}, {}, {}
        for k in range(num_exp_conds):
            # Get dictionaries of means and errors for all experiment conditions containing array of all cluster values
            m_slice = rich_bin_means[
                rich_bin_means["exp condition"] == all_exp_conditions[k]
            ][["cluster", "time on cluster"]]
            mean_values[all_exp_conditions[k]] = m_slice.set_index("cluster")[
                "time on cluster"
            ].to_dict()
            s_slice = rich_bin_err[
                rich_bin_err["exp condition"] == all_exp_conditions[k]
            ][["cluster", "time on cluster"]]
            err_values[all_exp_conditions[k]] = s_slice.set_index("cluster")[
                "time on cluster"
            ].to_dict()

            # Get extended version of these dictionaries for circular plot
            plot_means[all_exp_conditions[k]] = np.array(
                [mean_values[all_exp_conditions[k]][key] for key in x_bin_labels]
                + [
                    mean_values[all_exp_conditions[k]][x_bin_labels[0]]
                ]  # Add first value to the end of the list
            )
            plot_errs[all_exp_conditions[k]] = np.array(
                [err_values[all_exp_conditions[k]][key] for key in x_bin_labels]
                + [err_values[all_exp_conditions[k]][x_bin_labels[0]]]
            )

        # Plot means as lines and extract color of these lines
        colors = {}
        for k in plot_means:
            plot_handle = ax.plot(
                mid_angles, plot_means[k], linewidth=3, label=f"{k}", alpha=0.8
            )
            colors[k] = plot_handle[0].get_color()

        # Plot markers for each group
        marker_handles = []
        for k in plot_means:
            marker_handles.append(
                ax.plot(
                    mid_angles,
                    plot_means[k],
                    marker="o",
                    linestyle="",
                    color=colors[k],
                    linewidth=2,
                )
            )

        # Plot the error as lines above and below the mean values
        for k in plot_means:
            ax.plot(
                mid_angles,
                plot_means[k] + plot_errs[k],
                linestyle="",
                color=colors[k],
                alpha=0.8,
            )
            ax.plot(
                mid_angles,
                np.maximum(plot_means[k] - plot_errs[k], np.min(plot_means[k]) * 0.1),
                linestyle="",
                color=colors[k],
                alpha=0.8,
            )

        # Shade Error
        for k in plot_means:
            ax.fill_between(
                mid_angles,
                plot_means[k] + plot_errs[k],
                np.maximum(plot_means[k] - plot_errs[k], np.min(plot_means[k]) * 0.1),
                color=colors[k],
                alpha=0.15,
            )

    else:
        # Plot a barchart grouped per experimental conditions
        np.random.seed(42) #to ensure the outlier points are always jittered te same (relevant for automatic testing)
        sns.barplot(
            data=enrichment,
            x="cluster",
            y="time on cluster",
            hue="exp condition",
            hue_order=all_exp_conditions,
            ax=ax,
        )
        sns.stripplot(
            data=enrichment,
            x="cluster",
            y="time on cluster",
            hue="exp condition",
            hue_order=all_exp_conditions,
            color="black",
            ax=ax,
            dodge=True,
        )
        np.random.seed(None)

        ax.set_ylabel(y_axis_label)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[2:], labels[2:], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )

    if add_stats:
        # creating pairs containing information about which data gets compared
        pairs = list(
            product(
                set(
                    behavior_names,
                ),
                set(exp_conditions.values()),
            )
        )
        pairs = [
            [list(i) for i in list(combinations(list(map(tuple, p)), 2))]
            for p in np.array(pairs)
            .reshape([-1, len(set(exp_conditions.values())), 2])
            .tolist()
        ]
        pairs = [item for sublist in pairs for item in sublist]

        # Remove elements from pairs if clusters are not present in the enrichment data frame
        pairs = [
            p
            for p in pairs
            if p[0][0] in enrichment["cluster"].values
            and p[1][0] in enrichment["cluster"].values
        ]

        # do actual testing with annotator package
        annotator = Annotator(
            ax,
            pairs=pairs,
            data=enrichment,
            x="cluster",
            y="time on cluster",
            hue="exp condition",
            hide_non_significant=True,
        )
        annotator.configure(
            test=add_stats,
            text_format="star",
            loc="inside",
            comparisons_correction="fdr_bh",
            verbose=verbose,
        )
        # Automatic annotiation to plots does not work with polar plots
        test_dict = {}
        if polar_depiction:
            anni = annotator.apply_test()
            # Create dictionary containing test results
            for annotation in anni.annotations:
                test_dict[annotation.structs[0]["group"][0]] = annotation.text
        # Just annotate values for non-polar plot
        else:
            annotator.apply_and_annotate()

    # Adjustments for the polar plot
    if polar_depiction:
        # Set the the 0 angle to be at the top
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)  # Change to clockwise

        # Set custom ticks and hide labels for x axes
        ax.set_xticks(angles[0:-1])
        ax.set_xticklabels([])
        ax.set_rscale("log")  # Rescale to log for better visualization

        # Get overall max value in plot
        max_value = np.max([np.max(arr) for arr in plot_means.values()])
        # Customize y-axis ticks
        max_tick = np.ceil(np.max([np.log10(max_value),0])) + 0.5
        y_ticks = np.logspace(0, max_tick, num=int(max_tick * 2) + 1)
        ax.set_yticks(y_ticks)
        ax.set_rlabel_position(0)  # Set y-axis to top

        # Add labels manually at good positions
        z = 0
        for midangle, label in zip(mid_angles[0:-1], x_bin_labels):

            # Use different offset for every second x-axis label to avoid overlaps
            if np.mod(z, 2) == 0:
                offset = 1.5
            else:
                offset = 3.162
            # Add x-axis labels (cluster names)
            ax.text(
                midangle,
                ax.get_yticks()[-1] * offset,
                label,
                ha="center",
                va="center",
                fontsize="x-small",
                rotation=-np.flip(midangle * 180 / np.pi),
            )
            # Add stats annotations
            if add_stats:
                ax.text(
                    midangle,
                    np.sqrt(ax.get_yticks()[-1] * ax.get_yticks()[-2]),
                    test_dict[label],
                    ha="center",
                    va="center",
                    fontsize="x-small",
                    rotation=-np.flip(midangle * 180 / np.pi),
                )
            z += 1
        # Set R limits ( / Y limits)
        title = ""
        lower_lim = ax.get_ylim()[0]
        ax.set_rlim(lower_lim, ax.get_yticks()[-1])

    else:
        # set x-ticks
        if ax:
            bbox = ax.get_window_extent().transformed(
                plt.gcf().dpi_scale_trans.inverted()
            )
            X_size = bbox.width
            N_X_ticks = len(ax.xaxis.get_ticklabels())
            rotation_angle = int(np.clip((N_X_ticks / X_size - 1) * 30, 0, 90))
            ha = 'right' if rotation_angle != 0 else 'center'
            ax.set_xticks(
                ax.get_xticks(), 
                ax.get_xticklabels(), 
                rotation=rotation_angle, 
                ha=ha
            )
        else:
            X_size = plt.gcf().get_size_inches()[1]
            N_X_ticks = len(plt.xticks()[0])
            plt.xticks(
                rotation=int(
                    np.max([np.min([90.0, (N_X_ticks / X_size - 1) * 30]), 0.0])
                )
            )
        title = "deepOF - cluster enrichment"

    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_enrichment{}_bin_size={}_bin_index={}_test={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    bin_size,
                    bin_index,
                    add_stats,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    if ax is not None:
        ax.set_title(title, fontsize=15)
    else:
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.show()


def return_transitions(
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
    delta_T: float = 0.0,
    silence_diagonal: bool = False,
    diagonal_behavior_counting: str = "Transitions",
    normalize:bool = True,
    # Visualization parameters
    visualization="networks",

):
    """Returns data of plot_transitions with same Input options"""

    grouped_transitions, _, combined_columns, _, _ = _preprocess_transitions(
        coordinates=coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        bin_size=bin_size,
        bin_index=bin_index,
        precomputed_bins=precomputed_bins,
        samples_max=samples_max,
        roi_number=roi_number,
        animals_in_roi=animals_in_roi,      
        exp_condition=exp_condition,
        delta_T=delta_T,        
        silence_diagonal=silence_diagonal,
        diagonal_behavior_counting=diagonal_behavior_counting,
        normalize=normalize,
        visualization=visualization,
    )

    results={}
    for key in grouped_transitions.keys():

        results[key]=grouped_transitions[key].ravel()  # Flatten the matrix into a 1D array in row-major order  
    count_df = pd.DataFrame.from_dict(results, orient='index', columns=combined_columns)
    
    return count_df      


def plot_transitions(
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
    delta_T: float = 0.0,
    silence_diagonal: bool = False,
    diagonal_behavior_counting: str = "Transitions",
    normalize:bool = True,
    # Visualization parameters
    visualization="networks",
    ax: list = None,
    save: bool = False,
    **kwargs,
):
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
        ax (list): axes where to plot the current figure. If not provided, a new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.
        kwargs: additional arguments to pass to the seaborn kdeplot function.

    """
    grouped_transitions, columns, _, exp_conditions, normalize = _preprocess_transitions(
        coordinates=coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        bin_size=bin_size,
        bin_index=bin_index,
        precomputed_bins=precomputed_bins,
        samples_max=samples_max,
        roi_number=roi_number,
        animals_in_roi=animals_in_roi,      
        exp_condition=exp_condition,
        delta_T=delta_T,        
        silence_diagonal=silence_diagonal,
        diagonal_behavior_counting=diagonal_behavior_counting,
        normalize=normalize,
        visualization=visualization,
    )

    if exp_conditions is None:
        grouped_transitions = np.mean(
            np.concatenate(
                [np.expand_dims(i, axis=0) for i in grouped_transitions.values()]
            ),
            axis=0,
        )

    # Use seaborn to plot heatmaps across both conditions
    if ax is None:
        fig, ax = plt.subplots(
            1,
            (len(set(exp_conditions.values())) if exp_conditions is not None else 1),
            figsize=(16, 8),
        )

    if not isinstance(ax, np.ndarray) and not isinstance(ax, Sequence):
        ax = [ax]

    if exp_conditions is not None:
        iters = zip(set(exp_conditions.values()), ax)
    else:
        iters = zip([None], ax)

    if visualization == "networks":

        for exp_condition, ax in iters:

            if isinstance(grouped_transitions, dict):
                G = nx.DiGraph(grouped_transitions[exp_condition])
            else:
                G = nx.DiGraph(grouped_transitions)
            weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]

            pos = nx.spring_layout(G, scale=1, center=None, dim=2)

            nx.draw(
                G,
                ax=ax,
                arrows=True,
                with_labels=True,
                node_size=500,
                node_color=[plt.cm.tab20(i) for i in range(len(G.nodes))],
                font_size=18,
                font_weight="bold",
                width=weights,
                alpha=0.6,
                pos=pos,
                **kwargs,
            )
            ax.set_title(exp_condition)

    elif visualization == "heatmaps":

        if normalize:
            vmax=0.5 
        else:
            vmax=None
        for exp_condition, ax in iters:

            if isinstance(grouped_transitions, dict):
                clustered_transitions = grouped_transitions[exp_condition]
            else:
                clustered_transitions = grouped_transitions
            if soft_counts is not None:
                # Cluster rows and columns and reorder
                row_link = linkage(
                    clustered_transitions, method="average", metric="euclidean"
                )  # computing the linkage
                row_order = dendrogram(row_link, no_plot=True)["leaves"]
                clustered_transitions = pd.DataFrame(clustered_transitions).iloc[
                    row_order, row_order
                ]
                reordered_columns = np.array(columns)[row_order]
            else:
                reordered_columns = np.array(columns)

            sns.heatmap(
                clustered_transitions,
                cmap="coolwarm",
                vmin=0,
                vmax=vmax,
                ax=ax,
                **kwargs,
            )
            ax.set_title(exp_condition)
            ax.set_xticklabels(reordered_columns, rotation=90)
            ax.set_yticklabels(reordered_columns, rotation=0)

    if ax is None:

        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(
                    coordinates._project_path,
                    coordinates._project_name,
                    "Figures",
                    "deepof_transitions{}_viz={}_bin_size={}_bin_index={}_{}.pdf".format(
                        (f"_{save}" if isinstance(save, str) else ""),
                        visualization,
                        bin_size,
                        bin_index,
                        calendar.timegm(time.gmtime()),
                    ),
                )
            )

        plt.show()


def count_all_events(
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
    # Others
    counting_mode = "Events",
):
    """Counts all events in supervised or soft_counts dataset and returns a data table.

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
        counting_mode (str): How to count behaviors. Options: 
            - "Frames": Total frames where behavior is active (after extension)
            - "Time": Total time where behavior is active
            - "Events": number of instances of the behavior occuring 
            - "Transitions": number of frame-wise internal behavior transitions e.g. A behavior of 4 frames in length would have 3 transitions.     

    """
    _check_enum_inputs(
        coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
    )
    Counting_mode_options=["Frames","Time","Events","Transitions"]  
    if counting_mode not in Counting_mode_options:
        raise ValueError(
            '"diagonal_behavior_counting" needs to be one of the following: {}'.format(
                str(Counting_mode_options)[1:-1]
            )
        )
    if (supervised_annotations is None and soft_counts is None) or (supervised_annotations is not None and soft_counts is not None):
        raise ValueError(
            "Need either supervised_annotations or soft_counts, not both or neither!"
        )
    elif supervised_annotations is not None:
        tab_dict=supervised_annotations
    else:
        tab_dict=soft_counts

    # preprocess information given for time binning
    bin_info_time = _preprocess_time_bins(
        coordinates, bin_size, bin_index, precomputed_bins, tab_dict_for_binning=tab_dict, samples_max=samples_max, down_sample=False,
    )
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    # create tabdict dictionary to iterate over options
    load_range = None
            
    results={}
    for key in tab_dict.keys():

        # for each tab, first cut tab in requested shape based on bin_info
        if bin_info is not None:
            load_range = bin_info[key]["time"]
            if len(bin_info[key]) > 1:
                load_range=deepof.visuals_utils.get_behavior_frames_in_roi(None,bin_info[key],animals_in_roi)
        tab = get_dt(tab_dict,key,load_range=load_range)
        
        # in case tab is a numpy array (soft_counts), transform numpy array in analogous pandas datatable
        if isinstance(tab,np.ndarray):
            max_indices = tab.argmax(axis=1)
            tab_soft = np.zeros_like(tab, dtype=int)
            tab_soft[np.arange(tab.shape[0]), max_indices] = 1 # set maximum column to 1 for each row
            columns = [f"Cluster_{i}" for i in range(tab_soft.shape[1])] #create useful column names
            tab=pd.DataFrame(tab_soft, columns=columns)
        
        # count events in all columns
        column_counts={}
        for col in tab.columns:
            series = tab[col]
            series.fillna(0,inplace=True)
            # skip non-binary columns (e.g. speed column)
            if (series > 1.0001).any():
                continue
            column_counts[col]=deepof.utils.count_events(series, counting_mode=counting_mode, frame_rate=coordinates._frame_rate)   

        results[key] = pd.Series(column_counts)
    
    count_df = pd.DataFrame.from_dict(results, orient='index')
    
    return count_df

"""
def plot_associations(
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
    # Visualization parameters
    exp_condition: str = None,
    condition_values: list = None,
    experiment_id: str = None,
    behaviors: list = None,
    exclude_given_behaviors: bool = False,
    delta_T: float = 0.5,
    association_metric:str = "FSTTC",
    return_values = False,
    ax: list = None,
    save: bool = False,
    **kwargs,
):
    Compute and plots transition matrices for all data or per condition. Plots can be heatmaps or networks.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        supervised_annotations (table_dict): table dict with supervised annotations per video. new figure will be created.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                      
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.        
        condition_values (list): Experimental condition values to plot. If available, it filters the experiments to keep only those whose condition value matches the given string in the provided exp_condition. If two condition values are given as a list, the difference between both sets of corresponding experiments is plotted
        experiment_id (str): Name of the experiment to display. When given as "average" positiosn of all animals are averaged.
        behaviors (list): List of behaviors to include in the plot. Should be given as "[Cluster_0, Cluster_1..." in case of soft_counts.
        exclude_given_behaviors (bool): If True, will instead of only including given behaviors in the plot exclude these behaviors and plot all other behaviors. Defaults to False.
        delta_T (float): Maximum time delay after the end of any behavior instance during which following behaviors are still counted as associated. 
        association_metric (str): Association metric that should be used to determine if two behaviors are associated. Options are "odds_ratio" and "FSTTC". Defaults to FSTTC.
        get_values (bool): Determines if the plotted matrix should also be returned as an 2D array. Defaults to False.
        ax (list): axes where to plot the current figure. If not provided, a new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    
    if isinstance(condition_values,str):
        condition_values=[condition_values]  
    
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        behaviors=behaviors,
        experiment_ids=experiment_id,
        exp_condition=exp_condition,
        condition_values = condition_values,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
    )

    if animals_in_roi is None:
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animals_in_roi is only relevant if a ROI was selected!\033[0m'
        )
    
    # set active axes if provided
    if ax:
        plt.sca(ax)

    # Determine plot type and length of the whole dataset
    if soft_counts is None and supervised_annotations is not None:
        plot_type = "supervised"
        tab_dict = supervised_annotations
        # Extract the first key from the tab dictionary
        first_experiment_key = list(tab_dict.keys())[0]
        available_behaviors=get_dt(tab_dict,first_experiment_key,only_metainfo=True)["columns"]
  
        # preprocess information given for time binning
        bin_info_time = _preprocess_time_bins(
        coordinates, 
        bin_size, 
        bin_index, 
        precomputed_bins=precomputed_bins, 
        tab_dict_for_binning=supervised_annotations, 
        samples_max=samples_max,
        )
    elif soft_counts is not None and supervised_annotations is None:
        plot_type = "unsupervised"
        tab_dict = soft_counts
        # Extract the first key from the tab dictionary
        first_experiment_key = list(tab_dict.keys())[0]
        num_cols=get_dt(tab_dict,first_experiment_key,only_metainfo=True)["num_cols"]
        available_behaviors = ["Cluster_" + k for k in range(0,num_cols)]

        # preprocess information given for time binning
        bin_info_time = _preprocess_time_bins(
        coordinates, 
        bin_size, 
        bin_index, 
        precomputed_bins=precomputed_bins, 
        tab_dict_for_binning=soft_counts, 
        samples_max=samples_max,
        )
    else:
        plot_type = "mixed"
        raise NotImplementedError(
            "This function only accepts either supervised or unsupervised annotations as inputs, not both at the same time!"
        )

    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    if behaviors is None:
        behaviors=[]
    # invert behavior list to contain only excluded behaviors if non-exclusion was chosen
    elif not exclude_given_behaviors:
        behaviors = list(set(available_behaviors)-set(behaviors))
    # Always exclude
    always_exclude=["speed"]
    if coordinates._animal_ids is not None:
        always_exclude = [id + "_" + "speed" for id in coordinates._animal_ids]
    behaviors = behaviors + always_exclude

    if experiment_id is not None:
        # Filter to only the specified experiment ID
        tab_dicts = [{experiment_id: get_dt(tab_dict, experiment_id)}]
        
        if exp_condition is not None:
            # Warn about ignored exp_condition parameters
            warning_message = (
                "\033[38;5;208m\nWarning! Both exp_condition and experiment_id were provided. "
                "exp_condition related inputs will be ignored!\n\033[0m"
            )
            warnings.warn(warning_message)

    elif exp_condition is not None:

        if condition_values is None:
            condition_df = tab_dict.get_exp_conditions[first_experiment_key]
            condition_values = [condition_df.iloc[0, 0]]
            
            print(
                f"\033[33mInfo! Automatically set condition_value to {condition_values[0]} "
                f"as only exp_condition was provided!\033[0m"
            )
        elif len(condition_values) > 2:
            condition_values = [condition_values[0]]
            
            print(
                f"\033[33mInfo! Automatically set condition_value to {condition_values[0]} "
                f"as too many condition_values (max 2) were provided!\033[0m"
            )

        if condition_values is not None and (len(condition_values)==1 or len(condition_values)==2):
            # Filter experiments based on condition value
            filtered_experiments = [
                experiment_key
                for experiment_key, conditions in coordinates.get_exp_conditions.items()
                if conditions[exp_condition].values.astype(str) == condition_values[0]
            ]
            tab_dicts = [tab_dict.filter_videos(filtered_experiments)]
            if len(condition_values)==2: 
                filtered_experiments = [
                    experiment_key
                    for experiment_key, conditions in coordinates.get_exp_conditions.items()
                    if conditions[exp_condition].values.astype(str) == condition_values[1]
                ]
                tab_dicts.append(tab_dict.filter_videos(filtered_experiments))

    else:
        tab_dicts = [tab_dict]

    #collect transitions for all experiments
    first_key=list(tab_dicts[0].keys())[0]
    num_behaviors=get_dt(tab_dicts[0],first_key,only_metainfo=True)['num_cols']
    if behaviors is not None:
        num_behaviors=num_behaviors-len(np.unique(behaviors))

    for z, tab_dict in enumerate(tab_dicts): 
        associations = np.zeros([num_behaviors,num_behaviors])    
        for key in tab_dict.keys():
            load_range = bin_info[key]["time"]
            if roi_number is not None:
                load_range=deepof.visuals_utils.get_behavior_frames_in_roi(None,bin_info[key],animals_in_roi)
            tab = copy.deepcopy(get_dt(tab_dict,key,load_range=load_range))
            #reformat tab in unsupervised case
            if plot_type=="unsupervised":
                pass
            #remove excluded behaviors
            if behaviors is not None:
                tab=tab.drop(columns=behaviors)
            #save behavior order
            if key == first_key:
                included_behaviors=tab.columns
            if association_metric=="FSTTC":
                tab_numpy=np.nan_to_num(tab.to_numpy().T)
                extended_behaviors=deepof.utils.extend_behaviors_numba(tab_numpy,delta_T,coordinates._frame_rate)

            for i in range(0,tab.shape[1]):
                for j in range(0, tab.shape[1]):
                    if i==j:
                        association_ij=np.NaN
                    else: 
                        if association_metric=="FSTTC":
                            preceding_behavior=extended_behaviors[i,:]
                            proximate_behavior=extended_behaviors[j,:]
                            association_ij=calculate_FSTTC(
                                preceding_behavior,
                                proximate_behavior,
                                coordinates._frame_rate,
                                delta_T
                                )
                        elif association_metric=="odds_ratio":
                            preceding_behavior=tab.iloc[:,i]
                            proximate_behavior=tab.iloc[:,j]
                            association_ij=calculate_simple_association(
                                np.nan_to_num(preceding_behavior.to_numpy()).astype(bool),
                                np.nan_to_num(proximate_behavior.to_numpy()).astype(bool),
                                coordinates._frame_rate,
                                )
                        else:
                            raise NotImplementedError(
                            "currently, only FSTTC is implemented as a valid association metric!"
                        )
                    #skip cases in which nans are returned (treat them as zeros i.e. no association)
                    if not np.isnan(association_ij):
                        associations[i,j]+=association_ij

        associations=associations/len(tab_dict)
        if z==0:
            associations_prev=copy.copy(associations)
        else:
            associations=associations_prev-associations

    # Use seaborn to plot heatmaps across both conditions
    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(16, 8),
        )

    sns.heatmap(
        associations,
        cmap=sns.diverging_palette(145, 20, as_cmap=True),
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
        **kwargs,
    )
    if experiment_id is not None:
        ax.set_title(experiment_id)
    elif exp_condition is not None and len(condition_values)==1:
        ax.set_title(condition_values[0])
    elif exp_condition is not None and len(condition_values)==2:
        ax.set_title(condition_values[0]+ " - "+ condition_values[1])
    else:
        ax.set_title("all")
    ax.set_xticklabels(included_behaviors, rotation=90)
    ax.set_yticklabels(included_behaviors, rotation=0)


    if ax is None:

        plt.tight_layout()

        if save:
            plt.savefig(
                os.path.join(
                    coordinates._project_path,
                    coordinates._project_name,
                    "Figures",
                    "deepof_associations{}_bin_size={}_bin_index={}_{}.pdf".format(
                        (f"_{save}" if isinstance(save, str) else ""),
                        bin_size,
                        bin_index,
                        calendar.timegm(time.gmtime()),
                    ),
                )
            )

        plt.show() 

        if return_values:
            return associations
        else:
            return None
"""                                

def plot_stationary_entropy(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict,
    # Time selection parameters
    bin_size: Union[int, str] = None,
    bin_index: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max=20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    # Visualization parameters
    add_stats: str = "Mann-Whitney",
    exp_condition: str = None,
    verbose: bool = False,
    ax: Any = None,
    save: bool = False,
):
    """Compute and plots transition stationary distribution entropy per condition.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                           
        add_stats (str): test to use. Mann-Whitney (non-parametric) by default. See statsannotations documentation for details.        
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.        
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        exp_condition=exp_condition,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
    )
    if animals_in_roi is None:
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animal_id is only relevant if a ROI was selected!\033[0m'
        )


    # Get requested experimental condition. If none is provided, default to the first one available.
    if exp_condition is None:
        exp_conditions = {
            key: str(val.iloc[:, 0].values[0])
            for key, val in coordinates.get_exp_conditions.items()
        }
    else:
        exp_conditions = {
            key: str(val.loc[:, exp_condition].values[0])
            for key, val in coordinates.get_exp_conditions.items()
        }

    soft_counts = soft_counts.filter_videos(embeddings.keys())

    # preprocess information given for time binning
    bin_info_time = _preprocess_time_bins(
        coordinates, bin_size, bin_index, precomputed_bins, tab_dict_for_binning=embeddings, samples_max=samples_max, down_sample=False,
    )
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    if (any([np.sum(bin_info[key]["time"]) < 2 for key in bin_info.keys()])):
        raise ValueError("precomputed_bins or bin_size need to be > 1")

    # Get ungrouped entropy scores for the full videos
    ungrouped_transitions = deepof.post_hoc.compute_transition_matrix_per_condition(
        soft_counts,
        exp_conditions,
        bin_info=bin_info,
        roi_number=roi_number,
        animals_in_roi=animals_in_roi,
        aggregate=False,
        normalize=True,
    )
    ungrouped_entropy_scores = deepof.post_hoc.compute_steady_state(
        ungrouped_transitions, return_entropy=True, n_iters=10000
    )

    ungrouped_entropy_scores = pd.DataFrame(ungrouped_entropy_scores, index=[0]).melt(
        value_name="entropy"
    )
    ungrouped_entropy_scores["exp condition"] = ungrouped_entropy_scores.variable.map(
        exp_conditions
    )
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # sort for uniform plotting
    ungrouped_entropy_scores.sort_values(
        by=ungrouped_entropy_scores.columns[2], inplace=True
    )
    # Draw violin/strip plots with full-video entropy
    np.random.seed(42)
    sns.violinplot(
        data=ungrouped_entropy_scores,
        y="exp condition",
        x="entropy",
        ax=ax,
        linewidth=2,
    )
    sns.stripplot(
        data=ungrouped_entropy_scores,
        y="exp condition",
        x="entropy",
        ax=ax,
        color="black",
    )
    np.random.seed(None)
    plt.ylabel("experimental condition")

    if add_stats:
        pairs = list(combinations(set(exp_conditions.values()), 2))

        annotator = Annotator(
            ax,
            pairs=pairs,
            data=ungrouped_entropy_scores,
            x="entropy",
            y="exp condition",
            orient="h",
        )
        annotator.configure(
            test=add_stats,
            text_format="star",
            loc="inside",
            comparisons_correction="fdr_bh",
            verbose=verbose,
        )
        annotator.apply_and_annotate()

    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_entropy{}_bin_size={}_bin_index={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    bin_size,
                    bin_index,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    if ax is None:
        plt.show()


def plot_normative_log_likelihood(
    embeddings: table_dict,
    exp_condition: str,
    embedding_dataset: pd.DataFrame,
    normative_model: str,
    ax: Any,
    add_stats: str,
    verbose: bool,
):
    """Plot a bar chart with normative log likelihoods per experimental condition, and compute statistics.

    Args:
        embeddings (table_dict): table dictionary containing supervised annotations or unsupervised embeddings per animal.
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.
        embedding_dataset (pd.DataFrame): global animal embeddings, alongside their respective experimental conditions
        normative_model (str): Name of the cohort to use as controls. If provided, fits a Gaussian density to the control global animal embeddings, and reports the difference in likelihood across all instances of the provided experimental condition. Statistical parameters can be controlled via **kwargs (see full documentation for details).
        ax (plt.AxesSubplot): matplotlib axes where to render the plot
        add_stats (str): test to use. Mann-Whitney (non-parametric) by default. See statsannotations documentation for details.
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.

    Returns:
        embedding_dataset (pd.DataFrame): embedding data frame with added normative scores per sample

    """
    # Fit normative model to animals belonging to the control cohort
    norm_density = deepof.post_hoc.fit_normative_global_model(
        embedding_dataset.loc[
            embedding_dataset["experimental condition"] == normative_model,
            ["PCA-1", "PCA-2"],
        ]
    )

    # Add normative log likelihood to the dataset
    embedding_dataset["norm_scores"] = norm_density.score_samples(
        embedding_dataset.loc[:, ["PCA-1", "PCA-2"]].values
    )

    # Center log likelihood values around the control mean
    embedding_dataset["norm_scores"] -= embedding_dataset.loc[
        embedding_dataset["experimental condition"] == normative_model,
        "norm_scores",
    ].mean()

    # Add a second axis to the right of the main plot, and show the corresponding bar charts
    if ax is None:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]}
        )

    elif isinstance(ax, list):
        ax, ax2 = ax

    else:
        raise ValueError(
            "Passing normative_model produces two plots: a scatterplot with a PCA of the embeddings"
            "themselves, and a barplot depicting the normative likelihood per condition. Instead of"
            "a single ax, pass a list with two."
        )
    np.random.seed(42)
    sns.boxplot(
        data=embedding_dataset.sort_values(
            "experimental condition",
            key=lambda x: x == normative_model,
            ascending=False,
        ),
        x="experimental condition",
        y="norm_scores",
        ax=ax2,
    )
    sns.stripplot(
        data=embedding_dataset.sort_values(
            "experimental condition",
            key=lambda x: x == normative_model,
            ascending=False,
        ),
        x="experimental condition",
        y="norm_scores",
        dodge=True,
        color="black",
        ax=ax2,
    )
    np.random.seed(None)

    ax2.set_xlabel("")
    ax2.set_ylabel("centered normative log likelihood")

    # Add statistics
    if exp_condition is None:
        exp_conditions = {
            key: str(val.iloc[:, 0].values[0])
            for key, val in embeddings._exp_conditions.items()
        }
    else:
        exp_conditions = {
            key: str(val.loc[:, exp_condition].values[0])
            for key, val in embeddings._exp_conditions.items()
        }

    embedding_dataset.index = embeddings._exp_conditions.keys()
    embedding_dataset.sort_values(
        "experimental condition",
        key=lambda x: x == normative_model,
        ascending=False,
        inplace=True,
    )

    pairs = [
        pair
        for pair in list(combinations(set(exp_conditions.values()), 2))
        if normative_model in pair
    ]

    annotator = Annotator(
        pairs=pairs,
        data=embedding_dataset,
        x="experimental condition",
        y="norm_scores",
        ax=ax2,
    )
    annotator.configure(
        test=add_stats,
        verbose=verbose,
    )
    annotator.apply_and_annotate()

    return embedding_dataset, False, ax


def plot_embeddings(
    coordinates: coordinates,
    embeddings: table_dict = None,
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    # Time selection parameters
    bin_size: Union[int, str] = None,
    bin_index: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max=20000,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: Union[str,list] = None,
    roi_mode: str = "mousewise",
    # Quality selection parameters
    min_confidence: float = 0.0,
    # Normative modelling
    normative_model: str = None,
    add_stats: str = "Mann-Whitney",
    verbose: bool = False,
    # Visualization design and data parameters
    exp_condition: str = None,
    aggregate_experiments: str = None,
    samples: int = 500,
    show_aggregated_density: bool = True,
    colour_by: str = "exp_condition",
    ax: Any = None,
    save: bool = False,
):
    """Return a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation, and changepoint detection size visualization.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        supervised_annotations (table_dict): table dict with supervised annotations per experiment.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                                          
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
        min_confidence (float): minimum confidence in cluster assignments used for quality control filtering.                
        normative_model (str): Name of the cohort to use as controls. If provided, fits a Gaussian density to the control global animal embeddings, and reports the difference in likelihood across all instances of the provided experimental condition. Statistical parameters can be controlled via **kwargs (see full documentation for details).
        add_stats (str): test to use. Mann-Whitney (non-parametric) by default. See statsannotations documentation for details.
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one available is used.    
        aggregate_experiments (str): Whether to aggregate embeddings by experiment (by time on cluster, mean, or median) or not (default).
        samples (int): Number of samples to take from the time embeddings. None leads to plotting all time-points, which may hurt performance.
        show_aggregated_density (bool): if True, a density plot is added to the aggregated embeddings.
        colour_by (str): hue by which to colour the embeddings. Can be one of 'cluster' (default), 'exp_condition', or 'exp_id'.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        normative_model=normative_model,
        exp_condition=exp_condition,
        aggregate_experiments=aggregate_experiments,
        colour_by=colour_by,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
        roi_mode=roi_mode,
    )
    if supervised_annotations is not None and roi_number is not None and animals_in_roi is not None:
        raise ValueError(
            '"No animal_id can be selected when supeprvised_annotations are analyzed with a ROI as this would result in empty aggregations!"'
        )
    if animals_in_roi is None or roi_mode == "behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animal_id is only relevant if a ROI was selected!\033[0m'
        )
    if type(animals_in_roi)==str:
        animals_in_roi=[animals_in_roi]
    # prevents crash due to axis issues
    if (
        not aggregate_experiments
        and embeddings is not None
        and normative_model
        or not aggregate_experiments
        and embeddings is not None
        and type(ax) == list
    ):
        raise ValueError(
            '"normative_model" cannot be used without "aggregate_experiments", hence "ax" cannot be a list'
        )
    # Checks to throw errors or warn about conflicting inputs
    if supervised_annotations is not None and any(
        [embeddings is not None, soft_counts is not None]
    ):
        raise ValueError(
            "This function only accepts either supervised or unsupervised annotations as inputs, not both at the same time!"
        )      

    # preprocess information given for time binning
    if supervised_annotations is not None:
        bin_info_time = _preprocess_time_bins(
            coordinates, bin_size, bin_index, precomputed_bins, 
            tab_dict_for_binning=supervised_annotations, samples_max=samples_max,
        )
    else:
        bin_info_time = _preprocess_time_bins(
            coordinates, bin_size, bin_index, precomputed_bins, 
            tab_dict_for_binning=embeddings, samples_max=samples_max,
        )

    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)
    

    # Filter embeddings, soft_counts and supervised_annotations based on the provided keys and experimental condition
    (
        emb_to_plot,
        counts_to_plot,
        sup_annots_to_plot,
        concat_hue,
    ) = _filter_embeddings(
        coordinates,
        copy.deepcopy(embeddings),
        copy.deepcopy(soft_counts),
        copy.deepcopy(supervised_annotations),
        exp_condition,
    )
    show = True

    # Plot unravelled temporal embeddings
    if not aggregate_experiments and emb_to_plot is not None:

        samples_dict={}

        #set samples to a maximum of samples_max
        if samples is None:
            samples=samples_max

        # make sure that not more samples are drawn than are available
        shortest = samples
        for key in bin_info.keys():
            for aid in bin_info[key].keys():

                if aid == "time":
                    num_rows=len(bin_info[key]["time"])
                elif roi_number is not None and aid in animals_in_roi:
                    num_rows=np.sum(bin_info[key][aid])

                if num_rows < shortest:
                    shortest = num_rows
        if samples > shortest:
            assert shortest > 0, "Selected time bin and / or ROI are too restrictive, cannot draw enough samples for all experiments!" 
            samples = shortest
            print(
                "\033[33mInfo! Set samples to {} to not exceed data length!\033[0m".format(
                    samples
                )
            )

        # Sample per animal, to avoid alignment issues
        valid_samples={}
        for key in emb_to_plot.keys():

            #get correct section of current embedding 
            current_emb=get_dt(emb_to_plot,key)
            if roi_number is not None:
                valid_samples[key]=get_behavior_frames_in_roi(behavior=None,local_bin_info=bin_info[key],animal_ids=animals_in_roi)
            else:
                valid_samples[key]=bin_info[key]["time"]
            current_emb=current_emb[valid_samples[key]]

            sample_ids = np.random.choice(
                range(current_emb.shape[0]), samples, replace=False
            )
            samples_dict[key] = sample_ids
            #reduced section is kept in memory
            emb_to_plot[key] = current_emb[sample_ids]
        get_behavior_frames_in_roi._warning_issued = False
               

        # Concatenate experiments and align experimental conditions
        concat_embeddings = np.concatenate(
            [get_dt(emb_to_plot,key) 
                for key in emb_to_plot],
            axis=0
        )

        # Get cluster assignments from soft counts
        cluster_assignments = np.argmax(
            np.concatenate(
                [get_dt(counts_to_plot,key)[valid_samples[key]][samples_dict[key]]
                for key in counts_to_plot], 
                axis=0
            ), 
            axis=1
        )

        # Compute confidence in assigned clusters
        confidence = np.concatenate(
            [
                np.max(get_dt(counts_to_plot,key)[valid_samples[key]][samples_dict[key]], axis=1)
                for key in counts_to_plot
            ]
        )

        # Reduce the dimensionality of the embeddings using UMAP. Set n_neighbors to a large
        # value to see a more global picture
        reducers = deepof.post_hoc.compute_UMAP(concat_embeddings, cluster_assignments)
        reduced_embeddings = reducers[1].transform(
            reducers[0].transform(concat_embeddings)
        )

        # Generate unifier dataset using the reduced embeddings, experimental conditions
        # and the corresponding break lengths and cluster assignments

        
        lens = np.zeros(len(emb_to_plot), dtype=int)
        for i, key in enumerate(emb_to_plot.keys()):
            lens[i] = len(emb_to_plot[key])

        embedding_dataset = pd.DataFrame(
            {
                "UMAP-1": reduced_embeddings[:, 0],
                "UMAP-2": reduced_embeddings[:, 1],
                "exp_id": np.repeat(list(range(len(emb_to_plot))), lens),
                "confidence": confidence,
                "cluster": cluster_assignments,
                "experimental condition": np.repeat(concat_hue, lens),
            }
        )

        # Filter values with low confidence
        embedding_dataset = embedding_dataset.loc[
            embedding_dataset.confidence > min_confidence
        ]

    else:

        # set aggregate_experiments default based on type of plot
        if (
            not aggregate_experiments or aggregate_experiments == "time on cluster"
        ) and sup_annots_to_plot is not None:
            aggregate_experiments = "mean"  # makes more sense to capture 0-1 behaviors
            print(
                f"\033[33mInfo! Set aggregate_experiments to -mean- since supervised annotations were given!\033[0m"
            )

        # Aggregate experiments by time on cluster
        if aggregate_experiments == "time on cluster":
            aggregated_embeddings = deepof.post_hoc.get_time_on_cluster(
                counts_to_plot, reduce_dim=True, bin_info=bin_info, roi_number=roi_number, animals_in_roi=animals_in_roi,
            )

        else:
            if emb_to_plot is not None:
                aggregated_embeddings = deepof.post_hoc.get_aggregated_embedding(
                    emb_to_plot, agg=aggregate_experiments, reduce_dim=True, bin_info=bin_info, roi_number=roi_number, animals_in_roi=animals_in_roi, roi_mode=roi_mode,
                )
            else:
                aggregated_embeddings = deepof.post_hoc.get_aggregated_embedding(
                    sup_annots_to_plot, agg=aggregate_experiments, reduce_dim=True, bin_info=bin_info, roi_number=roi_number, animals_in_roi=animals_in_roi, roi_mode=roi_mode,
                )

        # Generate unifier dataset using the reduced aggregated embeddings and experimental conditions
        embedding_dataset = pd.DataFrame(
            {
                "PCA-1": aggregated_embeddings[0],
                "PCA-2": aggregated_embeddings[1],
                "experimental condition": concat_hue,
            }
        )
        embedding_dataset=embedding_dataset.dropna()
        embedding_dataset.sort_values(by=embedding_dataset.columns[2], inplace=True)

        if normative_model:
            embedding_dataset, show, ax = plot_normative_log_likelihood(
                (embeddings if embeddings is not None else supervised_annotations),
                exp_condition,
                embedding_dataset,
                normative_model,
                ax,
                add_stats,
                verbose,
            )

    # set hue for plot
    if colour_by != "exp_condition" and aggregate_experiments:
        colour_by = "exp_condition"
        print(
            "\033[33mInfo! Set colour_by to {} as aggregate_experiments were given!\033[0m".format(
                colour_by
            )
        )
        hue = "experimental condition"
    elif aggregate_experiments or colour_by == "exp_condition":
        hue = "experimental condition"
    else:
        hue = colour_by

    # resort
    embedding_dataset.sort_values(by=embedding_dataset.columns[2], inplace=True)
    # Plot selected embeddings using the specified settings
    sns.scatterplot(
        data=embedding_dataset,
        x="{}-1".format("PCA" if aggregate_experiments else "UMAP"),
        y="{}-2".format("PCA" if aggregate_experiments else "UMAP"),
        ax=ax,
        hue=hue,
        size=None,
        s=(50 if not aggregate_experiments else 100),
        edgecolor="black",
        palette=(
            None if aggregate_experiments or colour_by == "exp_condition" else "tab20"
        ),
    )

    if aggregate_experiments and show_aggregated_density:
        # group dataset according to all experiment condition combinations
        grouped = embedding_dataset.groupby("experimental condition", group_keys=True)
        # check colinearit for each condition combination
        is_colinear = False
        for key in grouped.groups.keys():
            data = np.array(grouped.get_group(key)[["PCA-1", "PCA-2"]])
            data -= data[0]
            if np.linalg.matrix_rank(data, tol=0.00001) < 2:
                is_colinear = True
                break
        # if no colinearity was detected, plot kdeplot
        if not (is_colinear):
            sns.kdeplot(
                data=embedding_dataset,
                x="PCA-1",
                y="PCA-2",
                hue="experimental condition",
                zorder=0,
                ax=ax,
            )
        else:
            warning_message = (
                "\033[38;5;208m\n"  # Set text color to orange
                "Warning! Failed to plot continuous probability density curve!\n"
                "Some Experimental condition combinations do not span at least two dimensions!\n"
                "This Error may happen due to an insufficient amount of data."
                "\033[0m"  # Reset text color
            )
            warnings.warn(warning_message)

    if not aggregate_experiments:
        if ax is None:
            plt.legend("", frameon=False)
        else:
            ax.get_legend().remove()

    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_embeddings{}_colour={}_agg={}_min_conf={}_bin_size={}_bin_index={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    colour_by,
                    aggregate_experiments,
                    min_confidence,
                    bin_size,
                    bin_index,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    title = "deepOF - {}supervised {}embedding".format(
        ("un" if sup_annots_to_plot is None else ""),
        ("aggregated " if aggregate_experiments else ""),
    )
    if ax is not None or not show:
        ax.set_title(title, fontsize=15)

    else:
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.show()


# noinspection PyTypeChecker
def animate_skeleton(
    coordinates: coordinates,
    experiment_id: str,
    embeddings: table_dict = None,
    soft_counts: table_dict = None,
    # Time selection parameters
    bin_size: Union[int, str] = None,
    bin_index: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max: int =20000,  
    # ROI functionality
    roi_number: int = None,  
    animals_in_roi: list = None,
    # other parameters
    animal_id: list = None,
    center: str = "arena",
    align: str = None,
    sampling_rate: float = None, 
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    selected_cluster: np.ndarray = None,
    display_arena: bool = True,
    legend: bool = True,
    save: bool = None,
    dpi: int = 100,
):
    """Render a FuncAnimation object with embeddings and/or motion trajectories over time.

    Args:
        coordinates (coordinates): deepof Coordinates object.
        experiment_id (str): Name of the experiment to display.
        embeddings (Union[List, np.ndarray]): UMAP 2D embedding of the datapoints provided. If not None, a second animation shows a parallel animation with the currently selected embedding, colored by cluster if cluster_assignments are available.
        soft_counts (np.ndarray): contain sorted cluster assignments for all instances in data. If provided together with selected_cluster, only instances of the specified component are returned. Defaults to None.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                                                  
        animal_id (list): ID list of animals to display. If None (default) it shows all animals.
        center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).       
        sampling_rate (float): Sampling rate for the video. If None is given, the same one as in the video recordings will be used.
        min_confidence (float): Minimum confidence threshold to render a cluster assignment bout.
        min_bout_duration (int): Minimum number of frames to render a cluster assignment bout.
        selected_cluster (int): cluster to filter. If provided together with cluster_assignments,
        display_arena (bool): whether to plot a dashed line with an overlying arena perimeter. Defaults to True.
        legend (bool): whether to add a color-coded legend to multi-animal plots. Defaults to True when there are more than one animal in the representation, False otherwise.
        save (str): name of the file where to save the produced animation.
        dpi (int): dots per inch of the figure to create.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        experiment_ids=experiment_id,
        animal_id=animal_id,
        center=center,
        roi_number=roi_number,
    )
    if animal_id is None:
        animal_id = coordinates._animal_ids
    if type(animal_id)==str:
        animal_id=[animal_id]
    if animals_in_roi is None:
        animals_in_roi = coordinates._animal_ids
    if type(animals_in_roi)==str:
        animals_in_roi=[animals_in_roi]


    bin_info_time = _preprocess_time_bins(
    coordinates, bin_size, bin_index, precomputed_bins, samples_max=samples_max,
    )
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)

    if sampling_rate is None:
        sampling_rate=coordinates._frame_rate

    if embeddings is not None:
        #Get data for requested experiment
        cur_embeddings=get_dt(embeddings, experiment_id)
        cur_soft_counts=get_dt(soft_counts, experiment_id)
    else:
        cur_embeddings=None
        cur_soft_counts=None

    #scales is now a dictionary which simplifies things
    coords = coordinates.get_coords_at_key(center=center, align=align, scale=coordinates._scales[experiment_id], key=experiment_id)

    # Filter requested animals
    coords_list=[]
    for aid in animal_id:
        coords_list.append(deepof.utils.filter_animal_id_in_table(table=coords, selected_id=aid))
    coords=pd.concat(coords_list,axis=1)

    # Sort column index to allow for multiindex slicing
    coords = coords.sort_index(ascending=True, inplace=False, axis=1)

    # Get output scale
    x_dv = np.maximum(
        np.abs(coords.loc[:, (slice("x"), ["x"])].min().mean()),
        np.abs(coords.loc[:, (slice("x"), ["x"])].max().mean()),
    )
    y_dv = np.maximum(
        np.abs(coords.loc[:, (slice("x"), ["y"])].min().mean()),
        np.abs(coords.loc[:, (slice("x"), ["y"])].max().mean()),
    )

    if embeddings is not None:
        # Get and process data to plot from coordinates object
        (
            coords,
            cur_embeddings,
            cluster_embedding,
            concat_embedding,
            hard_counts,
        ) = _process_animation_data(
            coords,
            cur_embeddings,
            cur_soft_counts,
            min_confidence,
            min_bout_duration,
            selected_cluster,
        )

    # Define canvas
    fig = plt.figure(figsize=((16 if cur_embeddings is not None else 8), 8), dpi=dpi)

    # If embeddings are provided, add projection plot to the left
    if cur_embeddings is not None:
        ax1 = fig.add_subplot(121)

        _scatter_embeddings(concat_embedding, hard_counts, ax1, show=False)

        # Plot current position
        umap_scatter = {}
        for i, emb in enumerate(cur_embeddings):
            umap_scatter[i] = ax1.scatter(
                emb[0, 0],
                emb[0, 1],
                color=(
                    "red"
                    if len(cur_embeddings) == 1
                    else list(sns.color_palette("tab10"))[i]
                ),
                s=200,
                linewidths=2,
                edgecolors="black",
            )

        ax1.set_title("UMAP projection of time embedding", fontsize=15)
        ax1.set_xlabel("UMAP-1")
        ax1.set_ylabel("UMAP-2")

    # Add skeleton animation
    ax2 = fig.add_subplot((122 if cur_embeddings is not None else 111))

    # Plot!
    init_x = coords.loc[:, (slice("x"), ["x"])].iloc[0, :]
    init_y = coords.loc[:, (slice("x"), ["y"])].iloc[0, :]

    # If there are more than one animal in the representation, display each in a different color
    hue = None
    cmap_all = ListedColormap(sns.color_palette("tab10", len(coordinates._animal_ids)))
    positions = [coordinates._animal_ids.index(item) for item in animal_id]
    cmap = ListedColormap([cmap_all(pos)for pos in positions])
    #cmap = cmap[]

    polygons = [_get_polygon_coords(coords, aid) for aid in animal_id]

    hue = np.zeros(len(np.array(init_x)))
    for i, id in enumerate(animal_id):

        hue[coords.columns.levels[0].str.startswith(id)] = i

        # Set a custom legend outside the plot, with the color of each animal

        if legend:
            custom_labels = [
                plt.scatter(
                    [np.inf],
                    [np.inf],
                    color=cmap(i / len(animal_id)),
                    lw=3,
                )
                for i in range(len(animal_id))
            ]
            ax2.legend(custom_labels, animal_id, loc="upper right")

    skeleton_scatter = ax2.scatter(
        x=np.array(init_x),
        y=np.array(init_y),
        cmap=cmap,
        label="Original",
        c=hue,
    )

    tail_lines = []
    for p, aid in enumerate(polygons):
        ax2.add_patch(
            patches.Polygon(
                aid[0][0, :].reshape(-1, 2),
                closed=True,
                fc=cmap.colors[p],
                ec=cmap.colors[p],
                alpha=0.5,
            )
        )
        ax2.add_patch(
            patches.Polygon(
                aid[1][0, :].reshape(-1, 2),
                closed=True,
                fc=cmap.colors[p],
                ec=cmap.colors[p],
                alpha=0.5,
            )
        )
        tail_lines.append(ax2.plot(*aid[2][0, :].reshape(-1, 2).T))

    if display_arena and center in [False, "arena"] and align is None:
        plot_arena(coordinates, center, "black", ax2, key=experiment_id)

    # Update data in main plot
    def animation_frame(i):

        if cur_embeddings is not None:
            # Update umap scatter
            for j, xy in umap_scatter.items():
                umap_x = cluster_embedding[j][i, 0]
                umap_y = cluster_embedding[j][i, 1]

                umap_scatter[j].set_offsets(np.c_[umap_x, umap_y])

        # Update skeleton scatter plot
        x = coords.loc[:, (slice("x"), ["x"])].iloc[i, :]
        y = coords.loc[:, (slice("x"), ["y"])].iloc[i, :]

        skeleton_scatter.set_offsets(np.c_[x, y])

        for p, aid in enumerate(polygons):
            # Update polygons
            ax2.patches[2 * p].set_xy(aid[0][i, :].reshape(-1, 2))
            ax2.patches[2 * p + 1].set_xy(aid[1][i, :].reshape(-1, 2))

            # Update tails
            tail_lines[p][0].set_xdata(aid[2][i, :].reshape(-1, 2)[:, 0])
            tail_lines[p][0].set_ydata(aid[2][i, :].reshape(-1, 2)[:, 1])

        if cur_embeddings is not None:
            return umap_scatter, skeleton_scatter

        return skeleton_scatter

    #get frames to display based on binning
    if roi_number is None:
        frames = bin_info[experiment_id]["time"]
    else:
        #a pseudo behavior gets constructed from the animal ids that contains all ids intended to be inside the roi.
        frames = get_behavior_frames_in_roi('_'.join(animals_in_roi) + '_', bin_info[experiment_id], animal_ids=animals_in_roi)
        get_behavior_frames_in_roi._warning_issued = False

    animation = FuncAnimation(
        fig,
        func=animation_frame,
        frames=frames,
        interval=int(np.round(1000 // sampling_rate)),
    )

    ax2.set_title(
        f"deepOF animation - {(f'{str(animal_id)} - ')}{experiment_id}",
        fontsize=15,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    if center not in [False, "arena"]:

        ax2.set_xlim(-1.5 * x_dv, 1.5 * x_dv)
        ax2.set_ylim(-1.5 * y_dv, 1.5 * y_dv)

    ax2.invert_yaxis()

    plt.tight_layout()

    if save is not None:
        save = os.path.join(
            coordinates._project_path,
            coordinates._project_name,
            "Out_videos",
            "deepof_embedding_animation{}_{}_start{}-duration{}_{}.mp4".format(
                (f"_{save}" if isinstance(save, str) else ""),
                (
                    "cluster={}".format(selected_cluster)
                    if selected_cluster is not None
                    else experiment_id
                ),
                str(bin_index) if bin_index is not None else "",
                str(bin_size) if bin_size is not None else "",
                calendar.timegm(time.gmtime()),
            ),
        )

        writevideo = FFMpegWriter(fps=sampling_rate)
        animation.save(save, writer=writevideo)

    return animation.to_html5_video()


@_suppress_warning(
    [
        "iteritems is deprecated and will be removed in a future version. Use .items instead."
    ]
)
def plot_cluster_detection_performance(
    coordinates: coordinates,
    chunk_stats: pd.DataFrame,
    cluster_gbm_performance: dict,
    hard_counts: np.ndarray,
    groups: list,
    save: bool = False,
    visualization: str = "confusion_matrix",
    ax: plt.Axes = None,
):
    """Plot either a confusion matrix or a bar chart with balanced accuracy for cluster detection cross validated models.

    Designed to be run after deepof.post_hoc.train_supervised_cluster_detectors (see documentation for details).

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        chunk_stats (pd.DataFrame): table with descriptive statistics for a series of sequences ('chunks').
        cluster_gbm_performance (dict): cross-validated dictionary containing trained estimators and performance metrics.
        hard_counts (np.ndarray): cluster assignments for the corresponding 'chunk_stats' table.
        groups (list): cross-validation indices. Data from the same animal are never shared between train and test sets.
        save (bool): name of the file where to save the produced figure.
        visualization (str): plot to render. Must be one of 'confusion_matrix', or 'balanced_accuracy'.
        ax (plt.Axes): axis where to plot the figure. If None, a new figure is created.

    """
    _check_enum_inputs(
        coordinates,
        origin="plot_cluster_detection_performance",
        visualization=visualization,
    )

    n_clusters = len(np.unique(hard_counts))
    confusion_matrices = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    for clf, fold in zip(cluster_gbm_performance["estimator"], groups):
        cm = confusion_matrix(
            hard_counts.values[fold[1]],
            clf.predict(chunk_stats.values[fold[1]]),
            labels=np.unique(hard_counts),
        )

        confusion_matrices.append(cm)

    cluster_names = ["cluster {}".format(i) for i in sorted(list(set(hard_counts)))]

    if visualization == "confusion_matrix":

        cm = np.stack(confusion_matrices).sum(axis=0)
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = pd.DataFrame(cm, index=cluster_names, columns=cluster_names)

        # Cluster rows and columns and reorder to put closer similar clusters
        row_link = linkage(
            cm, method="average", metric="euclidean"
        )  # computing the linkage
        row_order = dendrogram(row_link, no_plot=True)["leaves"]
        cm = cm.iloc[row_order, row_order]

        ax.set_title("Confusion matrix for multiclass state prediction")
        sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)
        ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), rotation=0)

    elif visualization == "balanced_accuracy":

        def compute_balanced_accuracy(cm, cluster_index):
            """

            Compute balanced accuracy for a specific cluster given a confusion matrix.

            Formula: ((( TP / (TP+FN) + (TN/(TN+FP))) / 2

            """
            TP = cm[cluster_index, cluster_index]
            FP = cm[:, cluster_index].sum() - TP
            FN = cm[cluster_index, :].sum() - TP
            TN = cm.sum() - TP - FP - FN

            return ((TP / (TP + FN)) + (TN / (TN + FP))) / 2

        dataset = defaultdict(list)

        for cluster in range(n_clusters):
            for cm in confusion_matrices:
                ba = compute_balanced_accuracy(cm, cluster)
                dataset[cluster].append(ba)

        dataset = pd.DataFrame(dataset)

        ax.set_title("Supervised cluster mapping performance")
        xticklabels = [str(column) for column in dataset.columns]

        # both throw iteritems deprecation warning
        np.random.seed(42)
        sns.barplot(
            data=dataset, ci=95, color=sns.color_palette("Blues").as_hex()[-3], ax=ax
        )
        sns.stripplot(data=dataset, color="black", ax=ax)
        np.random.seed(None)

        ax.axhline(1 / n_clusters, linestyle="--", color="black")
        ax.set_ylim(0, 1)

        ax.set_xlabel("Cluster")
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("Balanced accuracy")

    else:
        raise ValueError(
            "Invalid plot selected. Visualization should be one of 'confusion_matrix' or 'balanced_accuracy'. See documentation for details."
        )

    if ax is None:
        plt.tight_layout()

    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_supervised_cluster_detection_type={}{}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    visualization,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    if ax is None:
        plt.show()


@_suppress_warning(
    [
        "No data for colormapping provided via 'c'. Parameters 'vmin', 'vmax' will be ignored"
    ]
)
def plot_shap_swarm_per_cluster(
    coordinates: coordinates,
    data_to_explain: pd.DataFrame,
    shap_values: list,
    cluster: Union[str, int] = "all",
    max_display: int = 10,
    save: str = False,
    show: bool = True,
):
    """Plot a swarm plot of the SHAP values for a given cluster.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        data_to_explain (pd.DataFrame): table with descriptive statistics for a series of sequences ('chunks').
        shap_values (list): shap_values per cluster.
        cluster (int): cluster to plot. If "all" (default) global feature importance across all clusters is depicted in a bar chart.
        max_display (int): maximum number of features to display.
        save (str): if provided, saves the figure to the specified file.
        show (bool): if True, shows the figure.

    """
    shap_vals = copy.deepcopy(shap_values)

    if cluster != "all":
        shap_vals = shap_vals[cluster]

    shap.summary_plot(
        shap_vals,
        data_to_explain,
        max_display=max_display,
        show=False,
        feature_names=data_to_explain.columns,
    )

    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_supervised_cluster_detection_SHAP_cluster={}{}_{}.pdf".format(
                    cluster,
                    (f"_{save}" if isinstance(save, str) else ""),
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    if show:
        plt.show()


def export_annotated_video(
    coordinates: coordinates,
    soft_counts: dict = None,
    supervised_annotations: table_dict = None,
    # Time selection parameters
    bin_size: Union[int, str] = None,
    bin_index: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    frame_limit_per_video: int = None,
    # ROI functionality
    roi_number: int =None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    #others
    behaviors: list = None,
    experiment_id: str = None,
    min_confidence: float = 0.75,
    min_bout_duration: int = None,
    display_time: bool = False,
    display_counter: bool = False,
    display_arena: bool = False,
    display_markers: bool = False,
    display_mouse_labels: bool = False,
    exp_conditions: dict = {},
    cluster_names: str = None,
):
    """Export annotated videos from both supervised and unsupervised pipelines.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        soft_counts (dict): dictionary with soft_counts per experiment.
        supervised_annotations (table_dict): table dict with supervised annotations per experiment.
        bin_size (Union[int,str]): bin size for time filtering.
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        frame_limit_per_video (int): number of frames to render per video. If None, all frames are included for all videos.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded)       
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                                                  
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
        behaviors (list): Behaviors or Clusters to that get exported. If none is given, all are exported for softcounts and only nose2nose is exported for supervised annotations. If multiple behaviors are given as a list, one video can get annotated with multiple different behaviors
        experiment_id (str): if provided, data coming from a particular experiment is used. If not, all experiments are exported.
        min_confidence (float): minimum confidence threshold for a frame to be considered part of a cluster.
        min_bout_duration (int): Minimum number of frames to render a cluster assignment bout.
        display_time (bool): Displays current time in top left corner of the video frame
        display_counter (bool): Displays event counter for each displayed event.
        display_arena (bool): Displays arena for each video.
        display_markers (bool): Displays mouse body parts on top of the mice.
        display_mouse_labels (bool): Displays identities of the mice
        exp_conditions (dict): if provided, data coming from a particular condition is used. If not, all conditions are exported. If a dictionary with more than one entry is provided, the intersection of all conditions (i.e. male, stressed) is used.
        cluster_names (dict): dictionary with user-defined names for each cluster (useful to output interpretation).

    """
    if isinstance(behaviors,str):
        behaviors=[behaviors]
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        experiment_ids=experiment_id,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
        roi_mode=roi_mode,
    )
    # Create video config
    video_export_config = VideoExportConfig(
        display_time=display_time,
        display_counter=display_counter,
        display_arena=display_arena,
        display_markers=display_markers,
        display_mouse_labels=display_mouse_labels,
    )   

    if animals_in_roi is None or roi_mode=="behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For the video export animal_id is only relevant if a ROI was selected!\033[0m'
        )

    # Create output directory if it doesn't exist
    proj_path = os.path.join(coordinates._project_path, coordinates._project_name)
    out_path = os.path.join(proj_path, "Out_videos")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # If no bout duration is provided, use half the frame rate
    if min_bout_duration is None:
        min_bout_duration = int(np.round(coordinates._frame_rate // 2))  
    
    # set cluster names dependend on tab dict type (supervised or soft counts)
    if soft_counts is not None:

        first_key=list(soft_counts.keys())[0]
        if isinstance(soft_counts[first_key], dict):
            soft_counts[first_key] = get_dt(soft_counts,first_key)

        if cluster_names is None or len(cluster_names) != soft_counts[first_key].shape[1]:
            cluster_names = ["Cluster_"+ str(k) for k in range(soft_counts[first_key].shape[1])]
        #unify tab_dict name
        tab_dict=soft_counts
 
    else:
        first_key=list(supervised_annotations.keys())[0]
        if cluster_names is None or len(cluster_names) != supervised_annotations[first_key].shape[1]:
                cluster_names=supervised_annotations[first_key].columns
        tab_dict=supervised_annotations

    #preprocess time bins            
    bin_info_time = _preprocess_time_bins(
        coordinates, bin_size, bin_index, precomputed_bins, tab_dict_for_binning=tab_dict,
        )
    
    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)
    
    # special case: an experiment id was given
    if experiment_id is not None:

        # get frames for this experiment id
        if behaviors is None and supervised_annotations is not None:
            cur_tab=copy.deepcopy(get_dt(tab_dict, experiment_id))
            behaviors = [cur_tab.columns[0]]
        if roi_number is not None:
            if roi_mode == "behaviorwise":
                behavior_in=behaviors[0]
                if len(behaviors)>1:
                    print('\033[33mInfo! The first behavior was automatically chosen for ROI application!\033[0m')
            else:
                behavior_in=None
            frames=get_behavior_frames_in_roi(behavior=behavior_in, local_bin_info=bin_info[experiment_id], animal_ids=animals_in_roi)
        else:
            frames=bin_info[experiment_id]["time"]
        # get current tab and video path
        cur_tab=copy.deepcopy(get_dt(tab_dict, experiment_id))
        
        # reformat current tab into data table with cluster names as column names
        if soft_counts is not None:
            cur_tab=pd.DataFrame(cur_tab,columns=cluster_names)
        else: 
            cur_tab.columns = cluster_names
        # handle defaults
        if frame_limit_per_video is None:
            frame_limit_per_video = np.inf

        if len(frames) >= frame_limit_per_video:
                frames = frames[0:frame_limit_per_video]

        video = output_annotated_video(
            coordinates=coordinates,
            experiment_id=experiment_id,                
            tab=cur_tab,
            behaviors=behaviors,
            config=video_export_config,
            frames=frames,
            out_path=out_path,
        )
        get_behavior_frames_in_roi._warning_issued = False

        return video

    else:
        # If experiment_id is not provided, output a video per cluster for each experiment
        if frame_limit_per_video is None:
            frame_limit_per_video = 250

        output_videos_per_cluster(
            coordinates,
            exp_conditions,
            tab_dict,
            behaviors,
            behavior_names=cluster_names,
            single_output_resolution=(500, 500),
            frame_limit_per_video=frame_limit_per_video,
            bin_info=bin_info,
            roi_number=roi_number,
            animals_in_roi=animals_in_roi,
            min_confidence=min_confidence,
            min_bout_duration=min_bout_duration,
            out_path=out_path,
            config=video_export_config,
            roi_mode=roi_mode,
        )

        return None


def plot_distance_between_conditions(
    # Model selection parameters
    coordinates: coordinates,
    embedding: dict,
    soft_counts: dict,
    exp_condition: str,
    embedding_aggregation_method: str = "median",
    distance_metric: str = "wasserstein",
    n_jobs: int = -1,
    save: bool = False,
    ax: Any = None,
):
    """Plot the distance between conditions across a growing time window.

    Finds an optimal separation binning based on the distance between conditions, and plots it across all non-overlapping bins.
    Useful, for example, to measure habituation over time.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        embedding (dict): embedding object for the current project. Used to get video paths.
        soft_counts (dict): dictionary with soft_counts per experiment.
        exp_condition (str): experimental condition to use for the distance calculation.
        embedding_aggregation_method (str): method to use for aggregating the embedding. Options are 'time_on_cluster' and 'mean'.
        distance_metric (str): distance metric to use for the distance calculation. Options are 'wasserstein' and 'euclidean'.
        n_jobs (int): number of jobs to use for the distance calculation.
        save (bool): if True, saves the figure to the project directory.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        exp_condition=exp_condition,
    )

    min_len=np.min([get_dt(soft_counts, key, only_metainfo=True)['num_rows'] for key in soft_counts.keys()])

    # Get distance between distributions across the growing window
    distance_array = deepof.post_hoc.condition_distance_binning(
        embedding,
        soft_counts,
        {
            key: val[exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        },
        int(np.round(10 * coordinates._frame_rate)),
        min_len,
        int(np.round(coordinates._frame_rate)),
        agg=embedding_aggregation_method,
        metric=distance_metric,
        n_jobs=n_jobs,
    )

    optimal_bin = np.argmax(savgol_filter(distance_array, 10, 2)) + 10
    print("Found an optimal_bin at {} seconds".format(optimal_bin))

    distance_per_bin = deepof.post_hoc.condition_distance_binning(
        embedding,
        soft_counts,
        {
            key: val[exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        },
        int(np.round(10 * coordinates._frame_rate)),
        min_len,
        int(np.round(optimal_bin * coordinates._frame_rate)),
        agg=embedding_aggregation_method,
        scan_mode="per-bin",
        metric=distance_metric,
        n_jobs=n_jobs,
    )

    # Concatenate both arrays and create a px compatible data frame
    distance_df = pd.DataFrame(
        {
            exp_condition: distance_array,
            "Time": np.linspace(
                10,
                min_len,
                len(distance_array),
            )
            / coordinates._frame_rate,
        }
    ).melt(
        id_vars=["Time"],
        value_name=distance_metric,
        var_name="experimental setting",
    )

    bin_distance_df = pd.DataFrame(
        {
            exp_condition: distance_per_bin,
            "Time": np.concatenate(
                [
                    optimal_bin * np.arange(1, len(distance_per_bin)),
                    [
                        min_len
                        / coordinates._frame_rate
                    ],
                ]
            ),
        }
    ).melt(
        id_vars=["Time"],
        value_name=distance_metric,
        var_name="experimental setting",
    )

    # Plot the obtained distance array
    sns.lineplot(
        data=distance_df,
        x="Time",
        y=distance_metric,
        color="#d6dbd2",
        ax=ax,
    )
    sns.lineplot(
        data=bin_distance_df,
        x="Time",
        y=distance_metric,
        color="#0b7189",
        zorder=100,
        ax=ax,
    )
    sns.scatterplot(
        data=bin_distance_df,
        x="Time",
        y=distance_metric,
        color="#0b7189",
        s=200,
        linewidth=1,
        zorder=100,
        ax=ax,
    )

    if ax is None:
        plt.title("deepOF - distance between conditions")
        plt.xlim(0, len(distance_array) + coordinates._frame_rate)
        plt.tight_layout()

    if save:  # pragma: no cover
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_distance_between_conditions_{}{}_{}_{}_{}.pdf".format(
                    exp_condition,
                    embedding_aggregation_method,
                    distance_metric,
                    (f"_{save}" if isinstance(save, str) else ""),
                    calendar.timegm(time.gmtime()),
                ),
            )
        )


@_suppress_warning(
    warn_messages=["Info! At least one of the selected groups has only one element!"],
    do_what=["once"]
)
def plot_behavior_trends(
    coordinates: coordinates,
    embeddings: table_dict = None,
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    # Time selection parameters
    N_time_bins: int = 24,
    custom_time_bins: List[List[Union[int, str]]] = None,
    # ROI functionality
    roi_number: int = None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    # Visualization
    hide_time_bins: List[bool] = None,
    polar_depiction: bool = True,
    show_histogram: bool = True,
    exp_condition: str = None,
    condition_values: list = None,
    behavior_to_plot: str = None,
    normalize: bool = False,
    add_stats: str = "Mann-Whitney",
    error_bars: str = "sem",
    ax: Any = None,
    save: bool = False,
):
    """
    Creates a polar plot or histogram of behavioral data over time.

    Args:
    coordinates (coordinates): deepOF project containing the stored data.
    embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
    soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
    supervised_annotations (table_dict): Table dict with supervised annotations per video.
    N_time_bins (int): Number of time bins for data separation. Defaults to 24.
    custom_time_bins (List[List[Union[int,str]]]): Custom time bins array consisting of pairs of start- and stop positions given as integers or time strings. Overrides N_time_bins if provided.
    roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded)       
    animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                                                  
    roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
    hide_time_bins (List[bool]): List of booleans denoting which bins should be visible (False) or hidden (True). Defaults to displaying all tiem bins.    
    polar_depiction (bool): if True, display as polar plot. Defaults to True.
    show_histogram (bool): If True, displays histogram with rough effect size estimations. Defaults to True.
    exp_condition (str): Experimental condition to compare.
    condition_values (list): List of two strings containing the condition values to compare.
    behavior_to_plot (str): Behavior to compare for selected condition.
    normalize (bool): If True, shows time on cluster relative to bin length instead of total time on cluster. Speed is always averaged. Defaults to False.
    add_stats (str): test to use. Mann-Whitney (non-parametric) by default. See statsannotations documentation for details.
    error_bars (str): Type of error bars to display (either standard deviation ("std") or standard error ("sem")). Defaults to standard error.
    ax (Any): Matplotlib axis for plotting. If None, creates a new figure.
    save (bool): If True, saves the plot to a file. Defaults to False.
    """


    # Initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        supervised_annotations=supervised_annotations,
        soft_counts=soft_counts,
        exp_condition=exp_condition,
        condition_values=condition_values,
        behaviors=behavior_to_plot,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
        roi_mode=roi_mode,
    )
    if animals_in_roi is None or roi_mode == "behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animal_id is only relevant if a ROI was selected!\033[0m'
        )

    #####
    # Set defaults based on inputs
    #####

    # Init exp_condition if not given
    if not exp_condition:
        exp_condition = coordinates.get_exp_conditions[
            next(iter(coordinates.get_exp_conditions))
        ].columns[0]

    # Init condition_values if not given
    if not condition_values:
        condition_values = np.unique(
            [
                str(val.loc[:, exp_condition].values[0])
                for key, val in coordinates.get_exp_conditions.items()
            ]
        )
    if len(condition_values) > 2:
        condition_values = condition_values[0:2]
        warning_message = (
            "\033[38;5;208m\n"
            "Warning! No exp conditions were chosen for comparison and the experiment contains more than two conditions!\n"
            f"Therefore, the following conditions were set to be compared automatically: {condition_values}\n"
            "You can manually change this by setting condition_values explicitely with a list of two conditions."
            "\033[0m"
        )
        warnings.warn(warning_message)

    # Init plot type based on inputs
    if (
        any([embeddings is None, soft_counts is None])
        and supervised_annotations is not None
    ):
        plot_type = "supervised"
        L_shortest = min(
            get_dt(supervised_annotations,key,only_metainfo=True)['num_rows'] for key in supervised_annotations.keys()
        )
    elif (
        embeddings is not None
        and soft_counts is not None
        and supervised_annotations is None
    ):
        plot_type = "unsupervised"
        L_shortest = min(
            get_dt(soft_counts,key,only_metainfo=True)['num_rows']  for key in soft_counts.keys()
        )
    else:
        raise ValueError(
            "This function only accepts either supervised or unsupervised annotations as inputs, not both at the same time!"
        )

    # Init bin ranges if not given
    if not custom_time_bins:
        custom_time_bins = deepof.visuals_utils.create_bin_pairs(
            L_shortest, N_time_bins
        )

    # Init hidden bins if not given
    if not hide_time_bins:
        hide_time_bins = [False] * len(custom_time_bins)
    elif not len(hide_time_bins) == len(custom_time_bins):
        raise ValueError(
            f'The variables "hide_time_bins" and "custom_time_bins" need to have the same length!'
        )

    # Set behavior ids
    if plot_type == "unsupervised":
        keys=list(soft_counts.keys())
        num_clusters=get_dt(soft_counts,keys[0],only_metainfo=True)['num_cols']
        behavior_ids = [f"Cluster {str(k)}" for k in range(0, num_clusters)]
        
    elif plot_type == "supervised":
        keys=list(supervised_annotations.keys())
        behavior_ids = get_dt(supervised_annotations,keys[0],only_metainfo=True)['columns']


    #####
    # Some validity checks and more formatting
    #####

    # Check validity of id
    if not (behavior_to_plot is not None and behavior_to_plot in behavior_ids):
        raise ValueError(
            f"The selected behavior '{behavior_to_plot}' is not valid! Please select one of the following:\n {behavior_ids}"
        )

    # Check custom_time_bin validity
    if len(
        custom_time_bins
    ) > 3 and all(  # list has at least 4 bins (less lead to failing of the interpol. function later)
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
            list(chain(*custom_time_bins)) == sorted(list(chain(*custom_time_bins)))
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
            f'At least 4 bins are required! If "custom_time_bins" is used, it needs to be a list of at least 4 elments with each element being a list!'
        )
    
    #####
    # Get ROI bin info
    ##### 

    if roi_number is not None:
        #create full time bins covering entire signal
        if supervised_annotations is not None:
            bin_info_time = _preprocess_time_bins(
            coordinates, None, None, None, 
            tab_dict_for_binning=supervised_annotations,
            )
        else:            
            bin_info_time = _preprocess_time_bins(
                coordinates, None, None, None,  
                tab_dict_for_binning=soft_counts,
            )
        # Create ROI bins
        roi_bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)



    #####
    # Collect data for plotting
    #####

    # Initialize table
    columns = ["time_bin", "exp_condition", behavior_to_plot]
    df = pd.DataFrame(columns=columns)

    # Iterate over all experiments via keys
    for key in keys:

        cond = coordinates.get_exp_conditions[key][exp_condition].values[0]
        #skip excluded experiment condition values
        if cond not in condition_values:
            continue

        if plot_type == "unsupervised":
            data_set=get_dt(soft_counts,key)
            if roi_number is not None:
                data_set=get_unsupervised_behaviors_in_roi(cur_unsupervised=data_set, local_bin_info=roi_bin_info[key],animal_ids=animals_in_roi)
            index_dict_fn = lambda x: x[
                :, int(re.search(r"\d+", behavior_to_plot).group())
            ]
        elif plot_type == "supervised":
            data_set=get_dt(supervised_annotations,key)
            if roi_number is not None:
                data_set=get_supervised_behaviors_in_roi(cur_supervised=data_set, local_bin_info=roi_bin_info[key],animal_ids=animals_in_roi, roi_mode=roi_mode)
            index_dict_fn = lambda x: x[
                behavior_to_plot
            ]  # Specialized index functions to handle differing data_snippet formatting

        # Iterate over all time bins and collect average behavior data for all bins over all exp conditions
        for i, (bin_start, bin_end) in enumerate(custom_time_bins):

            #get current snippet
            data_snippet=data_set[bin_start:bin_end]

            behavior_timebin = np.nansum(index_dict_fn(data_snippet))
            if normalize or behavior_to_plot == "speed":
                behavior_timebin = behavior_timebin / len(
                    index_dict_fn(data_snippet)
                )
            
            new_row = pd.DataFrame(
                [
                    {
                        "time_bin": i,
                        "exp_condition": str(cond),
                        behavior_to_plot: behavior_timebin,
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)
    get_unsupervised_behaviors_in_roi._warning_issued = False

    # Normalize frames to reflect seconds
    df[behavior_to_plot] = df[behavior_to_plot] / coordinates._frame_rate
    
    assert np.sum(df[behavior_to_plot])>0.000001, "None of the selected behavior was measured within the given time bins and ROI!"    

    # Calculate mean values and errors accross samples
    time_bin_means = (
        df.groupby(["time_bin", "exp_condition"]).mean(numeric_only=True).reset_index()
    )
    if error_bars == "sem":
        time_bin_err = (
            df.groupby(["time_bin", "exp_condition"])
            .sem(numeric_only=True)
            .reset_index()
        )
    else:
        time_bin_err = (
            df.groupby(["time_bin", "exp_condition"])
            .std(numeric_only=True)
            .reset_index()
        )

    # Estimate effect sizes based on cohens d
    hourly_effect_sizes_df = pd.DataFrame(
        columns=["time_bin", "Absolute_Cohens_d", "Effect_Size_Category"]
    )
    for k in range(0, len(custom_time_bins)):
        # Extract arrays for both exp conditions and time bins
        array_a = df.loc[
            (df["exp_condition"] == condition_values[0]) & (df["time_bin"] == k),
            behavior_to_plot,
        ].values
        array_b = df.loc[
            (df["exp_condition"] == condition_values[1]) & (df["time_bin"] == k),
            behavior_to_plot,
        ].values
        d = abs(deepof.visuals_utils.cohend(array_a, array_b))  # Calc d
        d_effect_size = deepof.visuals_utils.cohend_effect_size(d)  # Est. effect size
        # Collect data
        new_row = pd.DataFrame(
            [
                {
                    "time_bin": k,
                    "Absolute_Cohens_d": d,
                    "Effect_Size_Category": d_effect_size,
                }
            ]
        )
        hourly_effect_sizes_df = pd.concat(
            [hourly_effect_sizes_df, new_row], ignore_index=True
        )

    # Extract mean and error values for chosen behavior
    mean_values = [
        time_bin_means[time_bin_means["exp_condition"] == condition_values[0]][
            behavior_to_plot
        ].values,
        time_bin_means[time_bin_means["exp_condition"] == condition_values[1]][
            behavior_to_plot
        ].values,
    ]
    error_values = [
        time_bin_err[time_bin_err["exp_condition"] == condition_values[0]][
            behavior_to_plot
        ].values,
        time_bin_err[time_bin_err["exp_condition"] == condition_values[1]][
            behavior_to_plot
        ].values,
    ]

    #####
    # Handle present or absent axes of different types
    #####

    show = False
    # Update active axes, if axes are given
    if ax and polar_depiction:
        # Switch out the input axes object with a polar axis
        fig = ax.figure
        position = ax.get_position()
        fig.delaxes(ax)
        new_ax = fig.add_axes(position, projection="polar")
        # Update the original ax reference to point to the new axis
        ax.__dict__.clear()
        ax.__dict__.update(new_ax.__dict__)
        ax.__class__ = new_ax.__class__
        # Replace the new_ax with ax in the figure's axes list
        fig.axes[fig.axes.index(new_ax)] = ax
        del new_ax

    elif ax and not polar_depiction:
        plt.sca(ax)

    elif polar_depiction:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
        show = True

    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        show = True

    #####
    # Stats
    #####

    # Get stats annotations if required
    if add_stats:

        # Initialize a set to keep track of seen pairs
        pairs = df.groupby("time_bin").apply(
            lambda x: list(dict.fromkeys(zip(x["time_bin"], x["exp_condition"])))
        )
        # Exclude hidden bins and convert to list
        pairs = pairs[np.invert(hide_time_bins)].tolist()
        # Do actual testing with annotator package
        annotator = Annotator(
            ax,
            pairs=pairs,
            data=df,
            x="time_bin",
            y=behavior_to_plot,
            hue="exp_condition",
            hide_non_significant=True,
        )
        annotator.configure(
            test=add_stats,
            text_format="star",
            loc="inside",
            comparisons_correction="fdr_bh",
            verbose=False,
        )
        # Automatic annotiation to plots does not work with polar plots
        # Hence test results get extracted manually and collected in a dict
        test_dict = {}
        anni = annotator.apply_test()
        for annotation in anni.annotations:
            test_dict[annotation.structs[0]["group"][0]] = annotation.text

    #####
    # Line plot
    #####

    sns.set_style("whitegrid")
    num_bins = len(custom_time_bins)

    # Define the angles and mid_angles (angle in the middle of two angles) for each bin
    lengths = [sublist[1] - sublist[0] for sublist in custom_time_bins]
    cumsum_lengths = np.cumsum([0] + lengths)
    angles = cumsum_lengths[:-1] / cumsum_lengths[-1] * 2 * np.pi
    rotation = angles[0]
    # Ensure that no angle exceeds 2 pi
    angles = np.mod(angles + rotation, 2 * np.pi)
    mid_angles = np.mod(
        angles + np.diff(np.concatenate((angles, [angles[0] + 2 * np.pi]))) / 2,
        2 * np.pi,
    )

    # Define colors for each group
    colors = ["#1f77b4", "#ff7f0e"]

    # Init boolean mask to hide data segments based on hide_time_bins input
    mask = np.full(
        ((len(mid_angles) - 1) * 10 - len(mid_angles) + 2), False, dtype=bool
    )
    smooth_mean_angles = np.linspace(
        mid_angles[0], mid_angles[-1], (len(mid_angles) - 1) * 10 - len(mid_angles) + 2
    )
    int_pos = np.argmin(np.abs(smooth_mean_angles[:, np.newaxis] - mid_angles), axis=0)
    # Iterate over all bins
    for i in range(0, len(hide_time_bins)):
        if hide_time_bins[i]:
            if i < len(hide_time_bins) - 1:
                mask[int_pos[i] : int_pos[i + 1] - 1] = True
            if i > 0:
                mask[int_pos[i - 1] + 1 : int_pos[i]] = True

    # Plot the mean value lines for each group
    for i in range(2):
        # Interpolate the data to create a smooth line
        interp_func = interp1d(mid_angles, mean_values[i], kind="cubic")
        smooth_mean_values = interp_func(smooth_mean_angles)
        masked_angles = np.ma.masked_array(smooth_mean_angles, mask)  # mask lines
        masked_values = np.ma.masked_array(smooth_mean_values, mask)
        ax.plot(
            masked_angles,
            masked_values,
            linewidth=3,
            label=f"{[condition_values[0],condition_values[1]][i]}",
            color=colors[i],
            linestyle="-",
            alpha=0.8,
        )

    # Plot markers for each group
    marker_handles = [0, 0]
    for i in range(2):
        masked_mid_angles = np.ma.masked_array(mid_angles, hide_time_bins)
        masked_mean_values = np.ma.masked_array(mean_values[i], hide_time_bins)
        marker_handles[i] = ax.plot(
            masked_mid_angles,
            masked_mean_values,
            marker="o",
            linestyle="",
            color=ax.lines[i].get_color(),
            linewidth=2,
        )  # Use the same color as the line

    # Interpolate error bars
    smooth_err_values = []
    for i in range(2):
        interp_sem_func = interp1d(mid_angles, error_values[i], kind="cubic")
        smooth_err_values.append(
            interp_sem_func(mid_angles)
        )  # Use the original angles array

    # Plot the error as lines above and below the mean values
    for i in range(2):
        ax.plot(
            masked_mid_angles,
            mean_values[i] + smooth_err_values[i],
            linestyle="",
            color=colors[i],
            alpha=0.8,
        )
        ax.plot(
            masked_mid_angles,
            mean_values[i] - smooth_err_values[i],
            linestyle="",
            color=colors[i],
            alpha=0.8,
        )

    # Shade error
    for i in range(2):
        ax.fill_between(
            masked_mid_angles,
            mean_values[i] + smooth_err_values[i],
            mean_values[i] - smooth_err_values[i],
            color=colors[i],
            alpha=0.15,
        )

    # Set custom ticks and labels for the y axes
    ax.set_title(f"DeepOF - {behavior_to_plot}", fontsize=18, y=1.15)
    max_value = np.max(mean_values)
    y_ticks = np.arange(0, max_value * 1.5, max_value * 1.5 / 6)
    ax.set_yticks(y_ticks)

    # Set custom xticklabels
    xticklabels = [str(i) for i in range(1, num_bins + 1)]

    # Special modifications for the polar plot
    if polar_depiction:

        # Set xticks to angles and hide labels
        ax.set_xticks(angles)
        ax.set_xticklabels([])
        # Set the direction of angle 0 and labels to the top
        ax.set_theta_zero_location("N")
        ax.set_rlabel_position(0)
        # Set the direction to clockwise
        ax.set_theta_direction(-1)
        # Start position of histograms on y axis
        top = max_value * 1.5  # change inside circle size

        # Add legend of first part of plot
        legend_1 = ax.legend(
            handles=[marker_handles[0][0], marker_handles[1][0]],
            labels=[condition_values[0], condition_values[1]],
            fontsize=12,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.1),
        )
    else:
        # Set xticks to mid_angles and display labels
        ax.set_xticks(mid_angles)
        ax.set_xticklabels(xticklabels)
        # Start position of histograms on y axis
        top = ax.get_ylim()[0]

        # Add axis labels
        ax.set_xlabel("Time Bins", fontsize=12)
        ax.set_ylabel(f"{behavior_to_plot} [s]", fontsize=12)

        # Add legend
        legend_1 = ax.legend(
            handles=[marker_handles[0][0], marker_handles[1][0]],
            labels=[condition_values[0], condition_values[1]],
            fontsize=12,
            loc="upper right",
        )

    ax.add_artist(legend_1)

    #####
    # Histogram
    #####

    # Some inits
    ax.grid(True)
    values = hourly_effect_sizes_df["Effect_Size_Category"] * max_value * 0.1
    num_bins = len(values)
    # Calculate widths of histogram bars
    widths = lengths / np.sum(lengths) * (2 * np.pi)

    # Set colors
    cmap = [
        "#9370DB",
        "#6A5ACD",
        "#4B0082",
    ]  # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    colors = [
        cmap[val]
        for val in hourly_effect_sizes_df["Effect_Size_Category"].astype(int).values - 1
    ]
    for k in range(0, len(colors)):
        if hide_time_bins[k]:
            colors[k] = "#C0C0C0"
            values[k] = 1 * max_value * 0.1

    # Plot histogram if required
    stat_text_col = "k"
    if show_histogram:
        bars = ax.bar(mid_angles, values, width=widths, bottom=top)
        # Change color of text of stat annotations for better contrast
        stat_text_col = "#FFFF00"

        # Use custom colors and opacity
        for color, bar in zip(colors, bars):
            bar.set_facecolor(color)
            bar.set_alpha(0.8)

        # create legend for hist with color patches
        bar_handles = [0, 0, 0]
        legend_labels = ["large", "medium", "small"]
        legend_colors = cmap[::-1]
        for i, label in enumerate(legend_labels):
            bar_handles[i] = Patch(color=legend_colors[i], label=label)

    # Special modifications for the polar plot
    if polar_depiction:

        # Add xticklabels manually for each circle segment in the middle between ticks
        for midangle, label in zip(mid_angles, xticklabels):
            ax.text(midangle, ax.get_rmax() * 1.05, label, ha="center", va="center")

        # Add stat annotations as text in plot
        if add_stats:
            z = 0
            # Add annotation for each circle segment
            for label in test_dict:
                ax.text(
                    mid_angles[int(label)] + 0.02,
                    ax.get_yticks()[-1]
                    + (ax.get_yticks()[-1] - ax.get_yticks()[-2]) * 1.166,
                    test_dict[label],
                    ha="center",
                    va="center",
                    fontsize="small",
                    color=stat_text_col,
                    rotation=-np.flip(mid_angles[int(label)] * 180 / np.pi),
                )
                z += 1
        # Update limits
        lower_lim = ax.get_ylim()[0]
        ax.set_rlim(lower_lim, ax.get_rmax())

        # Only show histogram legend if required
        if show_histogram:
            legend_2 = ax.legend(
                handles=[bar_handles[0], bar_handles[1], bar_handles[2]],
                title="Effect Size",
                loc="upper left",
                bbox_to_anchor=(0.0, 1.1),
                fontsize=8,
            )
            ax.add_artist(legend_2)
    else:

        # Add stat annotations as text in plot
        if add_stats:
            z = 0
            # Add annotation for each plot segment
            for label in test_dict:
                ax.text(
                    mid_angles[int(label)],
                    (ax.get_yticks()[-1] - ax.get_yticks()[-2]) * 0.166,
                    test_dict[label],
                    ha="center",
                    va="center",
                    color=stat_text_col,
                    fontsize="small",
                )
                z += 1

        # Only show histogram legend if required
        if show_histogram:
            ax.legend(
                handles=[bar_handles[0], bar_handles[1], bar_handles[2]],
                title="Effect Size",
                loc="upper left",
                fontsize=8,
            )  # , bbox_to_anchor=(1.3, 0.65), fontsize=8)

    # Save plot if required
    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_time_plot{}_behavior={}_error_bars={}_test={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    behavior_to_plot,
                    error_bars,
                    add_stats,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    # If no axes are given, show plot
    if show:
        plt.show()


def get_roi_data(
    coordinates: coordinates,
    table_dict: table_dict,
    # ROI functionality
    roi_number: int,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    # Time selection parameters
    bin_index: Union[int, str] = None,
    bin_size: Union[int, str] = None,
    precomputed_bins: np.ndarray = None,
    samples_max: int =100000,
    # Visualization parameters
    experiment_id: str = None,
):
    """get data in Rois.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        table_dict (table_dict): table dict with information for ROi extraction. Can be supervised or unsupervised data.
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded)       
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                                                  
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
        bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
        bin_size (Union[int,str]): bin size for time filtering.
        precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.     
        experiment_id (str): Name of the experiment id to extract. If None (default) a dictionary of all entries will be exported.

    """
    # initial check if enum-like inputs were given correctly
    _check_enum_inputs(
        coordinates,
        experiment_ids=experiment_id,
        animals_in_roi=animals_in_roi,
        roi_number=roi_number,
        roi_mode=roi_mode,
    )
    if coordinates._very_large_project and experiment_id is None:
        raise NotImplementedError(
            "Iteration accross all experiments is currently not supported for very large projects! However, by setting \"experiment_id\" you can still export single tables"
        )

    if animals_in_roi is None or roi_mode == "behaviorwise":
        animals_in_roi = coordinates._animal_ids
    elif roi_number is None:
        print(
        '\033[33mInfo! For this plot animal_id is only relevant if a ROI was selected!\033[0m'
        )
    # Get requested experimental condition. If none is provided, default to the first one available.
    if experiment_id is None:
        exp_ids = list(table_dict.keys())
    else:
        exp_ids = [experiment_id]

    # Preprocess information given for time binning
    bin_info_time = _preprocess_time_bins(
        coordinates, bin_size, bin_index, precomputed_bins, 
        tab_dict_for_binning=table_dict, samples_max=samples_max,
    )

    bin_info = _apply_rois_to_bin_info(coordinates, roi_number, bin_info_time)    

    object_out={}
    for key in exp_ids:
        tab = get_dt(table_dict, key)
        if type(tab)==pd.DataFrame:
            supervised_binned = pd.DataFrame(tab.iloc[bin_info[key]["time"]])
            time_binned = get_supervised_behaviors_in_roi(supervised_binned, bin_info[key], animals_in_roi, roi_mode=roi_mode)
        elif type(tab)==np.ndarray:
            unsupervised_binned = tab[bin_info[key]["time"]]
            time_binned = get_unsupervised_behaviors_in_roi(unsupervised_binned, bin_info[key], animals_in_roi)
        else:
            raise NotImplementedError(
                "This function only supports supervised and unsupervides table dicts!"
            )
        if len(exp_ids)>1:
            object_out[key]=time_binned
        else:
            object_out=time_binned

    get_unsupervised_behaviors_in_roi._warning_issued = False

    return object_out
