"""General plotting functions for the deepof package."""
import collections

# @author lucasmiranda42
# encoding: utf-8
# module deepof

from collections import defaultdict, Sequence
from itertools import cycle, product, combinations
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from statannotations.Annotator import Annotator
from typing import Any, List, NewType, Union
import calendar
import copy
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import shap
import tensorflow as tf
import time
import umap
import warnings

import deepof.post_hoc

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# PLOTTING FUNCTIONS #


def plot_arena(
    coordinates: coordinates, center: str, color: str, ax: Any, i: Union[int, str]
):
    """Plots the arena in the given canvas.

    Args:
        coordinates (coordinates): deepof Coordinates object.
        center (str): Name of the body part to which the positions will be centered. If false,
        the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        color (str): color of the displayed arena.
        ax (Any): axes where to plot the arena.
        i (Union[int, str]): index of the animal to plot.
    """
    if isinstance(i, np.int64):
        arena = coordinates._arena_params[i]

    if "circular" in coordinates._arena:

        if i == "average":
            arena = [
                np.mean(np.array([i[0] for i in coordinates._arena_params]), axis=0),
                np.mean(np.array([i[1] for i in coordinates._arena_params]), axis=0),
                np.mean(np.array([i[2] for i in coordinates._arena_params]), axis=0),
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

        if center == "arena" and i == "average":
            arena = np.stack(coordinates._arena_params)
            arena -= np.expand_dims(
                np.array(coordinates._scales[:, :2]).astype(int), axis=1
            )
            arena = arena.mean(axis=0)

        elif center == "arena":
            arena -= np.expand_dims(
                np.array(coordinates._scales[i, :2]).astype(int), axis=1
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
    xlim: tuple,
    ylim: tuple,
    title: str,
    save: str = False,
    dpi: int = 200,
    ax: Any = None,
    **kwargs,
) -> plt.figure:
    """Returns a heatmap of the movement of a specific bodypart in the arena.

    If more than one bodypart is passed, it returns one subplot for each.

    Args:
        dframe (pandas.DataFrame): table_dict value with info to plot
        bodyparts (List): bodyparts to represent (at least 1)
        xlim (float): limits of the x-axis
        ylim (float): limits of the y-axis
        save (str): if provided, saves the figure to the specified file.
        dpi (int): dots per inch of the figure to create.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided,
        new figure will be created.

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

    for i, bpart in enumerate(bodyparts):
        heatmap = dframe[bpart]

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
                x=heatmap.x, y=heatmap.y, cmap="magma", fill=True, alpha=1, ax=ax
            )
            ax = np.array([ax])

    for x, bp in zip(ax, bodyparts):
        x.set_xlim(xlim)
        x.set_ylim(ylim)
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


# noinspection PyTypeChecker
def plot_heatmaps(
    coordinates: coordinates,
    bodyparts: list,
    center: str = "arena",
    align: str = None,
    exp_condition: str = None,
    display_arena: bool = True,
    xlim: float = None,
    ylim: float = None,
    save: bool = False,
    experiment_id: int = "average",
    dpi: int = 100,
    ax: Any = None,
    show: bool = True,
    **kwargs,
) -> plt.figure:  # pragma: no cover
    """Plots heatmaps of the specified body parts (bodyparts) of the specified animal (i).

    Args:
        coordinates (coordinates): deepof Coordinates object.
        bodyparts (list): list of body parts to plot.
        center (str): Name of the body part to which the positions will be centered. If false,
        the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        align (str): Selects the body part to which later processes will align the frames with
        (see preprocess in table_dict documentation).
        exp_condition (str): Experimental condition to plot. If available, it filters the experiments
        to keep only those whose condition matches the given string.
        display_arena (bool): whether to plot a dashed line with an overlying arena perimeter. Defaults to True.
        xlim (float): x-axis limits.
        ylim (float): y-axis limits.
        save (str):  if provided, the figure is saved to the specified path.
        experiment_id (str): index of the animal to plot.
        dpi (int): resolution of the figure.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided,
        a new figure will be created.
        show (bool): whether to show the created figure. If False, returns al axes.

    Returns:
        heatmaps (plt.figure): figure with the specified characteristics
    """
    coords = coordinates.get_coords(center=center, align=align)

    if exp_condition is not None:
        coords = coords.filter_videos(
            [k for k, v in coordinates.get_exp_conditions.items() if v == exp_condition]
        )

    if not center:  # pragma: no cover
        warnings.warn("Heatmaps look better if you center the data")

    # Add experimental conditions to title, if provided
    title_suffix = experiment_id
    if coordinates.get_exp_conditions is not None and exp_condition is None:
        title_suffix += (
            " - " + coordinates.get_exp_conditions[list(coords.keys())[experiment_id]]
        )

    elif exp_condition is not None:
        title_suffix += f" - {exp_condition}"

    if experiment_id != "average":

        i = np.argmax(np.array(list(coords.keys())) == experiment_id)
        coords = coords[experiment_id]

    else:
        i = experiment_id
        coords = pd.concat([val for val in coords.values()], axis=0).reset_index(
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
            plot_arena(coordinates, center, "white", hmap, i)

    if show:
        plt.show()
    else:
        return heatmaps


def plot_gantt(
    coordinates: project,
    experiment_id: str,
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    additional_checkpoints: pd.DataFrame = None,
    save: bool = False,
):
    """Returns a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation,
    and changepoint detection size visualization.

    Args:
        coordinates (project): deepOF project where the data is stored.
        experiment_id (str): Name of the experiment to display.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        supervised_annotations (table_dict): table dict with supervised annotations per video.
        new figure will be created.
        additional_checkpoints (pd.DataFrame): table with additional checkpoints to plot.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """  # TODO: extend to supervised annotations, and add time below
    # Determine plot type
    if soft_counts is None and supervised_annotations is not None:
        plot_type = "supervised"
    elif soft_counts is not None and supervised_annotations is None:
        plot_type = "unsupervised"
    else:
        plot_type = "mixed"

    hard_counts = soft_counts[experiment_id].argmax(axis=1)
    gantt = np.zeros([hard_counts.max() + 1, hard_counts.shape[0]])

    # If available, add additional checkpoints to the Gantt matrix
    if additional_checkpoints is not None:
        additional_checkpoints = additional_checkpoints.iloc[:, : gantt.shape[1]]
        gantt = np.concatenate([gantt, additional_checkpoints], axis=0)

    colors = np.tile(
        list(sns.color_palette("tab20").as_hex()), int(np.ceil(gantt.shape[0] / 20))
    )

    # Iterate over unsupervised clusters and plot
    rows = 0
    for cluster, color in zip(range(hard_counts.max() + 1), colors):
        gantt[cluster] = hard_counts == cluster
        gantt_cp = gantt.copy()
        gantt_cp[[i for i in range(gantt.shape[0]) if i != cluster]] = np.nan
        plt.axhline(y=cluster, color="k", linewidth=0.5)
        rows += 1

        sns.heatmap(
            data=gantt_cp,
            cbar=False,
            cmap=LinearSegmentedColormap.from_list("deepof", ["white", color], N=2),
        )

    # Iterate over additional checkpoints and plot
    if additional_checkpoints is not None:
        for checkpoint in range(additional_checkpoints.shape[0]):
            gantt_cp = gantt.copy()
            gantt_cp[
                [i for i in range(gantt.shape[0]) if i != rows + checkpoint]
            ] = np.nan
            plt.axhline(y=rows + checkpoint, color="k", linewidth=0.5)

            sns.heatmap(
                data=gantt_cp,
                cbar=False,
                cmap=LinearSegmentedColormap.from_list(
                    "deepof", ["white", "black"], N=2
                ),
            )

    plt.xticks([])
    plt.yticks(
        np.array(range(gantt.shape[0])) + 0.5,
        # Concatenate cluster IDs and checkpoint names if they exist
        np.concatenate(
            [
                np.arange(hard_counts.max() + 1),
                np.array(additional_checkpoints.index)
                if additional_checkpoints is not None
                else [],
            ]
        ),
        rotation=0,
        fontsize=10,
    )

    plt.axhline(y=0, color="k", linewidth=1)
    plt.axhline(y=gantt.shape[0], color="k", linewidth=2)
    plt.axvline(x=0, color="k", linewidth=1)
    plt.axvline(x=gantt.shape[1], color="k", linewidth=2)

    plt.xlabel("Time", fontsize=10)
    plt.ylabel("Cluster", fontsize=10)

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

    title = "deepOF - Gantt chart of {} behaviors - {}".format(plot_type, experiment_id)
    plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_cluster_enrichment(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict,
    breaks: table_dict = None,
    add_stats: str = "Mann-Whitney",
    # Quality selection parameters
    min_confidence: float = 0.0,
    # Time selection parameters
    bin_size: int = None,
    bin_index: int = 0,
    precomputed: np.ndarray = None,
    # Visualization parameters
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
        breaks (table_dict): table dict with changepoint detection breaks per experiment.
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one
        available is used.
        exp_condition_order (list): Order in which to plot experimental conditions. If None (default), the order
        is determined by the order of the keys in the table dict.
        min_confidence (float): minimum confidence in cluster assignments used for quality control filtering.
        bin_size (int): bin size for time filtering.
        bin_index (int): index of the bin of size bin_size to select along the time dimension.
        precomputed (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
        add_stats (bool): whether to add stats to the plots. Defaults to True.
        may hurt performance.
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.
        normalize (bool): whether to represent time fractions or actual time in seconds on the y axis.

    """
    # Get requested experimental condition. If none is provided, default to the first one available.
    if exp_condition is None:
        exp_conditions = {
            key: val.iloc[:, 0].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        }
    else:
        exp_conditions = {
            key: val.loc[:, exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        }

    # Get cluster enrichment across conditions for the desired settings
    enrichment = deepof.post_hoc.cluster_enrichment_across_conditions(
        embedding=embeddings,
        soft_counts=soft_counts,
        breaks=breaks,
        exp_conditions=exp_conditions,
        bin_size=(coordinates._frame_rate * bin_size if bin_size is not None else None),
        bin_index=bin_index,
        precomputed=precomputed,
        normalize=normalize,
    )

    if exp_condition_order is not None:
        enrichment["exp condition"] = pd.Categorical(
            enrichment["exp condition"], exp_condition_order
        )
        enrichment.sort_values(by=["exp condition", "cluster"], inplace=True)

    enrichment["cluster"] = enrichment["cluster"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot a barchart grouped per experimental conditions
    sns.barplot(
        data=enrichment,
        x="cluster",
        y="time on cluster",
        hue="exp condition",
        ax=ax,
    )
    sns.stripplot(
        data=enrichment,
        x="cluster",
        y="time on cluster",
        hue="exp condition",
        color="black",
        ax=ax,
        dodge=True,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[2:], labels[2:], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
    )

    if add_stats:
        pairs = list(
            product(
                set(np.concatenate(list(soft_counts.values())).argmax(axis=1)),
                set(exp_conditions.values()),
            )
        )
        pairs = [
            list(map(tuple, p))
            for p in np.array(pairs)
            .reshape([-1, 2, len(set(exp_conditions.values()))])
            .tolist()
        ]

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
        annotator.apply_and_annotate()

    if save:
        plt.savefig(
            os.path.join(
                coordinates._project_path,
                coordinates._project_name,
                "Figures",
                "deepof_enrichment{}_min_conf={}_bin_size={}_bin_index={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    min_confidence,
                    bin_size,
                    bin_index,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    title = "deepOF - cluster enrichment"

    if ax is not None:
        plt.title(title, fontsize=15)
    else:
        ax.set_title(title, fontsize=15)
        plt.tight_layout()
        plt.show()


def plot_transitions(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict,
    breaks: table_dict = None,
    # Time selection parameters
    bin_size: int = None,
    bin_index: int = 0,
    # Visualization parameters
    exp_condition: str = None,
    visualization="networks",
    silence_diagonal=False,
    cluster: bool = True,
    axes: list = None,
    save: bool = False,
    **kwargs,
):
    """Computes and plots transition matrices for all data or per condition. Plots can be heatmaps or networks.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        breaks (table_dict): table dict with changepoint detection breaks per experiment.
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one
        available is used.
        bin_size (int): bin size for time filtering.
        bin_index (int): index of the bin of size bin_size to select along the time dimension.
        new figure will be created.
        visualization (str): visualization mode. Can be either 'networks', or 'heatmaps'.
        silence_diagonal (bool): If True, diagonals are set to zero.
        cluster (bool): If True (default) rows and columns on heatmaps are hierarchically clustered.
        axes (list): axes where to plot the current figure. If not provided, a new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # Get requested experimental condition. If none is provided, default to the first one available.
    if exp_condition is None:
        exp_conditions = exp_condition
    else:
        exp_conditions = {
            key: val.loc[:, exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        }

    grouped_transitions = deepof.post_hoc.compute_transition_matrix_per_condition(
        embeddings,
        soft_counts,
        breaks,
        exp_conditions,
        bin_size=bin_size,
        bin_index=bin_index,
        silence_diagonal=silence_diagonal,
        aggregate=(exp_conditions is not None),
        normalize=True,
    )

    if exp_conditions is None:
        grouped_transitions = np.mean(
            np.concatenate(
                [np.expand_dims(i, axis=0) for i in grouped_transitions.values()]
            ),
            axis=0,
        )

    # Use seaborn to plot heatmaps across both conditions
    if axes is None:
        fig, axes = plt.subplots(
            1,
            (len(set(exp_conditions.values())) if exp_conditions is not None else 1),
            figsize=(16, 8),
        )

    if not isinstance(axes, np.ndarray) and not isinstance(axes, Sequence):
        axes = [axes]

    if exp_conditions is not None:
        iters = zip(set(exp_conditions.values()), axes)
    else:
        iters = zip([None], axes)

    if visualization == "networks":

        for exp_condition, ax in iters:

            try:
                G = nx.DiGraph(grouped_transitions[exp_condition])
            except:
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

        for exp_condition, ax in iters:

            if cluster:
                if isinstance(grouped_transitions, dict):
                    clustered_transitions = grouped_transitions[exp_condition]
                else:
                    clustered_transitions = grouped_transitions
                # Cluster rows and columns and reorder
                row_link = linkage(
                    clustered_transitions, method="average", metric="euclidean"
                )  # computing the linkage
                row_order = dendrogram(row_link, no_plot=True)["leaves"]
                clustered_transitions = pd.DataFrame(clustered_transitions).iloc[
                    row_order, row_order
                ]

            sns.heatmap(
                clustered_transitions,
                cmap="coolwarm",
                vmin=0,
                vmax=0.35,
                ax=ax,
                **kwargs,
            )
            ax.set_title(exp_condition)

    if axes is None:

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


def plot_stationary_entropy(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict,
    breaks: table_dict = None,
    add_stats: str = "Mann-Whitney",
    # Time selection parameters
    bin_size: int = None,
    bin_index: int = 0,
    # Visualization parameters
    exp_condition: str = None,
    verbose: bool = False,
    ax: Any = None,
    save: bool = False,
):
    """Computes and plots transition stationary distribution entropy per condition.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        breaks (table_dict): table dict with changepoint detection breaks per experiment.
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one
        available is used.
        add_stats (bool): whether to add stats to the plots. Defaults to True.
        bin_size (int): bin size for time filtering.
        bin_index (int): index of the bin of size bin_size to select along the time dimension.
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided,
        new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # Get requested experimental condition. If none is provided, default to the first one available.
    if exp_condition is None:
        exp_conditions = {
            key: val.iloc[:, 0].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        }
    else:
        exp_conditions = {
            key: val.loc[:, exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        }

    # Get ungrouped entropy scores for the full videos
    ungrouped_transitions = deepof.post_hoc.compute_transition_matrix_per_condition(
        embeddings,
        soft_counts,
        breaks,
        exp_conditions,
        bin_size=bin_size,
        bin_index=bin_index,
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

    # Draw violin/strip plots with full-video entropy
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

    plt.show()


def plot_embeddings(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict,
    breaks: table_dict = None,
    # Quality selection parameters
    min_confidence: float = 0.0,
    # Time selection parameters
    bin_size: int = None,
    bin_index: int = 0,
    # Visualization design and data parameters
    exp_condition: str = None,
    use_keys: list = None,
    aggregate_experiments: str = False,
    samples: int = 500,
    show_aggregated_density: bool = True,
    colour_by: str = "cluster",
    show_break_size_as_radius: bool = False,
    ax: Any = None,
    save: bool = False,
):
    """Returns a scatter plot of the passed projection. Allows for temporal and quality filtering, animal aggregation,
    and changepoint detection size visualization.

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        breaks (table_dict): table dict with changepoint detection breaks per experiment.
        exp_condition (str): Name of the experimental condition to use when plotting. If None (default) the first one
        available is used.
        use_keys (list): list of keys to use for plotting. If None (default), all keys are used.
        min_confidence (float): minimum confidence in cluster assignments used for quality control filtering.
        bin_size (int): bin size for time filtering.
        bin_index (int): index of the bin of size bin_size to select along the time dimension.
        aggregate_experiments (str): Whether to aggregate embeddings by experiment (by time on cluster, mean, or median) or not (default).
        samples (int): Number of samples to take from the time embeddings. None leads to plotting all time-points, which
        may hurt performance.
        show_aggregated_density (bool): if True, a density plot is added to the aggregated embeddings.
        colour_by (str): hue by which to colour the embeddings. Can be one of 'cluster' (default), 'exp_condition', or 'exp_id'.
        show_break_size_as_radius (bool): Only usable when embeddings come from a model using changepoint detection. If True,
        the size of each chunk is depicted as the radius of each dot.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided,
        new figure will be created.
        save (bool): Saves a time-stamped vectorized version of the figure if True.

    """
    # Get experimental conditions per video
    if exp_condition is None:
        exp_condition = list(coordinates.get_exp_conditions.values())[0].columns[0]

    concat_hue = [
        i[exp_condition].values[0]
        for i in list(coordinates.get_exp_conditions.values())
    ]

    if use_keys is not None:
        embeddings = {key: embeddings[key] for key in use_keys}
        soft_counts = {key: soft_counts[key] for key in use_keys}
        breaks = {key: breaks[key] for key in use_keys}
        concat_hue = [
            j
            for i, j in zip(list(coordinates.get_exp_conditions.keys()), concat_hue)
            if i in use_keys
        ]

    # Restrict embeddings, soft_counts and breaks to the selected time bin
    if bin_size is not None:
        embeddings, soft_counts, breaks = deepof.post_hoc.select_time_bin(
            embeddings,
            soft_counts,
            breaks,
            coordinates._frame_rate * bin_size,
            bin_index,
        )

    # Keep only those experiments for which we have an experimental condition assigned
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
    breaks = {
        key: val
        for key, val in breaks.items()
        if key in coordinates.get_exp_conditions.keys()
    }

    # Plot unravelled temporal embeddings
    if not aggregate_experiments:

        if samples is not None:

            # Sample per animal, to avoid alignment issues
            for key in embeddings.keys():

                sample_ids = np.random.choice(
                    range(embeddings[key].shape[0]), samples, replace=False
                )
                embeddings[key] = embeddings[key][sample_ids]
                soft_counts[key] = soft_counts[key][sample_ids]
                breaks[key] = breaks[key][sample_ids]

        # Concatenate experiments and align experimental conditions
        concat_embeddings = np.concatenate(list(embeddings.values()), 0)

        # Concatenate breaks
        concat_breaks = tf.concat(list(breaks.values()), 0)

        # Get cluster assignments from soft counts
        cluster_assignments = np.argmax(
            np.concatenate(list(soft_counts.values()), 0), axis=1
        )

        # Compute confidence in assigned clusters
        confidence = np.concatenate(
            [np.max(val, axis=1) for val in soft_counts.values()]
        )

        break_lens = tf.stack([len(i) for i in list(breaks.values())], 0)

        # Reduce the dimensionality of the embeddings using UMAP. Set n_neighbors to a large
        # value to see a more global picture
        reducers = deepof.post_hoc.compute_UMAP(concat_embeddings, cluster_assignments)
        reduced_embeddings = reducers[1].transform(
            reducers[0].transform(concat_embeddings)
        )

        # Generate unifier dataset using the reduced embeddings, experimental conditions
        # and the corresponding break lengths and cluster assignments

        embedding_dataset = pd.DataFrame(
            {
                "UMAP-1": reduced_embeddings[:, 0],
                "UMAP-2": reduced_embeddings[:, 1],
                "exp_id": np.repeat(list(range(len(embeddings))), break_lens),
                "breaks": concat_breaks,
                "confidence": confidence,
                "cluster": cluster_assignments,
                "experimental condition": np.repeat(concat_hue, break_lens),
            }
        )

        # Filter values with low confidence
        embedding_dataset = embedding_dataset.loc[
            embedding_dataset.confidence > min_confidence
        ]
        embedding_dataset.sort_values("cluster", inplace=True)

    else:

        # Aggregate experiments by time on cluster
        if aggregate_experiments == "time on cluster":
            aggregated_embeddings = deepof.post_hoc.get_time_on_cluster(
                soft_counts, breaks, reduce_dim=True
            )

        else:
            aggregated_embeddings = deepof.post_hoc.get_aggregated_embedding(
                embeddings, agg=aggregate_experiments, reduce_dim=True
            )

        aggregated_embeddings = aggregated_embeddings.loc[
            (coordinates.get_exp_conditions.keys() if use_keys is None else use_keys), :
        ]

        # Generate unifier dataset using the reduced aggregated embeddings and experimental conditions
        embedding_dataset = pd.DataFrame(
            {
                "PCA-1": aggregated_embeddings[0],
                "PCA-2": aggregated_embeddings[1],
                "experimental condition": concat_hue,
            }
        )

        embedding_dataset.index = (
            coordinates.get_exp_conditions.keys() if use_keys is None else use_keys
        )
        embedding_dataset.sort_values("experimental condition", inplace=True)

    # Plot selected embeddings using the specified settings
    sns.scatterplot(
        data=embedding_dataset,
        x="{}-1".format("PCA" if aggregate_experiments else "UMAP"),
        y="{}-2".format("PCA" if aggregate_experiments else "UMAP"),
        ax=ax,
        hue=(
            "experimental condition"
            if aggregate_experiments or colour_by == "exp_contition"
            else colour_by
        ),
        size=(
            "breaks"
            if show_break_size_as_radius and not aggregate_experiments
            else None
        ),
        s=(50 if not aggregate_experiments else 100),
        edgecolor="black",
        palette=(
            None if aggregate_experiments or colour_by == "exp_condition" else "tab20"
        ),
    )

    if aggregate_experiments and show_aggregated_density:
        sns.kdeplot(
            data=embedding_dataset,
            x="PCA-1",
            y="PCA-2",
            hue="experimental condition",
            zorder=0,
            ax=ax,
        )

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
                "deepof_embeddings{}_colour={}_agg={}_min_conf={}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    colour_by,
                    aggregate_experiments,
                    min_confidence,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    title = "deepOF - unsupervised {}embedding".format(
        ("aggregated " if aggregate_experiments else "")
    )
    if ax is not None:
        ax.set_title(title, fontsize=15)

    else:
        plt.title(title, fontsize=15)
        plt.tight_layout()
        plt.show()


def _scatter_embeddings(
    embeddings: np.ndarray,
    cluster_assignments: np.ndarray = None,
    ax: Any = None,
    save: str = False,
    show: bool = True,
    dpi: int = 200,
) -> plt.figure:
    """Returns a scatter plot of the passed projection. Each dot represents the trajectory of an entire animal.

    If labels are propagated, it automatically colours all data points with their respective condition.

    Args:
        embeddings (tuple): sequence embeddings obtained with the unsupervised pipeline within deepof
        cluster_assignments (tuple): labels of the clusters. If None, aggregation method should be provided.
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
        c=(cluster_assignments if cluster_assignments is not None else None),
        cmap=("tab20" if cluster_assignments is not None else None),
        edgecolor="black",
        linewidths=0.25,
    )

    plt.tight_layout()

    if save:
        plt.savefig(save)

    if not show:
        return ax

    plt.show()


# noinspection PyTypeChecker
def animate_skeleton(
    coordinates: coordinates,
    experiment_id: str,
    animal_id: list = None,
    center: str = "arena",
    align: str = None,
    frame_limit: int = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    cluster_assignments: np.ndarray = None,
    embedding: Union[List, np.ndarray] = None,
    selected_cluster: np.ndarray = None,
    display_arena: bool = True,
    legend: bool = True,
    save: bool = None,
    dpi: int = 300,
):
    """Renders a FuncAnimation object with embeddings and/or motion trajectories over time.

    Args:
        coordinates (coordinates): deepof Coordinates object.
        experiment_id (str): Name of the experiment to display.
        animal_id (list): ID list of animals to display. If None (default) it shows all animals.
        center (str): Name of the body part to which the positions will be centered. If false,
        the raw data is returned; if 'arena' (default), coordinates are centered in the pitch.
        align (str): Selects the body part to which later processes will align the frames with
        (see preprocess in table_dict documentation).
        frame_limit (int): Number of frames to plot. If None, the entire video is rendered.
        min_confidence (float): Minimum confidence threshold to render a cluster assignment bout.
        min_bout_duration (int): Minimum number of frames to render a cluster assignment bout.
        cluster_assignments (np.ndarray): contain sorted cluster assignments for all instances in data.
        If provided together with selected_cluster, only instances of the specified component are returned.
        Defaults to None.
        only instances of the specified component are returned. Defaults to None.
        embedding (Union[List, np.ndarray]): UMAP 2D embedding of the datapoints provided. If not None, a second animation
        shows a parallel animation showing the currently selected embedding, colored by cluster if cluster_assignments
        are available.
        selected_cluster (int): cluster to filter. If provided together with cluster_assignments,
        display_arena (bool): whether to plot a dashed line with an overlying arena perimeter. Defaults to True.
        legend (bool): whether to add a color-coded legend to multi-animal plots. Defaults to True when there are more
        than one animal in the representation, False otherwise.
        save (str): name of the file where to save the produced animation.
        dpi (int): dots per inch of the figure to create.
    """
    # Get data to plot from coordinates object
    data = coordinates.get_coords(center=center, align=align)

    # Filter requested animals
    if animal_id:
        data = data.filter_id(animal_id)

    # Select requested experiment and frames
    data = data[experiment_id]

    # Sort column index to allow for multiindex slicing
    data = data.sort_index(ascending=True, inplace=False, axis=1)

    # Get output scale
    x_dv = np.maximum(
        np.abs(data.loc[:, (slice("x"), ["x"])].min().mean()),
        np.abs(data.loc[:, (slice("x"), ["x"])].max().mean()),
    )
    y_dv = np.maximum(
        np.abs(data.loc[:, (slice("x"), ["y"])].min().mean()),
        np.abs(data.loc[:, (slice("x"), ["y"])].max().mean()),
    )

    # Filter assignments and embeddings
    if isinstance(cluster_assignments, dict):
        cluster_confidence = cluster_assignments[experiment_id].max(axis=1)
        cluster_assignments = cluster_assignments[experiment_id].argmax(axis=1)
        confidence_indices = np.ones(cluster_assignments.shape[0], dtype=bool)

        # Compute bout lengths, and filter out bouts shorter than min_bout_duration
        full_confidence_indices = deepof.utils.filter_short_bouts(
            cluster_assignments,
            cluster_confidence,
            confidence_indices,
            min_confidence,
            min_bout_duration,
        )
        confidence_indices = full_confidence_indices.copy()

    if isinstance(embedding, dict):

        embedding = embedding[experiment_id]
        reducers = deepof.post_hoc.compute_UMAP(embedding, cluster_assignments)
        embedding = reducers[1].transform(reducers[0].transform(embedding))

    # Checks that all shapes and passed parameters are correct
    if embedding is not None:

        # Center sliding window instances
        try:
            win_size = data.shape[0] - embedding.shape[0]
        except AttributeError:
            win_size = data.shape[0] - embedding[0].shape[1]
        data = data[win_size // 2 : -win_size // 2]

        if isinstance(embedding, np.ndarray):
            assert (
                embedding.shape[0] == data.shape[0]
            ), "there should be one embedding per row in data"

            concat_embedding = embedding
            embedding = [embedding]

        elif isinstance(embedding, list):

            assert len(embedding) == len(coordinates._animal_ids)

            for emb in embedding:
                assert (
                    emb.shape[0] == data.shape[0]
                ), "there should be one embedding per row in data"

            concat_embedding = np.concatenate(embedding)

        if selected_cluster is not None:
            cluster_embedding = [embedding[0][cluster_assignments == selected_cluster]]
            confidence_indices = confidence_indices[
                cluster_assignments == selected_cluster
            ]

        else:
            cluster_embedding = embedding

    if cluster_assignments is not None:

        assert (
            len(cluster_assignments) == data.shape[0]
        ), "there should be one cluster assignment per row in data"

        # Filter data to keep only those instances assigned to a given cluster
        if selected_cluster is not None:

            assert selected_cluster in set(
                cluster_assignments
            ), "selected cluster should be in the clusters provided"

            data = data.loc[cluster_assignments == selected_cluster, :]
            data = data.loc[confidence_indices, :]
            cluster_embedding = [cluster_embedding[0][confidence_indices]]
            concat_embedding = concat_embedding[full_confidence_indices]
            cluster_assignments = cluster_assignments[full_confidence_indices]

    def get_polygon_coords(data, animal_id=""):
        """Generates polygons to animate for the indicated animal in the provided dataframe."""

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

    # Define canvas
    fig = plt.figure(figsize=((16 if embedding is not None else 8), 8), dpi=dpi)

    # If embeddings are provided, add projection plot to the left
    if embedding is not None:
        ax1 = fig.add_subplot(121)

        _scatter_embeddings(concat_embedding, cluster_assignments, ax1, show=False)

        # Plot current position
        umap_scatter = {}
        for i, emb in enumerate(embedding):
            umap_scatter[i] = ax1.scatter(
                emb[0, 0],
                emb[0, 1],
                color=(
                    "red"
                    if len(embedding) == 1
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
    ax2 = fig.add_subplot((122 if embedding is not None else 111))

    # Plot!
    init_x = data.loc[:, (slice("x"), ["x"])].iloc[0, :]
    init_y = data.loc[:, (slice("x"), ["y"])].iloc[0, :]

    # If there are more than one animal in the representation, display each in a different color
    hue = None
    cmap = ListedColormap(sns.color_palette("tab10", len(coordinates._animal_ids)))

    if not animal_id and coordinates._animal_ids[0]:
        animal_ids = coordinates._animal_ids

    else:
        animal_ids = [animal_id]

    polygons = [get_polygon_coords(data, aid) for aid in animal_ids]

    if animal_id is None:
        hue = np.zeros(len(np.array(init_x)))
        for i, id in enumerate(coordinates._animal_ids):

            hue[data.columns.levels[0].str.startswith(id)] = i

            # Set a custom legend outside the plot, with the color of each animal

            if legend:
                custom_labels = [
                    plt.scatter(
                        [np.inf],
                        [np.inf],
                        color=cmap(i / len(coordinates._animal_ids)),
                        lw=3,
                    )
                    for i in range(len(coordinates._animal_ids))
                ]
                ax2.legend(custom_labels, coordinates._animal_ids, loc="upper right")

    skeleton_scatter = ax2.scatter(
        x=np.array(init_x),
        y=np.array(init_y),
        cmap=(cmap if animal_id is None else None),
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
        i = np.argmax(np.array(list(coordinates.get_coords().keys())) == experiment_id)
        plot_arena(coordinates, center, "black", ax2, i)

    # Update data in main plot
    def animation_frame(i):

        if embedding is not None:
            # Update umap scatter
            for j, xy in umap_scatter.items():
                umap_x = cluster_embedding[j][i, 0]
                umap_y = cluster_embedding[j][i, 1]

                umap_scatter[j].set_offsets(np.c_[umap_x, umap_y])

        # Update skeleton scatter plot
        x = data.loc[:, (slice("x"), ["x"])].iloc[i, :]
        y = data.loc[:, (slice("x"), ["y"])].iloc[i, :]

        skeleton_scatter.set_offsets(np.c_[x, y])

        for p, aid in enumerate(polygons):
            # Update polygons
            ax2.patches[2 * p].set_xy(aid[0][i, :].reshape(-1, 2))
            ax2.patches[2 * p + 1].set_xy(aid[1][i, :].reshape(-1, 2))

            # Update tails
            tail_lines[p][0].set_xdata(aid[2][i, :].reshape(-1, 2)[:, 0])
            tail_lines[p][0].set_ydata(aid[2][i, :].reshape(-1, 2)[:, 1])

        if embedding is not None:
            return umap_scatter, skeleton_scatter

        return skeleton_scatter

    animation = FuncAnimation(
        fig,
        func=animation_frame,
        frames=np.minimum(data.shape[0], frame_limit),
        interval=2000 // coordinates._frame_rate,
    )

    ax2.set_title(
        f"deepOF animation - {(f'{animal_id} - ' if animal_id is not None else '')}{experiment_id}",
        fontsize=15,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    if center not in [False, "arena"]:

        ax2.set_xlim(-1.5 * x_dv, 1.5 * x_dv)
        ax2.set_ylim(-1.5 * y_dv, 1.5 * y_dv)

    plt.tight_layout()

    if save is not None:
        writevideo = FFMpegWriter(fps=15)
        animation.save(save, writer=writevideo)

    return animation.to_html5_video()


def plot_cluster_detection_performance(
    coordinates: coordinates,
    chunk_stats: pd.DataFrame,
    cluster_gbm_performance: dict,
    hard_counts: np.ndarray,
    groups: list,
    save: bool = False,
    visualization: str = "confusion_matrix",
):
    """Plots either a confusion matrix or a bar chart with balanced accuracy for cluster detection cross validated models.
    Designed to be run after deepof.post_hoc.train_supervised_cluster_detectors (see documentation for details).

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        chunk_stats (pd.DataFrame): table with descriptive statistics for a series of sequences ('chunks').
        cluster_gbm_performance (dict): cross-validated dictionary containing trained estimators and performance metrics.
        hard_counts (np.ndarray): cluster assignments for the corresponding 'chunk_stats' table.
        groups (list): cross-validation indices. Data from the same animal are never shared between train and test sets.
        save:
        visualization (str): plot to render. Must be one of 'confusion_matrix', or 'balanced_accuracy'.

    """
    n_clusters = len(np.unique(hard_counts))
    confusion_matrices = []

    for clf, fold in zip(cluster_gbm_performance["estimator"], groups):
        cm = confusion_matrix(
            hard_counts.values[fold[1]],
            clf.predict(chunk_stats.values[fold[1]]),
            labels=np.unique(hard_counts),
        )

        confusion_matrices.append(cm)

    cluster_names = ["cluster {}".format(i) for i in range(10)]

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

        plt.title("Confusion matrix for multiclass state prediction")
        sns.heatmap(cm, annot=True, cmap="Blues")

        plt.yticks(rotation=0)

    elif visualization == "balanced_accuracy":

        def compute_balanced_accuracy(cm, cluster_index):
            """

            Computes balanced accuracy for a specific cluster given a confusion matrix

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

        plt.title("Supervised cluster mapping performance")

        sns.barplot(data=dataset, ci=95, color=sns.color_palette("Blues").as_hex()[-3])
        plt.axhline(1 / n_clusters, linestyle="--", color="black")

        plt.xlabel("Cluster")
        plt.ylabel("Balanced accuracy")

        plt.ylim(0, 1)

    else:
        raise ValueError(
            "Invalid plot selected. Visualization should be one of 'confusion_matrix' or 'balanced_accuracy'. See documentation for details."
        )

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
    plt.show()


def plot_shap_swarm_per_cluster(
    coordinates: coordinates,
    data_to_explain: pd.DataFrame,
    shap_values: list,
    cluster: Union[str, int] = "all",
    max_display: int = 10,
    save: str = False,
    show: bool = True,
):
    """

    Args:
        coordinates (coordinates): deepOF project where the data is stored.
        data_to_explain (pd.DataFrame): table with descriptive statistics for a series of sequences ('chunks').
        shap_values (list): shap_values per cluster.
        cluster: cluster to plot. If "all" (default) global feature importance
        across all clusters is depicted in a bar chart.
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
                "deepof_supervised_cluster_detection_SHAP{}_{}.pdf".format(
                    (f"_{save}" if isinstance(save, str) else ""),
                    visualization,
                    calendar.timegm(time.gmtime()),
                ),
            )
        )

    if show:
        plt.show()


def output_cluster_video(
    cap: Any,
    out: Any,
    frame_mask: list,
    v_width: int,
    v_height: int,
    path: str,
    frame_limit: int = np.inf,
):
    """Outputs a video with the frames corresponding to the cluster.

    Args:
        cap: video capture object
        out: video writer object
        frame_mask: list of booleans indicating whether a frame should be written
        v_width: video width
        v_height: video height
        path: path to the video file
        frame_limit: maximum number of frames to render

    """
    i = 0
    j = 0
    while cap.isOpened() and j < frame_limit:
        ret, frame = cap.read()
        if ret == False:
            break

        try:
            if frame_mask[i]:

                res_frame = cv2.resize(frame, [v_width, v_height])
                re_path = re.findall(".+/(.+)DLC", path)[0]

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
                j += 1

            i += 1
        except IndexError:
            ret = False

    cap.release()
    cv2.destroyAllWindows()


def output_videos_per_cluster(
    video_paths: list,
    breaks: list,
    soft_counts: list,
    frame_rate: int = 25,
    frame_limit_per_video: int = np.inf,
    single_output_resolution: tuple = None,
    window_length: int = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    out_path: str = ".",
):
    """Given a list of videos, and a list of soft counts per video, outputs a video for each cluster.

    Args:
        video_paths: list of paths to the videos
        breaks: list of breaks between videos
        soft_counts: list of soft counts per video
        frame_rate: frame rate of the videos
        frame_limit_per_video: number of frames to render per video.
        single_output_resolution: if single_output is provided, this is the resolution of the output video.
        window_length: window length used to compute the soft counts.
        min_confidence: minimum confidence threshold for a frame to be considered part of a cluster.
        min_bout_duration: minimum duration of a bout to be considered.
        out_path: path to the output directory.

    """
    # Iterate over all clusters, and output a masked video for each
    for cluster_id in range(soft_counts[0].shape[1]):

        out = cv2.VideoWriter(
            os.path.join(
                out_path,
                "deepof_unsupervised_annotation_cluster={}_threshold={}_{}.mp4".format(
                    cluster_id, min_confidence, calendar.timegm(time.gmtime())
                ),
            ),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            single_output_resolution,
        )

        for i, path in enumerate(video_paths):

            # Get hard counts and confidence estimates per cluster
            hard_counts = np.argmax(soft_counts[i], axis=1)
            confidence = np.max(soft_counts[i], axis=1)
            confidence_indices = np.ones(hard_counts.shape[0], dtype=bool)

            # Given a frame mask, output a subset of the given video to disk, corresponding to a particular cluster
            cap = cv2.VideoCapture(path)
            v_width, v_height = single_output_resolution

            # Compute confidence mask, filtering out also bouts that are too short
            confidence_indices = deepof.utils.filter_short_bouts(
                hard_counts,
                confidence,
                confidence_indices,
                min_confidence,
                min_bout_duration,
            )
            confidence_mask = (hard_counts == cluster_id) & confidence_indices

            # Extend confidence mask using the corresponding breaks, to select and output all relevant video frames
            # Add a prefix of zeros to the mask, to account for the frames lost by the sliding window
            frame_mask = np.repeat(confidence_mask, breaks[i])
            frame_mask = np.concatenate(
                (np.zeros(window_length, dtype=bool), frame_mask)
            )

            output_cluster_video(
                cap,
                out,
                frame_mask,
                v_width,
                v_height,
                path,
                frame_limit_per_video,
            )


def output_unsupervised_annotated_video(
    video_path: str,
    breaks: list,
    soft_counts: np.ndarray,
    frame_rate: int = 25,
    frame_limit: int = np.inf,
    window_length: int = None,
    cluster_names: dict = {},
    out_path: str = ".",
):
    """Given a video, and soft_counts per frame, outputs a video with the frames annotated with the cluster they belong to.

    Args:
        video_path: full path to the video
        breaks: dictionary with break lengths for each video
        soft_counts: soft cluster assignments for a specific video
        frame_rate: frame rate of the video
        frame_limit: maximum number of frames to output.
        window_length: window length used to compute the soft counts.
        cluster_names: dictionary with user-defined names for each cluster (useful to output interpretation).
        out_path: out_path: path to the output directory.

    """
    # Get cluster assignment per frame
    hard_counts = np.argmax(soft_counts, axis=1)
    assignments_per_frame = np.repeat(hard_counts, breaks)

    # Name clusters, and update names using the provided dictionary
    cluster_labels = {i: str(i) for i in set(hard_counts)}
    cluster_labels.update(cluster_names)

    # Given a frame mask, output a subset of the given video to disk, corresponding to a particular cluster
    cap = cv2.VideoCapture(video_path)

    # Get width and height of current video
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = os.path.join(
        out_path,
        video_path[:-4].split("/")[-1]
        + "_unsupervised_annotated_{}.mp4".format(calendar.timegm(time.gmtime())),
    )

    out = cv2.VideoWriter(
        video_out, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (v_width, v_height)
    )

    i, j = 0, 0
    while cap.isOpened() and i < frame_limit:
        if j >= window_length:
            j += 1

        else:
            ret, frame = cap.read()
            if ret == False:
                break

            try:
                cv2.putText(
                    frame,
                    "Cluster {}".format(cluster_labels[assignments_per_frame[i]]),
                    (int(v_width * 0.3 / 10), int(v_height / 1.05)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                out.write(frame)

                i += 1

            except IndexError:
                ret = False

    cap.release()
    cv2.destroyAllWindows()


def output_supervised_annotated_video():
    """Given a video, and supervised assignments per frame, outputs a video with the frames annotated.

    Args:

    """
    pass


def export_annotated_video(
    coordinates: coordinates,
    soft_counts: dict = None,
    breaks: dict = None,
    experiment_id: str = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    frame_limit_per_video: int = np.inf,
    exp_conditions: dict = {},
    cluster_names: dict = {},
):
    """Generic function to export annotated videos from both supervised and unsupervised pipelines.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        soft_counts (dict): dictionary with soft_counts per experiment.
        breaks (dict): dictionary with break lengths for each video.r
        experiment_id (str): if provided, data coming from a particular experiment is used. If not, all experiments are exported.
        min_confidence (float): minimum confidence threshold for a frame to be considered part of a cluster.
        min_bout_duration (int): Minimum number of frames to render a cluster assignment bout.
        frame_limit_per_video (int): number of frames to render per video. If None, all frames are included for all videos.
        exp_conditions (dict): if provided, data coming from a particular condition is used. If not, all conditions are exported.
        If a dictionary with more than one entry is provided, the intersection of all conditions (i.e. male, stressed) is used.
        cluster_names (dict): dictionary with user-defined names for each cluster (useful to output interpretation).

    """
    # Create output directory if it doesn't exist
    proj_path = os.path.join(coordinates._project_path, coordinates._project_name)
    out_path = os.path.join(proj_path, "Out_videos")
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Compute sliding window lenth, to determine the frame/annotation offset
    first_key = list(coordinates.get_quality().keys())[0]
    window_length = (
        coordinates.get_quality()[first_key].shape[0]
        - soft_counts[first_key].shape[0]
        + 1
    )

    def filter_experimental_conditions(
        coordinates: coordinates, videos: list, conditions: list
    ):
        """Returns a list of videos that match the provided experimental conditions."""

        filtered_videos = videos

        for condition, state in conditions.items():

            filtered_videos = [
                video
                for video in filtered_videos
                if state
                == np.array(
                    coordinates.get_exp_conditions[re.findall("(.+)DLC", video)[0]][
                        condition
                    ]
                )
            ]

        return filtered_videos

    # Unsupervised annotation output
    if soft_counts is not None:
        if experiment_id is not None:
            # If experiment_id is provided, only output a video for that experiment
            deepof.visuals.output_unsupervised_annotated_video(
                os.path.join(
                    proj_path,
                    "Videos",
                    [
                        video
                        for video in coordinates.get_videos()
                        if experiment_id in video
                    ][0],
                ),
                breaks[experiment_id],
                soft_counts[experiment_id],
                frame_rate=coordinates._frame_rate,
                window_length=window_length,
                cluster_names=cluster_names,
                out_path=out_path,
                frame_limit=frame_limit_per_video,
            )
        else:
            # If experiment_id is not provided, output a video per cluster for each experiment
            filtered_videos = filter_experimental_conditions(
                coordinates, coordinates.get_videos(), exp_conditions
            )

            deepof.visuals.output_videos_per_cluster(
                [
                    os.path.join(
                        proj_path,
                        "Videos",
                        video,
                    )
                    for video in filtered_videos
                ],
                [
                    val
                    for key, val in breaks.items()
                    if key
                    in [re.findall("(.+)DLC", video)[0] for video in filtered_videos]
                ],
                [
                    val
                    for key, val in soft_counts.items()
                    if key
                    in [re.findall("(.+)DLC", video)[0] for video in filtered_videos]
                ],
                frame_rate=coordinates._frame_rate,
                single_output_resolution=(500, 500),
                window_length=window_length // 2,
                frame_limit_per_video=frame_limit_per_video,
                min_confidence=min_confidence,
                min_bout_duration=min_bout_duration,
                out_path=out_path,
            )

    # Supervised annotation output
    else:
        raise NotImplementedError


def plot_distance_between_conditions(
    # Model selection parameters
    coordinates: coordinates,
    embedding: dict,
    soft_counts: dict,
    breaks: dict,
    exp_condition: str,
    embedding_aggregation_method: str = "median",
    distance_metric: str = "wasserstein",
    n_jobs: int = -1,
    save: bool = False,
    ax: Any = None,
):
    """Plots the distance between conditions across a growing time window. Finds an optimal separation binning based on the
    distance between conditions, and plots the distance between conditions across all non-overlapping bins. Useful, for example,
    to measure habituation across time.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        embedding (dict): embedding object for the current project. Used to get video paths.
        soft_counts (dict): dictionary with soft_counts per experiment.
        breaks (dict): dictionary with break lengths for each video.
        exp_condition (str): experimental condition to use for the distance calculation.
        embedding_aggregation_method (str): method to use for aggregating the embedding. Options are 'time_on_cluster' and 'mean'.
        distance_metric (str): distance metric to use for the distance calculation. Options are 'wasserstein' and 'euclidean'.
        n_jobs (int): number of jobs to use for the distance calculation.
        save (bool): if True, saves the figure to the project directory.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided, new figure will be created.
    """
    # Get distance between distributions across the growing window
    distance_array = deepof.post_hoc.condition_distance_binning(
        embedding,
        soft_counts,
        breaks,
        {
            key: val[exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        },
        10 * coordinates._frame_rate,
        np.min([val.shape[0] for val in soft_counts.values()]),
        coordinates._frame_rate,
        agg=embedding_aggregation_method,
        metric=distance_metric,
        n_jobs=n_jobs,
    )

    optimal_bin = np.argmax(savgol_filter(distance_array, 10, 2)) + 10
    print("Found an optimal_bin at {} seconds".format(optimal_bin))

    distance_per_bin = deepof.post_hoc.condition_distance_binning(
        embedding,
        soft_counts,
        breaks,
        {
            key: val[exp_condition].values[0]
            for key, val in coordinates.get_exp_conditions.items()
        },
        10 * coordinates._frame_rate,
        np.min([val.shape[0] for val in soft_counts.values()]),
        optimal_bin * coordinates._frame_rate,
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
                np.min([val.shape[0] for val in soft_counts.values()]),
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
                        np.min([val.shape[0] for val in soft_counts.values()])
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


def tag_annotated_frames(
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
):
    """Helper function for annotate_video. Annotates a given frame with on-screen information
    about the recognised patterns"""

    arena, w, h = arena

    def write_on_frame(text, pos, col=(255, 255, 255)):
        """Partial closure over cv2.putText to avoid code repetition"""
        return cv2.putText(frame, text, pos, font, 0.75, col, 2)

    def conditional_flag():
        """Returns a tag depending on a condition"""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return left_flag
        return right_flag

    def conditional_pos():
        """Returns a position depending on a condition"""
        if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
            return corners["downleft"]
        return corners["downright"]

    def conditional_col(cond=None):
        """Returns a colour depending on a condition"""
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

        if tag_dict["nose2nose"][fnum]:
            write_on_frame("Nose-Nose", conditional_pos())
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

        if tag_dict[animal_ids[0] + "_nose2body"][fnum] and left_flag:
            write_on_frame("nose2body", corners["downleft"])
            left_flag = False

        if tag_dict[animal_ids[1] + "_nose2body"][fnum] and right_flag:
            write_on_frame("nose2body", corners["downright"])
            right_flag = False

        if tag_dict[animal_ids[0] + "_nose2tail"][fnum] and left_flag:
            write_on_frame("Nose-Tail", corners["downleft"])
            left_flag = False

        if tag_dict[animal_ids[1] + "_nose2tail"][fnum] and right_flag:
            write_on_frame("Nose-Tail", corners["downright"])
            right_flag = False

        if tag_dict["sidebyside"][fnum] and left_flag and conditional_flag():
            write_on_frame("Side-side", conditional_pos())
            if frame_speeds[animal_ids[0]] > frame_speeds[animal_ids[1]]:
                left_flag = False
            else:
                right_flag = False

        if tag_dict["sidereside"][fnum] and left_flag and conditional_flag():
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

            if tag_dict[_id + undercond + "climbing"][fnum]:
                write_on_frame("climbing", down_pos)
            # elif tag_dict[_id + undercond + "huddle"][fnum]:
            #     write_on_frame("huddling", down_pos)
            elif tag_dict[_id + undercond + "sniffing"][fnum]:
                write_on_frame("sniffing", down_pos)

        # Define the condition controlling the colour of the speed display
        if len(animal_ids) > 1:
            colcond = frame_speeds[_id] == max(list(frame_speeds.values()))
        else:
            colcond = hparams["huddle_speed"] < frame_speeds

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
    tag_dict: pd.DataFrame,
    vid_index: int,
    frame_limit: int = np.inf,
    debug: bool = False,
    params: dict = {},
) -> True:
    """Renders a version of the input video with all supervised taggings in place.

    Parameters:
        - coordinates (deepof.preprocessing.coordinates): coordinates object containing the project information
        - debug (bool): if True, several debugging attributes (such as used body parts and arena) are plotted in
        the output video
        - vid_index: for internal usage only; index of the video to tag in coordinates._videos
        - frame_limit (float): limit the number of frames to output. Generates all annotated frames by default
        - params (dict): dictionary to overwrite the default values of the hyperparameters of the functions
        that the supervised pose estimation utilizes.

    Returns:
        True

    """

    # Extract useful information from coordinates object
    tracks = list(coordinates._tables.keys())
    videos = coordinates._videos
    path = os.path.join(coordinates._project_path, coordinates._project_name, "Videos")

    params = deepof.annotation_utils.get_hparameters(params)
    animal_ids = coordinates._animal_ids
    undercond = "_" if len(animal_ids) > 1 else ""

    try:
        vid_name = re.findall("(.*)DLC", tracks[vid_index])[0]
    except IndexError:
        vid_name = tracks[vid_index]

    arena_params = coordinates._arena_params[vid_index]
    h, w = coordinates._video_resolution[vid_index]
    corners = deepof.annotation_utils.frame_corners(h, w)

    cap = cv2.VideoCapture(os.path.join(path, videos[vid_index]))
    # Keep track of the frame number, to align with the tracking data
    fnum = 0
    writer = None
    frame_speeds = (
        {_id: -np.inf for _id in animal_ids} if len(animal_ids) > 1 else -np.inf
    )

    # Loop over the frames in the video
    while cap.isOpened() and fnum < frame_limit:

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:  # pragma: no cover
            print("Can't receive frame (stream end?). Exiting ...")
            break

        font = cv2.FONT_HERSHEY_DUPLEX

        # Capture speeds
        try:
            if (
                list(frame_speeds.values())[0] == -np.inf
                or fnum % params["speed_pause"] == 0
            ):
                for _id in animal_ids:
                    frame_speeds[_id] = tag_dict[_id + undercond + "speed"][fnum]
        except AttributeError:
            if frame_speeds == -np.inf or fnum % params["speed_pause"] == 0:
                frame_speeds = tag_dict["speed"][fnum]

        # Display all annotations in the output video
        tag_annotated_frames(
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
            coordinates.get_coords(center=False)[vid_name],
        )

        if writer is None:
            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            # Define the FPS. Also frame size is passed.
            writer = cv2.VideoWriter()
            writer.open(
                os.path.join(
                    coordinates._project_path,
                    coordinates._project_name,
                    "Out_videos",
                    vid_name + "_supervised_tagged.avi",
                ),
                cv2.VideoWriter_fourcc(*"MJPG"),
                coordinates._frame_rate,
                (frame.shape[1], frame.shape[0]),
                True,
            )

        writer.write(frame)
        fnum += 1

    cap.release()
    cv2.destroyAllWindows()

    return True
