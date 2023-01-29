"""General plotting functions for the deepof package."""
# @author lucasmiranda42
# encoding: utf-8
# module deepof

from itertools import cycle, product, combinations
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statannotations.Annotator import Annotator
from typing import Any, List, NewType, Union
import calendar
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import seaborn as sns
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
    colors = np.tile(
        list(sns.color_palette("tab20").as_hex()), int(np.ceil(gantt.shape[0] / 20))
    )

    for cluster, color in zip(range(hard_counts.max() + 1), colors):
        gantt[cluster] = hard_counts == cluster
        gantt_cp = gantt.copy()
        gantt_cp[[i for i in range(hard_counts.max()) if i != cluster]] = np.nan
        plt.axhline(y=cluster, color="k", linewidth=0.5)

        sns.heatmap(
            data=gantt_cp,
            cbar=False,
            cmap=LinearSegmentedColormap.from_list("deepof", ["white", color], N=2),
        )

    plt.xticks([])
    plt.yticks(
        np.array(range(hard_counts.max() + 1)) + 0.5,
        range(hard_counts.max() + 1),
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
    # Visualization parameters
    exp_condition: str = None,
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
        min_confidence (float): minimum confidence in cluster assignments used for quality control filtering.
        bin_size (int): bin size for time filtering.
        bin_index (int): index of the bin of size bin_size to select along the time dimension.
        add_stats (bool): whether to add stats to the plots. Defaults to True.
        may hurt performance.
        verbose (bool): if True, prints test results and p-value cutoffs. False by default.
        ax (plt.AxesSubplot): axes where to plot the current figure. If not provided,
        new figure will be created.
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
        normalize=normalize,
    )

    enrichment["cluster"] = enrichment["cluster"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot a barchart grouped per experimental conditions
    sns.violinplot(
        data=enrichment,
        x="cluster",
        y="time on cluster",
        hue="exp condition",
        ax=ax,
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

    if ax is None:
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
    save: bool = False,
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

    grouped_transitions = deepof.post_hoc.compute_transition_matrix_per_condition(
        embeddings,
        soft_counts,
        breaks,
        exp_conditions,
        bin_size=bin_size,
        bin_index=bin_index,
        silence_diagonal=silence_diagonal,
        aggregate=True,
        normalize=True,
    )

    # Use seaborn to plot heatmaps across both conditions
    fig, axes = plt.subplots(1, len(set(exp_conditions.values())), figsize=(16, 8))

    if visualization == "networks":

        for exp_condition, ax in zip(set(exp_conditions.values()), axes):

            G = nx.DiGraph(grouped_transitions[exp_condition])
            weights = [G[u][v]["weight"] * 10 for u, v in G.edges()]

            pos = nx.circular_layout(G, scale=1, center=None, dim=2)
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
            )
            ax.set_title(exp_condition)

    elif visualization == "heatmaps":

        for exp_condition, ax in zip(set(exp_conditions.values()), axes):

            if cluster:
                clustered_transitions = grouped_transitions[exp_condition]
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
            )
            ax.set_title(exp_condition)

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
            coordinates.get_exp_conditions.keys(), :
        ]

        # Generate unifier dataset using the reduced aggregated embeddings and experimental conditions
        embedding_dataset = pd.DataFrame(
            {
                "PCA-1": aggregated_embeddings[0],
                "PCA-2": aggregated_embeddings[1],
                "experimental condition": concat_hue,
            }
        )

        embedding_dataset.index = coordinates.get_exp_conditions.keys()
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
            None
            if aggregate_experiments or colour_by == "exp_condition"
            else sns.color_palette("tab20").as_hex()[: len(set(cluster_assignments))]
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

    if not aggregate_experiments:
        if ax is None:
            plt.legend("", frameon=False)
        else:
            ax.get_legend().remove()

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
    animal_id: list = "",
    center: str = "arena",
    align: str = None,
    frame_limit: int = None,
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
        cluster_assignments = cluster_assignments[experiment_id].argmax(axis=1)

    if isinstance(embedding, dict):

        embedding = embedding[experiment_id]
        reducers = deepof.post_hoc.compute_UMAP(embedding, cluster_assignments)
        embedding = reducers[1].transform(reducers[0].transform(embedding))

    # Checks that all shapes and passed parameters are correct
    if embedding is not None:

        data = data[: embedding.shape[0]]

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

    def get_polygon_coords(data, animal_id=""):
        """Generates polygons to animate for the indicated animal in the provided dataframe."""

        if animal_id:
            animal_id += "_"

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

    if animal_id and coordinates._animal_ids[0]:
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
