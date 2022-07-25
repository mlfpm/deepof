# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

General plotting functions for the deepof package

"""

from itertools import cycle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# PLOTTING FUNCTIONS #


def plot_heatmap(
    dframe: pd.DataFrame,
    bodyparts: List,
    xlim: tuple,
    ylim: tuple,
    save: str = False,
    dpi: int = 200,
) -> plt.figure:
    """Returns a heatmap of the movement of a specific bodypart in the arena.
    If more than one bodypart is passed, it returns one subplot for each

     Parameters:
         - dframe (pandas.DataFrame): table_dict value with info to plot
         - bodyparts (List): bodyparts to represent (at least 1)
         - xlim (float): limits of the x-axis
         - ylim (float): limits of the y-axis
         - save (str): name of the file to which the figure should be saved
         - dpi (int): dots per inch of the returned image

     Returns:
         - heatmaps (plt.figure): figure with the specified characteristics"""

    # noinspection PyTypeChecker
    heatmaps, ax = plt.subplots(1, len(bodyparts), sharex=True, sharey=True, dpi=dpi)

    for i, bpart in enumerate(bodyparts):
        heatmap = dframe[bpart]
        if len(bodyparts) > 1:
            sns.kdeplot(
                x=heatmap.x, y=heatmap.y, cmap=None, shade=True, alpha=1, ax=ax[i]
            )
        else:
            sns.kdeplot(x=heatmap.x, y=heatmap.y, cmap=None, shade=True, alpha=1, ax=ax)
            ax = np.array([ax])

    for x, bp in zip(ax, bodyparts):
        x.set_xlim(xlim)
        x.set_ylim(ylim)
        x.set_title(bp)

    if save:  # pragma: no cover
        plt.savefig(save)

    return heatmaps


def plot_projection(projection: tuple, save=False, dpi=200) -> plt.figure:
    """
    Returns a scatter plot of the passed projection. Each dot represents the trajectory of an entire animal.
    If labels are propagated, it automatically colours all data points with their respective condition.

     Args:
         - projection (tuple): tuple containing the projection and the associated conditions when available
         - save (str): name of the file to which the figure should be saved
         - dpi (int): dots per inch of the returned image

     Returns:
         - projection_scatter (plt.figure): figure with the specified characteristics
    """

    pass


def plot_unsupervised_embeddings(
    embeddings,
    exp_labels=None,
    cluster_labels=None,
    aggregation_method=None,
    save=False,
    dpi=200,
) -> plt.figure:
    """
    Returns a scatter plot of the passed projection. Each dot represents the trajectory of an entire animal.
    If labels are propagated, it automatically colours all data points with their respective condition.

     Parameters:
         - embeddings (tuple): sequence embeddings obtained with the unsupervised pipeline within deepof
         - exp_labels (tuple): labels of the experiments. If None, aggregation method must be None as well.
         - cluster_labels (tuple): labels of the clusters. If None, aggregation method should be provided.
         - aggregation_method (str): method to aggregate the data. If None, exp_labels must be None as well.
         Must be one of [None, "mean", "cluster_population"].

     Returns:
         - projection_scatter (plt.figure): figure with the specified characteristics"""

    pass
