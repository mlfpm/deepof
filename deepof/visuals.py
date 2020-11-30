# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

General plotting functions for the deepof package

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
from typing import List


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
            sns.kdeplot(heatmap.x, heatmap.y, cmap=None, shade=True, alpha=1, ax=ax[i])
        else:
            sns.kdeplot(heatmap.x, heatmap.y, cmap=None, shade=True, alpha=1, ax=ax)
            ax = np.array([ax])

    [x.set_xlim(xlim) for x in ax]
    [x.set_ylim(ylim) for x in ax]
    [x.set_title(bp) for x, bp in zip(ax, bodyparts)]

    if save:  # pragma: no cover
        plt.savefig(save)

    return heatmaps


def model_comparison_plot(
    bic: list,
    m_bic: list,
    n_components_range: range,
    cov_plot: str,
    save: str = False,
    cv_types: tuple = ("spherical", "tied", "diag", "full"),
    dpi: int = 200,
) -> plt.figure:
    """

    Plots model comparison statistics for Gaussian Mixture Model analysis.
    Similar to https://scikit-learn.org/stable/modules/mixture.html, it shows
    an upper panel with BIC per number of components and covariance matrix type
    in a bar plot, and a lower panel with box plots showing bootstrap runs of the
    models corresponding to one of the covariance types.

        Parameters:
            - bic (list): list with BIC for all used models
            - m_bic (list): list with minimum bic across cov matrices
            for all used models
            - n_components_range (range): range of components to evaluate
            - cov_plot (str): covariance matrix to use in the lower panel
            - save (str): name of the file to which the figure should be saved
            - cv_types (tuple): tuple indicating which covariance matrix types
            to use. All (spherical, tied, diag and full) used by default.
            - dpi (int): dots per inch of the returned image

        Returns:
            - modelcomp (plt.figure): figure with all specified characteristics

    """

    m_bic = np.array(m_bic)
    color_iter = cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    bars = []

    # Plot the BIC scores
    modelcomp = plt.figure(dpi=dpi)
    spl = plt.subplot(2, 1, 1)
    covplot = np.repeat(cv_types, len(m_bic) / 4)

    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            spl.bar(
                xpos,
                m_bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                color=color,
                width=0.2,
            )
        )

    spl.set_xticks(n_components_range)
    plt.title("BIC score per model")
    xpos = (
        np.mod(m_bic.argmin(), len(n_components_range))
        + 0.5
        + 0.2 * np.floor(m_bic.argmin() / len(n_components_range))
    )
    # noinspection PyArgumentList
    spl.text(xpos, m_bic.min() * 0.97 + 0.1 * m_bic.max(), "*", fontsize=14)
    spl.legend([b[0] for b in bars], cv_types)
    spl.set_ylabel("BIC value")

    spl2 = plt.subplot(2, 1, 2, sharex=spl)
    spl2.boxplot(list(np.array(bic)[covplot == cov_plot]), positions=n_components_range)
    spl2.set_xlabel("Number of components")
    spl2.set_ylabel("BIC value")

    if save:  # pragma: no cover
        plt.savefig(save)

    return modelcomp
