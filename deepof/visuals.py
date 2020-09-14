# @author lucasmiranda42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
from typing import List, Dict


# PLOTTING FUNCTIONS #


def plot_speed(behaviour_dict: dict, treatments: Dict[List]) -> plt.figure:
    """Plots a histogram with the speed of the specified mouse.
       Treatments is expected to be a list of lists with mice keys per treatment"""

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))

    for Treatment, Mice_list in treatments.items():
        hist = pd.concat([behaviour_dict[mouse] for mouse in Mice_list])
        sns.kdeplot(hist["bspeed"], shade=True, label=Treatment, ax=ax1)
        sns.kdeplot(hist["wspeed"], shade=True, label=Treatment, ax=ax2)

    ax1.set_xlim(0, 7)
    ax2.set_xlim(0, 7)
    ax1.set_title("Average speed density for black mouse")
    ax2.set_title("Average speed density for white mouse")
    plt.xlabel("Average speed")
    plt.ylabel("Density")
    plt.show()


def plot_heatmap(
    dframe: pd.DataFrame, bodyparts: List, xlim: float, ylim: float, save: str = False
) -> plt.figure:
    """Returns a heatmap of the movement of a specific bodypart in the arena.
       If more than one bodypart is passed, it returns one subplot for each"""

    # noinspection PyTypeChecker
    fig, ax = plt.subplots(1, len(bodyparts), sharex=True, sharey=True)

    for i, bpart in enumerate(bodyparts):
        heatmap = dframe[bpart]
        if len(bodyparts) > 1:
            sns.kdeplot(heatmap.x, heatmap.y, cmap="jet", shade=True, alpha=1, ax=ax[i])
        else:
            sns.kdeplot(heatmap.x, heatmap.y, cmap="jet", shade=True, alpha=1, ax=ax)
            ax = np.array([ax])

    [x.set_xlim(xlim) for x in ax]
    [x.set_ylim(ylim) for x in ax]
    [x.set_title(bp) for x, bp in zip(ax, bodyparts)]

    if save:
        plt.savefig(save)

    plt.show()


def model_comparison_plot(
    bic: list,
    m_bic: list,
    n_components_range: range,
    cov_plot: str,
    save: str,
    cv_types: tuple = ("spherical", "tied", "diag", "full"),
) -> plt.figure:
    """Plots model comparison statistics over all tests"""

    m_bic = np.array(m_bic)
    color_iter = cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(12, 8))
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

    plt.tight_layout()

    if save:
        plt.savefig(save)

    plt.show()
