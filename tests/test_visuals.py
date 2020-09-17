# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals

"""


from deepof.utils import *
import deepof.preprocess
import deepof.visuals
import matplotlib.figure


def test_plot_heatmap():
    prun = (
        deepof.preprocess.project(
            path=os.path.join(".", "tests", "test_examples"),
            arena="circular",
            arena_dims=tuple([380]),
            angles=False,
            video_format=".mp4",
            table_format=".h5",
        )
        .run()
        .get_coords()
    )

    assert (
        type(
            deepof.visuals.plot_heatmap(
                prun["test"],
                ["Center"],
                tuple([-100, 100]),
                tuple([-100, 100]),
                dpi=200,
            )
        )
        == matplotlib.figure.Figure
    )


def test_model_comparison_plot():
    prun = (
        deepof.preprocess.project(
            path=os.path.join(".", "tests", "test_examples"),
            arena="circular",
            arena_dims=tuple([380]),
            angles=False,
            video_format=".mp4",
            table_format=".h5",
        )
        .run()
        .get_coords()
    )

    gmm_run = gmm_model_selection(
        prun["test"], n_components_range=range(1, 3), n_runs=1, part_size=100
    )

    assert (
        type(
            deepof.visuals.model_comparison_plot(
                gmm_run[0], gmm_run[1], range(1, 3), cov_plot="full"
            )
        )
        == matplotlib.figure.Figure
    )
