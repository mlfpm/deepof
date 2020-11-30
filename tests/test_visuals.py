# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals

"""

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from deepof.utils import *
import deepof.data
import deepof.visuals
import matplotlib.figure


@settings(deadline=None)
@given(bparts=st.one_of(st.just(["Center"]), st.just(["Center", "Nose"])))
def test_plot_heatmap(bparts):
    prun = (
        deepof.data.project(
            path=os.path.join(".", "tests", "test_examples"),
            arena="circular",
            arena_dims=tuple([380]),
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
                bparts,
                tuple([-100, 100]),
                tuple([-100, 100]),
                dpi=200,
            )
        )
        == matplotlib.figure.Figure
    )


def test_model_comparison_plot():
    prun = (
        deepof.data.project(
            path=os.path.join(".", "tests", "test_examples"),
            arena="circular",
            arena_dims=tuple([380]),
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
