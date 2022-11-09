# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.visuals

"""

import os

import matplotlib.figure
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import deepof.data
import deepof.utils
import deepof.visuals


@settings(deadline=None)
@given(bparts=st.one_of(st.just(["Center"]), st.just(["Center", "Nose"])))
def test_plot_heatmap(bparts):
    prun = (
        deepof.data.Project(
            path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
            arena="circular-autodetect",
            video_scale=tuple([380]),
            video_format=".mp4",
            table_format=".h5",
        )
        .run()
        .get_coords()
    )

    assert isinstance(
        deepof.visuals.heatmap(
            prun["test"], bparts, tuple([-100, 100]), tuple([-100, 100]), dpi=200
        ),
        matplotlib.figure.Figure,
    )
