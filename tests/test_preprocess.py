# @author lucasmiranda42

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from scipy.spatial import distance
from deepof.utils import *
import deepof.preprocess
import pytest


@given(table_type=st.integers(min_value=0, max_value=2))
def test_project_init(table_type):

    table_type = [".h5", ".csv", ".foo"][table_type]

    prun = deepof.preprocess.project(
        path=os.path.join(".", "tests", "test_examples"),
        arena="circular",
        arena_dims=[380],
        angles=False,
        video_format=".mp4",
        table_format=table_type,
    )

    if table_type != ".foo":
        assert type(prun) == deepof.preprocess.project
        assert type(prun.load_tables(verbose=True)) == tuple
        print(prun)
    else:
        with pytest.raises(NotImplementedError):
            prun.load_tables(verbose=True)
