# @author lucasmiranda42

import sys

sys.path.append("../")
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from itertools import combinations
from scipy.spatial import distance
from source.utils import *


@given(
    tab=data_frames(
        index=range_indexes(min_size=1),
        columns=columns(["X", "y"], dtype=float),
        rows=st.tuples(
            st.floats(
                min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
            ),
        ),
    )
)
def test_bp2polar(tab):
    polar = bp2polar(tab)
    assert np.allclose(polar["rho"], np.sqrt(tab["X"] ** 2 + tab["y"] ** 2))
    assert np.allclose(polar["phi"], np.arctan2(tab["y"], tab["X"]))


@given(
    mult=st.integers(min_value=1, max_value=10),
    cartdf=data_frames(
        index=range_indexes(min_size=1),
        columns=columns(["X", "y"], dtype=float),
        rows=st.tuples(
            st.floats(
                min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
            ),
        ),
    ),
)
def test_tab2polar(mult, cartdf):
    cart_df = pd.concat([cartdf] * mult, axis=0)
    idx = pd.MultiIndex.from_product(
        [list(cart_df.columns[: len(cart_df.columns) // 2]), ["X", "y"]],
        names=["bodyparts", "coords"],
    )
    cart_df.columns = idx

    assert cart_df.shape == tab2polar(cart_df).shape


@given(
    pair_array=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=1, max_value=1000),
            st.integers(min_value=4, max_value=4),
        ),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    ),
    arena_abs=st.integers(min_value=1, max_value=1000),
    arena_rel=st.integers(min_value=1, max_value=1000),
)
def test_compute_dist(pair_array, arena_abs, arena_rel):
    assert np.allclose(
        compute_dist(pair_array, arena_abs, arena_rel),
        pd.DataFrame(distance.cdist(pair_array[:, :2], pair_array[:, 2:]).diagonal())
        * arena_abs
        / arena_rel,
    )


@given(
    cordarray=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=2, max_value=5).map(lambda x: 4 * x),
        ),
        elements=st.floats(
            min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_bpart_distance(cordarray):
    cord_df = pd.DataFrame(cordarray)
    idx = pd.MultiIndex.from_product(
        [list(cord_df.columns[: len(cord_df.columns) // 2]), ["X", "y"]],
        names=["bodyparts", "coords"],
    )
    cord_df.columns = idx

    bpart = bpart_distance(cord_df)

    assert bpart.shape[0] == cord_df.shape[0]
    assert bpart.shape[1] == len(list(combinations(range(cord_df.shape[1] // 2), 2)))


@given(
    abc=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=3, max_value=3),
            st.integers(min_value=5, max_value=100),
            st.integers(min_value=2, max_value=2),
        ),
        elements=st.floats(
            min_value=1, max_value=10, allow_nan=False, allow_infinity=False
        ).map(lambda x: x + np.random.uniform(0, 10)),
    ),
)
def test_angle(abc):

    a, b, c = abc

    angles = []
    for i, j, k in zip(a, b, c):
        ang = np.arccos(
            (np.dot(i - j, k - j) / (np.linalg.norm(i - j) * np.linalg.norm(k - j)))
        )
        angles.append(ang)

    print(angle(a, b, c), np.array(angles))

    assert np.allclose(angle(a, b, c), np.array(angles))


@given(
    array=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=3, max_value=3),
            st.integers(min_value=5, max_value=100),
            st.integers(min_value=2, max_value=2),
        ),
        elements=st.floats(
            min_value=1, max_value=10, allow_nan=False, allow_infinity=False
        ).map(lambda x: x + np.random.uniform(0, 10)),
    )
)
def test_angle_trio(array):
    assert len(angle_trio(array)) == 3


@given(
    p=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=2, max_value=100),
            st.integers(min_value=2, max_value=2),
        ),
        elements=st.floats(
            min_value=1, max_value=10, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_rotate(p):

    assert np.allclose(rotate(p, 2 * np.pi), p)
    assert np.allclose(rotate(p, np.pi), -p)


@given(
    data=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=1, max_value=100),
            st.integers(min_value=3, max_value=100),
            st.integers(min_value=1, max_value=10).map(lambda x: 2 * x),
        ),
        elements=st.floats(
            min_value=1, max_value=10, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_align_trajectories(data):
    mode = np.random.choice(["center", "all"], 1)
    aligned = align_trajectories(data, mode)
    assert aligned.shape == data.shape
    if mode == "center":
        assert np.allclose(aligned[:, (data.shape[1] - 1) // 2, 0], 0)
    elif mode == "all":
        assert np.allclose(aligned[:, :, 0], 0)


@given(a=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=3, max_value=1000))))
def test_smooth_boolean_array(a):
    smooth = smooth_boolean_array(a)
    trans = lambda x: sum([i + 1 != i for i in range(x.shape[0] - 1)])
    assert trans(a) >= trans(smooth)


@given(
    a=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=1000, max_value=10000),
            st.integers(min_value=1, max_value=10).map(lambda x: 2 * x),
        ),
        elements=st.floats(
            min_value=1, max_value=10, allow_nan=False, allow_infinity=False
        ),
    ),
    window=st.data(),
)
def test_rolling_window(a, window):
    window_step = window.draw(st.integers(min_value=1, max_value=10))
    window_size = window.draw(
        st.integers(min_value=1, max_value=10).map(lambda x: x * window_step)
    )

    rolled_shape = rolling_window(a, window_size, window_step).shape

    assert len(rolled_shape) == a.shape + 1
    assert rolled_shape[1] == window_size
