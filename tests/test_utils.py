# @author lucasmiranda42

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from scipy.spatial import distance
from deepof.utils import *
import deepof.preprocess
import pytest

# AUXILIARY FUNCTIONS #


def autocorr(x, t=1):
    return np.round(np.corrcoef(np.array([x[:-t], x[t:]]))[0, 1], 5)


# QUALITY CONTROL AND PREPROCESSING #


@settings(deadline=None)
@given(
    mult=st.integers(min_value=1, max_value=10),
    dframe=data_frames(
        index=range_indexes(min_size=1),
        columns=columns(["X", "y", "likelihood"], dtype=float),
        rows=st.tuples(
            st.floats(
                min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
            ),
            st.floats(
                min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        ),
    ),
    threshold=st.data(),
)
def test_likelihood_qc(mult, dframe, threshold):
    thresh1 = threshold.draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False))
    thresh2 = threshold.draw(
        st.floats(min_value=thresh1, max_value=1.0, allow_nan=False)
    )

    dframe = pd.concat([dframe] * mult, axis=0)
    idx = pd.MultiIndex.from_product(
        [list(dframe.columns[: len(dframe.columns) // 3]), ["X", "y", "likelihood"]],
        names=["bodyparts", "coords"],
    )
    dframe.columns = idx

    filt1 = likelihood_qc(dframe, thresh1)
    filt2 = likelihood_qc(dframe, thresh2)

    assert np.sum(filt1) <= dframe.shape[0]
    assert np.sum(filt2) <= dframe.shape[0]
    assert np.sum(filt1) >= np.sum(filt2)


@settings(deadline=None)
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


@settings(deadline=None)
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


@settings(deadline=None)
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


@settings(deadline=None)
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


@settings(deadline=None)
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


@settings(deadline=None)
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


@settings(deadline=None)
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
    assert np.allclose(rotate(p, 0), p)


@settings(deadline=None)
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
    ),
    mode_idx=st.integers(min_value=0, max_value=2),
)
def test_align_trajectories(data, mode_idx):
    mode = ["center", "all", "none"][mode_idx]
    aligned = align_trajectories(data, mode)
    assert aligned.shape == data.shape
    if mode == "center":
        assert np.allclose(aligned[:, (data.shape[1] - 1) // 2, 0], 0)
    elif mode == "all":
        assert np.allclose(aligned[:, :, 0], 0)
    elif mode == "none":
        assert np.allclose(aligned, data)


@settings(deadline=None)
@given(a=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=3, max_value=1000))))
def test_smooth_boolean_array(a):
    smooth = smooth_boolean_array(a)

    def trans(x):
        return sum([i + 1 != i for i in range(x.shape[0] - 1)])

    assert trans(a) >= trans(smooth)


@settings(deadline=None)
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

    assert len(rolled_shape) == len(a.shape) + 1
    assert rolled_shape[1] == window_size


@settings(deadline=None)
@given(
    alpha=st.data(),
    series=arrays(
        dtype=float,
        shape=st.tuples(st.integers(min_value=10, max_value=1000),),
        elements=st.floats(
            min_value=1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_smooth_mult_trajectory(alpha, series):
    alpha1 = alpha.draw(
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    alpha2 = alpha.draw(
        st.floats(
            min_value=alpha1, max_value=1.0, allow_nan=False, allow_infinity=False
        )
    )

    series *= +np.random.normal(0, 1, len(series))

    smoothed1 = smooth_mult_trajectory(series, alpha1)
    smoothed2 = smooth_mult_trajectory(series, alpha2)

    assert autocorr(smoothed1) >= autocorr(series)
    assert autocorr(smoothed2) >= autocorr(series)
    assert autocorr(smoothed2) <= autocorr(smoothed1)


# BEHAVIOUR RECOGNITION FUNCTIONS #


@settings(deadline=None)
@given(
    pos_dframe=data_frames(
        index=range_indexes(min_size=5),
        columns=columns(["X1", "y1", "X2", "y2"], dtype=float),
        rows=st.tuples(
            st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False),
        ),
    ),
    tol=st.floats(min_value=0.01, max_value=4.98),
)
def test_close_single_contact(pos_dframe, tol):

    idx = pd.MultiIndex.from_product(
        [["bpart1", "bpart2"], ["X", "y"]], names=["bodyparts", "coords"],
    )
    pos_dframe.columns = idx
    close_contact = close_single_contact(pos_dframe, "bpart1", "bpart2", tol)
    assert close_contact.dtype == bool
    assert np.array(close_contact).shape[0] <= pos_dframe.shape[0]


@settings(deadline=None)
@given(
    pos_dframe=data_frames(
        index=range_indexes(min_size=5),
        columns=columns(["X1", "y1", "X2", "y2", "X3", "y3", "X4", "y4"], dtype=float),
        rows=st.tuples(
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
        ),
    ),
    tol=st.floats(min_value=0.01, max_value=4.98),
    rev=st.booleans(),
)
def test_close_double_contact(pos_dframe, tol, rev):

    idx = pd.MultiIndex.from_product(
        [["bpart1", "bpart2", "bpart3", "bpart4"], ["X", "y"]],
        names=["bodyparts", "coords"],
    )
    pos_dframe.columns = idx
    close_contact = close_double_contact(
        pos_dframe, "bpart1", "bpart2", "bpart3", "bpart4", tol, rev
    )
    assert close_contact.dtype == bool
    assert np.array(close_contact).shape[0] <= pos_dframe.shape[0]


@settings(deadline=None)
@given(indexes=st.data())
def test_recognize_arena_and_subfunctions(indexes):

    path = "./tests/test_examples/Videos/"
    videos = [i for i in os.listdir(path) if i.endswith("mp4")]

    vid_index = indexes.draw(st.integers(min_value=0, max_value=len(videos) - 1))
    recoglimit = indexes.draw(st.integers(min_value=1, max_value=10))

    assert recognize_arena(videos, vid_index, path, recoglimit, "") == 0
    assert len(recognize_arena(videos, vid_index, path, recoglimit, "circular")) == 3


@settings(deadline=None)
@given(
    arena=st.lists(
        min_size=3, max_size=3, elements=st.integers(min_value=300, max_value=500)
    ),
    tol=st.data(),
)
def test_climb_wall(arena, tol):

    tol1 = tol.draw(st.floats(min_value=0.001, max_value=10))
    tol2 = tol.draw(st.floats(min_value=tol1, max_value=10))

    prun = (
        deepof.preprocess.project(
            path="./tests/test_examples",
            arena="circular",
            arena_dims=[arena[0]],
            angles=False,
            video_format=".mp4",
            table_format=".h5",
        )
        .run(verbose=False)
        .get_coords()
    )

    climb1 = climb_wall("circular", arena, prun["test"], tol1, nose="Nose")
    climb2 = climb_wall("circular", arena, prun["test"], tol2, nose="Nose")

    assert climb1.dtype == bool
    assert climb2.dtype == bool
    assert np.sum(climb1) >= np.sum(climb2)

    with pytest.raises(NotImplementedError):
        climb_wall("", arena, prun["test"], tol1, nose="Nose")


@settings(deadline=None)
@given(
    dframe=data_frames(
        index=range_indexes(min_size=50),
        columns=columns(["X1", "y1", "X2", "y2"], dtype=float),
        rows=st.tuples(
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
            st.floats(min_value=1, max_value=10),
        ),
    ),
    sampler=st.data(),
)
def test_rolling_speed(dframe, sampler):

    dframe *= np.random.uniform(0, 1, dframe.shape)

    order1 = sampler.draw(st.integers(min_value=1, max_value=3))
    order2 = sampler.draw(st.integers(min_value=order1, max_value=3))
    window2 = sampler.draw(st.integers(min_value=10, max_value=25))

    idx = pd.MultiIndex.from_product(
        [["bpart1", "bpart2"], ["X", "y"]], names=["bodyparts", "coords"],
    )
    dframe.columns = idx

    speeds1 = rolling_speed(dframe, 5, 10, order1)
    speeds2 = rolling_speed(dframe, 5, 10, order2)
    speeds3 = rolling_speed(dframe, window2, 10, order1)

    assert speeds1.shape[0] == dframe.shape[0]
    assert speeds1.shape[1] == dframe.shape[1] // 2
    assert np.all(np.std(speeds1) >= np.std(speeds2))
    for i in range(speeds1.shape[1]):
        assert autocorr(np.array(speeds1.iloc[:, i])) <= autocorr(
            np.array(speeds3.iloc[:, i])
        )
