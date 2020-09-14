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

    idx = pd.MultiIndex.from_product(
        [["bpart1", "bpart2"], ["X", "y"]], names=["bodyparts", "coords"],
    )
    dframe.columns = idx

    speeds1 = rolling_speed(dframe, 5, 10, order1)
    speeds2 = rolling_speed(dframe, 5, 10, order2)

    assert speeds1.shape[0] == dframe.shape[0]
    assert speeds1.shape[1] == dframe.shape[1] // 2
    assert np.all(np.std(speeds1) >= np.std(speeds2))


@settings(deadline=None)
@given(
    pos_dframe=data_frames(
        index=range_indexes(min_size=5),
        columns=columns(
            [
                "X1",
                "y1",
                "X2",
                "y2",
                "X3",
                "y3",
                "X4",
                "y4",
                "X5",
                "y5",
                "X6",
                "y6",
                "X7",
                "y7",
                "X8",
                "y8",
            ],
            dtype=float,
            elements=st.floats(min_value=-20, max_value=20),
        ),
    ),
    tol_forward=st.floats(min_value=0.01, max_value=4.98),
    tol_spine=st.floats(min_value=0.01, max_value=4.98),
)
def test_huddle(pos_dframe, tol_forward, tol_spine):

    idx = pd.MultiIndex.from_product(
        [
            [
                "Left_ear",
                "Right_ear",
                "Left_fhip",
                "Right_fhip",
                "Spine1",
                "Center",
                "Spine2",
                "Tail_base",
            ],
            ["X", "y"],
        ],
        names=["bodyparts", "coords"],
    )
    pos_dframe.columns = idx
    hudd = huddle(pos_dframe, tol_forward, tol_spine)

    assert hudd.dtype == bool
    assert np.array(hudd).shape[0] == pos_dframe.shape[0]
    assert np.sum(np.array(hudd)) <= pos_dframe.shape[0]


@settings(deadline=None)
@given(
    distance_dframe=data_frames(
        index=range_indexes(min_size=20, max_size=20),
        columns=columns(
            ["d1", "d2", "d3", "d4"],
            dtype=float,
            elements=st.floats(min_value=-20, max_value=20),
        ),
    ),
    position_dframe=data_frames(
        index=range_indexes(min_size=20, max_size=20),
        columns=columns(
            ["X1", "y1", "X2", "y2", "X3", "y3", "X4", "y4"],
            dtype=float,
            elements=st.floats(min_value=-20, max_value=20),
        ),
    ),
    frames=st.integers(min_value=1, max_value=20),
    tol=st.floats(min_value=0.01, max_value=4.98),
)
def test_following_path(distance_dframe, position_dframe, frames, tol):

    bparts = [
        "A_Nose",
        "B_Nose",
        "A_Tail_base",
        "B_Tail_base",
    ]

    pos_idx = pd.MultiIndex.from_product(
        [bparts, ["X", "y"]], names=["bodyparts", "coords"],
    )

    position_dframe.columns = pos_idx
    distance_dframe.columns = [c for c in combinations(bparts, 2) if c[0][0] != c[1][0]]

    follow = following_path(
        distance_dframe,
        position_dframe,
        follower="A",
        followed="B",
        frames=frames,
        tol=tol,
    )

    assert follow.dtype == bool
    assert len(follow) == position_dframe.shape[0]
    assert len(follow) == distance_dframe.shape[0]
    assert np.sum(follow) <= position_dframe.shape[0]
    assert np.sum(follow) <= distance_dframe.shape[0]


@settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow],
)
@given(sampler=st.data())
def test_single_behaviour_analysis(sampler):
    behaviours = sampler.draw(
        st.lists(min_size=2, elements=st.text(min_size=5), unique=True)
    )
    treatments = sampler.draw(
        st.lists(min_size=2, max_size=4, elements=st.text(min_size=5), unique=True)
    )

    behavioural_dict = sampler.draw(
        st.dictionaries(
            min_size=2,
            keys=st.text(min_size=5),
            values=data_frames(
                index=range_indexes(min_size=50, max_size=50),
                columns=columns(behaviours, dtype=bool),
            ),
        )
    )

    ind_dict = {vid: np.random.choice(treatments) for vid in behavioural_dict.keys()}
    treatment_dict = {treat: [] for treat in set(ind_dict.values())}
    for vid, treat in ind_dict.items():
        treatment_dict[treat].append(vid)

    ylim = sampler.draw(st.floats(min_value=0, max_value=10))
    stat_tests = sampler.draw(st.booleans())

    out = single_behaviour_analysis(
        behaviours[0],
        treatment_dict,
        behavioural_dict,
        plot=0,
        stat_tests=stat_tests,
        save=None,
        ylim=ylim,
    )

    assert len(out) == 1 if stat_tests == 0 else len(out) == 2
    assert type(out[0]) == dict
    if stat_tests:
        assert type(out[0]) == dict


@settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow],
)
@given(
    behaviour_dframe=data_frames(
        index=range_indexes(min_size=100, max_size=1000),
        columns=columns(
            ["d1", "d2", "d3", "d4", "speed1"], dtype=bool, elements=st.booleans(),
        ),
    ),
    window_size=st.data(),
    stepped=st.booleans(),
)
def test_max_behaviour(behaviour_dframe, window_size, stepped):
    wsize1 = window_size.draw(st.integers(min_value=5, max_value=50))
    wsize2 = window_size.draw(st.integers(min_value=wsize1, max_value=50))

    maxbe1 = max_behaviour(behaviour_dframe, wsize1, stepped)
    maxbe2 = max_behaviour(behaviour_dframe, wsize2, stepped)

    assert type(maxbe1) == np.ndarray
    assert type(maxbe2) == np.ndarray
    if not stepped:
        assert type(maxbe1[wsize1 // 2 + 1]) == str
        assert type(maxbe1[wsize2 // 2 + 1]) == str
        assert maxbe1[wsize1 // 2 + 1] in behaviour_dframe.columns
        assert maxbe2[wsize2 // 2 + 1] in behaviour_dframe.columns
        assert len(maxbe1) >= len(maxbe2)


@settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow],
)
@given(
    x=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=1000),
            st.integers(min_value=10, max_value=1000),
        ),
        elements=st.floats(min_value=1.0, max_value=1.0,),
    ).map(lambda x: x * np.random.uniform(0, 2, x.shape)),
    n_components=st.integers(min_value=1, max_value=10),
    cv_type=st.integers(min_value=0, max_value=3),
)
def test_gmm_compute(x, n_components, cv_type):
    cv_type = ["spherical", "tied", "diag", "full"][cv_type]
    assert len(gmm_compute(x, n_components, cv_type)) == 2


@settings(
    deadline=None, suppress_health_check=[HealthCheck.too_slow],
)
@given(
    x=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=1000),
            st.integers(min_value=10, max_value=1000),
        ),
        elements=st.floats(min_value=1.0, max_value=1.0,),
    ).map(lambda x: x * np.random.uniform(0, 2, x.shape)),
    sampler=st.data(),
)
def test_gmm_model_selection(x, sampler):
    n_component_range = range(1, sampler.draw(st.integers(min_value=2, max_value=5)))
    part_size = sampler.draw(
        st.integers(min_value=x.shape[0] // 2, max_value=x.shape[0] * 2)
    )
    assert (
        len(
            gmm_model_selection(pd.DataFrame(x), n_component_range, part_size, n_runs=1)
        )
        == 3
    )


@settings(deadline=None)
@given(sampler=st.data(), autocorrelation=st.booleans(), return_graph=st.booleans())
def test_cluster_transition_matrix(sampler, autocorrelation, return_graph):

    nclusts = sampler.draw(st.integers(min_value=1, max_value=10))
    cluster_sequence = sampler.draw(
        arrays(
            dtype=int,
            shape=st.tuples(st.integers(min_value=10, max_value=1000)),
            elements=st.integers(min_value=1, max_value=nclusts),
        ).filter(lambda x: len(set(x)) != 1)
    )

    trans = cluster_transition_matrix(
        cluster_sequence, nclusts, autocorrelation, return_graph
    )

    if autocorrelation:
        assert len(trans) == 2

        if return_graph:
            assert type(trans[0]) == nx.Graph
        else:
            assert type(trans[0]) == np.ndarray

        assert type(trans[1]) == np.ndarray

    else:
        if return_graph:
            assert type(trans) == nx.Graph
        else:
            assert type(trans) == np.ndarray
