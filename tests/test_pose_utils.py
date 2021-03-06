# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.pose_utils

"""

from hypothesis import given
from hypothesis import HealthCheck
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import range_indexes, columns, data_frames
from deepof.pose_utils import *
import deepof.data
import matplotlib.figure
import pytest
import string


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
        [["bpart1", "bpart2"], ["X", "y"]],
        names=["bodyparts", "coords"],
    )
    pos_dframe.columns = idx
    close_contact = close_single_contact(pos_dframe, "bpart1", "bpart2", tol, 1, 1)
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
        pos_dframe, "bpart1", "bpart2", "bpart3", "bpart4", tol, 1, 1, rev
    )
    assert close_contact.dtype == bool
    assert np.array(close_contact).shape[0] <= pos_dframe.shape[0]


@settings(deadline=None)
@given(
    center=st.tuples(
        st.integers(min_value=300, max_value=500),
        st.integers(min_value=300, max_value=500),
    ),
    axes=st.tuples(
        st.integers(min_value=300, max_value=500),
        st.integers(min_value=300, max_value=500),
    ),
    angle=st.floats(min_value=0, max_value=360),
    tol=st.data(),
)
def test_climb_wall(center, axes, angle, tol):

    arena = (center, axes, np.radians(angle))
    tol1 = tol.draw(st.floats(min_value=0.001, max_value=10))
    tol2 = tol.draw(st.floats(min_value=tol1, max_value=10))

    prun = (
        deepof.data.project(
            path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
            arena="circular",
            arena_dims=tuple([arena[2]]),
            video_format=".mp4",
            table_format=".h5",
        )
        .run(verbose=True)
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
    pos_dframe=data_frames(
        index=range_indexes(min_size=5),
        columns=columns(
            [
                "X0",
                "y0",
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
            ],
            dtype=float,
            elements=st.floats(min_value=-20, max_value=20),
        ),
    ),
    tol_forward=st.floats(min_value=0.01, max_value=4.98),
    tol_speed=st.floats(min_value=0.01, max_value=4.98),
    animal_id=st.text(min_size=0, max_size=15, alphabet=string.ascii_lowercase),
)
def test_single_animal_traits(pos_dframe, tol_forward, tol_speed, animal_id):

    _id = animal_id
    if animal_id != "":
        _id += "_"

    idx = pd.MultiIndex.from_product(
        [
            [
                _id + "Nose",
                _id + "Left_bhip",
                _id + "Right_bhip",
                _id + "Left_fhip",
                _id + "Right_fhip",
                _id + "Center",
            ],
            ["X", "y"],
        ],
        names=["bodyparts", "coords"],
    )
    pos_dframe.columns = idx
    hudd = huddle(
        pos_dframe,
        pos_dframe.xs("X", level="coords", axis=1, drop_level=True),
        tol_forward,
        tol_speed,
        animal_id,
    )
    digging = dig(
        pos_dframe.xs("X", level="coords", axis=1, drop_level=True),
        pos_dframe.xs("X", level="coords", axis=1, drop_level=True),
        tol_speed,
        0.85,
        animal_id,
    )

    assert hudd.dtype == bool
    assert digging.dtype == bool
    assert np.array(hudd).shape[0] == pos_dframe.shape[0]
    assert np.array(digging).shape[0] == pos_dframe.shape[0]
    assert np.sum(np.array(hudd)) <= pos_dframe.shape[0]
    assert np.sum(np.array(digging)) <= pos_dframe.shape[0]


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
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
        [bparts, ["X", "y"]],
        names=["bodyparts", "coords"],
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
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
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

    plot = sampler.draw(st.integers(min_value=0, max_value=200))

    out = single_behaviour_analysis(
        behaviours[0],
        treatment_dict,
        behavioural_dict,
        plot=plot,
        stat_tests=stat_tests,
        save=None,
        ylim=ylim,
    )

    assert len(out) == 1 if (stat_tests == 0 and plot == 0) else len(out) >= 2
    assert type(out[0]) == dict
    if plot:
        assert np.any(np.array([type(i) for i in out]) == matplotlib.figure.Figure)
    if stat_tests:
        assert type(out[0]) == dict


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    behaviour_dframe=data_frames(
        index=range_indexes(min_size=100, max_size=1000),
        columns=columns(
            ["d1", "d2", "d3", "d4", "speed1"],
            dtype=bool,
            elements=st.booleans(),
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


def test_get_hparameters():
    assert type(get_hparameters()) == dict
    assert get_hparameters({"speed_pause": 20})["speed_pause"] == 20


@settings(deadline=None)
@given(
    w=st.integers(min_value=300, max_value=500),
    h=st.integers(min_value=300, max_value=500),
)
def test_frame_corners(w, h):
    assert len(frame_corners(w, h)) == 4
    assert frame_corners(w, h, {"downright": "test"})["downright"] == "test"


@settings(deadline=None)
@given(
    multi_animal=st.booleans(),
    video_output=st.booleans(),
)
def test_rule_based_tagging(multi_animal, video_output):

    if video_output:
        video_output = ["test"]

    path = os.path.join(
        ".",
        "tests",
        "test_examples",
        "test_{}_topview".format("multi" if multi_animal else "single"),
    )

    prun = deepof.data.project(
        path=path,
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
        animal_ids=(["B", "W"] if multi_animal else [""]),
    ).run(verbose=True)

    hardcoded_tags = prun.rule_based_annotation(
        video_output=video_output, frame_limit=50
    )

    assert type(hardcoded_tags) == deepof.data.table_dict
    assert list(hardcoded_tags.values())[0].shape[1] == (21 if multi_animal else 6)
