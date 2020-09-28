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
import deepof.preprocess
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
        [["bpart1", "bpart2"], ["X", "y"]], names=["bodyparts", "coords"],
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
            path=os.path.join(".", "tests", "test_examples"),
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
    tol_speed=st.floats(min_value=0.01, max_value=4.98),
    animal_id=st.text(min_size=0, max_size=15, alphabet=string.ascii_lowercase),
)
def test_huddle(pos_dframe, tol_forward, tol_spine, tol_speed, animal_id):

    _id = animal_id
    if animal_id != "":
        _id += "_"

    idx = pd.MultiIndex.from_product(
        [
            [
                _id + "Left_ear",
                _id + "Right_ear",
                _id + "Left_fhip",
                _id + "Right_fhip",
                _id + "Spine_1",
                _id + "Center",
                _id + "Spine_2",
                _id + "Tail_base",
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
        tol_spine,
        tol_speed,
        animal_id,
    )

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


def test_get_hparameters():
    assert get_hparameters() == {
        "speed_pause": 10,
        "close_contact_tol": 15,
        "side_contact_tol": 15,
        "follow_frames": 20,
        "follow_tol": 20,
        "huddle_forward": 15,
        "huddle_spine": 10,
        "huddle_speed": 0.1,
        "fps": 24,
    }
    assert get_hparameters({"speed_pause": 20}) == {
        "speed_pause": 20,
        "close_contact_tol": 15,
        "side_contact_tol": 15,
        "follow_frames": 20,
        "follow_tol": 20,
        "huddle_forward": 15,
        "huddle_spine": 10,
        "huddle_speed": 0.1,
        "fps": 24,
    }


@settings(deadline=None)
@given(
    w=st.integers(min_value=300, max_value=500),
    h=st.integers(min_value=300, max_value=500),
)
def test_frame_corners(w, h):
    assert len(frame_corners(w, h)) == 4
    assert frame_corners(w, h, {"downright": "test"})["downright"] == "test"


def test_rule_based_tagging():

    prun = deepof.preprocess.project(
        path=os.path.join(".", "tests", "test_examples"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
        animal_ids=[""],
    ).run(verbose=True)

    hardcoded_tags = rule_based_tagging(
        list([i + "_" for i in prun.get_coords().keys()]),
        ["test_video_circular_arena.mp4"],
        prun,
        vid_index=0,
        path=os.path.join(".", "tests", "test_examples", "Videos"),
    )

    assert type(hardcoded_tags) == pd.DataFrame
    assert hardcoded_tags.shape[1] == 3


def test_rule_based_video():

    prun = deepof.preprocess.project(
        path=os.path.join(".", "tests", "test_examples"),
        arena="circular",
        arena_dims=tuple([380]),
        video_format=".mp4",
        table_format=".h5",
        animal_ids=[""],
    ).run(verbose=True)

    hardcoded_tags = rule_based_tagging(
        list([i + "_" for i in prun.get_coords().keys()]),
        ["test_video_circular_arena.mp4"],
        prun,
        vid_index=0,
        path=os.path.join(".", "tests", "test_examples", "Videos"),
    )

    rule_based_video(
        coordinates=prun,
        tracks=list([i + "_" for i in prun.get_coords().keys()]),
        videos=["test_video_circular_arena.mp4"],
        vid_index=0,
        frame_limit=100,
        tag_dict=hardcoded_tags,
        path=os.path.join(".", "tests", "test_examples", "Videos"),
    )
