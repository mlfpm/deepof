# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.utils

"""

import os
from itertools import combinations
from shutil import rmtree

import networkx as nx
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from scipy.spatial import distance
from shapely.geometry import Point, Polygon


import deepof.data
import deepof.utils

# AUXILIARY FUNCTIONS #


def autocorr(x, t=1):
    """Computes autocorrelation of the given array with a lag of t"""
    return np.round(np.corrcoef(np.array([x[:-t], x[t:]]))[0, 1], 5)


# QUALITY CONTROL AND PREPROCESSING #


@settings(deadline=None)
@given(
    v=st.one_of(
        st.just("yes"),
        st.just("true"),
        st.just("t"),
        st.just("y"),
        st.just("1"),
        st.just("no"),
        st.just("false"),
        st.just("f"),
        st.just("n"),
        st.just("0"),
    )
)
def test_str2bool(v):
    assert isinstance(deepof.utils.str2bool(v), bool)


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
    polar = deepof.utils.bp2polar(tab)
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

    assert cart_df.shape == deepof.utils.tab2polar(cart_df).shape


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
        deepof.utils.compute_dist(pair_array, arena_abs, arena_rel),
        pd.DataFrame(distance.cdist(pair_array[:, :2], pair_array[:, 2:]).diagonal())
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
    )
)
def test_bpart_distance(cordarray):
    cord_df = pd.DataFrame(cordarray)
    idx = pd.MultiIndex.from_product(
        [list(cord_df.columns[: len(cord_df.columns) // 2]), ["X", "y"]],
        names=["bodyparts", "coords"],
    )
    cord_df.columns = idx

    bpart = deepof.utils.bpart_distance(cord_df)

    assert bpart.shape[0] == cord_df.shape[0]
    assert bpart.shape[1] == len(list(combinations(range(cord_df.shape[1] // 2), 2)))


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
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
    )
)
def test_angle(abc):
    a, b, c = abc

    angles = []
    for i, j, k in zip(a, b, c):
        ang = np.arccos(
            (np.dot(i - j, k - j) / (np.linalg.norm(i - j) * np.linalg.norm(k - j)))
        )
        angles.append(ang)

    assert np.allclose(deepof.utils.angle([a, b, c]), np.array(angles))


@settings(max_examples=10, deadline=None)
@given(
    p=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=2)
        ),
        elements=st.floats(
            min_value=1, max_value=10, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_rotate(p):
    assert np.allclose(deepof.utils.rotate_numba(p, 2 * np.pi), p)
    assert np.allclose(deepof.utils.rotate_numba(p, np.pi), -p)
    assert np.allclose(deepof.utils.rotate_numba(p, 0), p)
    assert np.allclose(deepof.utils.rotate(p, 2 * np.pi), p)
    assert np.allclose(deepof.utils.rotate(p, np.pi), -p)
    assert np.allclose(deepof.utils.rotate(p, 0), p)


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
    aligned = deepof.utils.align_trajectories(data, mode)
    assert aligned.shape == data.shape
    if mode == "center":
        assert np.allclose(aligned[:, (data.shape[1] - 1) // 2, 0], 0)
    elif mode == "all":
        assert np.allclose(aligned[:, :, 0], 0)
    elif mode == "none":
        assert np.allclose(aligned, data)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    a=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=3, max_value=100))),
    lag=st.integers(min_value=1, max_value=10),
)
def test_smooth_boolean_array_and_binary_moving_median(a, lag):
    a[0] = True  # make sure we have at least one True
    smooth = deepof.utils.smooth_boolean_array(a)
    filtered = deepof.utils.binary_moving_median_numba(a, lag)

    def trans(x):
        """In situ function for computing boolean transitions"""
        return sum([i + 1 != i for i in range(x.shape[0] - 1)])

    # Binary median filtered arrays have always less or equal the number of 0-1 and 0-1 transitions
    assert trans(a) >= trans(smooth)
    assert trans(a) >= trans(filtered)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    tab_numpy=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=3, max_value=30),st.integers(min_value=3, max_value=100))),
    frame_rate=st.floats(min_value=1, max_value=100),
    delta_T=st.floats(min_value=0.0, max_value=100),
)
def test_extend_behaviors_numba(tab_numpy,frame_rate,delta_T):

    def _has_no_short_ones(matrix, N):
        """Counts consecutive ones and checks if the at least as long as N except for edge case (Borderline too complicated function for a test case)"""
        for row in range(0,matrix.shape[0]):
            # Count number of consecutive Ones
            c_row=matrix[row,:]
            c_row=np.concatenate(( [0], c_row, [0] ))
            idx = np.flatnonzero(c_row[1:] != c_row[:-1])
            len_ones=np.abs(idx[::2] - idx[1::2])
            #remove last segment, if it ended at 1 (nd hence was correctly cut off)
            if c_row[-2]==1:
                len_ones=len_ones[:-1]
            # Notify, if any segment was too short
            if (len_ones<N).any():
                return False
            
        return True

    tab_extended=deepof.utils.extend_behaviors_numba(tab_numpy,frame_rate,delta_T)
    extension=int(frame_rate*delta_T)

    # An extended table always has equal or more ones than their unextended equivalent
    assert(np.sum(tab_numpy)<=np.sum(tab_extended))
    # All original 1s still exist
    assert np.sum(tab_extended[tab_numpy])==np.sum(tab_numpy)
    # An extended table cannot have more ones than all ones in the original table including extension length
    assert(np.sum(tab_numpy)+np.sum(tab_numpy)*extension)>=np.sum(tab_extended)
    # Make sure that no section of 1s remain that are shorter than extension +1 except the end segemnts
    assert _has_no_short_ones(tab_extended, extension+1)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    pos_in=st.lists(elements=st.integers(min_value=1, max_value=4), min_size=3, max_size=100),
    min_length=st.integers(min_value=3,max_value=100),
    get_both=st.booleans(),
)
def test_multi_step_paired_smoothing(pos_in,min_length,get_both):
    pos_in[0] = 1  # make sure we have at least one True

    if min_length > len(pos_in):
        min_length = len(pos_in)

    #construct non-overlapping oolean arrays from input integers
    a=np.zeros(len(pos_in)).astype(bool)
    b=np.zeros(len(pos_in)).astype(bool)
    c=np.zeros(len(pos_in)).astype(bool)
    f=np.array(pos_in)
    a[f==1]=True
    b[f==2]=True
    c[f==3]=True
    if not b.any()==True:
        b=None
    if not c.any()==True:
        c=None

    non_behavior=None
    if get_both:
        behavior, non_behavior = deepof.utils.multi_step_paired_smoothing(behavior_in=a, not_behavior=b, exclude=c, min_length=min_length, get_both=get_both)
    else:
        behavior = deepof.utils.multi_step_paired_smoothing(behavior_in=a, not_behavior=b, exclude=c, min_length=min_length, get_both=get_both)

    def trans(x):
        """In situ function for computing boolean transitions"""
        return sum([i + 1 != i for i in range(x.shape[0] - 1)])

    # make sure all arrays are smoothed
    assert trans(a) >= trans(behavior)
    if b is not None:
        assert trans(b) >= trans(behavior)
    if c is not None:
        assert trans(c) >= trans(behavior)

    # make sure behaviors and non_behaviors do not overlap
    if get_both:
        assert np.sum(non_behavior) + np.sum(behavior) <= len(pos_in)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    array=arrays(dtype=bool, shape=st.tuples(st.integers(min_value=3, max_value=100))),
    min_length=st.integers(min_value=1, max_value=10),
    )
def test_filter_short_true_segments(array, min_length):
    
    filtered_a = deepof.utils.filter_short_true_segments(array, min_length)
    filtered_b = deepof.utils.filter_short_true_segments_numba(array, min_length)

    # Numba and non-numba version return identical results
    assert (filtered_a==filtered_b).all()
    # Arrays with removed short segments always have less or equal the number of "True" entries
    assert np.sum(filtered_a) <= np.sum(array)


@settings(deadline=None)
@given(
    window=st.data(),
)
def test_rolling_window(window):
    window_step = window.draw(st.integers(min_value=1, max_value=5))
    window_size = 5 * window_step

    a = np.random.uniform(-10, 10, size=(100, 4))

    rolled_a = deepof.utils.rolling_window(
        a, window_size, window_step
    )


    assert len(rolled_a.shape) == len(a.shape) + 1
    assert rolled_a.shape[1] == window_size


@settings(deadline=None)
@given(
    alpha=st.data(),
    series=arrays(
        dtype=float,
        shape=st.tuples(st.integers(min_value=300, max_value=1000)),
        elements=st.floats(
            min_value=1.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    ),
)
def test_smooth_mult_trajectory(alpha, series):
    alpha1 = alpha.draw(st.integers(min_value=3, max_value=6))
    alpha2 = alpha.draw(st.integers(min_value=alpha1 + 2, max_value=10))

    series *= +np.random.normal(0, 1, len(series))

    smoothed1 = deepof.utils.smooth_mult_trajectory(series, alpha1)
    smoothed2 = deepof.utils.smooth_mult_trajectory(series, alpha2)

    assert autocorr(smoothed1) >= autocorr(series)
    assert autocorr(smoothed2) >= autocorr(series)
    assert autocorr(smoothed2) >= autocorr(smoothed1)


@settings(deadline=None)
@given(mode=st.one_of(st.just("or")))
def test_remove_outliers(mode):

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format=".csv",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
        iterative_imputation="full",
    ).create(force=True, test=True)

    coords = prun.get_coords()
    lkhood = prun.get_quality()
    coords_name = list(coords.keys())[0]

    high_std_sum=deepof.utils.full_outlier_mask(
                    coords[coords_name],
                    lkhood[coords_name],
                    likelihood_tolerance=0.9,
                    exclude="Center",
                    lag=5,
                    n_std=3,
                    mode=mode,
                    ).sum().sum()
    
    low_std_sum=deepof.utils.full_outlier_mask(
                    coords[coords_name],
                    lkhood[coords_name],
                    likelihood_tolerance=0.9,
                    exclude="Center",
                    lag=5,
                    n_std=1,
                    mode=mode,
                    ).sum().sum()

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    assert (high_std_sum < low_std_sum)


@settings(deadline=None, max_examples=10)
@given(
    detection_mode=st.one_of(
        st.just("polygonal-autodetect"), st.just("circular-autodetect")
    ),
    video_key=st.one_of(
        st.just("test"), st.just("test2")
    ),
)
def test_recognize_arena_and_subfunctions(detection_mode,video_key):

    if detection_mode=="circular-autodetect":
        arena_type="test_single_topview"
    else:
        arena_type="test_square_arena_topview"

    path = os.path.join(".", "tests", "test_examples", arena_type, "Videos")

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", arena_type),
        video_path=os.path.join(
            ".", "tests", "test_examples", arena_type, "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", arena_type, "Tables"
        ),
        arena=detection_mode,
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
        iterative_imputation="partial",
    )
    #get pre-defined arenas with test-setting
    coords = prun.create(force=True, test=True)

    #actually detect arenas 
    arena_parameters, h, w = deepof.utils.automatically_recognize_arena(
        coordinates=prun,
        videos=coords.get_videos(),
        tables=coords._tables,
        vid_key=video_key,
        path=path,
        segmentation_model=deepof.utils.load_segmentation_model(None),
        arena_type=detection_mode,
    )
    #adjust scaling
    scaling_ratio = coords._scales[video_key][3]/coords._scales[video_key][2]
    if "polygonal" in detection_mode:
        arena_parameters=np.array(arena_parameters)*scaling_ratio
    elif "circular" in detection_mode:
        arena_parameters=(tuple(np.array(arena_parameters[0])*scaling_ratio),tuple(np.array(arena_parameters[1])*scaling_ratio),arena_parameters[2])

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", arena_type, "deepof_project"
        )
    )
    #check if height and width of detected arenas are identical to expected arenas within 10% tolarance
    tolerance=0.1
    assert prun.video_resolution[video_key][0]<(h*(1+tolerance))
    assert prun.video_resolution[video_key][0]>(h*(1-tolerance))
    assert prun.video_resolution[video_key][1]<(w*(1+tolerance))
    assert prun.video_resolution[video_key][1]>(w*(1-tolerance))


    if detection_mode=="circular-autodetect":

        #check if the detected circular areas are sufficiently similar
        for i in range(3):
            assert np.linalg.norm(
                np.array(coords._arena_params[video_key][i]) - np.array(arena_parameters[i])
                ) < 1

        pass
    else:
        #roughly checks if the ideal corner points are somewhat close to the detected edges
        dist_min=np.inf
        predefined_arena=coords._arena_params[video_key]
        detected_arena =np.array(arena_parameters)
        for point in predefined_arena:
            for k in range(len(detected_arena)):
                dist=np.linalg.norm(point-detected_arena[k,:])
                if dist<dist_min:
                    dist_min=dist
            assert dist_min<25


@settings(deadline=None, max_examples=3)
@given(
    detection_mode=st.one_of(
        st.just("polygonal-autodetect"), st.just("circular-autodetect")
    ),
)
def test_detection_modes(detection_mode):

    if detection_mode=="circular-autodetect":
        arena_type="test_single_topview"
    else:
        arena_type="test_square_arena_topview"

    path = os.path.join(".", "tests", "test_examples", arena_type, "Videos")

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", arena_type),
        video_path=os.path.join(
            ".", "tests", "test_examples", arena_type, "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", arena_type, "Tables"
        ),
        arena=detection_mode,
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
        exp_conditions={"test": "test_cond", "test2": "test_cond"},
        iterative_imputation="partial",
    )
    #get arenas with actual detection
    coords = prun.create(force=True, test="detect_arena")

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", arena_type, "deepof_project"
        )
    )
    
    #check if height and width of detected arenas are identical to expected arenas within 10% tolarance
    assert len(coords._arena_params)==2
    if detection_mode=="polygonal-autodetect":
        assert isinstance(coords._arena_params['test'], np.ndarray)
        assert isinstance(coords._arena_params['test2'], np.ndarray)
    else:
        assert isinstance(coords._arena_params['test'], tuple)
        assert isinstance(coords._arena_params['test2'], tuple)
    assert (coords._video_resolution['test'][0]<coords._video_resolution['test'][1])
    assert (coords._video_resolution['test2'][0]<coords._video_resolution['test2'][1])
    assert (len(coords._scales['test'])==4)
    assert (len(coords._scales['test2'])==4)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
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
        [["bpart1", "bpart2"], ["X", "y"]], names=["bodyparts", "coords"]
    )
    dframe.columns = idx

    speeds1 = deepof.utils.rolling_speed(dframe, 10, 5, 10, order1)
    speeds2 = deepof.utils.rolling_speed(dframe, 10, 5, 10, order2)

    assert speeds1.shape[0] == dframe.shape[0]
    assert speeds1.shape[1] == dframe.shape[1] // 2
    assert np.all(np.std(speeds1) >= np.std(speeds2))


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    x=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=1000),
            st.integers(min_value=10, max_value=1000),
        ),
        elements=st.floats(min_value=1.0, max_value=1.0),
    ).map(lambda x: x * np.random.uniform(0, 2, x.shape)),
    n_components=st.integers(min_value=1, max_value=10),
    cv_type=st.integers(min_value=0, max_value=3),
)
def test_gmm_compute(x, n_components, cv_type):
    cv_type = ["spherical", "tied", "diag", "full"][cv_type]
    assert len(deepof.utils.gmm_compute(x, n_components, cv_type)) == 2


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    x=arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=10, max_value=100),
            st.integers(min_value=10, max_value=100),
        ),
        elements=st.floats(min_value=1.0, max_value=1.0),
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
            deepof.utils.gmm_model_selection(
                pd.DataFrame(x), n_component_range, part_size, n_runs=1
            )
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

    trans = deepof.utils.cluster_transition_matrix(
        cluster_sequence, nclusts, autocorrelation, return_graph
    )

    if autocorrelation:
        assert len(trans) == 2

        if return_graph:
            assert isinstance(trans[0], nx.Graph)
        else:
            assert isinstance(trans[0], np.ndarray)

        assert isinstance(trans[1], np.ndarray)

    else:
        if return_graph:
            assert isinstance(trans, nx.Graph)
        else:
            assert isinstance(trans, np.ndarray)

# list of valid polygons as ill-defined polygons (e.g. lines) can lead to deviations
polygons = [
    [[0, 0], [1, 0], [1, 1], [0, 1]],  # Square
    [[0, 0], [2, 0], [1, 2], [0, 0]],  # Triangle
    [[-4, 0], [2, 0], [2, 2], [-4, 2], [-4, 0]],  # Rectangle
    [[1, 1], [3, 1], [4, 3], [2, 4], [1, 3], [1, 1]],  # Complex polygon
    [[0, 0], [6, 0], [6, 4], [4, 4], [4, 2], [2, 2], [2, 4], [0, 4], [0, 0]],  # U-shape
    [
        [0, 0],
        [6, 0],
        [6, 6],
        [0, 6],
        [0, 3],
        [2, 3],
        [2, 4],
        [4, 4],
        [4, 2],
        [2, 2],
        [2, 3],
        [0, 3],
        [0, 0],
    ],  # ring polygon
]

@settings(max_examples=100, deadline=None)
@given(
    points=st.lists(
        st.lists(
            st.floats(min_value=-100, max_value=100, width=32), min_size=2, max_size=2
        ),
        min_size=1,
        max_size=100,
    ),
    polygons=st.sampled_from(polygons),
)
def test_point_in_polygon(points, polygons):

    points = np.array(points)
    polygons = np.array(polygons)
    assert all(
        deepof.utils.point_in_polygon_numba(points, polygons)
        == deepof.utils.point_in_polygon(points, Polygon(polygons))
    )


test_point_in_polygon()
