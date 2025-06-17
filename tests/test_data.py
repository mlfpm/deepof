# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.preprocess

"""

import os
import random
import re
import string
from shutil import rmtree, copy

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from deepof.data import TableDict
import deepof.data
import deepof.utils


@settings(max_examples=20, deadline=None)
@given(
    table_type=st.one_of(
        st.just("analysis.h5"),
        st.just("h5"),
        st.just("csv"),
        st.just("npy"),
        st.just("slp"),
    ),
    arena_detection=st.one_of(
        st.just("circular-autodetect"), st.just("polygonal-autodetect")
    ),
    custom_bodyparts=st.booleans(),
)
def test_project_init(table_type, arena_detection, custom_bodyparts):

    if custom_bodyparts or table_type == "npy":
        custom_bodyparts = [
            "".join(random.choice(string.ascii_lowercase) for _ in range(10))
            for _ in range(14)
        ]

    # Add path to SLEAP tables if necessary
    tables_path = "Tables"
    if table_type in ["slp", "analysis.h5", "npy"]:
        tables_path = os.path.join(tables_path, "SLEAP")

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", tables_path
        ),
        project_name=f"test_{table_type[1:]}",
        rename_bodyparts=(None if not custom_bodyparts else custom_bodyparts),
        bodypart_graph=(
            "deepof_14"
            if not custom_bodyparts
            else {custom_bodyparts[0]: custom_bodyparts[1:]}
        ),
        arena=arena_detection,
        video_scale=380,
        video_format=".mp4",
        table_format=table_type,
    )

    assert isinstance(prun, deepof.data.Project)
    assert isinstance(prun.preprocess_tables(), tuple)

    prun = prun.create(test=True, force=True)
    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_single_topview",
            f"test_{table_type[1:]}",
        )
    )

    assert isinstance(prun, deepof.data.Coordinates)


def test_project_extend():

    #create a new folder with only one video and table  
    # Define the base path
    base_path = os.path.join('.','tests', 'test_examples')

    # Create folder under the local path './tests'
    to_extend_path = os.path.join(base_path, 'to_extend')
    os.makedirs(to_extend_path)

    # Create 'Tables' and 'Videos' folders
    tables_path = os.path.join(to_extend_path, 'Tables')
    videos_path = os.path.join(to_extend_path, 'Videos')
    os.makedirs(tables_path)
    os.makedirs(videos_path)

    # Define source file paths
    source_table_file = os.path.join(base_path, 'test_single_topview', 'Tables', 'testDLC_h5_table.h5')
    source_video_file = os.path.join(base_path, 'test_single_topview', 'Videos', 'testDLC_video_circular_arena.mp4')

    # Copy files to the new folders
    copy(source_table_file, tables_path)
    copy(source_video_file, videos_path)
    
    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "to_extend"),
        video_path=videos_path,
        table_path=tables_path,
        project_name=f"test_extend",
        rename_bodyparts=None,
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format="h5",
    )

    video_extend = os.path.join(
        ".", "tests", "test_examples", "test_single_topview", "Videos"
    )
    table_extend = os.path.join(
        ".", "tests", "test_examples", "test_single_topview", "Tables"
    )
    ext_prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "to_extend"),
        video_path=video_extend,
        table_path=table_extend,
        project_name=f"test_extend",
        rename_bodyparts=None,
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        table_format="h5",
    )

    prun_path = os.path.join(
        ".",
        "tests",
        "test_examples",
        "to_extend",
        "test_extend",
    )

    prun.create(test=True, force=True)

    ext_prun.extend(prun_path, video_path=video_extend, table_path=table_extend)

    # ensure that new project has all four datasets from both sources
    
    rmtree(prun_path)
    rmtree(to_extend_path)

    assert len(prun.tables) == 1
    assert len(prun.videos) == 1

    assert len(ext_prun.tables) == 2
    assert len(ext_prun.videos) == 2
    assert len(ext_prun.arena_params) == 2




def test_project_properties():

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
        table_format=".h5",
    )

    assert prun.distances == "all"
    prun.distances = "testing"
    assert prun.distances == "testing"

    assert not prun.ego
    prun.ego = "testing"
    assert prun.ego == "testing"

    assert prun.angles
    prun.angles = False
    assert not prun.angles


def test_project_filters():

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
        table_format=".h5",
    ).create(force=True, test=True)

    # Update experimental conditions with mock values
    prun._exp_conditions = {
        key: pd.DataFrame(
            {"CSDS": np.random.choice(["Case", "Control"], size=1)[0]}, index=[0]
        )
        for key in prun.get_coords().keys()
    }

    coords = prun.get_coords()

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )
    assert isinstance(coords.filter_id("B"), dict)
    assert isinstance(coords.filter_videos(coords.keys()), dict)
    assert isinstance(coords.filter_condition(exp_filters={"CSDS": "Control"}), dict)


@settings(max_examples=5, deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_distances(nodes, ego):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

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
        table_format=".h5",
    )
    prun.create(force=True, test=True)

    tables, _ = prun.preprocess_tables()
    prun.scales, prun.arena_params, prun.roi_dicts, prun.video_resolution = prun.get_arena(
        tables=tables, test=True,
    )
    prun.distances = nodes
    prun.ego = ego
    prun = prun.get_distances(prun.preprocess_tables()[0])

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    assert isinstance(prun, dict)


@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
)
def test_get_angles(nodes, ego):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

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
        table_format=".h5",
    )

    prun.distances = nodes
    prun.ego = ego
    prun = prun.get_angles(prun.preprocess_tables()[0])

    assert isinstance(prun, dict)


@settings(max_examples=5, deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    ego=st.integers(min_value=0, max_value=2),
    use_numba=st.booleans(),  # intended to be so low that numba runs (10) or not
)
def test_run(nodes, ego, use_numba):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]
    fast_implementations_threshold = 100000
    if use_numba:
        fast_implementations_threshold = 10

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
        iterative_imputation="full",
        fast_implementations_threshold=fast_implementations_threshold,
    )

    prun.distances = nodes
    prun.ego = ego
    prun = prun.create(force=True, test=True)
    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    assert isinstance(prun, deepof.data.Coordinates)


@settings(max_examples=8, deadline=None)
@given(
    use_numba=st.booleans(),  # intended to be so low that numba runs (10) or not
    detection_mode=st.one_of(
        st.just("polygonal-autodetect"), st.just("circular-autodetect")
    ),
    bodypart_graph=st.one_of(
        st.just("deepof_14"), st.just("deepof_8")
    ),
)
def test_get_supervised_annotation(use_numba,detection_mode,bodypart_graph):

    if detection_mode=="circular-autodetect":
        arena_type="test_single_topview"
    else:
        arena_type="test_square_arena_topview"

    fast_implementations_threshold = 100000
    if use_numba:
        fast_implementations_threshold = 10

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", arena_type),
        video_path=os.path.join(
            ".", "tests", "test_examples", arena_type, "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", arena_type, "Tables"
        ),
        arena=detection_mode,
        bodypart_graph=bodypart_graph,
        exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
        fast_implementations_threshold=fast_implementations_threshold,
    ).create(force=True, test=True)

    prun = prun.supervised_annotation()

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", arena_type, "deepof_project"
        )
    )

    assert isinstance(prun, deepof.data.TableDict)
    assert prun._type == "supervised"


def test_supervised_parameters():

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_single_topview"),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "Tables"
        ),
        arena="circular-autodetect",
        exclude_bodyparts=["Tail_1", "Tail_2", "Tail_tip"],
        video_scale=380,
        video_format=".mp4",
        table_format=".h5",
    ).create(force=True, test=True)

    #get and update parameters, get supervised with parameters
    params=prun.get_supervised_parameters()
    params['sniff_arena_tol']=50
    params['stationary_threshold']=100
    params['non_existing']=7
    prun.set_supervised_parameters(params)
    supervised_a = prun.supervised_annotation()

    # reset parameters, get second supervised with parameters
    prun.reset_supervised_parameters()
    supervised_b = prun.supervised_annotation()

    rmtree(
        os.path.join(
            ".", "tests", "test_examples", "test_single_topview", "deepof_project"
        )
    )

    #ensure that more behavior was detected with more generous parameters
    assert np.sum(supervised_a['test']['sniff-arena']) > np.sum(supervised_b['test']['sniff-arena'])
    

@settings(deadline=None)
@given(
    nodes=st.integers(min_value=0, max_value=1),
    mode=st.one_of(st.just("single"), st.just("multi"), st.just("madlc")),
    ego=st.integers(min_value=0, max_value=1),
    exclude=st.one_of(st.just(tuple([""])), st.just(["Tail_tip"])),
    sampler=st.data(),
    random_id=st.text(alphabet=string.ascii_letters, min_size=50, max_size=50),
    use_numba=st.booleans(),  # intended to be so low that numba runs (10) or not
)
def test_get_table_dicts(nodes, mode, ego, exclude, sampler, random_id, use_numba):

    nodes = ["all", ["Center", "Nose", "Tail_base"]][nodes]
    ego = [False, "Center", "Nose"][ego]

    fast_implementations_threshold = 100000
    if use_numba:
        fast_implementations_threshold = 10

    if mode == "multi":
        animal_ids = ["B", "W"]
    elif mode == "madlc":
        animal_ids = ["mouse_black_tail", "mouse_white_tail"]
    else:
        animal_ids = [""]

    prun = deepof.data.Project(
        project_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode), "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode), "Tables"
        ),
        project_name=f"deepof_project_{random_id}",
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={
            "test": pd.DataFrame({"CSDS": "test_cond"}, index=[0]),
            "test2": pd.DataFrame({"CSDS": "test_cond"}, index=[0]),
        },
        fast_implementations_threshold=fast_implementations_threshold,
    )

    #also use large table handling 
    if use_numba:
        prun.very_large_project=True

    if mode == "single":
        prun.distances = nodes
        prun.ego = ego

    prun = prun.create(force=True, test=True)

    selected_id = None
    if mode == "multi" and nodes == "all" and not ego:
        selected_id = "B"
    elif mode == "madlc" and nodes == "all" and not ego:
        selected_id = "mouse_black_tail"

    center = sampler.draw(st.one_of(st.just("arena"), st.just("Center")))
    algn = sampler.draw(st.one_of(st.just(False), st.just("Spine_1")))
    polar = sampler.draw(st.booleans())
    speed = sampler.draw(st.integers(min_value=1, max_value=3))
    rois = sampler.draw(st.one_of(st.just(None),st.integers(min_value=1, max_value=2)))
    animals_in_roi = sampler.draw(st.one_of(st.just(None),st.just(selected_id)))

    #get table info
    start_times_dict=prun.get_start_times()
    end_times_dict=prun.get_end_times()
    table_lengths_dict=prun.get_table_lengths()

    coords = prun.get_coords(
        center=center,
        polar=polar,
        align=(algn if center == "Center" and not polar else False),
        selected_id=selected_id,
        roi_number = rois,
        animals_in_roi = animals_in_roi,
    )
    speeds = prun.get_coords(
        speed=(speed if not ego and nodes == "all" else 0),
        selected_id=selected_id,
        roi_number = rois,
        animals_in_roi = animals_in_roi,
    )
    distances = prun.get_distances(
        speed=sampler.draw(st.integers(min_value=0, max_value=2)),
        selected_id=selected_id,
        roi_number = rois,
        animals_in_roi = animals_in_roi,
    )
    angles = prun.get_angles(
        degrees=sampler.draw(st.booleans()),
        speed=sampler.draw(st.integers(min_value=0, max_value=2)),
        selected_id=selected_id, 
        roi_number = rois,
        animals_in_roi = animals_in_roi,
    )
    areas = prun.get_areas(
        roi_number = rois,
        animals_in_roi = animals_in_roi,
    )
    merged = coords.merge(speeds, distances, angles, areas)


    # deepof.table testing
    samples_max=sampler.draw(st.integers(min_value=10, max_value=500000))
    bin_info_time=deepof.visuals_utils._preprocess_time_bins(coordinates=prun, bin_size=None, bin_index=None, samples_max=samples_max)

    # at least two entries per column need to be not nan to make sure that not the entire entire table is filtered out due to low variance
    if (np.sum(speeds['test'].iloc[bin_info_time['test'],:].notnull())>1).all() and (np.sum(speeds['test2'].iloc[bin_info_time['test2'],:].notnull())>1).all() :
        prep = coords.preprocess(
            prun,
            window_size=11,
            window_step=1,
            scale=sampler.draw(
                st.one_of(st.just("standard"), st.just("minmax"), st.just("robust"))
            ),
            test_videos=1,
            verbose=2,
            filter_low_variance=1e-3,
            interpolate_normalized=5,
            shuffle=sampler.draw(st.booleans()),
            samples_max=samples_max,
        )
        first_key=list(prep[0][0].keys())[0]
        prep_data=deepof.data_loading.get_dt(prep[0][0],first_key)

        assert isinstance(prep[0][0], dict)
        assert isinstance(prep_data, np.ndarray)

        # deepof dimensionality reduction testing

        assert isinstance(coords.random_projection(n_components=2), tuple)
        assert isinstance(coords.pca(n_components=2), tuple)

    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            f"deepof_project_{random_id}",
        )
    )
    

    #table info
    assert all(
        [int(
            ''.join(re.findall(r'\d+', start_times_dict[key])))
            <int(''.join(re.findall(r'\d+', end_times_dict[key]))) 
            for key 
            in start_times_dict.keys()
            ])
    assert all(
        table_lengths_dict[key] > 0
        for key 
        in table_lengths_dict.keys() 
        )

    # deepof.coordinates testing
    assert isinstance(coords, deepof.data.TableDict)
    assert isinstance(speeds, deepof.data.TableDict)
    assert isinstance(distances, deepof.data.TableDict)
    assert isinstance(angles, deepof.data.TableDict)
    assert isinstance(areas, deepof.data.TableDict)
    assert isinstance(merged, deepof.data.TableDict)
    assert isinstance(prun.get_videos(), dict)
    assert prun.get_exp_conditions is not None
    assert isinstance(prun.get_quality(), deepof.data.TableDict)
    assert isinstance(prun.get_arenas, tuple)


@settings(deadline=None)
@given(
    mode=st.one_of(st.just("single"), st.just("multi"), st.just("madlc")),
    sampler=st.data(),
    random_id=st.text(alphabet=string.ascii_letters, min_size=50, max_size=50),
)
def test_get_graph_dataset(mode, sampler, random_id):

    if mode == "multi":
        animal_ids = ["B", "W"]
    elif mode == "madlc":
        animal_ids = ["mouse_black_tail", "mouse_white_tail"]
    else:
        animal_ids = [""]

    prun = deepof.data.Project(
        project_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode)
        ),
        video_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode), "Videos"
        ),
        table_path=os.path.join(
            ".", "tests", "test_examples", "test_{}_topview".format(mode), "Tables"
        ),
        project_name=f"deepof_project_{random_id}",
        arena="circular-autodetect",
        video_scale=380,
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
    ).create(force=True, test=True)

    graph_dset, shapes, adj_matrix, to_preprocess, global_scaler = prun.get_graph_dataset(
        animal_id=sampler.draw(st.one_of(st.just(None), st.just(animal_ids[0]))),
        scale=sampler.draw(
            st.one_of(
                st.just("standard"),
                st.just("minmax"),
                st.just("robust"),
                st.just(False),
            )
        ),
        test_videos=1,
    )

    rmtree(
        os.path.join(
            ".",
            "tests",
            "test_examples",
            "test_{}_topview".format(mode),
            f"deepof_project_{random_id}",
        )
    )
    
    assert isinstance(graph_dset, tuple)
    assert isinstance(adj_matrix, np.ndarray)
    assert isinstance(to_preprocess, deepof.data.TableDict)


@settings(deadline=None)
@given(
    use_bin_info=st.booleans(),
    N_windows_tab=st.integers(min_value=10, max_value=100),
    return_edges=st.booleans(),
    no_nans=st.booleans(),
    dtype=st.one_of(st.just("numpy"), st.just("pandas")),
    is_tab_tuple=st.booleans(),
)
def test_sample_windows_from_data(use_bin_info, N_windows_tab, return_edges, no_nans, dtype, is_tab_tuple):

    #create bin_info object
    time_bin_info={}
    if use_bin_info:
        time_bin_info={i: np.arange(4,N_windows_tab-4) for i in range(10)}

    my_dict = {i: np.random.normal(size=[100, 10]) for i in range(10)}
    #add nans
    num_nans=50
    for key in my_dict:
        indices = np.random.choice(my_dict[key].shape[0], num_nans, replace=False)
        my_dict[key][indices,0] = np.nan 

    #create different types of Table dicts
    if is_tab_tuple:
        if dtype == "numpy":
            tab_dict= TableDict({i: (my_dict[i],my_dict[i]) for i in range(10)}, typ='test')
        else:
            tab_dict= TableDict({i: (pd.DataFrame(my_dict[i]),pd.DataFrame(my_dict[i])) for i in range(10)}, typ='test')
    else:
        if dtype == "numpy":
            tab_dict= TableDict({i: my_dict[i] for i in range(10)}, typ='test')
        else:
            tab_dict= TableDict({i: pd.DataFrame(my_dict[i]) for i in range(10)}, typ='test')
    

    a_data=None
    if return_edges:
        X_data, a_data, bin_info_out = tab_dict.sample_windows_from_data(time_bin_info, N_windows_tab, return_edges, no_nans)
    else:
        X_data, bin_info_out = tab_dict.sample_windows_from_data(time_bin_info, N_windows_tab, return_edges, no_nans)


    if use_bin_info:
        assert X_data.shape[0]==np.sum([len(time_bin_info[i]) for i in time_bin_info.keys()])
    else:
        assert X_data.shape[0]<=10*N_windows_tab 

    if a_data is not None:
        if use_bin_info:
            assert a_data.shape[0]==np.sum([len(time_bin_info[i]) for i in time_bin_info.keys()])
        else:
            assert a_data.shape[0]<=10*N_windows_tab


    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        table_type=st.just("h5"),
    )
    def test_deep_unsupervised_embedding(table_type):

        tables_path = "Tables"

        prun = deepof.data.Project(
            project_path=os.path.join(".", "tests", "test_examples", "test_multi_topview"),
            video_path=os.path.join(
                ".", "tests", "test_examples", "test_multi_topview", "Videos"
            ),
            table_path=os.path.join(
                ".", "tests", "test_examples", "test_multi_topview", tables_path
            ),
            project_name=f"test_{table_type[1:]}",
            animal_ids=["B","W"],
            bodypart_graph="deepof_11",
            arena="circular-autodetect",
            video_scale=380,
            video_format=".mp4",
            table_format=table_type,
        )

        prun = prun.create(test=True, force=True)

        (
        graph_preprocessed_coords, shapes, adj_matrix, to_preprocess, global_scaler
        ) = prun.get_graph_dataset(
            animal_id="B",  # Comment out for multi-animal embeddings
            center="Center",
            align="Spine_1",
            window_size=25,
            window_step=1,
            test_videos=1,
            preprocess=True,
            scale="standard",
        )

        trained_model = prun.deep_unsupervised_embedding(
            preprocessed_object=graph_preprocessed_coords,  # Use graph-preprocessed embeddings
            adjacency_matrix=adj_matrix,
            embedding_model="VaDE", # Can also be set to 'VQVAE' and 'Contrastive'
            epochs=10,
            encoder_type="recurrent", # Can also be set to 'TCN' and 'transformer'
            n_components=10,
            latent_dim=8,
            batch_size=16,
            verbose=True, # Set to True to follow the training loop
            interaction_regularization=0.0,
            pretrained=False, # Set to False to train a new model!
        )

        embeddings, soft_counts = deepof.model_utils.embedding_per_video(
            coordinates=prun,
            to_preprocess=to_preprocess,
            model=trained_model,
            animal_id="B",
            global_scaler=global_scaler,
        )

        assert embeddings['test'].shape==(76,8)
        assert embeddings['test2'].shape==(76,8)