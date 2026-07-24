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
        st.just("autodetect"),
    ),
    arena_detection=st.one_of(
        st.just("circular-autodetect"), st.just("polygonal-autodetect")
    ),
    table_bodyparts=st.booleans(),
    bit_precision=st.one_of(
        st.just(64), st.just(32), st.just(None)
    ),
)
def test_project_init(table_type, arena_detection, table_bodyparts, bit_precision):

    if table_bodyparts or table_type == "npy":
        table_bodyparts = ["Nose", "Left_ear", "Right_ear", "Spine_1", "Center", "Spine_2", "Left_fhip", "Right_fhip", "Left_bhip", "Right_bhip", "Tail_base", "Tail_1", "Tail_2", "Tail_tip"]

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
        rename_bodyparts=(None if not table_bodyparts else table_bodyparts),
        bodypart_graph=(
            "deepof_14"
            if not table_bodyparts
            else {table_bodyparts[0]: table_bodyparts[1:]}
        ),
        arena=arena_detection,
        video_scale="380 mm",
        video_format=".mp4",
        table_format=table_type,
        bit_precision=bit_precision,
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

    # verify correct bit precision for all created table objects
    stores = ("_tables", "_quality", "_angles", "_areas", "_distances")

    for store_name in stores:
        store = getattr(prun, store_name)
        for key in store:
            tab = deepof.data_loading.get_dt(store, key)
            
            for col_name in tab:
                datatype = tab[col_name].dtype

                if bit_precision==64 or bit_precision is None:
                    assert datatype==np.float64
                # Since duckdb cannot store 16 bit pyarrow tables natively, they get upcasted
                elif bit_precision==32:
                    assert datatype==np.float32


@settings(max_examples=20, deadline=None)
@given(
    table_type=st.one_of(
        st.just("analysis.h5"),
        st.just("h5"),
        st.just("csv"),
        st.just("slp"),
        st.just("autodetect"),
    ),
    rename_len=st.sampled_from([8, 11, 14]),
)
def test_rename_bodyparts(table_type, rename_len):

    base_path = os.path.join(".", "tests", "test_examples", "test_single_topview")

    # Match existing test structure for table paths
    tables_path = "Tables"
    if table_type in ["slp", "analysis.h5"]:
        tables_path = os.path.join(tables_path, "SLEAP")

    # Fake bodypart names. In normal usage, this has to correspond the the actually correct DeepOF naming schemas
    #(The purpose of rename_bodyparts is to FIX naming errors in the table after all, not cause them)
    rename_bodyparts = [f"custom_bp_{i}" for i in range(rename_len)]

    prun = deepof.data.Project(
        project_path=base_path,
        project_name=f"test_rename_bodyparts_{table_type}_{rename_len}",
        video_path=os.path.join(base_path, "Videos"),
        table_path=os.path.join(base_path, tables_path),
        arena="circular-autodetect",
        animal_ids="",
        video_scale="380 mm",
        video_format=".mp4",
        table_format=table_type,
        rename_bodyparts=rename_bodyparts,
        bodypart_graph=f"deepof_{rename_len}",
    )

    # Ensure dict was created
    assert isinstance(prun.rename_bodyparts_dict, dict)
    assert set(prun.rename_bodyparts_dict.keys()) == set(rename_bodyparts)
    assert len(prun.rename_bodyparts_dict) == rename_len

    # Ensure mapping order matches connect_mouse(node order) for that preset
    expected_nodes = list(
        deepof.utils.connect_mouse(animal_ids="", graph_preset=f"deepof_{rename_len}").nodes
    )
    for custom_name, deepof_name in zip(rename_bodyparts, expected_nodes):
        assert prun.rename_bodyparts_dict[custom_name] == deepof_name


def test_arena_loading():

    base_path = os.path.join(".", "tests", "test_examples", "test_single_topview")
    video_path = os.path.join(base_path, "Videos")
    table_path = os.path.join(base_path, "Tables")

    tmp_dir = os.path.join(base_path, "_tmp_load_arena_data")
    arena_file = os.path.join(tmp_dir, "arena_data.pkl")

    project_name = "test_get_arena_loading_branch"
    out_project_dir = os.path.join(base_path, project_name)

    try:
        # 1) Create and save arena data (file has 3 ROIs)
        pr_raw = deepof.data.Project(
            project_path=base_path,
            project_name="test_get_arena_loading_save",
            video_path=video_path,
            table_path=table_path,
            arena="polygonal-autodetect",
            video_scale="380 mm",
            video_format=".mp4",
            table_format=".h5",
            number_of_rois=2,
        )
        
        pr_save=pr_raw.create(force=True, test=True) # Note: test mode will result in 2 rois independent from the number chosen

        keys = list(pr_save._tables.keys())

        arena_params = pr_save._arena_params
        scales = pr_save._scales
        video_resolution = pr_save._video_resolution
        roi_dicts = pr_save._roi_dicts

        pr_raw.save_arena_data(
            arena_path=arena_file,
            arena_params=arena_params,
            roi_dicts=roi_dicts,
            scales=scales,
            video_resolution=video_resolution,
        )

        # 2) Loader project expects only 1 ROI -> load_arena_data will truncate -> skip_detection=True
        if os.path.exists(out_project_dir):
            rmtree(out_project_dir)

        pr_load = deepof.data.Project(
            project_path=base_path,
            project_name=project_name,
            video_path=video_path,
            table_path=table_path,
            arena="polygonal-autodetect",
            video_scale="380 mm",
            video_format=".mp4",
            table_format=".h5",
            number_of_rois=1,
        )
        pr_load.set_up_project_directory(debug=True)  # ensures Arena_detection/Coordinates exist

        got_scales, got_arena, got_rois, got_res = pr_load.get_arena(
            tables={k: None for k in keys},
            arena_path=arena_file,
            test=True,
            load_also_rois=True,  # no UI
        )

        # Minimal correctness checks
        k0 = keys[0]
        assert list(got_rois[k0].keys()) == [1]
        assert (got_rois[k0][1] == roi_dicts[k0][1]).all()
        assert (got_arena[k0] == arena_params[k0]).all()
        assert got_scales[k0] == scales[k0]
        assert got_res[k0] == video_resolution[k0]

        # get_arena always saves arena_data.pkl into the project
        assert os.path.isfile(os.path.join(out_project_dir, "Coordinates", "arena_data.pkl"))

    finally:
        if os.path.exists(tmp_dir):
            rmtree(tmp_dir)
        if os.path.exists(out_project_dir):
            rmtree(out_project_dir)


def test_start_markers():

    base_path = os.path.join(".", "tests", "test_examples", "test_single_topview")
    video_path = os.path.join(base_path, "Videos")
    table_path = os.path.join(base_path, "Tables")
    project_name = "test_start_markers"
    keys = ['test2', 'test']

    # 2) Define start markers 
    # dict[key] -> DataFrame with one row, columns = marker names, values = time strings
    marker_name = "trial_start"
    start_time_str = "00:00:01.000"  # 1 second
    start_markers = {k: pd.DataFrame({marker_name: [start_time_str]}) for k in keys}

    try:
        # 3) Create project WITH start_markers
        coords = deepof.data.Project(
            project_path=base_path,
            project_name=project_name,
            video_path=video_path,
            table_path=table_path,
            arena="polygonal-autodetect",
            video_scale="380 mm",
            video_format=".mp4",
            table_format=".h5",
            start_markers=start_markers,
        ).create(force=True, test=True)

        # 4) Validate start marker frame values
        start_frames = coords.get_start_marker_values(marker_name, return_frames=True)
        expected_start_frame = int(np.round(coords._frame_rate))  # 1 second * fps

        for k in keys:
            assert start_frames[k] == expected_start_frame

        # 5) Validate table lengths are shortened accordingly
        full_lengths = coords.get_table_lengths()
        shortened_lengths = coords.get_table_lengths(start_marker=marker_name)

        for k in keys:
            assert shortened_lengths[k] == full_lengths[k] - expected_start_frame

    finally:
        out_dir = os.path.join(base_path, project_name)
        if os.path.exists(out_dir):
            rmtree(out_dir)


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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
        video_scale="380 mm",
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
    to_video=st.booleans()
)
def test_get_table_dicts(nodes, mode, ego, exclude, sampler, random_id, use_numba, to_video):

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
        video_scale="380 mm",
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
        exclude_bodyparts=exclude,
        exp_conditions={
            "test": pd.DataFrame({"CSDS": "test_cond"}, index=[0]),
            "test2": pd.DataFrame({"CSDS": "test_cond"}, index=[0]),
        },
        fast_implementations_threshold=fast_implementations_threshold,
        frame_rate=25,
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
        to_video = to_video,
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
    assert prun.get_condition_values("CSDS") is not None
    assert isinstance(prun.get_quality(), deepof.data.TableDict)
    assert isinstance(prun.get_arenas, tuple)


@settings(deadline=None)
@given(
    mode=st.one_of(st.just("single"), st.just("multi"), st.just("madlc")),
    sampler=st.data(),
    random_id=st.text(alphabet=string.ascii_letters, min_size=50, max_size=50),
    test_videos=st.one_of(st.just(1),st.just(["test"])),
    full_nan_table=st.booleans(),
    dist_standardize_groups=st.booleans(),
    speed_standardize_groups=st.booleans(),
    bit_precision=st.one_of(
        st.just(64), st.just(32), st.just(None)
    ),
)
def test_get_graph_dataset(mode, sampler, random_id, test_videos, full_nan_table, dist_standardize_groups,speed_standardize_groups, bit_precision):

    if mode == "multi":
        animal_ids = ["B", "W"]
    elif mode == "madlc":
        animal_ids = ["mouse_black_tail", "mouse_white_tail"]
    else:
        animal_ids = [""]
    dist_standardize="per_column"
    if dist_standardize_groups:
        dist_standardize="groupwise"
    speed_standardize="per_column"
    if speed_standardize_groups:
        speed_standardize="groupwise"

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
        video_scale="380 mm",
        video_format=".mp4",
        animal_ids=animal_ids,
        table_format=".h5",
        bit_precision=bit_precision,
    ).create(force=True, test=True)
    prun._frame_rate=25

    if full_nan_table:
       #simulate missing data
       key_with_nans=list(prun._tables.keys())[0]
       prun._tables[key_with_nans].iloc[::]=np.nan 
       prun._distances[key_with_nans].iloc[::]=np.nan 
       prun._angles[key_with_nans].iloc[::]=np.nan
    if bit_precision==32:
        pass

    graph_dset, meta_info, adj_matrix, to_preprocess, global_scaler = prun.get_graph_dataset(
        animal_id=sampler.draw(st.one_of(st.just(None), st.just(animal_ids[0]))),
        scale=sampler.draw(
            st.one_of(
                st.just("standard"),
                st.just("minmax"),
                st.just("robust"),
                st.just(False),
            )
        ),
        test_videos=test_videos,
        dist_standardize=dist_standardize,
        speed_standardize=speed_standardize,
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

    # verify correct bit precision for all created table objects
    tab_dicts=(graph_dset[0], graph_dset[1], to_preprocess)

    for tab_dict in tab_dicts:
        for key in tab_dict:
            tabs = deepof.data_loading.get_dt(tab_dict, key)
            
            for tab in tabs:
                if isinstance(tabs, pd.DataFrame):
                    datatypes = tabs.dtypes
                else:
                    datatypes = [tab.dtype]

                if bit_precision==64 or bit_precision is None:
                    assert all([datatype==np.float64 for datatype in datatypes])
                # Since duckdb cannot store 16 bit pyarrow tables natively, they get upcasted
                elif bit_precision==32:
                    assert all([datatype==np.float32 for datatype in datatypes])
    
    # data from nan table was removed
    if full_nan_table:
        assert len(graph_dset[0])==0
        assert isinstance(meta_info, dict) and len(meta_info['shape_train'])==3


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
        video_scale="380 mm",
        video_format=".mp4",
        table_format=table_type,
    )

    prun = prun.create(test=True, force=True)

    (
    graph_preprocessed_coords, meta_info, adj_matrix, to_preprocess, global_scaler
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

    model_val, model_score, model_part, log_summary = prun.deep_unsupervised_embedding(
        preprocessed_object=graph_preprocessed_coords,  # Use graph-preprocessed embeddings
        adjacency_matrix=adj_matrix,
        meta_info=meta_info,
        embedding_model="VaDE", # Can also be set to 'VQVAE' and 'Contrastive'
        epochs=10,
        encoder_type="recurrent", # Can also be set to 'TCN' and 'transformer'
        n_clusters=10,
        latent_dim=8,
        batch_size=16,
        interaction_regularization=0.0,
        pretrained=False, # Set to False to train a new model!
        use_turtle_teacher = False,
    )

    embeddings, soft_counts = deepof.clustering.model_utils_new.embedding_per_video(
        coordinates=prun,
        meta_info=meta_info,
        to_preprocess=to_preprocess,
        model=model_val,
        animal_id="B",
        global_scaler=global_scaler,
    )

    assert embeddings['test'].shape==(76,8)
    assert embeddings['test2'].shape==(76,8)
