"""Data structures for preprocessing and wrangling of motion tracking output data. This is the main module handled by the user.

There are three main data structures to pay attention to:
- :class:`~deepof.data.Project`, which serves as a configuration hub for the whole pipeline
- :class:`~deepof.data.Coordinates`, which acts as an intermediary between project configuration and data, and contains
a plethora of processing methods to apply, and
- :class:`~deepof.data.TableDict`, which is the main data structure to store the data, having experiment IDs as keys
and processed time-series as values in a dictionary-like object.

For a detailed tutorial on how to use this module, see the advanced tutorials in the main section.
"""
# @author lucasmiranda42
# encoding: utf-8
# module deepof

#bug fix for linux cv2 issue
import os
import sys
import subprocess
import cv2

def is_display_available(): # pragma: no cover
    # Check for Linux and X display
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    
    # Test OpenCV in a subprocess to avoid crashing the main script
    check_script = """
import cv2
try:
    cv2.imshow('test', 1)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
except Exception:
    exit(1)
exit(0)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

if is_display_available():
    cv2.imshow("test",1)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
else:
    print("Display not available, skipping cv2 init.")


import copy
import pickle
import re
import shutil
import warnings
from shutil import rmtree
from time import time
from typing import Any, Dict, List, NewType, Tuple, Union, Optional

import networkx as nx
import numpy as np
import pandas as pd
import psutil

import umap
from natsort import os_sorted
from pkg_resources import resource_filename
from sklearn import random_projection
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from tqdm import tqdm

import deepof.annotation_utils
from deepof.config import PROGRESS_BAR_FIXED_WIDTH, suppress_warnings_context
import deepof.model_utils
import deepof.models
import deepof.clustering.models_new
import deepof.clustering.model_utils_new
import deepof.utils
import deepof.arena_utils
import deepof.visuals
from deepof.visuals_utils import _preprocess_time_bins
from deepof.data_loading import get_dt, save_dt
from concurrent.futures import ThreadPoolExecutor


# SET DEEPOF VERSION
current_deepof_version="0.8.2"

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)

# CLASSES FOR PREPROCESSING AND DATA WRANGLING


def load_project(
        project_path: str,
        animal_ids: List = None,
        arena: str = "polygonal-autodetect",
        bodypart_graph: Union[str, dict] = "deepof_14",
        iterative_imputation: str = "partial",
        exclude_bodyparts: List = tuple([""]),
        exp_conditions: dict = None,
        remove_outliers: bool = True,
        interpolation_limit: int = 5,
        interpolation_std: int = 3,
        likelihood_tol: float = 0.75,
        model: str = "mouse_topview",
        project_name: str = "deepof_project",
        video_path: str = None,
        table_path: str = None,
        rename_bodyparts: list = None,
        sam_checkpoint_path: str = None,
        smooth_alpha: float = 1,
        table_format: str = "autodetect",
        video_format: str = ".mp4",
        video_scale: int = 1,
        number_of_rois = 0,
        fast_implementations_threshold: int = 50000,) -> coordinates:  # pragma: no cover
    """Load a pre-saved pickled Coordinates object. Will update Coordinate objects from older versions of deepof (down to 0.7) to work with this version. 
    Very old projects will be recreated during loading with the current version of Deepof. For this purpose input arguments can be set just as in a recular project definition.

    Args:

        animal_ids (list): list of animal ids.
        arena (str): arena type. Can be one of "circular-autodetect", "circular-manual", "polygonal-autodetect", or "polygonal-manual".
        bodypart_graph (str): body part scheme to use for the analysis. Defaults to None, in which case the program will attempt to select it automatically based on the available body parts.
        iterative_imputation (str): whether to use iterative imputation for occluded body parts, options are "full" and "partial". if set to None, no imputation takes place.
        exclude_bodyparts (list): list of bodyparts to exclude from analysis.
        exp_conditions (dict): dictionary with experiment IDs as keys and experimental conditions as values.
        remove_outliers (bool): whether outliers should be removed during project creation.
        interpolation_limit (int): maximum number of missing frames to interpolate.
        interpolation_std (int): maximum number of standard deviations to interpolate.
        likelihood_tol (float): likelihood threshold for outlier detection.
        model (str): model to use for pose estimation. Defaults to 'mouse_topview' (as described in the documentation).
        project_name (str): name of the current project.
        project_path (str): path to the folder containing the motion tracking output data.
        video_path (str): path where to find the videos to use. If not specified, deepof, assumes they are in your project path.
        table_path (str): path where to find the tracks to use. If not specified, deepof, assumes they are in your project path.
        rename_bodyparts (list): list of names to use for the body parts in the provided tracking files. The order should match that of the columns in your DLC tables or the node dimensions on your (S)LEAP .npy files.
        sam_checkpoint_path (str): path to the checkpoint file for the SAM model. If not specified, the model will be saved in the installation folder.
        smooth_alpha (float): smoothing intensity. The higher the value, the more smoothing.
        table_format (str): format of the table. Defaults to 'autodetect', but can be set to "csv" or "h5" for DLC output, and "npy", "slp" or "analysis.h5" for (S)LEAP.
        video_format (str): video format. Defaults to '.mp4'.
        video_scale (int): diameter of the arena in mm (if the arena is round) or length of the first specified arena side (if the arena is polygonal).            
        number_of_rois (int): number of behavior rois to be drawn during project creation, default = 0,
        fast_implementations_threshold (int): If the total number of frames in the project is larger than this, numba implementations of all functions with a numba option will be used.

    Returns:
        coordinates (deepof_coordinates): Pre-run coordinates object.

    """
    with open(
        os.path.join(project_path, "Coordinates", "deepof_coordinates.pkl"), "rb"
    ) as handle:
        coordinates = pickle.load(handle)

    coordinates._project_path = os.path.split(project_path[0:-1])[0]
    # Error for not compatible versions
    if not (hasattr(coordinates, "_run_numba")):

        raise ValueError(
            """You are trying to load a deepOF project that was created with version 0.6.x or earlier.\n
            These older versions are not compatible with the current version"""
        )
    # Compatibility fixes versions 0.7.0 to 0.7.2
    if isinstance(coordinates._table_paths, List):
        
        print(f"Initiate project upgrade to 0.8")
        print(f"IMPORTANT: The following original input options cannot be transferred and the following values will be used:")
        print(f"(you can change these values using load_project input variables)")
        print(f"- iterative_imputation: " + str(iterative_imputation))
        print(f"- remove_outliers: " + str(remove_outliers))
        print(f"- interpolation_limit: " + str(interpolation_limit))
        print(f"- interpolation_std: " + str(interpolation_std))
        print(f"- likelihood_tol: " + str(likelihood_tol))
        print(f"- model: " + str(model))
        print(f"- rename_bodyparts: " + str(rename_bodyparts))
        print(f"- sam_checkpoint_path: " + str(sam_checkpoint_path))
        print(f"- smooth_alpha: " + str(smooth_alpha))
        print(f"- fast_implementations_threshold: " + str(fast_implementations_threshold))

        #get file endings
        table_extension=".h5"
        video_extension=".mp4"

        redone_project = deepof.data.Project(
            animal_ids=coordinates._animal_ids,
            arena=coordinates._arena,
            bodypart_graph=coordinates._bodypart_graph, 
            iterative_imputation=iterative_imputation,
            exclude_bodyparts = coordinates._excluded,
            exp_conditions=coordinates._exp_conditions,
            remove_outliers = remove_outliers,
            interpolation_limit = interpolation_limit,
            interpolation_std = interpolation_std,
            likelihood_tol = likelihood_tol,
            model = model,
            project_name=coordinates._project_name,
            project_path=project_path,
            video_path=os.path.join(project_path, 'Videos'),
            table_path=os.path.join(project_path, 'Tables'),
            rename_bodyparts=rename_bodyparts,
            sam_checkpoint_path=sam_checkpoint_path,
            smooth_alpha=smooth_alpha,
            table_format=table_extension,
            video_format=video_extension,
            video_scale=coordinates._arena_dims,
            number_of_rois = number_of_rois,
            fast_implementations_threshold=fast_implementations_threshold,
        )

        coordinates=redone_project.create(force=True)

    return coordinates


class Project:
    """Class for loading and preprocessing motion tracking data of individual and multiple animals.

    All main computations are handled from here.

    """

    def __init__(
        self,
        animal_ids: List = None,
        arena: str = "polygonal-autodetect",
        bodypart_graph: Union[str, dict] = "deepof_14",
        iterative_imputation: str = "partial",
        exclude_bodyparts: List = tuple([""]),
        exp_conditions: Union[str, dict] = None,
        remove_outliers: bool = True,
        interpolation_limit: int = 5,
        interpolation_std: int = 3,
        likelihood_tol: float = 0.75,
        model: str = "mouse_topview",
        project_name: str = "deepof_project",
        project_path: str = os.path.join("."),
        video_path: str = None,
        table_path: str = None,
        rename_bodyparts: list = None,
        sam_checkpoint_path: str = None,
        smooth_alpha: float = 1,
        table_format: str = "autodetect",
        video_format: str = ".mp4",
        video_scale: int = 1,
        number_of_rois: int = 0,
        fast_implementations_threshold: int = 50000,
    ):
        """Initialize a Project object.

        Args:
            animal_ids (list): list of animal ids.
            arena (str): arena type. Can be one of "circular-autodetect", "circular-manual", "polygonal-autodetect", or "polygonal-manual".
            bodypart_graph (str): body part scheme to use for the analysis. Defaults to None, in which case the program will attempt to select it automatically based on the available body parts.
            iterative_imputation (str): whether to use iterative imputation for occluded body parts, options are "full" and "partial". if set to None, no imputation takes place.
            exclude_bodyparts (list): list of bodyparts to exclude from analysis.
            exp_conditions (dict): dictionary with experiment IDs as keys and experimental conditions as values.
            remove_outliers (bool): whether outliers should be removed during project creation.
            interpolation_limit (int): maximum number of missing frames to interpolate.
            interpolation_std (int): maximum number of standard deviations to interpolate.
            likelihood_tol (float): likelihood threshold for outlier detection.
            model (str): model to use for pose estimation. Defaults to 'mouse_topview' (as described in the documentation).
            project_name (str): name of the current project.
            project_path (str): path to the folder containing the motion tracking output data.
            video_path (str): path where to find the videos to use. If not specified, deepof, assumes they are in your project path.
            table_path (str): path where to find the tracks to use. If not specified, deepof, assumes they are in your project path.
            rename_bodyparts (list): list of names to use for the body parts in the provided tracking files. The order should match that of the columns in your DLC tables or the node dimensions on your (S)LEAP .npy files.
            sam_checkpoint_path (str): path to the checkpoint file for the SAM model. If not specified, the model will be saved in the installation folder.
            smooth_alpha (float): smoothing intensity. The higher the value, the more smoothing.
            table_format (str): format of the table. Defaults to 'autodetect', but can be set to "csv" or "h5" for DLC output, and "npy", "slp" or "analysis.h5" for (S)LEAP.
            video_format (str): video format. Defaults to '.mp4'.
            video_scale (int): diameter of the arena in mm (if the arena is round) or length of the first specified arena side (if the arena is polygonal).            
            number_of_rois (int): number of behavior rois to be drawn during project creation, default = 0,
            fast_implementations_threshold (int): If the total number of frames in the project is larger than this, numba implementations of all functions with a numba option will be used.

        """
        # Set version
        self.version=current_deepof_version
        # Set working paths
        self.project_path = project_path
        self.project_name = project_name
        self.video_path = video_path
        #for later separation into path to source tables and path tables generated with deepof
        self.table_path = table_path
        self.source_table_path = table_path
        self.trained_path = resource_filename(__name__, "trained_models")

        # Detect files to load from disk
        self.table_format = table_format
        if self.table_format != "analysis.h5":
            self.table_format = table_format.replace(".", "")
        if self.table_format == "autodetect":
            ex = [
                i
                for i in os.listdir(self.source_table_path)
                if (
                    os.path.isfile(os.path.join(self.source_table_path, i))
                    and not i.startswith(".")
                )
            ][0]
            self.table_format = ex.split(".")[-1]
        video_list = os_sorted(
            [
                vid
                for vid in os.listdir(self.video_path)
                if vid.endswith(video_format) and not vid.startswith(".")
            ]
        )
        table_list = os_sorted(
            [
                tab
                for tab in os.listdir(self.source_table_path)
                if tab.endswith(self.table_format) and not tab.startswith(".")
            ]
        )
        assert len(video_list) == len(
            table_list
        ), "Unequal number of videos and tables. Please check your file structure"

        #turn tables and videos into dictionaries with same keys
        self.tables={}
        self.videos={}
        for i, tab in enumerate(table_list):
            # Remove the DLC suffix from the table name
            try:
                tab_name = deepof.utils.re.findall("(.*?)DLC", tab)[0]
            except IndexError:
                tab_name = tab.split(".")[0]
            self.tables[tab_name]=table_list[i]
            self.videos[tab_name]=video_list[i]

        # Loads arena details and (if needed) detection models
        self.arena = arena
        self.arena_dims = video_scale
        self.number_of_rois = number_of_rois
        self.ellipse_detection = None

        # check if there are enough frames to use numba compilation and / or use memory efficient implementations
        self.run_numba = False
        self.very_large_project = False
        video_paths = {key: os.path.join(video_path, video) for key, video in self.videos.items()}
        total_frames = deepof.utils.get_total_Frames(video_paths)
        frames_sum=np.sum(total_frames)
        frames_max=np.max(total_frames)

        if frames_sum > fast_implementations_threshold:
            self.run_numba = True
        if frames_max > 360000 or frames_sum > 900000: #roughly one 4 hour video at 25 fps or 10 hours of recording material in total
            self.very_large_project = True

        # Init the rest of the parameters
        self.angles = True
        self.animal_ids = animal_ids if animal_ids is not None else [""]
        self.areas = True
        self.bodypart_graph = bodypart_graph
        self.connectivity = None
        self.distances = "all"
        self.ego = False
        if isinstance(exp_conditions, str):
            self.load_exp_conditions(exp_conditions)
        else:
            self.exp_conditions = exp_conditions
        self.remove_outliers = remove_outliers
        self.interpolation_limit = interpolation_limit
        self.interpolation_std = interpolation_std
        self.likelihood_tolerance = likelihood_tol
        self.model = model
        self.smooth_alpha = smooth_alpha
        self.frame_rate = None
        self.video_format = video_format
        self.iterative_imputation = iterative_imputation
        self.exclude_bodyparts = exclude_bodyparts
        self.segmentation_path = sam_checkpoint_path
        self.rename_bodyparts = rename_bodyparts

    def __str__(self):  # pragma: no cover
        """Print the object to stdout."""
        return "deepof analysis of {} videos".format(len(self.videos))

    def __repr__(self):  # pragma: no cover
        """Print the object to stdout."""
        return "deepof analysis of {} videos".format(len(self.videos))

    def set_up_project_directory(self, debug=False):
        """Create a project directory where to save all produced results."""
        # Create a project directory, as well as subfolders for videos and tables
        project_path = os.path.join(self.project_path, self.project_name)

        if (
            len(
                [
                    i
                    for i in os.listdir(self.video_path)
                    if i.endswith(self.video_format)
                ]
            )
            == 0
        ):
            raise FileNotFoundError(
                "There are no compatible videos in the specified directory."
            )  # pragma: no cover
        if (
            len(
                [
                    i
                    for i in os.listdir(self.source_table_path)
                    if i.endswith(self.table_format)
                ]
            )
            == 0
        ):
            raise FileNotFoundError(
                "There are no compatible tracks in the specified directory."
            )  # pragma: no cover

        if not os.path.exists(project_path):
            os.makedirs(project_path)
            os.makedirs(os.path.join(self.project_path, self.project_name, "Tables"))
            os.makedirs(
                os.path.join(self.project_path, self.project_name, "Coordinates")
            )
            os.makedirs(os.path.join(self.project_path, self.project_name, "Figures"))
            if debug and "auto" in self.arena:
                os.makedirs(
                    os.path.join(
                        self.project_path, self.project_name, "Arena_detection"
                    )
                )


        else:
            raise OSError(
                "Project already exists. Delete it or specify a different name."
            )  # pragma: no cover
    
    def load_exp_conditions(self, filepath):  # pragma: no cover
        """Load experimental conditions from a wide-format csv table.

        Args:
            filepath (str): Path to the file containing the experimental conditions.

        """
        exp_conditions = pd.read_csv(filepath, index_col=0)
        exp_conditions = {
            exp_id: pd.DataFrame(
                exp_conditions.loc[exp_conditions.iloc[:, 0] == exp_id, :].iloc[0, 1:]
            ).T
            for exp_id in exp_conditions.iloc[:, 0]
        }
        self.exp_conditions = exp_conditions

    @property
    def distances(self):
        """Returns distances table_dict"""
        return self._distances

    @property
    def ego(self):
        """String, name of a body part. If True, computes only the distances between the specified body part and the rest."""
        return self._ego

    @property
    def angles(self):
        """Returns angles table_dict"""
        return self._angles

    def get_arena(
        self,
        tables: dict,
        debug: str = False,
        test: bool = False,
    ) -> np.array:
        """Return the arena as recognised from the videos.

        Args:
            tables (dict): dictionary containing coordinate tables
            debug (str): if True, saves intermediate results to disk
            test (bool): if True, runs the function in test mode

        Returns:
            arena (np.ndarray): arena parameters, as recognised from the videos. The shape depends on the arena type

        """
        #if verbose:
        #    print("Detecting arena...")

        return deepof.arena_utils.get_arenas(
            self,
            tables,
            self.arena,
            self.arena_dims,
            self.number_of_rois,
            self.segmentation_path,
            self.video_path,
            self.videos,
            debug,
            test,
        )


    def _update_progress(self, pbar: tqdm, step: str, key: str):
        """Updates the progress bar with the current step."""
        # A little fun, just like in the original!
        funny_messages = [
            "Reticulating splines", "Calibrating flux capacitor", 
            "Aligning warp core", "Polishing the monocle", "Planning AI uprising"
        ]
        if (pbar.n + 1) % 20 == 0 and step == "Updating time index":
            step = np.random.choice(funny_messages)
        
        pbar.set_postfix(file=f"{key[:10]}...", step=step)

    def _load_and_prepare_table(self, key: str, found_individuals_before: bool) -> Tuple[pd.DataFrame, bool]:
        """Loads a table and handles multi-animal formatting."""
        table = deepof.utils.load_table(
            self.tables[key], self.source_table_path, self.table_format,
            self.rename_bodyparts, self.animal_ids
        )

        is_multi_animal = "individuals" in table.index
        
        # Check for consistent header format across all tables
        if list(self.tables.keys()).index(key) > 0:
            assert is_multi_animal == found_individuals_before, \
                f"Table {key} has inconsistent 'individuals' formatting!"

        if is_multi_animal:
            # Update animal IDs and adapt table for the pipeline
            self.animal_ids = list(table.loc["individuals"].unique())
            table.loc["bodyparts"] = table.loc["individuals"] + "_" + table.loc["bodyparts"]
            table.drop("individuals", axis=0, inplace=True)
        
        return table, is_multi_animal
    
    def _format_table_header(self, table: pd.DataFrame) -> pd.DataFrame:
        """Converts the top rows of a dataframe to a MultiIndex header."""
        table.columns = pd.MultiIndex.from_arrays(
            [table.iloc[i] for i in range(2)],
            names=['bodyparts', 'coords']
        )
        # Drop the original header rows and convert to float
        formatted_table = table.iloc[2:].astype(float).reset_index(drop=True)
        return formatted_table

    def _update_connectivity_graph(self):
        """Updates body part connectivity graph based on current animal_ids and bodyparts."""
        # Reinstate "vanilla" bodyparts without animal ids
        reinstated_bps = list(set(
            bp[len(aid) + 1:] if bp.startswith(f"{aid}_") else bp
            for aid in self.animal_ids for bp in self.exclude_bodyparts
        ))

        model_dict = {
            f"{aid}mouse_topview": deepof.utils.connect_mouse(
                aid, exclude_bodyparts=reinstated_bps, graph_preset=self.bodypart_graph
            ) for aid in self.animal_ids
        }
        self.connectivity = {aid: model_dict[f"{aid}{self.model}"] for aid in self.animal_ids}

        if len(self.animal_ids) > 1 and reinstated_bps != [""]:
            self.exclude_bodyparts = [f"{aid}_{bp}" for aid in self.animal_ids for bp in reinstated_bps]

    def _filter_irrelevant_bodyparts(self, table: pd.DataFrame) -> pd.DataFrame:
        """Removes bodyparts not present in the connectivity graph or explicitly excluded."""
        all_bodyparts = table.columns.get_level_values('bodyparts').unique()
        
        relevant_nodes = set()
        for aid in self.animal_ids:
            relevant_nodes.update(self.connectivity[aid].nodes)
        
        relevant_bodyparts = relevant_nodes - set(self.exclude_bodyparts)
        irrelevant_bodyparts = list(set(all_bodyparts) - relevant_bodyparts)

        if not irrelevant_bodyparts:
            return table

        table = table.drop(columns=irrelevant_bodyparts, level="bodyparts").sort_index(axis=1)
        # Recreate a clean MultiIndex to avoid potential gaps
        table.columns = pd.MultiIndex.from_product(
            [
                os_sorted(table.columns.get_level_values('bodyparts').unique()),
                os_sorted(table.columns.get_level_values('coords').unique()),
            ],
            names=table.columns.names
        )
        return table

    def _apply_optional_transforms(self, table_dict: Dict, lik_dict: Dict, pbar: tqdm) -> Tuple[Dict, int]:
        """Applies smoothing, outlier removal, and imputation if configured."""
        key = list(table_dict.keys())[0]
        table = table_dict[key]
        warn_nans_count = 0

        # lik_dict needs to be converted into a TableDict for legacy reasons
        lik_dict = TableDict(lik_dict, 
                        typ="quality", 
                        table_path=os.path.join(self.project_path, self.project_name, "Tables"), 
                        animal_ids=self.animal_ids)

        # 1. Smoothing
        if self.smooth_alpha:
            self._update_progress(pbar, "Smoothing trajectories", key)
            smoothed_data = deepof.utils.smooth_mult_trajectory(
                table.values, alpha=self.smooth_alpha, w_length=15
            )
            table = pd.DataFrame(smoothed_data, index=table.index, columns=table.columns)

        # 2. Outlier Removal
        if self.remove_outliers:
            self._update_progress(pbar, "Removing outliers", key)
            table, warn_nans = deepof.utils.remove_outliers(
                table, lik_dict[key],
                likelihood_tolerance=self.likelihood_tolerance,
                mode="or", n_std=self.interpolation_std,
            )
            warn_nans_count += warn_nans
        
        # Update table_dict with potentially modified table
        table_dict[key] = table

        # 3. Imputation
        if self.iterative_imputation:
            self._update_progress(pbar, "Iterative imputation of ocluded bodyparts", key)
            full_imputation = self.iterative_imputation == "full"
            table_dict = deepof.utils.iterative_imputation(
                self, table_dict, lik_dict, full_imputation=full_imputation
            )

        # 4. Set missing animals
        table_dict = deepof.utils.set_missing_animals(self, table_dict, lik_dict)

        return table_dict, warn_nans_count

    def preprocess_tables(self) -> Tuple[table_dict, table_dict]:
        """
        Loads and preprocesses tracking data through a series of modular steps,
        then saves the results and returns table dictionaries.
        """
        if self.table_format not in ["h5", "csv", "npy", "slp", "analysis.h5"]:
            raise NotImplementedError("Tracking files must be in h5, csv, npy, or slp format")

        final_tab_dict, final_lik_dict = {}, {}
        total_warnings = 0
        found_individuals = False

        with tqdm(total=len(self.tables), desc=f"{'Preprocessing tables':<{PROGRESS_BAR_FIXED_WIDTH}}") as pbar:
            for key in self.tables.keys():
                # 1. Load and Standardize
                self._update_progress(pbar, "Loading trajectories", key)
                table, found_individuals = self._load_and_prepare_table(key, found_individuals)
                
                # 2. Format Header
                self._update_progress(pbar, "Adjusting headers", key)
                table = self._format_table_header(table)
                
                # 3. Update Connectivity Graph
                self._update_progress(pbar, "Updating graphs", key)
                self._update_connectivity_graph()

                # 4. Add Time Index
                if self.frame_rate:
                    #self._update_progress(pbar, "Updating time index", key)
                    #ätime_index = pd.to_timedelta(np.arange(len(table)) / self.frame_rate, unit="s")
                    #table.index = time_index.map(lambda t: str(t.round('ms'))[7:])

                    table.index = pd.timedelta_range(
                        "00:00:00",
                        pd.to_timedelta(
                            int(np.round(table.shape[0] // self.frame_rate)), unit="sec"
                        ),
                        periods=table.shape[0] + 1,
                        closed="left",
                    ).map(lambda t: str(t)[7:])

                    #freq_in_nanoseconds = np.round(1e9 / self.frame_rate)
                    #time_index = pd.timedelta_range(
                    #    start="0s",
                    #    periods=len(table),
                    #    freq=f"{freq_in_nanoseconds}ns"
                    #)
                    # Perform rounding on the ENTIRE index at once. This is a fast, vectorized operation.
                    #rounded_index = time_index.round('ms')
                    # Now, apply the string conversion. The slow .map() is now doing the minimum work.
                    #table.index = rounded_index.map(lambda t: str(t)[7:])

                # 5. Split coordinates from likelihood and filter bodyparts
                self._update_progress(pbar, "Filter bodyparts", key)
                x = table.xs("x", level="coords", axis=1)
                y = table.xs("y", level="coords", axis=1)
                likelihood_table = table.xs("likelihood", level="coords", axis=1, drop_level=True).fillna(0.0)
                
                coords_table = table.drop(columns='likelihood', level='coords')
                processed_table = self._filter_irrelevant_bodyparts(coords_table)
                processed_table.sort_index(axis=1, inplace=True)
                
                # 6. Apply Optional Transformations (Smoothing, Outliers, Imputation)
                table_dict_single = {key: processed_table}
                lik_dict_single = {key: likelihood_table}
                table_dict_single, warn_count = self._apply_optional_transforms(table_dict_single, lik_dict_single, pbar)
                total_warnings += warn_count

                # 7. Save Processed Data
                self._update_progress(pbar, "Saving data", key)
                save_dir = os.path.join(self.project_path, self.project_name, 'Tables', key)
                os.makedirs(save_dir, exist_ok=True)
                
                quality_path = os.path.join(save_dir, f"{key}_likelihood")
                table_path = os.path.join(save_dir, key)
                
                final_lik_dict[key] = save_dt(lik_dict_single[key], quality_path, self.very_large_project)
                final_tab_dict[key] = save_dt(table_dict_single[key], table_path, self.very_large_project)
                
                pbar.update(1)

        if total_warnings > 0:
            warnings.warn(
                f"\033[38;5;208m"
                f"More than 30% of position values were missing or outliers in {total_warnings} out of "
                f"{len(self.tables)} tables. This may be expected if subjects were obscured for long intervals."
                f"\033[0m"
            )

        self.table_path = os.path.join(self.project_path, self.project_name, "Tables")
        
        lik_table_dict = TableDict(
            final_lik_dict, typ="quality", table_path=self.table_path, animal_ids=self.animal_ids
        )
        
        return final_tab_dict, lik_table_dict
    

    def scale_tables(self, tab_dict: table_dict) -> table_dict:
        """Scales all tables to mm using scaling information from arena detection.
        
        Args:
            tab_dict (table_dict): Table dictionary of pandas DataFrames containing the trajectories of all bodyparts.

        Returns:
            tab_dict (table_dict): Scaled table dictionary of pandas DataFrames containing the trajectories of all bodyparts.
        """

        with tqdm(total=len(tab_dict), desc=f"{'Rescaling tables':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            for i, (key, tab) in enumerate(tab_dict.items()):

                #load active table
                tab = get_dt(tab_dict, key)  

                #determine ratio for scaling
                scaling_ratio=1
                if self.scales is not None:
                    scaling_ratio = self.scales[key][3]/self.scales[key][2]

                #scale tables
                tab=tab*scaling_ratio

                #save scaled table
                distance_path = os.path.join(self.project_path, self.project_name, 'Tables',key, key)
                tab_dict[key] = save_dt(tab,distance_path,self.very_large_project)

                pbar.update()

        return tab_dict


    #from memory_profiler import profile
    #@profile
    def get_distances(self, tab_dict: table_dict) -> dict:
        """Compute the distances between all selected body parts over time for a table dictionary.

        Args:
            tab_dict (table_dict): Table dictionary of pandas DataFrames containing the trajectories of all bodyparts.

        Returns:
            distance_dict: Table dictionary of pandas DataFrames containing the distances between all bodyparts.

        """
        #if verbose:
        #    print("Computing distances...")


        distance_dict = {}                
        with tqdm(total=len(tab_dict), desc=f"{'Computing distances':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            for key, tab in tab_dict.items():

                #load active table
                tab = get_dt(tab_dict, key)

                #get distances for this table
                distance_tab=self.get_distances_tab(tab)

                #save distances for active table
                distance_path = os.path.join(self.project_path, self.project_name, 'Tables',key, key + '_dist')
                distance_dict[key] = save_dt(distance_tab,distance_path,self.very_large_project)

                #clean up
                del distance_tab
                pbar.update()

        return distance_dict
    
    def get_distances_tab(self, tab: pd.DataFrame) -> dict:
        """Compute the distances between all selected body parts over time for a single table.

        Args:
            tab (pd.DataFrame): Pandas DataFrame containing the trajectories of all bodyparts.

        Returns:
            distance_tab: Pandas DataFrame containing the distances between all bodyparts.

        """


        nodes = self.distances
        if nodes == "all":
            nodes = tab.columns.levels[0]

        assert [
            i in tab.columns.levels[0] for i in nodes
        ], "Nodes should correspond to existent bodyparts"

        distance_tab = deepof.utils.bpart_distance(tab)
        distance_tab = distance_tab.loc[
                :, [np.all([i in nodes for i in j]) for j in distance_tab.columns]
            ]
        if self.ego:
            distance_tab = distance_tab.loc[
                    :, [dist for dist in distance_tab.columns if self.ego in dist]
                ]
        
        # Restore original index
        distance_tab.index = tab.index

        return distance_tab


    def get_angles(self, tab_dict: table_dict) -> dict:
        """Compute all the angles between adjacent bodypart trios per video and per frame in all datasets in the given table dictionary.

        Args:
            tab_dict (table_dict): Table dictionary of pandas DataFrames containing the trajectories of all bodyparts.

        Returns:
            angle_dict: Table dictionary of pandas DataFrames containing the angles between all bodyparts.

        """
        #if verbose:
        #    print("Computing angles...")

        # Add all three-element connected sequences on each mouse
        bridges = []
        for i in self.animal_ids:
            bridges += deepof.utils.enumerate_all_bridges(self.connectivity[i])
        bridges = [i for i in bridges if len(i) == 3]

        angle_dict = {}
        try:                                        
            with tqdm(total=len(tab_dict), desc=f"{'Computing angles':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
                for key in tab_dict.keys():

                    #load table 
                    tab = get_dt(tab_dict, key)

                    dats = []
                    for clique in bridges:
                        dat = pd.DataFrame(
                            deepof.utils.angle(
                                np.transpose(
                                    np.array(tab[clique]).reshape([tab.shape[0], 3, 2])
                                ,(1, 0, 2))
                            ).T
                        )

                        dat.columns = [tuple(clique)]
                        dats.append(dat)

                    dats = pd.concat(dats, axis=1)

                    # Restore original index
                    dats.index = tab.index

                    # get path for saving
                    angle_path = os.path.join(self.project_path, self.project_name, 'Tables',key, key + '_angle')
                    angle_dict[key] = save_dt(dats,angle_path,self.very_large_project)
                    pbar.update()


        except KeyError:
            set_of_required_bps=set(item for sublist in bridges for item in sublist)
            # Workaround to allow for line breaks in key error message (key error behaves differently than all other errors)
            error_message=deepof.utils.KeyErrorMessage(
                    "Could not find expected bodypart or bodyparts: " + str(set_of_required_bps-set(tab.columns.levels[0])) + ".\n "
                    "Are you using a custom labelling scheme? Our tutorials may help!\n "
                    "In case you're not, are there multiple animals in your single-animal DLC video?\n "
                    "Make sure to set the animal_ids parameter in deepof.data.Project\n"
            )
            raise KeyError(error_message)

        
        return angle_dict

    def get_areas(self, tab_dict: table_dict) -> dict:
        """Compute all relevant areas (head, torso, back) per video and per frame in the data.

        Args:
            tab_dict (table_dict): Table dictionary of pandas DataFrames containing the trajectories of all bodyparts.

        Returns:
            all_areas_dict: Table dictionary of pandas DataFrames containing the areas (head, torso, back) between sets of bodyparts.

        """

        # landmark combinations for valid areas
        body_part_patterns = {
            "head_area": ["Nose", "Left_ear", "Left_fhip", "Spine_1"],
            "torso_area": ["Spine_1", "Right_fhip", "Spine_2", "Left_fhip"],
            "back_area": ["Spine_1", "Right_bhip", "Spine_2", "Left_bhip"],
            "full_area": [
                "Nose",
                "Left_ear",
                "Left_fhip",
                "Left_bhip",
                "Tail_base",
                "Right_bhip",
                "Right_fhip",
                "Right_ear",
            ],
        }

        all_areas_dict = {}
        not_all_area_warn = False

        # iterate over all tables
        with tqdm(total=len(tab_dict),desc=f"{'Computing areas':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            for key in tab_dict.keys():

                #load table 
                tab = get_dt(tab_dict, key)

                current_table = pd.DataFrame()

                # iterate over all animals in each table
                for animal_id in self.animal_ids:

                    if animal_id == "":
                        animal_id = None

                    # get the current table for the current animal
                    current_animal_table = tab.loc[
                        :, deepof.utils.filter_columns(tab.columns, animal_id)
                    ]

                    # iterate over all types of areas to calculate list of polygon areas for each type of area
                    areas_animal_dict = {}
                    for bp_pattern_key, bp_pattern in body_part_patterns.items():

                        try:

                            # in case of multiple animals, add animal identifier to area keys
                            if animal_id is not None:
                                bp_pattern = [
                                    "_".join([animal_id, body_part])
                                    for body_part in bp_pattern
                                ]

                            # special case full area: Use all bodyparts in that list that are available
                            if bp_pattern_key == "full_area":
                                bp_pattern = [bp for bp in bp_pattern if bp in current_animal_table.columns.levels[0]]
                                #skip, if no area can be calculated
                                if len(bp_pattern) < 3:
                                    continue

                            # create list of keys containing all table columns relevant for the current area
                            bp_x_keys = [(body_part, "x") for body_part in bp_pattern]
                            bp_y_keys = [(body_part, "y") for body_part in bp_pattern]

                            # create a 3D numpy array [NFrames, NPoints, NDis]
                            x = current_animal_table[bp_x_keys].to_numpy()
                            y = current_animal_table[bp_y_keys].to_numpy()
                            y = y[:, :, np.newaxis]
                            polygon_xy_stack = np.dstack((x, y))

                            # dictionary of area lists (each list has dimensions [NFrames]),
                            # use faster calculation for large datasets
                            if self.run_numba:
                                areas_animal_dict[
                                    bp_pattern_key
                                ] = deepof.utils.compute_areas_numba(polygon_xy_stack)
                            else:
                                areas_animal_dict[
                                    bp_pattern_key
                                ] = deepof.utils.compute_areas(polygon_xy_stack)

                        except KeyError:
                            continue

                    # change dictionary to table and check size
                    areas_table = pd.DataFrame(
                        areas_animal_dict, index=current_animal_table.index
                    )
                    if animal_id is not None:
                        areas_table.columns = [
                            "_".join([animal_id, col]) for col in areas_table.columns
                        ]

                    if areas_table.shape[1] != 4:
                        not_all_area_warn = True

                    # collect area tables for all animals
                    current_table = pd.concat([current_table, areas_table], axis=1)

                area_path = os.path.join(self.project_path, self.project_name, 'Tables',key, key + '_area')
                all_areas_dict[key] = save_dt(current_table,area_path,self.very_large_project)
                pbar.update()

        if not_all_area_warn:
            warnings.warn(
                "\033[38;5;208m"
                "It seems you're using deepof_8 or a custom labelling scheme which is missing key body parts.\n"
                "You can proceed, but not all areas will be computed.\n"
                "\033[0m"
            )    

        return all_areas_dict


    def create(
        self,
        verbose: bool = True,
        force: bool = False,
        debug: bool = True,
        test: bool = False,
        _to_extend: coordinates = None,
    ) -> coordinates:
        """Generate a deepof.Coordinates dataset using all the options specified during initialization.

        Args:
            verbose (bool): If True, prints progress. Defaults to True.
            force (bool): If True, overwrites existing project. Defaults to False.
            debug (bool): If True, saves arena detection images to disk. Defaults to False.
            test (bool): If True, creates the project in test mode (which, for example, bypasses any manual input). Defaults to False.
            _to_extend (coordinates): Coordinates object to extend with the current dataset. For internal usage only.

        Returns:
            coordinates (coordinates): Deepof.Coordinates object containing the trajectories of all bodyparts.

        """
        if verbose:
            print("Setting up project directories...")

        if force and os.path.exists(os.path.join(self.project_path, self.project_name)):
            rmtree(os.path.join(self.project_path, self.project_name))

        if not os.path.exists(os.path.join(self.project_path, self.project_name)):
            self.set_up_project_directory(debug=debug)

        # load video info
        first_key=list(self.videos.keys())[0]
        current_video_cap = cv2.VideoCapture(os.path.join(self.video_path, self.videos[first_key]))
        self.frame_rate = float(current_video_cap.get(cv2.CAP_PROP_FPS))
        current_video_cap.release()

        # load table info
        tables, quality = self.preprocess_tables()

        if self.exp_conditions is not None:
            assert (
                tables.keys() == self.exp_conditions.keys()
            ), "experimental IDs in exp_conditions do not match"

        distances = None
        angles = None
        areas = None

        # noinspection PyAttributeOutsideInit
        self.scales, self.arena_params, self.roi_dicts, self.video_resolution = self.get_arena(
            tables, debug, test
        )

        tables=self.scale_tables(tables)

        if self.distances:
            distances = self.get_distances(tables)

        if self.angles:
            angles = self.get_angles(tables)

        if self.areas:
            areas = self.get_areas(tables)

        if _to_extend is not None:

            table_path=os.path.join(self.project_path, self.project_name, "Tables")
            # Merge and expand coordinate objects
            angles = TableDict({**_to_extend._angles, **angles}, typ="angles", table_path=table_path)
            areas = TableDict({**_to_extend._areas, **areas}, typ="areas", table_path=table_path)
            distances = TableDict(
                {**_to_extend._distances, **distances}, typ="distances", table_path=table_path
            )
            tables = TableDict({**_to_extend._tables, **tables}, typ="tables", table_path=table_path)
            quality = TableDict({**_to_extend._quality, **quality}, typ="quality", table_path=table_path)

            # Merge metadata
            self.tables.update(_to_extend._table_paths)
            self.videos.update(_to_extend._videos)
            self.arena_params.update(_to_extend._arena_params)
            self.scales.update(_to_extend._scales)
            if _to_extend._roi_dicts is not None:
                self.roi_dicts.update(_to_extend._roi_dicts)


            self.version= _to_extend._version

            # Optional
            try:
                self.exp_conditions = {
                    **_to_extend._exp_conditions,
                    **self.exp_conditions,
                }
            except TypeError:
                pass

        coords = Coordinates(
            project_path=self.project_path,
            project_name=self.project_name,
            angles=angles,
            animal_ids=self.animal_ids,
            areas=areas,
            arena=self.arena,
            arena_dims=self.arena_dims,
            bodypart_graph=self.bodypart_graph,
            distances=distances,
            connectivity=self.connectivity,
            excluded_bodyparts=self.exclude_bodyparts,
            frame_rate=self.frame_rate,
            exp_conditions=self.exp_conditions,
            path=self.project_path,
            quality=quality,
            scales=self.scales,
            arena_params=self.arena_params,
            roi_dicts=self.roi_dicts,
            tables=tables,
            table_paths=self.tables,
            source_table_path=self.source_table_path,
            trained_model_path=self.trained_path,
            videos=self.videos,
            video_path=self.video_path,
            video_resolution=self.video_resolution,
            number_of_rois=self.number_of_rois,
            run_numba=self.run_numba,
            very_large_project=self.very_large_project,
            version=self.version,
        )

        #set supervised parameters via initial reset (sets values to defaults)
        coords.reset_supervised_parameters()

        # Save created coordinates to the project directory
        coords.save(timestamp=False)

        if verbose:
            print("Done!")

        return coords

    @distances.setter
    def distances(self, value):
        self._distances = value

    @ego.setter
    def ego(self, value):
        self._ego = value

    @angles.setter
    def angles(self, value):
        self._angles = value

    def extend(
        self,
        project_to_extend: coordinates,
        video_path: str = None,
        table_path: str = None,
        verbose: bool = True,
        debug: bool = True,
        test: bool = False,
    ) -> coordinates:
        """Generate a deepof.Coordinates dataset using all the options specified during initialization.

        Args:
            project_to_extend (coordinates): Coordinates object to extend with the current dataset.
            video_path (str): Path to the videos. If not specified, defaults to the project path.
            table_path (str): Path to the tracks. If not specified, defaults to the project path.
            verbose (bool): Prints progress if True. Defaults to True.
            debug (bool): Saves arena detection images to disk if True. Defaults to False.
            test (bool): Runs the project in test mode if True. Defaults to False.

        Returns:
            coordinates (coordinates): Deepof.Coordinates object containing the trajectories of all body parts.
        """

        if not video_path:
            video_path=self.video_path
        if not table_path:
            table_path=self.source_table_path
        if verbose:
            print("Loading previous project...")

        previous_project = load_project(project_to_extend)

        assert (
            os.path.abspath(os.path.join(previous_project._project_path,previous_project._project_name)) == os.path.abspath(os.path.join(self.project_path,self.project_name))
            ), ("The project to be extended and the project used for extension\n"
        "need to have the same project paths and names! Table- and video paths can differ.\n"
        "This is because Videos and Tables from the \"new\" project will get copied into the \"old\" one.")

        
        #get keys that are in new project but not in old one
        new_keys=list(set(self.videos.keys()) - set(previous_project._videos.keys()))
        # Keep only those videos and tables that were not in the original dataset
        self.videos={key:self.videos[key] for key in new_keys}
        self.tables={key:self.tables[key] for key in new_keys}


        if verbose:
            print(f"Processing data from {len(self.videos)} experiments...")

        if len(self.videos) > 0:

            #for compatibility
            if hasattr(previous_project, '_video_path') and os.path.exists(previous_project._video_path): 
                previous_vid_path = previous_project._video_path
                previous_table_path = previous_project._source_table_path
            #older projects that do not have _video_path have the videos copied to this folder
            else:
                previous_vid_path = os.path.join(previous_project._project_path, "Videos")
                previous_table_path = os.path.join(previous_project._project_path, "Tables")

            if verbose:
                print(f"Copy video data from {os.path.join(video_path)}\n")
                print(f"to {os.path.join(previous_vid_path)}")

            # Copy new videos into old directory
            for vid in tqdm(self.videos, desc="Copying videos", unit="video"):
                if (vid not in previous_project._videos 
                    and os.path.abspath(video_path) != os.path.abspath(previous_vid_path)):
                    
                    shutil.copy2(
                        os.path.join(video_path, self.videos[vid]),
                        os.path.join(previous_vid_path, self.videos[vid]),
                    )

            if verbose:
                print(f"Copy table data from {os.path.join(table_path)}\n")
                print(f"to {os.path.join(previous_table_path)}")

            # Copy new tables into old directory
            for tab in tqdm(self.tables, desc="Copying tables", unit="table"):
                if (tab not in previous_project._videos 
                    and os.path.abspath(table_path) != os.path.abspath(previous_table_path)):
                    
                    shutil.copy2(
                        os.path.join(table_path, self.tables[tab]),
                        os.path.join(previous_table_path, self.tables[tab]),
                    )
            
            self.video_path = previous_vid_path
            self.source_table_path = previous_table_path

            if verbose:
                print(f"Evaluate new data...")

            # Use the same directory as the original project 
            return self.create(
                verbose, force=False, debug=debug, test=test, _to_extend=previous_project
            )

        else:

            return previous_project


class Coordinates:
    """Class for storing the results of a ran project. Methods are mostly setters and getters in charge of tidying up the generated tables."""

    def __init__(
        self,
        project_path: str,
        project_name: str,
        arena: str,
        arena_dims: np.array,
        bodypart_graph: str,
        path: str,
        quality: dict,
        scales: dict,
        frame_rate: float,
        arena_params: dict,
        roi_dicts: dict,
        tables: dict,
        source_table_path: str,
        table_paths: List,
        trained_model_path: str,
        videos: List,
        video_path: str,
        video_resolution: dict,
        angles: dict = None,
        animal_ids: List = tuple([""]),
        areas: dict = None,
        distances: dict = None,
        connectivity: nx.Graph = None,
        excluded_bodyparts: list = None,
        exp_conditions: dict = None,
        number_of_rois: int = 0,
        run_numba: bool = False,
        very_large_project: bool = False,
        version: str = None,
    ):
        """Class for storing the results of a ran project. Methods are mostly setters and getters in charge of tidying up the generated tables.

        Args:
            project_name (str): name of the current project.
            project_path (str): path to the folder containing the motion tracking output data.
            arena (str): Type of arena used for the experiment. See deepof.data.Project for more information.
            arena_dims (np.array): Dimensions of the arena. See deepof.data.Project for more information.
            bodypart_graph (nx.Graph): Graph containing the body part connectivity. See deepof.data.Project for more information.
            path (str): Path to the folder containing the results of the experiment.
            quality (dict): Dictionary containing the quality of the experiment. See deepof.data.Project for more information.
            scales (dict): Scales used for the experiment. See deepof.data.Project for more information.
            frame_rate (float): frame rate of the processed videos.
            arena_params (dict): Dictionary containing the parameters of the arena. See deepof.data.Project for more information.
            roi_dicts (dict): Dictionary containing all rois for all videos as determined byt he user.
            tables (dict): Dictionary containing the tables of the experiment. See deepof.data.Project for more information.
            table_paths (List): List containing the paths to the tables of the experiment. See deepof.data.Project for more information.f
            trained_model_path (str): Path to the trained models used for the supervised pipeline. For internal use only.
            videos (List): List containing the videos used for the experiment. See deepof.data.Project for more information.
            video_resolution (dict): Dictionary containing the automatically detected resolution of the videos used for the experiment.
            angles (dict): Dictionary containing the angles of the experiment. See deepof.data.Project for more information.
            animal_ids (List): List containing the animal IDs of the experiment. See deepof.data.Project for more information.
            areas (dict): dictionary with areas to compute. By default, it includes head, torso, and back.
            distances (dict): Dictionary containing the distances of the experiment. See deepof.data.Project for more information.
            excluded_bodyparts (list): list of bodyparts to exclude from analysis.
            exp_conditions (dict): Dictionary containing the experimental conditions of the experiment. See deepof.data.Project for more information.
            number_of_rois (int): number of behavior rois t be drawn during project creation, default = 0,
            run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)
            very_large_project (bool): Decides if memory efficient data loading and saving should be used
            version (str): version of deepof this object was created with

        """
        self._project_path = project_path
        self._project_name = project_name
        self._animal_ids = animal_ids
        self._arena = arena
        self._arena_params = arena_params
        self._roi_dicts = roi_dicts
        self._arena_dims = arena_dims
        self._bodypart_graph = bodypart_graph
        self._excluded = excluded_bodyparts
        self._exp_conditions = exp_conditions
        self._frame_rate = frame_rate
        self._path = path
        self._quality = quality
        self._scales = scales
        self._tables = tables
        self._source_table_path = source_table_path
        self._table_paths = table_paths
        self._trained_model_path = trained_model_path
        self._videos = videos
        self._video_path = video_path
        self._video_resolution = video_resolution
        self._angles = angles
        self._areas = areas
        self._distances = distances
        self._connectivity = connectivity
        self._number_of_rois = number_of_rois
        self._run_numba = run_numba
        self._very_large_project = very_large_project
        self._version = version

    def __str__(self):  # pragma: no cover
        """Print the object to stdout."""
        lens = len(self._videos)
        return "deepof analysis of {} video{}".format(lens, ("s" if lens > 1 else ""))

    def __repr__(self):  # pragma: no cover
        """Print the object to stdout."""
        lens = len(self._videos)
        return "deepof analysis of {} video{}".format(lens, ("s" if lens > 1 else ""))

    def get_table_keys(self):
        """get the keys to all experiments in this coordinates object"""
        return self._tables.keys()

    def get_coords(
        self,
        center: str = False,
        polar: bool = False,
        speed: int = 0,
        align: str = False,
        align_inplace: bool = True,
        selected_id: str = None,
        roi_number: int = None,
        animals_in_roi: str = None,
        in_roi_criterion: str = "Center",
        file_name: str = 'coords',
        return_path: bool = False,
    ) -> table_dict:
        """Return a table_dict object with the coordinates of each animal as values.

        Args:
            center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch
            polar (bool) States whether the coordinates should be converted to polar values.
            speed (int): States the derivative of the positions to report. Speed is returned if 1, acceleration if 2, jerk if 3, etc.
            align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).
            align_inplace (bool): Only valid if align is set. Aligns the vector that goes from the origin to the selected body part with the y-axis, for all timepoints (default).
            selected_id (str): Selects a single animal on multi animal settings. Defaults to None (all animals are processed).
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
            in_roi_criterion (str): Bodypart of a mouse that has to be in the ROI to count the mouse as "inside" the ROI.
            file_name (str): Name of the file for saving
            return_path (bool): if True, Return only the path to the saving location of the processed table, if false, return the full table. 
            
        Returns:
            table_dict: A table_dict object containing the coordinates of each animal as values.

        """

        # Additional old version error for better user feedback, can get removed in a few versions
        if not (hasattr(self, "_run_numba")):
            raise ValueError(
                """You are trying to use a deepOF project that was created with version 0.6.3 or earlier.\n
            This is not supported byt he current version of deepof"""
            )

        tab_dict={}
        for key in self._tables.keys():

            tab=self.get_coords_at_key(
                key = key,
                scale = self._scales[key], 
                center = center,
                polar = polar,
                speed = speed,
                align = align,
                align_inplace = align_inplace,
                selected_id = selected_id,
                roi_number = roi_number,
                animals_in_roi = animals_in_roi,
                in_roi_criterion = in_roi_criterion,
            )
            
            # save paths for modified tables
            table_path = os.path.join(self._project_path, self._project_name, 'Tables',key, key + '_' + file_name)
            tab_dict[key] = save_dt(tab,table_path,return_path)

            #cleanup
            del tab

        return TableDict(
            tab_dict,
            typ="coords",
            table_path=os.path.join(self._project_path, self._project_name, "Tables"),
            animal_ids=self._animal_ids,
            arena=self._arena,
            arena_dims=self._scales,
            center=center,
            connectivity=self._connectivity,
            polar=polar,
            exp_conditions=self._exp_conditions,
        )


    def _load_and_prepare_data(self, key: str, quality: pd.DataFrame, data:dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the primary DataFrame and associated quality data."""
        tab = deepof.utils.deepcopy(get_dt(data, key))
        
        if quality is None:
            quality = self.get_quality().filter_videos([key])
            quality[key] = get_dt(quality, key)
            
        return tab, quality
    
    
    def _validate_inputs(self, tab: pd.DataFrame, key: str, align: str, center: str, roi_number: int):
        """Performs initial validation of function arguments."""
        if align:
            if not any(center in bp for bp in tab.columns.levels[0]):
                raise ValueError("For alignment, 'center' must be the name of a body part.")
            if not any(align in bp for bp in tab.columns.levels[0]):
                raise ValueError("'align' must be the name of a body part.")
        
        if roi_number is not None:
            if self._roi_dicts is None:
                raise ValueError("ROIs not created for this project. Define ROIs during project creation.")
            if len(self._roi_dicts.get(key, [])) < roi_number:
                raise ValueError(f"ROI {roi_number} does not exist for key '{key}'.")
            
    
    def _filter_by_roi(self, tab: pd.DataFrame, key: str, roi_number: int, animals_in_roi: List[str], in_roi_criterion: str) -> pd.DataFrame:
        """Filters the DataFrame to include only data within the specified ROI."""
        if roi_number is None:
            return tab

        # Determine which animals to check for ROI inclusion
        if animals_in_roi and isinstance(animals_in_roi, str):
            animals_to_check = [animals_in_roi]
        elif animals_in_roi: # handles list case
             animals_to_check = animals_in_roi
        else:
            animals_to_check = self._animal_ids

        roi_polygon = self._roi_dicts[key][roi_number]
        tab_mouse_positions=get_dt(self._tables, key)

        for aid in animals_to_check:
            mouse_in_polygon = deepof.utils.mouse_in_roi(tab_mouse_positions, aid, in_roi_criterion, roi_polygon, self._run_numba)
            
            # Get columns for the current animal
            mask = [any([col_sec.startswith(aid) for col_sec in col]) if isinstance(col,tuple) else col.startswith(aid) for col in tab.columns]
            aid_cols = tab.loc[:, mask].columns
            
            # Set data outside the ROI to NaN
            tab.loc[~mouse_in_polygon, aid_cols] = np.nan
            
        return tab
        
    def _select_animal_data(self, tab: pd.DataFrame, selected_ids: Union[str,list]) -> pd.DataFrame:
        """Filters the DataFrame for a single animal if selected_id is provided."""
        if isinstance(selected_ids,str):
            selected_ids=[selected_ids]
        
        if selected_ids:
            
            # Create table from all selected animal ids
            tab_out = pd.DataFrame()
            for id in selected_ids:
                tab_id=tab.loc[:, deepof.utils.filter_columns(tab.columns, id)]
                tab_out = pd.concat([tab_out, tab_id], axis=1)
        else:
            tab_out = tab

        return tab_out

    def _transform_to_polar(self, tab: pd.DataFrame, scale: np.array) -> Tuple[pd.DataFrame, tuple]:
        """Converts coordinates to polar if requested."""
        polar_scale = deepof.utils.bp2polar(scale).to_numpy().reshape(-1)
        polar_tab = deepof.utils.tab2polar(tab)
        return polar_tab, ("rho", "phi"), polar_scale
        
    def _center_coordinates(self, tab: pd.DataFrame, center: str, scale: np.array, coords: Tuple[str, str], animal_ids: List[str]) -> pd.DataFrame:
        """Centers the coordinates either to the arena or a specific body part."""
        coord_1, coord_2 = coords

        if center == "arena":
            tab.loc[:, (slice("x"), [coord_1])] -= scale[0]
            tab.loc[:, (slice("x"), [coord_2])] -= scale[1]
        elif isinstance(center, str):
            for aid in animal_ids:
                center_bp_name = f"{aid}{'_' if aid else ''}{center}"
                
                # Filter columns for the current animal
                animal_cols = [col for col in tab.columns if col[0].startswith(aid)]
                animal_tab_view = tab.loc[:, animal_cols]

                # Center on x / rho
                tab.update(
                    animal_tab_view.loc[:, (slice("x"), [coord_1])]
                    .subtract(tab[center_bp_name, coord_1], axis=0)
                )
                # Center on y / phi
                tab.update(
                    animal_tab_view.loc[:, (slice("x"), [coord_2])]
                    .subtract(tab[center_bp_name, coord_2], axis=0)
                )
        return tab
        
    def _rescale_to_video(self, tab: pd.DataFrame, scale: np.array, coords: Tuple[str, str]) -> pd.DataFrame:
        """Rescales coordinates from mm back to pixels for video output."""
        # Assuming scale[2] is video dimension and scale[3] is arena dimension in mm
        pixel_ratio = scale[2] / scale[3]
        tab.loc[:, (slice(None), list(coords))] *= pixel_ratio
        return tab

    def _align_trajectories(self, tab: pd.DataFrame, align: str, align_inplace: bool, polar: bool, animal_ids: List[str]) -> pd.DataFrame:
        """Aligns animal trajectories to a reference body part."""
        if not (align and align_inplace and not polar):
            return tab

        all_aligned_parts = []
        all_columns = []

        for aid in animal_ids:
            align_bp_name = f"{aid}{'_' if aid else ''}{align}"
            
            # Define alignment columns and remaining columns for the animal
            align_cols = [(align_bp_name, "phi" if polar else "x"),
                          (align_bp_name, "rho" if polar else "y")]
            other_cols = [col for col in tab.columns if col[0].startswith(aid) and col[0] != align_bp_name]
            
            # Reorder columns to have the alignment body part first
            ordered_cols = align_cols + other_cols
            partial_tab = tab[ordered_cols]
            
            # Perform alignment
            aligned_data = deepof.utils.align_trajectories(
                np.array(partial_tab),
                mode="all",
                run_numba=self._run_numba,
            )
            aligned_data[np.abs(aligned_data) < 1e-5] = 0.0
            
            all_aligned_parts.append(pd.DataFrame(aligned_data))
            all_columns.extend(ordered_cols)

        if not all_aligned_parts:
            return tab

        # Combine all aligned parts back into a single DataFrame
        aligned_df = pd.concat(all_aligned_parts, axis=1)
        aligned_df.index = tab.index
        aligned_df.columns = pd.MultiIndex.from_tuples(all_columns)
        
        return aligned_df

    def _calculate_derivatives(self, tab: pd.DataFrame, speed: int, frame_rate: float = None, typ: str = "coords") -> pd.DataFrame:
        """Calculates speed, acceleration, jerk, etc., based on the 'speed' parameter."""
        if not speed:
            return tab
        if frame_rate is None:
            frame_rate=self._frame_rate
            
        return deepof.utils.rolling_speed(
            tab,
            frame_rate=frame_rate,
            deriv=speed,
            typ=typ,
        )


    def get_coords_at_key(
    self,
    key: str,
    scale: np.array,
    quality: table_dict = None,
    center: str = False,
    polar: bool = False,
    speed: int = 0,
    align: str = False,
    align_inplace: bool = True,
    to_video: bool = False,
    selected_id: str = None,
    roi_number: int = None,
    animals_in_roi: str = None,
    in_roi_criterion: str = "Center",
) -> pd.DataFrame:
        """Return a pandas dataFrame with the coordinates for the selected key as values.

        Args:
            key (str): key for requested distance
            scale (np.array): scale of the current arena.
            quality: (table_dict): Quality information for current data Frame
            center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch
            polar (bool) States whether the coordinates should be converted to polar values.
            speed (int): States the derivative of the positions to report. Speed is returned if 1, acceleration if 2, jerk if 3, etc.
            align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).
            align_inplace (bool): Only valid if align is set. Aligns the vector that goes from the origin to the selected body part with the y-axis, for all timepoints (default).
            to_video (bool): Undoes the scaling to mm back to the pixel scaling from the original video 
            selected_id (str): Selects a single animal on multi animal settings. Defaults to None (all animals are processed).
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
            in_roi_criterion (str): Bodypart of a mouse that has to be in the ROI to count the mouse as "inside" the ROI.
    
        Returns:
            tab (pd.DataFrame): A data frame containing the coordinates for the selected key as values.

        """
        # 1. Load data and perform initial validation
        tab, quality = self._load_and_prepare_data(key, quality, data=self._tables)
        self._validate_inputs(tab, key, align, center, roi_number)

        # 2. Apply ROI filtering (before coordinate transformations)
        tab = self._filter_by_roi(tab, key, roi_number, animals_in_roi, in_roi_criterion)

        # 3. Select a single animal if specified
        tab = self._select_animal_data(tab, selected_id)

        # 4. Determine coordinate system and transform if necessary
        coords, current_scale = ("x", "y"), scale
        if polar:
            tab, coords, current_scale = self._transform_to_polar(tab, scale)

        # 5. Determine which animals to process for subsequent steps
        animal_ids = [selected_id] if selected_id else self._animal_ids

        # 6. Center coordinates
        if center:
            tab = self._center_coordinates(tab, center, current_scale, coords, animal_ids)
            
        # 7. Rescale to video pixels if requested
        if to_video:
            tab = self._rescale_to_video(tab, scale, coords)

        # 8. Align trajectories
        if align:
            tab = self._align_trajectories(tab, align, align_inplace, polar, animal_ids)

        # 9. Calculate speed/derivatives
        if speed:
            tab = self._calculate_derivatives(tab, speed)

        # 10. Handle missing animals based on quality data
        table_dict = deepof.utils.set_missing_animals(self, {key: tab}, quality)
        
        return table_dict[key]
           
            
    def get_distances(
        self,
        speed: int = 0,
        selected_id: str = None,
        roi_number: int = None,
        animals_in_roi: str = None,
        filter_on_graph: bool = True,
        file_name: str = 'got_distances',
        return_path: bool = False,
    ) -> table_dict:
        """Return a table_dict object with the distances between body parts animal as values.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
            filter_on_graph (bool): If True, only distances between connected nodes in the DeepOF graph representations are kept. Otherwise, all distances between bodyparts are returned.
            file_name (str): Name of the file for saving
            return_path (bool): if True, Return only the path to the processed table, if false, return the full table. 

        Returns:
            table_dict: A table_dict object with the distances between body parts animal as values.

        """

        if self._distances is not None:

            #copy only the header info, not the tables
            tabs = {}

            for key in self._distances.keys():
                                 
                tab=self.get_distances_at_key(
                    key,
                    speed=speed,
                    selected_id=selected_id,
                    roi_number=roi_number,
                    animals_in_roi=animals_in_roi,
                    filter_on_graph=filter_on_graph,
                    )
                
                # save paths for modified tables
                table_path = os.path.join(self._project_path, self._project_name, 'Tables',key, key + '_' + file_name)
                tabs[key] = save_dt(tab,table_path,return_path)

                #cleanup
                del tab

            return TableDict(
                tabs,
                typ="dists",
                table_path=os.path.join(self._project_path, self._project_name, "Tables"),
                animal_ids=self._animal_ids,
                connectivity=self._connectivity,
                exp_conditions=self._exp_conditions,
            )

        raise ValueError(
            "Distances not computed. Read the documentation for more details"
        )  # pragma: no cover


    def get_distances_at_key(
        self,
        key: str,
        quality: table_dict = None,
        speed: int = 0,
        selected_id: str = None,
        roi_number: int = None,
        animals_in_roi: str = None,
        filter_on_graph: bool = True,
    ) -> pd.DataFrame:
        """Return a pd.DataFrame with the distances between body parts of one animal as values.

        Args:
            key (str): key for requested distance
            quality: (table_dict): Quality information for current data Frame
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
            filter_on_graph (bool): If True, only distances between connected nodes in the DeepOF graph representations are kept. Otherwise, all distances between bodyparts are returned.

        Returns:
            tab (pd.DataFrame): A pd.DataFrame with the distances between body parts of one animal as values.

        """

        # 1. Load data and perform initial validation
        tab, quality = self._load_and_prepare_data(key, quality, data=self._distances)
        self._validate_inputs(tab, key, None, None, roi_number)

        # 2. Apply ROI filtering (before coordinate transformations)
        tab = self._filter_by_roi(tab, key, roi_number, animals_in_roi, "Center")

        # 3. Select a single animal if specified
        tab = self._select_animal_data(tab, selected_id)

        # 4. Calculate speed/derivatives
        if speed:
            tab = self._calculate_derivatives(tab, speed + 1, frame_rate=1, typ="dists")

        # 5. Handle missing animals based on quality data
        tab = deepof.utils.set_missing_animals(self, {key: tab}, quality)[key]

        if filter_on_graph:
            mouse_edges=deepof.utils.connect_mouse(animal_ids=self._animal_ids, graph_preset=self._bodypart_graph).edges
            sorted_edges=[]
            for edge in mouse_edges:
                edge=tuple(sorted(edge))
                sorted_edges.append(edge)
            
            tab = tab.loc[:, list(set(sorted_edges) & set(tab.columns))]
      
        return tab
    
   
    def get_angles(
        self,
        degrees: bool = False,
        speed: int = 0,
        selected_id: str = None,
        roi_number: int = None,
        animals_in_roi: str = None,
        file_name: str = 'got_angles',
        return_path: bool = False,
    ) -> table_dict:
        """Return a table_dict object with the angles between body parts animal as values.

        Args:
            degrees (bool): If True (default), the angles will be in degrees. Otherwise they will be converted to radians.
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
            file_name (str): Name of the file for saving
            return_path (bool): if True, Return only the path to the processed table, if false, return the full table. 

        Returns:
            table_dict: A table_dict object with the angles between body parts animal as values.

        """

        if self._angles is not None:

            tabs = {}

            for key in self._angles.keys():

                tab = self.get_angles_at_key(
                    key=key, 
                    degrees=degrees,
                    speed=speed,
                    selected_id=selected_id,
                    roi_number = roi_number,
                    animals_in_roi=animals_in_roi,
                )

                # save paths for modified tables
                table_path = os.path.join(self._project_path, self._project_name, 'Tables',key, key + '_' + file_name)
                tabs[key] = save_dt(tab,table_path,return_path)

                #cleanup
                del tab

            return TableDict(
                tabs,
                typ="angles",
                table_path=os.path.join(self._project_path, self._project_name, "Tables"),
                animal_ids=self._animal_ids,
                connectivity=self._connectivity,
                exp_conditions=self._exp_conditions,
            )

        raise ValueError(
            "Angles not computed. Read the documentation for more details"
        )  # pragma: no cover


    def get_angles_at_key(
    self,
    key: str,
    quality: table_dict = None,
    degrees: bool = False,
    speed: int = 0,
    selected_id: str = None,
    roi_number: int = None,
    animals_in_roi: str = None,

    ) -> pd.DataFrame:
        """Return a Dataframe with the angles between body parts for one animal as values.

        Args:
            key (str): key for requested distance
            quality: (table_dict): Quality information for current data Frame
            degrees (bool): If True (default), the angles will be in degrees. Otherwise they will be converted to radians.
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 

        Returns:
            tab (pd.DataFrame): A pd.DataFrame with the angles between body parts of one animal as values.

        """  

        # 1. Load data and perform initial validation
        tab, quality = self._load_and_prepare_data(key, quality, data=self._angles)
        self._validate_inputs(tab, key, None, None, roi_number)

        # 2. Convert unit
        if degrees:
            tab = np.degrees(tab) 

        # 3. Apply ROI filtering (before coordinate transformations)
        tab = self._filter_by_roi(tab, key, roi_number, animals_in_roi, "Center")

        # 4. Select a single animal if specified
        tab = self._select_animal_data(tab, selected_id)

        # 5. Calculate speed/derivatives
        if speed:
            tab = self._calculate_derivatives(tab, speed + 1, frame_rate=1, typ="angles")

        # 6. Handle missing animals based on quality data
        tab = deepof.utils.set_missing_animals(self, {key: tab}, quality)[key]

        return tab
    
  
    def get_areas(
            self, 
            speed: int = 0,
            selected_id: str = "all",
            roi_number: int = None,
            animals_in_roi: str = None,
            file_name: str = 'got_areas',
            return_path: bool = False,
            ) -> table_dict:
        """Return a table_dict object with all relevant areas (head, torso, back, full). Unless specified otherwise, the areas are computed for all animals.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select. "all" (default) computes the areas for all animals. Declared in self._animal_ids.
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
            file_name (str): Name of the file for saving
            return_path (bool): if True, Return only the path to the processed table, if false, return the full table. 

        Returns:
            table_dict: A table_dict object with the areas of the body parts animal as values.
        """

        if self._areas is not None:
            
            tabs = {}

            for key in self._areas.keys():

                tab = self.get_areas_at_key(
                    key=key, 
                    speed = speed,
                    selected_id = selected_id,
                    roi_number = roi_number,
                    animals_in_roi = animals_in_roi,
                )

                # save paths for modified tables
                table_path = os.path.join(self._project_path, self._project_name, 'Tables',key, key + '_' + file_name)
                tabs[key] = save_dt(tab,table_path,return_path)

                #cleanup
                del tab

            areas = TableDict(
                tabs,
                typ="areas",
                table_path=os.path.join(self._project_path, self._project_name, "Tables"),
                animal_ids=self._animal_ids,
                connectivity=self._connectivity,
                exp_conditions=self._exp_conditions,
            )

            return areas

        raise ValueError(
            "Areas not computed. Read the documentation for more details"
        )  # pragma: no cover
    

    def get_areas_at_key(
        self,
        key: str,
        quality: table_dict = None,
        speed: int = 0,
        selected_id: str = "all",
        roi_number: int = None,
        animals_in_roi: str = None,
        ) -> table_dict:
        """Return a pd.DataFrame with all relevant areas (head, torso, back, full). Unless specified otherwise, the areas are computed for all animals.

        Args:
            key (str): key for requested distance
            quality: (table_dict): Quality information for current data Frame
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select. "all" (default) computes the areas for all animals. Declared in self._animal_ids.
            roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
            animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 

        Returns:
            tab (pd.DataFrame): A pd.DataFrame object with the areas of the body parts animal as values.
        """

        # 1. Load data and perform initial validation
        tab, quality = self._load_and_prepare_data(key, quality, data=self._areas)
        self._validate_inputs(tab, key, None, None, roi_number)

        # 2. Adjust ids
        if selected_id == "all":
            selected_ids = self._animal_ids
        else:
            selected_ids = [selected_id]

        # 3. Apply ROI filtering (before coordinate transformations)
        tab = self._filter_by_roi(tab, key, roi_number, animals_in_roi, "Center")

        # 4. Select a single animal if specified
        tab = self._select_animal_data(tab, selected_ids)

        # 5. Calculate speed/derivatives
        if speed:
            tab = self._calculate_derivatives(tab, speed + 1, frame_rate=1, typ="areas")

        # 6. Handle missing animals based on quality data
        tab = deepof.utils.set_missing_animals(self, {key: tab}, quality)[key]
    
        return tab


    def get_videos(self, full_paths: bool = False, play: bool = False):
        """Returns the videos associated with the dataset as a dictionary."""
        if play:  # pragma: no cover
            raise NotImplementedError  
        if full_paths:
            out={key: os.path.join(self._video_path, video) for key, video in self._videos.items()}
        else:
            out=self._videos

        return out

    def get_start_times(self):
        """Returns the start time for each table in a dictionary"""
        start_times = {}
        for key in self._tables:
            start_times[key] = get_dt(self._tables,key, only_metainfo=True, load_index=True)['start_time']
        return start_times

    def get_end_times(self):
        """Returns the end time for each table in a dictionary"""
        end_times = {}
        for key in self._tables:
            end_times[key] = get_dt(self._tables,key, only_metainfo=True, load_index=True)['end_time']
        return end_times

    def get_table_lengths(self):
        """Returns the length for each table in a dictionary"""
        table_lengths = {}
        for key in self._tables:
            table_lengths[key] = get_dt(self._tables,key, only_metainfo=True)['num_rows']
        return table_lengths

    @property
    def get_exp_conditions(self):
        """Return the stored dictionary with experimental conditions per subject."""
        return self._exp_conditions

    def load_exp_conditions(self, filepath):  # pragma: no cover
        """Load experimental conditions from a wide-format csv table.

        Args:
            filepath (str): Path to the file containing the experimental conditions.

        """
        exp_conditions = pd.read_csv(filepath, index_col=0)
        exp_conditions = {
            exp_id: pd.DataFrame(
                exp_conditions.loc[exp_conditions.iloc[:, 0] == exp_id, :].iloc[0, 1:]
            ).T
            for exp_id in exp_conditions.iloc[:, 0]
        }
        self._exp_conditions = exp_conditions

        # Save loaded conditions within project
        self.save(timestamp=False)


    def get_quality(self):
        """Retrieve a dictionary with the tagging quality per video, as reported by DLC or SLEAP."""
        return TableDict(
            self._quality,
            typ="quality",
            table_path=os.path.join(self._project_path, self._project_name, "Tables"),
            animal_ids=self._animal_ids)

    @property
    def get_arenas(self):
        """Retrieve all available information associated with the arena."""
        return self._arena, self._arena_dims, self._scales

    def edit_arenas(
        self, video_keys: list = None, arena_type: str = None, verbose: bool = True
    ):  # pragma: no cover
        """Tag the arena in the videos.

        Args:
            video_keys (list): A list of keys for videos to reannotate. If None, all videos are loaded.
            arena_type (str): The type of arena to use. Must be one of "polygonal-manual", "circular-manual", or "circular-autodetect". If None (default), the arena type specified when creating the project is used.
            verbose (bool): Whether to print the progress of the annotation.

        """
        if video_keys is None:
            video_keys = self._videos
        if arena_type is None:
            arena_type = self._arena

        #create dictionary based on entered keys if keys actually exist in videos
        videos_to_update={key: self._videos[key] for key in video_keys if key in self._videos.keys()}

        if verbose:
            print(
                "Editing {} arena{}".format(len(video_keys), "s" if len(video_keys) > 1 else "")
            )

        edited_scales, edited_arena_params, edited_roi_dicts, _ = deepof.arena_utils.get_arenas(
            coordinates=self,
            tables=self._tables,
            arena=arena_type,
            arena_dims=self._arena_dims,
            number_of_rois=self._number_of_rois,
            segmentation_model_path=None,
            video_path=self._video_path,
            videos=videos_to_update,
        )

        # update the scales and arena parameters
        for key in video_keys:
            self._scales[key] = edited_scales[key]
            self._arena_params[key] = edited_arena_params[key]
            self._roi_dicts[key] = edited_roi_dicts[key]

        self.save(timestamp=False)

        if verbose:
            print("Done!")

    def save(self, file=None, filename: str = None, timestamp: bool = True):
        """Save the current state of the Coordinates object to a pickled file.

        Args:
            file (obj): optional Objet to save, if None, project gets saved
            filename (str): Name of the pickled file to store. If no name is provided, a default is used.
            timestamp (bool): Whether to append a time stamp at the end of the output file name.
        """

        pkl_out = "{}{}.pkl".format(
            os.path.join(
                self._project_path,
                self._project_name,
                "Coordinates",
                (filename if filename is not None else "deepof_coordinates"),
            ),
            (f"_{int(time())}" if timestamp else ""),
        )

        with open(pkl_out, "wb") as handle:
            if file is None:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @deepof.data_loading._suppress_warning(
        warn_messages=[
            "adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0."
        ]
    )
    def get_graph_dataset(
        self,
        animal_id: str = None,
        #binning info
        bin_size=None,
        bin_index=None,
        precomputed_bins=None,
        samples_max: int = 227272,  #corresponds to 1GB of memory when using default settings
        #other info
        precomputed_tab_dict: table_dict = None,
        center: str = False,
        polar: bool = False,
        align: str = None,
        preprocess: bool = True,
        return_as_paths: bool = None,
        **kwargs,
    ) -> table_dict:
        """Generate a dataset with all specified features.

        Args:
            animal_id (str): Name of the animal to process. If None (default) all animals are included in a multi-animal graph.
            bin_size (Union[int,str]): bin size for time filtering. Will select (up to) the first 2.5 hours of data per default
            bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
            precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
            samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
            precomputed_tab_dict (table_dict): table_dict object for further graph processing. None (default) builds it on the spot.
            center (str): Name of the body part to which the positions will be centered. If false, raw data is returned; if 'arena' (default), coordinates are centered on the pitch.
            polar (bool) States whether the coordinates should be converted to polar values.
            align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).
            preprocess (bool): whether to preprocess the data to pass to autoencoders. If False, node features and distance-weighted adjacency matrices on the raw data are returned.
            return_as_paths (bool): wheter the preprocessed data should only returned as a path to the data storage location or loaded in the RAM in full

        Returns:
            merged_features: A graph-based dataset.

        """

        if return_as_paths is None:
            return_as_paths = self._very_large_project

        N_steps=5
        with tqdm(total=N_steps, desc=f"{'Loading tables':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="step") as pbar:
                               
            pbar.set_postfix(step="Loading coords")

            # Get all relevant features
            coords = self.get_coords(
                selected_id=animal_id, center=center, align=align, polar=polar, return_path=return_as_paths,
            )

            pbar.update()
            pbar.set_postfix(step="Loading speeds")

            speeds = self.get_coords(selected_id=animal_id, speed=1, file_name='speed', return_path=return_as_paths)

            pbar.update()
            pbar.set_postfix(step="Loading distances")

            dists = self.get_distances(selected_id=animal_id, return_path=return_as_paths)

            pbar.update()
            pbar.set_postfix(step="Loading angles")

            #angles = self.get_angles(selected_id=animal_id, return_path=return_as_paths)

            # Merge and extract names
            tab_dict = coords.merge(
                speeds,
                #angles,
                dists,
                save_as_paths=return_as_paths
                )
            
            pbar.update()
            pbar.set_postfix(step="Get graph info")

            if precomputed_tab_dict is not None:  # pragma: no cover
                tab_dict = precomputed_tab_dict

            # Get corresponding feature graph
            graph = deepof.utils.connect_mouse(
                animal_ids=(self._animal_ids if animal_id is None else animal_id),
                exclude_bodyparts=(
                    list(
                        set(
                            [
                                re.sub(
                                    r"|".join(
                                        map(re.escape, [i + "_" for i in self._animal_ids])
                                    ),
                                    "",
                                    bp,
                                )
                                for bp in self._excluded
                            ]
                        )
                    )
                    if (self._animal_ids is not None and self._animal_ids[0])
                    else self._excluded
                ),
                graph_preset=self._bodypart_graph,
            )
            # Compares with existing table and removes all nodes from graph that are not part of the table columns
            table_metadata = get_dt(self._tables, list(self._tables.keys())[0], only_metainfo=True)
            table_columns = table_metadata["columns"]
            table_column_names = [column[0] for column in table_columns]
            nodes_to_remove=list(set(list(graph.nodes)) - set(table_column_names))
            for node in nodes_to_remove:
                graph.remove_node(node)

            tab_dict._connectivity = graph

            #read table metadata
            if type(list(dists.values())[0]) == dict:
                edge_feature_names = get_dt(dists,list(dists.keys())[0], only_metainfo=True)['columns']
            else:
                edge_feature_names = list(list(dists.values())[0].columns)

            if type(list(tab_dict.values())[0]) == dict:
                feature_names = pd.Index(get_dt(tab_dict,list(tab_dict.keys())[0], only_metainfo=True)['columns'])
            else:
                feature_names = pd.Index([i for i in list(tab_dict.values())[0].columns])

            node_feature_names = (
                [(i, "x") for i in list(graph.nodes())]
                + [(i, "y") for i in list(graph.nodes())]
                + list(graph.nodes())
                #+ get_dt(angles,list(angles.keys())[0], only_metainfo=True)['columns'][0:11]
            )

            # Sort indices to have always the same node order
            node_sorting_indices = []
            edge_sorting_indices = []
            for n in node_feature_names:
                for j, f in enumerate(feature_names):
                    if n == f:
                        node_sorting_indices.append(j)

            inner_link_bool_mask = []
            for e in [tuple(sorted(e)) for e in list(graph.edges)]:
                for j, f in enumerate(edge_feature_names):
                    if e == f:
                        edge_sorting_indices.append(j)

                if len(self._animal_ids) > 1:
                    inner_link_bool_mask.append(
                        len(set([node.split("_")[0] for node in e])) == 1
                    )
            
            pbar.update()

        # Create graph datasets
        if preprocess:
            to_preprocess, shapes, global_scaler = tab_dict.preprocess(
                coordinates=self,
                #binning info, explicitely stated as otherwise warnings seem to get suppressed
                bin_size=bin_size,
                bin_index=bin_index,
                precomputed_bins=precomputed_bins,
                samples_max=samples_max,
                **kwargs,
                save_as_paths=return_as_paths
                )
   
            shapes=[]
            with tqdm(total=len(to_preprocess),desc=f"{'Reshaping':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
                for k in range(0,len(to_preprocess)):
                
                    num_rows=0
                    for key in to_preprocess[k].keys():

                        #load table if not already loaded
                        result = get_dt(to_preprocess[k], key, return_path=True)
                        if result and len(result)==2 :
                            tab = result[0]
                            table_path = result[1]
                        dataset = (
                            tab[:, :, ~feature_names.isin(edge_feature_names)][
                                :, :, node_sorting_indices
                            ],
                            tab[:, :, feature_names.isin(edge_feature_names)][
                                :, :, edge_sorting_indices
                            ],
                        )
                        num_rows=num_rows+tab.shape[0]
                    

                        # save paths for modified tables
                        if type(table_path) == dict:
                            table_path = os.path.join(os.path.dirname(table_path.get("duckdb_file")) , table_path.get("table"))                    
                        to_preprocess[k][key] = save_dt(dataset,table_path,return_as_paths) 
                    #collect shapes
                    if len(to_preprocess[k].keys())>0:
                        shapes=shapes+[(num_rows, dataset[0].shape[1],dataset[0].shape[2]),(num_rows, dataset[1].shape[1],dataset[1].shape[2])]
                    else:
                        shapes=(0,)
                    pbar.update()
                shapes=tuple(shapes)

        else:  # pragma: no cover
            to_preprocess = tab_dict #np.concatenate(list(tab_dict.values()))

            shapes=[]
            num_rows=0
            with tqdm(total=len(to_preprocess), desc=f"{'Reshaping':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="array") as pbar:
                for key in to_preprocess.keys():

                    tab, table_path = get_dt(to_preprocess, key, return_path=True) 

                    tab = np.array(tab)

                    # Split node features (positions, speeds) from edge features (distances)
                    dataset = (
                        tab[:, ~feature_names.isin(edge_feature_names)][
                            :, node_sorting_indices
                        ].reshape([tab.shape[0], len(graph.nodes()), -1], order="F"),
                        deepof.utils.edges_to_weighted_adj(
                            nx.adj_matrix(graph).todense(),
                            tab[:, feature_names.isin(edge_feature_names)][
                                :, edge_sorting_indices
                            ],
                        ),
                    )
                    num_rows=num_rows+dataset.shape[0]


                    # save paths for modified tables
                    to_preprocess[key] = save_dt(dataset,table_path,return_as_paths)
                    pbar.update()


                shapes=shapes+[(num_rows, dataset[0].shape[1],dataset[0].shape[2]),(num_rows, dataset[1].shape[1],dataset[1].shape[2])]
                shapes=tuple(shapes)
                
        try:
            return (
                to_preprocess,
                shapes,
                nx.adjacency_matrix(graph).todense(),
                tab_dict,
                global_scaler,
            )
        except UnboundLocalError:
            return to_preprocess, nx.adjacency_matrix(graph).todense(), tab_dict

    # noinspection PyDefaultArgument
    def get_supervised_parameters(self) -> dict:
        """Return the most frequent behaviour in a window of window_size frames.

        Args:
            hparams (dict): dictionary containing hyperparameters to overwrite

        Returns:
            defaults (dict): dictionary with overwritten parameters. Those not specified in the input retain their default values

        """
        if not hasattr(self, '_supervised_parameters'):
            self.reset_supervised_parameters()


        return copy.copy(self._supervised_parameters)


    # noinspection PyDefaultArgument
    def reset_supervised_parameters(self) -> dict:
        """Return the most frequent behaviour in a window of window_size frames.

        Args:
            hparams (dict): dictionary containing hyperparameters to overwrite

        Returns:
            defaults (dict): dictionary with overwritten parameters. Those not specified in the input retain their default values

        """
        defaults = {
            "close_contact_tol": 25,                           # Body parts need to be 25 mm apart or closer
            "side_contact_tol": 50,                            # Sides need to be 50 mm apart or closer
            "median_filter_width": int(self._frame_rate/2),    # Width of median filter, determins smoothing degree of behavior signals
            "follow_frames": int(self._frame_rate/2),          # Frames over which following is considered, Half of a second
            "min_follow_frames": int(self._frame_rate/4),      # Minimum time mouse needs to follow, Quarter of a second
            "follow_tol": 25,                                  # Tail base of followed mouse needs to be 25 mm or closer to Nose of following mouse up to follow_frames in the past
            "climb_tol": 0.15,                                 # If mouse nouse is 15% or more of it's length outside of the arena for it to count as climbing
            "sniff_arena_tol": 12.5,                           # Noses needs to be 12.5 mm apart from the arena edge or closer
            "min_immobility": int(self._frame_rate),           # Min Time interval the mouse needs to be immobile to be counted as immobility, 1 second 
            #"max_immobility": 120*int(self._frame_rate),      # Max Time interval the mouse needs to be immobile to be counted as immobility, 2 minutes (anything longer is counted as "sleeping")                              
            "stationary_threshold": 40,                        # 40 mm per s, Speed below which the mouse is considered to only move neglegibly, before: 2 pixel per frame
            "nose_likelihood": 0.85,                           # Minimum degree of certainty of the Nose position prediction, relevant for lookaround and sniffing
        }

        self._supervised_parameters = defaults 
        self.save(timestamp=False)   


    # noinspection PyDefaultArgument
    def set_supervised_parameters(self, hparams: dict = {}):
        """Return the most frequent behaviour in a window of window_size frames.

        Args:
            hparams (dict): dictionary containing hyperparameters to overwrite

        Returns:
            defaults (dict): dictionary with overwritten parameters. Those not specified in the input retain their default values

        """
        params = self.get_supervised_parameters()

        for k, v in hparams.items():
            if k in list(params.keys()):
                params[k] = v
            else:
                warning_message = (
                "\033[38;5;208m\n"
                "Warning! At least one of the given parameter names does not match any supervised parameter names!"
                "\nPlease check if you spelled the parameter name correctly!"
                "\033[0m"
            )
                warnings.warn(warning_message)

        self._supervised_parameters = params
        self.save(timestamp=False)  

    # noinspection PyDefaultArgument
    #from memory_profiler import profile
    #@profile
    @deepof.data_loading._suppress_warning(
        warn_messages=[
            "Creating an ndarray from ragged nested sequences .* is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray."
        ]
    )
    def supervised_annotation(
        self,
        center: str = "Center",
        align: str = "Spine_1",
    ) -> table_dict:
        """Annotates coordinates with behavioral traits using a supervised pipeline.

        Args:
            center (str): Body part to center coordinates on. "Center" by default.
            align (str): Body part to rotationally align the body parts with. "Spine_1" by default.
            video_output (bool): It outputs a fully annotated video for each experiment indicated in a list. If set to "all", it will output all videos. False by default.
            frame_limit (int): Only applies if video_output is not False. Indicates the maximum number of frames per video to output.
            debug (bool): Only applies if video_output is not False. If True, all videos will include debug information, such as the detected arena and the preprocessed tracking tags.
            n_jobs (int): Number of jobs to use for parallel processing. Only applies if video_output is not set to False. 

        Returns:
            table_dict: A table_dict object with all supervised annotations per experiment as values.

        """
        # Additional old version error for better user feedback, can get removed in a few versions
        if not (hasattr(self, "_run_numba")):
            raise ValueError(
                """You are trying to use a deepOF project that was created with version 0.6.3 or earlier.\n
            This is not supported by the current version of deepof"""
            )
        
        # get immobility classifer
        self._trained_model_path = resource_filename(__name__, "trained_models")    
        immobility_estimator = deepof.utils.load_precompiled_model(
            None,
            download_path="https://datashare.mpcdf.mpg.de/s/kiLpLy1dYNQrPKb/download",
            model_path=os.path.join("trained_models", "deepof_supervised","deepof_supervised_huddle_estimator.pkl"),
            model_name="Immobility classifier"
            ) 
    
        N_preprocessing_steps=2+len(self._animal_ids)
        N_processing_steps=len(self._tables.keys())
        
        with tqdm(total=N_preprocessing_steps, desc=f"{'data preprocessing':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="step") as pbar:

            pbar.set_postfix(step="Loading raw coords")

            tag_dict = {}
            params = self.get_supervised_parameters()

            #get all kinds of tables
            raw_coords = self.get_coords(center=None, file_name='raw', return_path=self._very_large_project)
            pbar.update()
            pbar.set_postfix(step="Loading coords")
            def load_coords():
                try:
                    return self.get_coords(center=center, align=align, return_path=self._very_large_project)
                except ValueError:
                    try:
                        return self.get_coords(center="Center", align="Spine_1", return_path=self._very_large_project)
                    except ValueError:
                        return self.get_coords(center="Center", align="Nose", return_path=self._very_large_project)

            # Disable warnings manually around ThreadPoolExecutor
            # Reason: ThreadPoolExecutor will break the warnings if warning decorator functions are accessed in parallel
            with warnings.catch_warnings(record=True) as caught_warnings:  

                # Disable warning decorators
                token = suppress_warnings_context.set(False)

                # Manually set warnings to ignore
                warning = "Creating an ndarray from ragged nested sequences .* is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray."
                ignore_warning=f"(\n)?.*{warning}.*"
                warnings.filterwarnings("ignore", message=ignore_warning)  
                
                # Parallel execution of data gathering
                with ThreadPoolExecutor() as executor:
                    future_coords = executor.submit(load_coords)
                    future_speeds = executor.submit(self.get_coords, speed=1, file_name='speeds', return_path=self._very_large_project)
                    future_dists = executor.submit(self.get_distances, return_path=self._very_large_project)
                    future_angles = executor.submit(self.get_angles, return_path=self._very_large_project)

                    coords = future_coords.result()
                    speeds = future_speeds.result()
                    dists = future_dists.result()
                    angles = future_angles.result()
                pbar.update()
                pbar.set_postfix(step="Loading distances")
            
            # Display caught and not ignored warnings
            for caught_warning in caught_warnings:
                warnings.warn(caught_warning.message)
            
            # Reset warning decorators
            suppress_warnings_context.reset(token)


            #get kinematics
            pbar.set_postfix(step="Loading kinematics")
            if len(self._animal_ids) <= 1:
                features_dict = (
                    deepof.post_hoc.align_deepof_kinematics_with_unsupervised_labels(
                        self, center=center, align=align, include_angles=False, return_path=self._very_large_project
                    )
                )
                pbar.update() 
            else:  # pragma: no cover
                features_dict={}
                for _id in self._animal_ids:
                    features_dict[_id]=deepof.post_hoc.align_deepof_kinematics_with_unsupervised_labels(
                        self,
                        center=center,
                        align=align,
                        animal_id=_id,
                        include_angles=False,
                        file_name='kinematics_'+_id,
                        return_path=self._very_large_project
                    )
                    pbar.update() 

        with tqdm(total=N_processing_steps, desc=f"{'supervised annotations':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            # noinspection PyTypeChecker
            for key in self._tables.keys():
                               
                pbar.set_postfix(step="supervised tagging")

                # Remove indices and add at the very end, to avoid conflicts if
                # frame_rate is specified in project
                duckdb_file = ''
                table = ''
                if isinstance(raw_coords[key], dict):
                    duckdb_file = raw_coords[key].get("duckdb_file")
                    table = raw_coords[key].get("table")
                if isinstance(raw_coords[key], str) or (isinstance(duckdb_file, str) and isinstance(table, str)):
                    tag_index = get_dt(raw_coords,key, only_metainfo=True, load_index=True)['index_column']
                else:
                    tag_index = raw_coords[key].index

                supervised_tags = deepof.annotation_utils.supervised_tagging(
                    self,
                    raw_coords=raw_coords,
                    coords=coords,
                    dists=dists,
                    angles=angles,
                    full_features=features_dict,
                    speeds=speeds,
                    key=key, 
                    immobility_estimator=immobility_estimator,
                    center=center,
                    params=params,
                    run_numba=self._run_numba,
                )

                supervised_tags.index = tag_index

                pbar.set_postfix(step="post processing")

                quality=self.get_quality().filter_videos([key])
                quality[key] = get_dt(quality,key)
                table_dict={key:supervised_tags}
                # Set table_dict to NaN if animals are missing
                table_dict = deepof.utils.set_missing_animals(
                    self, 
                    table_dict, 
                    quality,
                    animal_ids=self._animal_ids + ["supervised"]
                    )
                supervised_tags = table_dict[key]

                # Add missing tags to all animals
                presence_masks = deepof.utils.compute_animal_presence_mask(quality)   
                for animal in self._animal_ids:
                    supervised_tags[
                        "{}missing".format(("{}_".format(animal) if animal else ""))
                    ] = (1 - presence_masks[key][animal].values)

                # save paths for modified tables
                table_path = os.path.join(coords._table_path,key, key + '_' + "supervised_annotations")
                tag_dict[key] = save_dt(supervised_tags,table_path,self._very_large_project) 

                pbar.update() 

        del features_dict, dists, speeds, coords, raw_coords

        supervised_annotation_instance = TableDict(
            tag_dict,
            typ="supervised",
            table_path=os.path.join(self._project_path, self._project_name, "Tables"),
            animal_ids=self._animal_ids,
            arena=self._arena,
            arena_dims=self._arena_dims,
            connectivity=self._connectivity,
            exp_conditions=self._exp_conditions,
        )
        self.save(file=supervised_annotation_instance, filename="supervised_annotations", timestamp=False)

        return supervised_annotation_instance

    def deep_unsupervised_embedding(
        self,
        preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        adjacency_matrix: np.ndarray = None,
        #binning info
        bin_size=None,
        bin_index=None,
        precomputed_bins=None,
        samples_max=None,
        #model info
        embedding_model: str = "VaDE",
        encoder_type: str = "recurrent",
        batch_size: int = 64,
        latent_dim: int = 4,
        epochs: int = 150,
        log_history: bool = True,
        log_hparams: bool = False,
        n_components: int = 10,
        kmeans_loss: float = 0.0,
        temperature: float = 0.1,
        contrastive_similarity_function: str = "cosine",
        contrastive_loss_function: str = "nce",
        beta: float = 0.1,
        tau: float = 0.1,
        output_path: str = "",
        pretrained: str = False,
        save_checkpoints: bool = False,
        save_weights: bool = True,
        input_type: str = False,
        run: int = 0,
        kl_annealing_mode: str = "linear",
        kl_warmup: int = 15,
        reg_cat_clusters: float = 0.0,
        recluster: bool = False,
        interaction_regularization: float = 0.0,
        **kwargs,
    ) -> Tuple:  # pragma: no cover
        """Annotates coordinates using a deep unsupervised autoencoder.

        Args:
            preprocessed_object (tuple): Tuple containing a preprocessed object (X_train, y_train, X_test, y_test).
            adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
            bin_size (Union[int,str]): bin size for time filtering.
            bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
            precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
            samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
            embedding_model (str): Name of the embedding model to use. Must be one of VQVAE (default), VaDE, or contrastive.
            encoder_type (str): Encoder architecture to use. Must be one of "recurrent", "TCN", and "transformer".
            batch_size (int): Batch size for training.
            latent_dim (int): Dimention size of the latent space.
            epochs (int): Maximum number of epochs to train the model. Actual training might be shorter, as the model will stop training when validation loss stops decreasing.
            log_history (bool): Whether to log the history of the model to TensorBoard.
            log_hparams (bool): Whether to log the hyperparameters of the model to TensorBoard.
            n_components (int): Number of latent clusters for the embedding model to use.
            kmeans_loss (float): Weight of the gram loss, which adds a regularization term to VaDE and VQVAE models which penalizes the correlation between the dimensions in the latent space.
            temperature (float): temperature parameter for the contrastive loss functions. Higher values put harsher penalties on negative pair similarity.
            contrastive_similarity_function (str): similarity function between positive and negative pairs. Must be one of 'cosine' (default), 'euclidean', 'dot', and 'edit'.
            contrastive_loss_function (str): contrastive loss function. Must be one of 'nce' (default), 'dcl', 'fc', and 'hard_dcl'. See specific documentation for details.
            beta (float): Beta (concentration) parameter for the hard_dcl contrastive loss. Higher values lead to 'harder' negative samples.
            tau (float): Tau parameter for the dcl and hard_dcl contrastive losses, indicating positive class probability.
            output_path (str): Path to save the trained model and all log files.
            pretrained (str): Whether to load a pretrained model. If False, model is trained from scratch. If not, must be the path to a saved model.
            save_checkpoints (bool): Whether to save checkpoints of the model during training. Defaults to False.
            save_weights (bool): Whether to save the weights of the model during training. Defaults to True.
            input_type (str): Type of the preprocessed_object passed as the first parameter. See deepof.data.TableDict for more details.
            run (int): Run number for the model. Used to save the model and log files. Optional.
            kl_annealing_mode (str): Mode of the KL annealing. Must be one of "linear", or "sigmoid".
            kl_warmup (int): Number of epochs to warm up the KL annealing.
            reg_cat_clusters (bool): whether to penalize uneven cluster membership in the latent space, by minimizing the KL divergence between cluster membership and a uniform categorical distribution.
            recluster (bool): whether to recluster after training using a Gaussian Mixture Model. Only valid for VaDE.
            interaction_regularization (float): weight of the interaction regularization term for all encoders.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            Tuple: Tuple containing all trained models. See specific model documentation under deepof.models for details.
        """
        
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
        # not needed anymore after refactor
        # extract from Tuple
        preprocessed_train, _= preprocessed_object
        pt_shape=get_dt(preprocessed_train,list(preprocessed_train.keys())[0], only_metainfo=True)['shape']

        #get available memory -10% as buffer
        available_mem=psutil.virtual_memory().available*0.9
        #calculate maximum number of rows that fit in memory based on table info 
        N_rows_max=int(available_mem/((pt_shape[1]+11)*pt_shape[2]*8))
        if samples_max is None:
            samples_max=N_rows_max
        elif samples_max>N_rows_max:
            warning_message = (
            "\033[38;5;208m\n"
            "Warning! The selected number of samples may exceed your available memory."
            "\033[0m"
        )
            warnings.warn(warning_message)

        bin_info=_preprocess_time_bins(coordinates=self, bin_size=bin_size,bin_index=bin_index,precomputed_bins=precomputed_bins, tab_dict_for_binning=preprocessed_object[0], samples_max=samples_max)
        bin_info_test=_preprocess_time_bins(coordinates=self, bin_size=bin_size,bin_index=bin_index,precomputed_bins=precomputed_bins, tab_dict_for_binning=preprocessed_object[1], samples_max=samples_max)
        bin_info.update(bin_info_test)

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

        ###
        # Improve after refactor
        # Select path to apropriate pretrained model
        if pretrained:
            pretrained_path = os.path.join(
                self._project_path,
                self._project_name,
                "Trained_models",
                "trained_weights",
            )
            pretrained = os.path.join(
                pretrained_path,
                (
                    pretrained
                    if isinstance(pretrained, str)
                    else [
                        w
                        for w in os.listdir(pretrained_path)
                        if embedding_model in w
                        and encoder_type in w
                        and "encoding={}".format(latent_dim) in w
                        and "k={}".format(n_components)
                    ][0]
                ),
            )

        try:
            trained_models = deepof.clustering.model_utils_new.embedding_model_fitting(
                preprocessed_object=preprocessed_object,
                adjacency_matrix=adjacency_matrix,
                embedding_model=embedding_model,
                encoder_type=encoder_type,
                batch_size=batch_size,
                latent_dim=latent_dim,
                epochs=epochs,
                log_history=log_history,
                log_hparams=log_hparams,
                n_components=n_components,
                kmeans_loss=kmeans_loss,
                temperature=temperature,
                contrastive_similarity_function=contrastive_similarity_function,
                contrastive_loss_function=contrastive_loss_function,
                beta=beta,
                tau=tau,
                output_path=os.path.join(
                    self._project_path,
                    self._project_name,
                    output_path,
                    "Trained_models",
                ),
                data_path=os.path.join(self._project_path, self._project_name, 'Tables'),
                pretrained=pretrained,
                save_checkpoints=save_checkpoints,
                save_weights=save_weights,
                input_type=input_type,
                bin_info=bin_info,
                run=run,
                kl_annealing_mode=kl_annealing_mode,
                kl_warmup=kl_warmup,
                reg_cat_clusters=reg_cat_clusters,
                recluster=recluster,
                interaction_regularization=interaction_regularization,
                **kwargs,
            )
        except IndexError:
            raise ValueError(
                "No pretrained model found for the given parameters. Please train a model first."
            )

        # returns a list of trained tensorflow models
        return trained_models



class TableDict(dict):
    """Main class for storing a single dataset as a dictionary with individuals as keys and pandas.DataFrames as values.

    Includes methods for generating training and testing datasets for the supervised and unsupervised models.
    """

    def __init__(
        self,
        tabs: Dict,
        typ: str,
        table_path: str = None,
        arena: str = None,
        arena_dims: np.array = None,
        animal_ids: List = tuple([""]),
        center: str = None,
        connectivity: nx.Graph = None,
        polar: bool = None,
        exp_conditions: dict = None,
        shapes: Dict = {},
    ):
        """Store single datasets as dictionaries with individuals as keys and pandas.DataFrames as values.

        Includes methods for generating training and testing datasets for the autoencoders.

        Args:
            tabs (Dict): Dictionary of pandas.DataFrames with individual experiments as keys.
            typ (str): Type of the dataset. Examples are "coords", "dists", and "angles". For logging purposes only.
            table_path (str): Path to the root directory that is going to be used to save table iterations.
            arena (str): Type of the arena. Must be one of "circular-autodetect", "circular-manual", or "polygon-manual". Handled internally.
            arena_dims (np.array): Dimensions of the arena in mm.
            animal_ids (list): list of animal ids.
            center (str): Type of the center. Handled internally.
            connectivity (nx.Graph): Bodypart graph of a mouse.
            polar (bool): Whether the dataset is in polar coordinates. Handled internally.
            exp_conditions (dict): dictionary with experiment IDs as keys and experimental conditions as values.
            shapes (Dict): Dictionary containing the shapes of all stored tables

        """
        super().__init__(tabs)
        self._type = typ
        self._center = center
        self._connectivity = connectivity
        self._polar = polar
        self._arena = arena
        self._arena_dims = arena_dims
        self._animal_ids = animal_ids
        self._exp_conditions = exp_conditions
        self._table_path = table_path
        self._shapes = shapes


    def filter_videos(self, keys: list) -> table_dict:
        """Return a subset of the original table_dict object, containing only the specified keys.

        Useful, for example, to select data coming from videos of a specified condition.

        Args:
            keys (list): List of keys to keep.

        Returns:
            TableDict: Subset of the original table_dict object, containing only the specified keys.
        """
        table = deepof.utils.deepcopy(self)
        assert np.all([k in table.keys() for k in keys]), "Invalid keys selected"

        return self.new_dict_same_header({k: value for k, value in table.items() if k in keys})
            
    def filter_condition(self, exp_filters: dict) -> table_dict:
        """Return a subset of the original table_dict object, containing only videos belonging to the specified experimental condition.

        Args:
            exp_filters (dict): experimental conditions and values to filter on.

        Returns:
            TableDict: Subset of the original table_dict object, containing only the specified keys.
        """
        table = deepof.utils.deepcopy(self)

        for exp_condition, exp_value in exp_filters.items():

            filtered_table = {
                k: value
                for k, value in table.items()
                if self._exp_conditions[k][exp_condition].values == exp_value
            }
            table = TableDict(
                filtered_table,
                self._type,
                self._table_path,
                connectivity=self._connectivity,
                exp_conditions={
                    k: value
                    for k, value in self._exp_conditions.items()
                    if k in filtered_table.keys()
                },
            )

        return table
    
    def filter_id(self, selected_id: str = None) -> table_dict:
        """Filter a TableDict object to keep only those columns related to the selected id.

        Leave labels untouched if present.

        Args:
            selected_id (str): select a single animal on multi animal settings. Defaults to None (all animals are processed).

        Returns:
            table_dict: Filtered TableDict object, keeping only the selected animal.
        """
        tabs = self.copy()
        for key, val in tabs.items():

            tabs[key] = deepof.utils.filter_animal_id_in_table(val, selected_id, self._type)

        return self.new_dict_same_header(tabs)

    
    def new_dict_same_header(self, tabs: dict = None, only_keys: bool=False):
        """Creates a new table dict based on a given dictionary and the existing header information.

        Args:
            tabs (dict): Dictionary of table entries 
            only_keys (bool): Copy dictionary keys and create empty dictionary with same keys

        Returns:
            table_dict: New TableDict object, based on given tabs and existing header info.
        """


        #create empty dict with same keys
        if tabs is None and only_keys:
            tabs={key: None for key in self.keys()}
        #create empty dict
        elif tabs is None:
            tabs={}

        return TableDict(
            tabs,
            typ = self._type,
            table_path = self._table_path,
            arena = self._arena,
            arena_dims = self._arena_dims,
            animal_ids = self._animal_ids,
            center = self._center,
            connectivity = self._connectivity,
            polar = self._polar,
            exp_conditions=self._exp_conditions,
        )

    def _prepare_projection(self) -> np.ndarray:
        """Return a numpy ndarray from the preprocessing of the table_dict object, ready for projection into a lower dimensional space."""
        labels = None

        X = {k: np.mean(v, axis=0) for k, v in self.items()}
        X = np.concatenate(
            [np.array(exp)[:, np.newaxis] for exp in X.values()], axis=1
        ).T

        return X, labels

    def _projection(
        self,
        projection_type: str,
        n_components: int = 2,
        kernel: str = None,
    ) -> Tuple[Any, Any]:
        """Return a training set generated from the 2D original data (time x features) and a specified projection to an n_components space.

        The sample parameter allows the user to randomly pick a subset of the data for performance or visualization reasons. For internal usage only.

        Args:
            projection_type (str): Projection to be used.
            n_components (int): Number of components to project to.
            kernel (str): Kernel to be used for the random and PCA algorithms.

        Returns:
            tuple: Tuple containing projected data and projection type.
        """
        X, labels = self._prepare_projection()

        if projection_type == "random":
            projection_type = random_projection.GaussianRandomProjection(
                n_components=n_components
            )
        elif projection_type == "pca":
            projection_type = KernelPCA(n_components=n_components, kernel=kernel)
        elif projection_type == "umap":  # pragma: no cover
            projection_type = umap.UMAP(n_components=n_components)

        X = projection_type.fit_transform(X)

        if labels is not None:
            return X, labels, projection_type

        return X, projection_type

    def random_projection(
        self, n_components: int = 2, kernel: str = "linear"
    ) -> Tuple[Any, Any]:
        """Return a training set generated from the 2D original data (time x features) and a random projection to a n_components space.

        The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons.

        Args:
            n_components (int): Number of components to project to. Default is 2.
            kernel (str): Kernel to be used for projections. Defaults to linear.

        Returns:
            tuple: Tuple containing projected data and projection type.
        """
        return self._projection("random", n_components=n_components, kernel=kernel)

    def pca(self, n_components: int = 2, kernel: str = "linear") -> Tuple[Any, Any]:
        """Return a training set generated from the 2D original data (time x features) and a PCA projection to a n_components space.

        The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons.

        Args:
            n_components (int): Number of components to project to. Default is 2.
            kernel (str): Kernel to be used for projections. Defaults to linear.

        Returns:
            tuple: Tuple containing projected data and projection type.
        """
        return self._projection("pca", n_components=n_components, kernel=kernel)

    def umap(
        self,
        n_components: int = 2,
    ) -> Tuple[Any, Any]:  # pragma: no cover
        """Return a training set generated from the 2D original data (time x features) and a PCA projection to a n_components space.

        The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons.

        Args:
            n_components (int): Number of components to project to. Default is 2.

        Returns:
            tuple: Tuple containing projected data and projection type.

        """
        return self._projection(
            "umap",
            n_components=n_components,
        )

    def merge(self, *args, ignore_index=False, file_name='merged', save_as_paths=False):
        """Take a number of table_dict objects and merges them to the current one.

        Returns a table_dict object of type 'merged'.
        Only annotations of the first table_dict object are kept.

        Args:
            *args (table_dict): table_dict objects to be merged.
            ignore_index (bool): ignore index when merging. Defaults to False.
            file_name (str): Name that is used for saving the merged table
            save_as_paths (bool): If True, Saves merged datasets as paths to file locations instead of keeping tables in RAM

        Returns:
            table_dict: Merged table_dict object.
        """
        args = [copy.deepcopy(self)] + list(args)

        merged_dict={}
        for key in args[0]:
            merged_tab = []
            for tabdict in args:

                #load table 
                tab = get_dt(tabdict, key)

                merged_tab.append(tab)

            merged_tab=pd.concat(merged_tab, axis=1, ignore_index=ignore_index, join="inner")

            # save paths for modified tables
            table_path = os.path.join(self._table_path,key, key + '_' + file_name)
            merged_dict[key] = save_dt(merged_tab,table_path,save_as_paths)           

        merged_tables = TableDict(
            merged_dict,
            typ="merged",
            table_path=self._table_path,
            connectivity=self._connectivity,
        )

        # Retake original table dict properties
        merged_tables._animal_ids = self._animal_ids

        return merged_tables

    def get_training_set(
        self, current_table_dict: table_dict, test_videos: int = 0
    ) -> tuple:
        """Generate training and test sets as table_dicts for model training.

        Intended for internal usage only.

        Args:
            current_table_dict (table_dict): table_dict object containing the data to be used for training.
            test_videos (int): Number of videos to be used for testing. Defaults to 0.

        Returns:
            IF there are no test videos:
            X_train (table_dict): only training data
            ELSE:
            tuple: Tuple containing training data, test data (as table_dicts), and test keys (if any).
        """

        # Padding of videos with slightly different lengths
        # Making sure that the training and test sets end up balanced in terms of labels

        keys=np.array(list(current_table_dict.keys()))

        test_indices = np.random.choice(
            range(len(current_table_dict)), test_videos, replace=False
        )

        test_keys = keys[test_indices]
        train_keys = np.delete(keys, test_indices)

        X_test = TableDict({},current_table_dict._type, current_table_dict._table_path)
        if test_videos > 0:
            try:
                X_test = current_table_dict.filter_videos(test_keys)   
                X_train = current_table_dict.filter_videos(train_keys)  
            except ValueError:  # pragma: no cover
                test_keys = np.array([], dtype=int)
                X_train = copy.deepcopy(current_table_dict)
                warnings.warn(
                    "Could not find more than one sample for at least one condition. "
                    "Partition between training and test set was not possible."
                )

        else:
            X_train = copy.deepcopy(current_table_dict)

        return (
            X_train,
            X_test,
            test_keys,
        )

    # noinspection PyTypeChecker,PyGlobalUndefined
    def preprocess(
        self,
        coordinates: coordinates,
        handle_ids: str = "concat",
        window_size: int = 25,
        window_step: int = 1,
        #binning info
        bin_size=None,
        bin_index=None,
        precomputed_bins=None,
        samples_max: int = 227272,  #corresponds to 1GB of memory when using default settings
        #other parameters
        scale: str = "standard",
        pretrained_scaler: Any = None,
        test_videos: int = 0,
        verbose: int = 0,
        filter_low_variance: bool = False,
        interpolate_normalized: int = 10,
        file_name = 'preprocessed',
        save_as_paths = None,
        shuffle: bool = False,
    ) -> np.ndarray:
        """Preprocess the loaded dataset before feeding to unsupervised embedding models.

        Capable of returning training and test sets ready for model training.

        Args:
            coordinates (coordinates): project coordinates
            handle_ids (str): indicates the default action to handle multiple animals in the TableDict object. Must be one of "concat" (body parts from different animals are treated as features) and "split" (different sliding windows are created for each animal).
            window_size (int): Minimum size of the applied ruptures. 
            window_step (int): Specifies the minimum jump for the rupture algorithms. 
            bin_size (Union[int,str]): bin size for time filtering. Will select (up to) the first 2.5 hours of data per default
            bin_index (Union[int,str]): index of the bin of size bin_size to select along the time dimension. Denotes exact start position in the time domain if given as string.
            precomputed_bins (np.ndarray): precomputed time bins. If provided, bin_size and bin_index are ignored.
            samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
            scale (str): Data scaling method. Must be one of 'standard', 'robust' (default; recommended) and 'minmax'.
            pretrained_scaler (Any): Pre-fit global scaler, trained on the whole dataset. Useful to process single videos.
            test_videos (int): Number of videos to use for testing. If 0, no test set is generated.
            verbose (int): Verbosity level. 0 (default) is silent, 1 prints progress, 2 prints debug information.
            filter_low_variance (float): remove features with variance lower than the specified threshold. Useful to get rid of the x axis of the body part used for alignment (which would introduce noise after standardization).
            interpolate_normalized(int): if not 0, it specifies the number of standard deviations beyond which values will be interpolated after normalization. Only used if scale is set to "standard".
            file_name (str): Name that is used for saving the merged table
            save_as_paths (bool): If True, Saves merged datasets as paths to file locations instead of keeping tables in RAM
            shuffle (bool): Whether to shuffle the data for each dataset. Defaults to False.


        Returns:
            (X_train, X_test) (np.ndarray,np.ndarray): Table dict with 3D datasets with shape (instances, sliding_window_size, features) generated from all training and all test videos (0 by default).
            (train_shape, test_shape) (np.ndarray,np.ndarray): Shape information for all trainign and test tables, when stacked upon each other.
            global_scaler: global scaler that was used for scaling
        
        """
        
        #get available memory -10% as buffer
        available_mem=psutil.virtual_memory().available*0.9
        #calculate maximum number of rows that fit in memory based on table info 
        N_rows_max=int(available_mem/((33+11)*window_size*8))
        if samples_max is None:
            samples_max=N_rows_max
        elif samples_max>N_rows_max: # pragma: no cover
            warning_message = (
            "\033[38;5;208m\n"
            "Warning! The selected number of samples may exceed your available memory."
            "\033[0m"
        )
            warnings.warn(warning_message)
        
        # Create a temporary copy of the current TableDict object,
        # to avoid modifying it in place
        table_temp = copy.deepcopy(self)

        bin_info=_preprocess_time_bins(coordinates=coordinates, bin_size=bin_size,bin_index=bin_index,precomputed_bins=precomputed_bins, tab_dict_for_binning=self, samples_max=samples_max)


        #determine the number of rows to use
        #N_elements_max=int(1000000000/8) #up to 1GB in save space
        #num_cols=get_dt(self, list(self.keys())[0], only_metainfo=True)['num_cols'] 
        #samples_max=int(N_elements_max/(window_size*num_cols)*window_step)

        #save outputs as paths if first table is larger than a threshold
        if save_as_paths is None:
            save_as_paths=False
            first_key=list(table_temp.keys())[0]
            num_rows=get_dt(table_temp,first_key,only_metainfo=True)["num_rows"]
            if coordinates._very_large_project:
                save_as_paths=True

        assert handle_ids in [
            "concat",
            "split",
        ], "handle IDs should be one of 'concat', and 'split'. See documentation for more details."


        sampled_tabs = []
                                         
        with tqdm(total=len(table_temp.keys()), desc=f"{'Filtering':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            for key in table_temp.keys():

                #pbar.set_postfix("Rescaling")
                #load table if not already loaded
                tab = get_dt(table_temp, key) 

                #select given range
                tab=tab.iloc[bin_info[key]]
            
                if filter_low_variance:

                    # Remove body parts with extremely low variance (usually the result of vertical alignment).
                    tab = tab.iloc[
                        :,
                        list(np.where(tab.var(axis=0) > filter_low_variance)[0])
                        + list(np.where(["pheno" in str(col) for col in tab.columns])[0]),
                    ]

                    assert len(tab.columns) > 0, "Error! During preprocessing the entire table was filtered out due to low variance!\nThis may happen due to an exceedingly high number of NaNs in the section chosen for preprocessing!"

                if scale:
                    if verbose:
                        print("Scaling data...")

                    if scale not in ["robust", "standard", "minmax"]:
                        raise ValueError(
                            "Invalid scaler. Select one of standard, minmax or robust"
                        )  # pragma: no cover

                    # Scale each experiment independently, to control for animal size
                    current_tab = deepof.utils.scale_table(
                        feature_array=tab,
                        scale=scale,
                        global_scaler=None,
                    )

                    tab = pd.DataFrame(
                        current_tab,
                        columns=tab.columns,
                        index=tab.index,
                    ).apply(lambda x: pd.to_numeric(x, errors="ignore"), axis=0)

                    sampled_tabs.append(tab.sample(n=min(samples_max, len(tab)), random_state=42))

                # save paths for modified tables
                table_path = os.path.join(self._table_path,key, key + '_' + file_name)
                table_temp[key] = save_dt(tab,table_path,save_as_paths) 
                pbar.update()

        if scale:    
        
            # Scale all experiments together, to control for differential stats
            if scale == "standard":
                global_scaler = StandardScaler()
            elif scale == "minmax":
                global_scaler = MinMaxScaler()
            else:
                global_scaler = RobustScaler()

            if pretrained_scaler is None:
                concat_data = pd.concat(sampled_tabs, ignore_index=True)
                global_scaler.fit(
                    concat_data.loc[:, concat_data.dtypes == float].values
                )
            else:
                global_scaler = pretrained_scaler
        
        with tqdm(total=len(table_temp.keys()), desc=f"{'Rescaling':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            for key in table_temp.keys():

                #pbar.set_postfix("Rescaling")
                #load table if not already loaded
                tab = get_dt(table_temp, key)   
                    
                if scale:

                    current_tab = deepof.utils.scale_table(
                        feature_array=tab,
                        scale=scale,
                        global_scaler=global_scaler,
                    )

                    tab = pd.DataFrame(
                        current_tab, columns=tab.columns, index=tab.index
                    )

                else:
                    global_scaler = None

                if scale == "standard" and interpolate_normalized:

                    # Interpolate outliers after preprocessing
                    tab_interpol = copy.deepcopy(tab)
                    
                    cur_tab =tab_interpol.values

                    try:
                        cur_tab[cur_tab > interpolate_normalized] = np.nan
                        cur_tab[cur_tab < -interpolate_normalized] = np.nan

                    # Deal with the edge case of phenotype label propagation
                    except TypeError:  # pragma: no cover

                        cur_tab[
                            np.append(
                                (cur_tab[:, :-1].astype(float) > interpolate_normalized),
                                np.array([[False] * len(cur_tab)]).T,
                                axis=1,
                            )
                        ] = np.nan
                        cur_tab[
                            np.append(
                                (cur_tab[:, :-1].astype(float) < -interpolate_normalized),
                                np.array([[False] * len(cur_tab)]).T,
                                axis=1,
                            )
                        ] = np.nan

                    tab_interpol = (
                        pd.DataFrame(cur_tab, index=tab.index, columns=tab.columns)
                        .apply(lambda x: pd.to_numeric(x, errors="ignore"))
                        .interpolate(limit_direction="both")
                    )

                    tab = tab_interpol

                    # save paths for modified tables
                    table_path = os.path.join(self._table_path,key, key + '_' + file_name)
                    table_temp[key] = save_dt(tab,table_path,save_as_paths) 
                pbar.update()

        # Split videos and generate training and test sets
        X_train, X_test, test_index = self.get_training_set(
            table_temp, test_videos
        )

        if verbose:
            print("Breaking time series...")

        # Apply rupture method to each train experiment independently
        X_train, train_shape = deepof.utils.extract_windows(
            to_window=X_train,
            window_size=window_size,
            window_step=window_step,
            save_as_paths=save_as_paths,
            shuffle=shuffle,
            windows_desc = "Get training windows"
        )

        
        if test_videos and len(test_index) > 0:

            # Apply rupture method to each test experiment independently
            X_test, test_shape = deepof.utils.extract_windows(
                to_window=X_test,
                window_size=window_size,
                window_step=window_step,
                save_as_paths=save_as_paths,
                shuffle=shuffle,
                windows_desc="Get testing windows"
            )
        else:
            test_shape = (0,)


        if verbose:
            print("Done!")

        return (X_train, X_test), (train_shape, test_shape), global_scaler

    def _get_data_tables(self, key: str) -> Tuple[Union[np.ndarray, pd.DataFrame], Optional[Union[np.ndarray, pd.DataFrame]]]:
        """
        Retrieves the main data table and an optional edge data table for a given key.
        
        This helper standardizes the data retrieval, always returning a tuple of
        (main_table, edge_table), where edge_table can be None.
        """
        raw_data = get_dt(self, key)
        if isinstance(raw_data, tuple) and len(raw_data) > 0:
            return raw_data[0], raw_data[1] if len(raw_data) > 1 else None
        return raw_data, None

    def _get_sample_indices(self, table: Union[np.ndarray, pd.DataFrame], n_windows: int, no_nans: bool) -> np.ndarray:
        """
        Generates a contiguous block of sample indices for a single table.
        
        If no_nans is True, it first filters out rows containing NaNs before sampling.
        The returned indices are always relative to the original, unfiltered table.
        """
        if no_nans:
            if isinstance(table, pd.DataFrame):
                valid_rows_mask = ~table.isna().any(axis=1)
                source_table = table[valid_rows_mask]
            else: # np.ndarray
                valid_rows_mask = ~np.isnan(table).any(axis=tuple(range(1, table.ndim)))
                source_table = table[valid_rows_mask]
            original_indices = np.where(valid_rows_mask)[0]
        else:
            source_table = table
            original_indices = np.arange(len(table))

        # Ensure we don't sample more windows than available
        n_windows_to_sample = min(n_windows, len(source_table))

        # Determine the last possible start position
        max_start = len(source_table) - n_windows_to_sample
        
        # Select a random start position
        start = np.random.randint(low=0, high=max(1, max_start + 1))
        end = start + n_windows_to_sample

        # Map the relative slice back to the original table's indices
        return original_indices[start:end]

    def _slice_data(self, data: Union[np.ndarray, pd.DataFrame], indices: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
        """Slices a numpy array or pandas DataFrame using the provided indices."""
        if isinstance(data, pd.DataFrame):
            return data.iloc[indices]
        return data[indices]

    def _get_edge_slice(self, 
                        edge_table: Optional[Union[np.ndarray, pd.DataFrame]],
                        sampled_main_table: Union[np.ndarray, pd.DataFrame], 
                        indices: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
        """
        Gets the corresponding slice from the edge table or creates a zero-filled
        placeholder if the edge table does not exist.
        """
        if edge_table is not None:
            return self._slice_data(edge_table, indices)
        
        # Create a zero-filled placeholder with the same shape and type
        zeros = np.zeros_like(sampled_main_table)
        if isinstance(sampled_main_table, pd.DataFrame):
            return pd.DataFrame(zeros, columns=sampled_main_table.columns, index=sampled_main_table.index)
        return zeros


    def sample_windows_from_data(self,
                                 time_bin_info: Dict[str, np.ndarray] = None,
                                 N_windows_tab: int = 10000,
                                 return_edges: bool = False,
                                 no_nans: bool = False) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Samples a set of windows from data entries, enhancing readability and reducing complexity.

        Args:
            time_bin_info (dict, optional): Pre-defined indices to sample for each key. 
                                            If provided, sampling logic is bypassed. Defaults to None.
            N_windows_tab (int): Max number of windows to sample from each recording if time_bin_info is not given.
            return_edges (bool): If True, returns a second dataset for edges.
            no_nans (bool): If True and time_bin_info is not given, only samples from rows without NaNs.
                            Note: This may result in non-contiguous original indices.

        Returns:
            - np.array: The concatenated main dataset (X_data).
            - np.array: The concatenated edge dataset (a_data), if return_edges is True.
            - dict: A dictionary with the sampled indices for each key (time_bin_info).
        """
        if time_bin_info is None:
            time_bin_info = {}

        X_data_list, a_data_list = [], []
        output_time_bin_info = {}

        # Determine if we should use provided indices or generate new ones.
        use_provided_indices = time_bin_info and set(self.keys()).issubset(time_bin_info.keys())
        
        for key in self.keys():
            main_table, edge_table = self._get_data_tables(key)

            if use_provided_indices:
                indices = time_bin_info[key]
            else:
                indices = self._get_sample_indices(main_table, N_windows_tab, no_nans)
            
            output_time_bin_info[key] = indices

            # Slice the main data table and append to our list
            sampled_x = self._slice_data(main_table, indices)
            X_data_list.append(sampled_x)
            
            # Handle the edge data if requested
            if return_edges:
                sampled_a = self._get_edge_slice(edge_table, sampled_x, indices)
                a_data_list.append(sampled_a)

        # Concatenate all the pieces into final numpy arrays
        X_data = pd.concat(X_data_list).values if isinstance(X_data_list[0], pd.DataFrame) else np.concatenate(X_data_list, axis=0)

        if return_edges:
            a_data = pd.concat(a_data_list).values if isinstance(a_data_list[0], pd.DataFrame) else np.concatenate(a_data_list, axis=0)
            return X_data, a_data, output_time_bin_info
        
        return X_data, output_time_bin_info
    

if __name__ == "__main__":
    # Remove excessive logging from tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
