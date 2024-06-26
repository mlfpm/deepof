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


from collections import defaultdict
from difflib import get_close_matches
from natsort import os_sorted
from pkg_resources import resource_filename
from shutil import rmtree
from sklearn import random_projection
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder,
)
from time import time
from tqdm import tqdm
from typing import NewType, Union
from typing import Dict, List, Tuple, Any
import copy
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import pims
import re
import shutil
import umap
import warnings

import deepof.model_utils
import deepof.models
import deepof.annotation_utils
import deepof.utils
import deepof.visuals

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


#  CLASSES FOR PREPROCESSING AND DATA WRANGLING


def load_project(project_path: str) -> coordinates:  # pragma: no cover
    """Load a pre-saved pickled Coordinates object.

    Args:
        project_path (str): name of the file to load.

    Returns:
        Pre-run coordinates object.

    """
    with open(
        os.path.join(project_path, "Coordinates", "deepof_coordinates.pkl"), "rb"
    ) as handle:
        coordinates = pickle.load(handle)

    coordinates._project_path = os.path.split(project_path)[0]
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
        enable_iterative_imputation: bool = True,
        exclude_bodyparts: List = tuple([""]),
        exp_conditions: dict = None,
        interpolate_outliers: bool = True,
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
    ):
        """Initialize a Project object.

        Args:
            animal_ids (list): list of animal ids.
            arena (str): arena type. Can be one of "circular-autodetect", "circular-manual", "polygonal-autodetect", or "polygonal-manual".
            bodypart_graph (str): body part scheme to use for the analysis. Defaults to None, in which case the program will attempt to select it automatically based on the available body parts.
            enable_iterative_imputation (bool): whether to use iterative imputation for occluded body parts.
            exclude_bodyparts (list): list of bodyparts to exclude from analysis.
            exp_conditions (dict): dictionary with experiment IDs as keys and experimental conditions as values.
            interpolate_outliers (bool): whether to interpolate missing data.
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

        """
        # Set working paths
        self.project_path = project_path
        self.project_name = project_name
        self.video_path = video_path
        self.table_path = table_path
        self.trained_path = resource_filename(__name__, "trained_models")

        # Detect files to load from disk
        self.table_format = table_format
        if self.table_format != "analysis.h5":
            self.table_format = table_format.replace(".", "")
        if self.table_format == "autodetect":
            ex = [
                i
                for i in os.listdir(self.table_path)
                if (
                    os.path.isfile(os.path.join(self.table_path, i))
                    and not i.startswith(".")
                )
            ][0]
            self.table_format = ex.split(".")[-1]
        self.videos = os_sorted(
            [
                vid
                for vid in os.listdir(self.video_path)
                if vid.endswith(video_format) and not vid.startswith(".")
            ]
        )
        self.tables = os_sorted(
            [
                tab
                for tab in os.listdir(self.table_path)
                if tab.endswith(self.table_format) and not tab.startswith(".")
            ]
        )
        assert len(self.videos) == len(
            self.tables
        ), "Unequal number of videos and tables. Please check your file structure"

        # Loads arena details and (if needed) detection models
        self.arena = arena
        self.arena_dims = video_scale
        self.ellipse_detection = None

        # Set the rest of the init parameters
        self.angles = True
        self.animal_ids = animal_ids if animal_ids is not None else [""]
        self.areas = True
        self.bodypart_graph = bodypart_graph
        self.connectivity = None
        self.distances = "all"
        self.ego = False
        self.exp_conditions = exp_conditions
        self.interpolate_outliers = interpolate_outliers
        self.interpolation_limit = interpolation_limit
        self.interpolation_std = interpolation_std
        self.likelihood_tolerance = likelihood_tol
        self.model = model
        self.smooth_alpha = smooth_alpha
        self.frame_rate = None
        self.video_format = video_format
        self.enable_iterative_imputation = enable_iterative_imputation
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
                    for i in os.listdir(self.table_path)
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
            os.makedirs(os.path.join(self.project_path, self.project_name, "Videos"))
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

            # Copy videos and tables to the new directories
            for vid in os.listdir(self.video_path):
                if vid.endswith(self.video_format):
                    shutil.copy2(
                        os.path.join(self.video_path, vid),
                        os.path.join(
                            self.project_path, self.project_name, "Videos", vid
                        ),
                    )

            for tab in os.listdir(self.table_path):
                if tab.endswith(self.table_format):
                    shutil.copy2(
                        os.path.join(self.table_path, tab),
                        os.path.join(
                            self.project_path, self.project_name, "Tables", tab
                        ),
                    )

            # Re-set video and table directories to the newly created paths
            self.video_path = os.path.join(
                self.project_path, self.project_name, "Videos"
            )
            self.table_path = os.path.join(
                self.project_path, self.project_name, "Tables"
            )

        else:
            raise OSError(
                "Project already exists. Delete it or specify a different name."
            )  # pragma: no cover

    @property
    def distances(self):
        """List. If not 'all', sets the body parts among which the distances will be computed."""
        return self._distances

    @property
    def ego(self):
        """String, name of a body part. If True, computes only the distances between the specified body part and the rest."""
        return self._ego

    @property
    def angles(self):
        """Bool. Toggles angle computation. True by default. If turned off, enhances performance for big datasets."""
        return self._angles

    def get_arena(
        self,
        tables: list,
        verbose: bool = False,
        debug: str = False,
        test: bool = False,
    ) -> np.array:
        """Return the arena as recognised from the videos.

        Args:
            tables (list): list of coordinate tables
            verbose (bool): if True, logs to console
            debug (str): if True, saves intermediate results to disk
            test (bool): if True, runs the function in test mode

        Returns:
            arena (np.ndarray): arena parameters, as recognised from the videos. The shape depends on the arena type

        """
        if verbose:
            print("Detecting arena...")

        return deepof.utils.get_arenas(
            self,
            tables,
            self.arena,
            self.arena_dims,
            self.project_path,
            self.project_name,
            self.segmentation_path,
            self.videos,
            debug,
            test,
        )

    def load_tables(self, verbose: bool = True) -> Tuple:
        """Load videos and tables into dictionaries.

        Args:
            verbose (bool): If True, prints the progress of data loading.

        Returns:
            Tuple: A tuple containing the following a dictionary with all loaded tables per experiment,
            and another dictionary with motion tracking data quality.

        """
        if self.table_format not in ["h5", "csv", "npy", "slp", "analysis.h5"]:
            raise NotImplementedError(
                "Tracking files must be in h5 (DLC or SLEAP), csv, npy, or slp format"
            )  # pragma: no cover

        if verbose:
            print("Loading trajectories...")

        tab_dict = {}
        for tab in self.tables:

            loaded_tab = deepof.utils.load_table(
                tab,
                self.table_path,
                self.table_format,
                self.rename_bodyparts,
                self.animal_ids,
            )

            # Remove the DLC suffix from the table name
            try:
                tab_name = deepof.utils.re.findall("(.*?)DLC", tab)[0]
            except IndexError:
                tab_name = tab.split(".")[0]

            tab_dict[tab_name] = loaded_tab

        # Check in the files come from a multi-animal DLC project
        if "individuals" in list(tab_dict.values())[0].index:

            self.animal_ids = list(
                list(tab_dict.values())[0].loc["individuals", :].unique()
            )

            for key, tab in tab_dict.items():
                # Adapt each table to work with the downstream pipeline
                tab_dict[key].loc["bodyparts"] = (
                    tab.loc["individuals"] + "_" + tab.loc["bodyparts"]
                )
                tab_dict[key].drop("individuals", axis=0, inplace=True)

        # Convert the first rows of each dataframe to a multi-index
        for key, tab in tab_dict.items():

            tab_copy = tab.copy()

            tab_copy.columns = pd.MultiIndex.from_arrays(
                [tab.iloc[i] for i in range(2)]
            )
            tab_copy = tab_copy.iloc[2:].astype(float)
            tab_dict[key] = tab_copy.reset_index(drop=True)

        # Update body part connectivity graph, taking detected or specified body parts into account
        model_dict = {
            "{}mouse_topview".format(aid): deepof.utils.connect_mouse(
                aid,
                exclude_bodyparts=self.exclude_bodyparts,
                graph_preset=self.bodypart_graph,
            )
            for aid in self.animal_ids
        }
        self.connectivity = {
            aid: model_dict[aid + self.model] for aid in self.animal_ids
        }

        # Remove specified body parts from the mice graph
        if len(self.animal_ids) > 1 and self.exclude_bodyparts != tuple([""]):
            self.exclude_bodyparts = [
                aid + "_" + bp
                for aid in self.animal_ids
                for bp in self.exclude_bodyparts
            ]

        # Pass a time-based index, if specified in init
        if self.frame_rate is not None:
            for key, tab in tab_dict.items():
                tab_dict[key].index = pd.timedelta_range(
                    "00:00:00",
                    pd.to_timedelta((tab.shape[0] // self.frame_rate), unit="sec"),
                    periods=tab.shape[0] + 1,
                    closed="left",
                ).map(lambda t: str(t)[7:])

        lik_dict = defaultdict()

        for key, tab in tab_dict.items():

            x = tab.xs("x", level="coords", axis=1, drop_level=False)
            y = tab.xs("y", level="coords", axis=1, drop_level=False)
            lik = tab.xs("likelihood", level="coords", axis=1, drop_level=True)

            tab_dict[key] = pd.concat([x, y], axis=1).sort_index(axis=1)
            lik_dict[key] = lik.fillna(0.0)

        lik_dict = TableDict(lik_dict, typ="quality", animal_ids=self.animal_ids)

        if self.smooth_alpha:

            if verbose:
                print("Smoothing trajectories...")

            for key, tab in tab_dict.items():
                cur_idx = tab.index
                cur_cols = tab.columns
                smooth = pd.DataFrame(
                    deepof.utils.smooth_mult_trajectory(
                        np.array(tab), alpha=self.smooth_alpha, w_length=15
                    )
                ).reset_index(drop=True)
                smooth.columns = cur_cols
                smooth.index = cur_idx
                tab_dict[key] = smooth

        if self.exclude_bodyparts != tuple([""]):

            for k, tab in tab_dict.items():
                temp = tab.drop(self.exclude_bodyparts, axis=1, level="bodyparts")
                temp.sort_index(axis=1, inplace=True)
                temp.columns = pd.MultiIndex.from_product(
                    [os_sorted(list(set([i[j] for i in temp.columns]))) for j in range(2)]
                )
                tab_dict[k] = temp.sort_index(axis=1)

        if self.interpolate_outliers:

            if verbose:
                print("Interpolating outliers...")

            for k, tab in tab_dict.items():
                tab_dict[k] = deepof.utils.interpolate_outliers(
                    tab,
                    lik_dict[k],
                    likelihood_tolerance=self.likelihood_tolerance,
                    mode="or",
                    limit=self.interpolation_limit,
                    n_std=self.interpolation_std,
                )

        if self.enable_iterative_imputation:

            if verbose:
                print("Iterative imputation of ocluded bodyparts...")

            tab_dict = deepof.utils.iterative_imputation(self, tab_dict, lik_dict)

        # Set table_dict to NaN if animals are missing
        tab_dict = deepof.utils.set_missing_animals(self, tab_dict, lik_dict)

        return tab_dict, lik_dict

    def get_distances(self, tab_dict: dict, verbose: bool = True) -> dict:
        """Compute the distances between all selected body parts over time. If ego is provided, it only returns distances to a specified bodypart.

        Args:
            tab_dict (dict): Dictionary of pandas DataFrames containing the trajectories of all bodyparts.
            verbose (bool): If True, prints progress. Defaults to True.

        Returns:
            dict: Dictionary of pandas DataFrames containing the distances between all bodyparts.

        """
        if verbose:
            print("Computing distances...")

        nodes = self.distances
        if nodes == "all":
            nodes = tab_dict[list(tab_dict.keys())[0]].columns.levels[0]

        assert [
            i in list(tab_dict.values())[0].columns.levels[0] for i in nodes
        ], "Nodes should correspond to existent bodyparts"

        scales = self.scales[:, 2:]

        distance_dict = {
            key: deepof.utils.bpart_distance(tab, scales[i, 1], scales[i, 0])
            for i, (key, tab) in enumerate(tab_dict.items())
        }

        for key in distance_dict.keys():
            distance_dict[key] = distance_dict[key].loc[
                :, [np.all([i in nodes for i in j]) for j in distance_dict[key].columns]
            ]

        if self.ego:
            for key, val in distance_dict.items():
                distance_dict[key] = val.loc[
                    :, [dist for dist in val.columns if self.ego in dist]
                ]

        # Restore original index
        for key in distance_dict.keys():
            distance_dict[key].index = tab_dict[key].index

        return distance_dict

    def get_angles(self, tab_dict: dict, verbose: bool = True) -> dict:
        """Compute all the angles between adjacent bodypart trios per video and per frame in the data.

        Args:
            tab_dict (dict): Dictionary of pandas DataFrames containing the trajectories of all bodyparts.
            verbose (bool): If True, prints progress. Defaults to True.

        Returns:
            dict: Dictionary of pandas DataFrames containing the distances between all bodyparts.

        """
        if verbose:
            print("Computing angles...")

        # Add all three-element connected sequences on each mouse
        bridges = []
        for i in self.animal_ids:
            bridges += deepof.utils.enumerate_all_bridges(self.connectivity[i])
        bridges = [i for i in bridges if len(i) == 3]

        angle_dict = {}
        try:
            for key, tab in tab_dict.items():

                dats = []
                for clique in bridges:
                    dat = pd.DataFrame(
                        deepof.utils.angle(
                            np.array(tab[clique]).reshape([3, tab.shape[0], 2])
                        ).T
                    )

                    dat.columns = [tuple(clique)]
                    dats.append(dat)

                dats = pd.concat(dats, axis=1)

                angle_dict[key] = dats
        except KeyError:
            raise KeyError(
                "Are you using a custom labelling scheme? Out tutorials may help! "
                "In case you're not, are there multiple animals in your single-animal DLC video? Make sure to set the "
                "animal_ids parameter in deepof.data.Project"
            )

        # Restore original index
        for key in angle_dict.keys():
            angle_dict[key].index = tab_dict[key].index

        return angle_dict

    def get_areas(self, tab_dict: dict, verbose: bool = True) -> dict:
        """Compute all relevant areas (head, torso, back) per video and per frame in the data.

        Args:
            tab_dict (dict): Dictionary of pandas DataFrames containing the trajectories of all bodyparts.
            verbose (bool): If True, prints progress. Defaults to True.

        Returns:
            dict: Dictionary of pandas DataFrames containing the distances between all bodyparts.

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

        if verbose:
            print("Computing areas...")

        all_areas_dict = {}

        # iterate over all tables
        for key, tab in tab_dict.items():

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

                        # create list of keys containing all table columns relevant for the current area
                        bp_x_keys = [(body_part, "x") for body_part in bp_pattern]
                        bp_y_keys = [(body_part, "y") for body_part in bp_pattern]

                        # create a 3D numpy array [NFrames, NPoints, NDis]
                        x = current_animal_table[bp_x_keys].to_numpy()
                        y = current_animal_table[bp_y_keys].to_numpy()
                        y = y[:, :, np.newaxis]
                        polygon_xy_stack = np.dstack((x, y))

                        # dictionary of area lists (each list has dimensions [NFrames])
                        areas_animal_dict[bp_pattern_key] = deepof.utils.compute_areas(
                            polygon_xy_stack
                        )

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
                    warnings.warn(
                        "It seems you're using a custom labelling scheme which is missing key body parts. You can proceed, but not all areas will be computed."
                    )

                # collect area tables for all animals
                current_table = pd.concat([current_table, areas_table], axis=1)

            all_areas_dict[key] = current_table

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
            coordinates: Deepof.Coordinates object containing the trajectories of all bodyparts.

        """
        if verbose:
            print("Setting up project directories...")

        if force and os.path.exists(os.path.join(self.project_path, self.project_name)):
            rmtree(os.path.join(self.project_path, self.project_name))

        if not os.path.exists(os.path.join(self.project_path, self.project_name)):
            self.set_up_project_directory(debug=debug)

        # load video info
        self.frame_rate = int(
            np.round(
                pims.ImageIOReader(
                    os.path.join(self.video_path, self.videos[0])
                ).frame_rate
            )
        )

        # load table info
        tables, quality = self.load_tables(verbose)
        if self.exp_conditions is not None:
            assert (
                tables.keys() == self.exp_conditions.keys()
            ), "experimental IDs in exp_conditions do not match"

        distances = None
        angles = None
        areas = None

        # noinspection PyAttributeOutsideInit
        self.scales, self.arena_params, self.video_resolution = self.get_arena(
            tables, verbose, debug, test
        )

        if self.distances:
            distances = self.get_distances(tables, verbose)

        if self.angles:
            angles = self.get_angles(tables, verbose)

        if self.areas:
            areas = self.get_areas(tables, verbose)

        if _to_extend is not None:

            # Merge and expand coordinate objects
            angles = TableDict({**_to_extend._angles, **angles}, typ="angles")
            # areas = TableDict({**_to_extend._areas, **areas}, typ="areas")
            distances = TableDict(
                {**_to_extend._distances, **distances}, typ="distances"
            )
            tables = TableDict({**_to_extend._tables, **tables}, typ="tables")
            quality = TableDict({**_to_extend._quality, **quality}, typ="quality")

            # Merge metadata
            self.tables = _to_extend._table_paths + self.tables
            self.videos = _to_extend._videos + self.videos
            self.arena_params = _to_extend._arena_params + self.arena_params
            self.scales = np.vstack([_to_extend._scales, self.scales])

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
            tables=tables,
            table_paths=self.tables,
            trained_model_path=self.trained_path,
            videos=self.videos,
            video_resolution=self.video_resolution,
        )

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
        verbose: bool = True,
        debug: bool = True,
        test: bool = False,
    ) -> coordinates:
        """Generate a deepof.Coordinates dataset using all the options specified during initialization.

        Args:
            project_to_extend (coordinates): Coordinates object to extend with the current dataset.
            verbose (bool): If True, prints progress. Defaults to True.
            debug (bool): If True, saves arena detection images to disk. Defaults to False.
            test (bool): If True, creates the project in test mode (which, for example, bypasses any manual input). Defaults to False.

        Returns:
            coordinates: Deepof.Coordinates object containing the trajectories of all bodyparts.

        """
        if verbose:
            print("Loading previous project...")

        previous_project = load_project(project_to_extend)

        # Keep only those videos and tables that were not in the original dataset
        self.videos = [
            vid for vid in self.videos if vid not in previous_project._videos
        ]
        self.tables = [
            tab for tab in self.tables if tab not in previous_project._table_paths
        ]

        if verbose:
            print(f"Processing data from {len(self.videos)} experiments...")

        if len(self.videos) > 0:

            # Use the same directory as the original project
            extended_coords = self.create(
                verbose,
                force=False,
                debug=debug,
                test=test,
                _to_extend=previous_project,
            )

            # Copy old files to the new directory
            for vid, tab in zip(
                previous_project._videos, previous_project._table_paths
            ):
                if vid not in self.tables:

                    shutil.copy(
                        os.path.join(
                            previous_project._project_path,
                            "Videos",
                            vid,
                        ),
                        os.path.join(self.project_path, self.project_name, "Videos"),
                    )
                    shutil.copy(
                        os.path.join(
                            previous_project._project_path,
                            "Tables",
                            tab,
                        ),
                        os.path.join(self.project_path, self.project_name, "Tables"),
                    )

            return extended_coords

        else:
            print("No new experiments to process. Exiting...")


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
        scales: np.ndarray,
        frame_rate: int,
        arena_params: List,
        tables: dict,
        table_paths: List,
        trained_model_path: str,
        videos: List,
        video_resolution: List,
        angles: dict = None,
        animal_ids: List = tuple([""]),
        areas: dict = None,
        distances: dict = None,
        connectivity: nx.Graph = None,
        excluded_bodyparts: list = None,
        exp_conditions: dict = None,
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
            scales (np.ndarray): Scales used for the experiment. See deepof.data.Project for more information.
            frame_rate (int): frame rate of the processed videos.
            arena_params (List): List containing the parameters of the arena. See deepof.data.Project for more information.
            tables (dict): Dictionary containing the tables of the experiment. See deepof.data.Project for more information.
            table_paths (List): List containing the paths to the tables of the experiment. See deepof.data.Project for more information.f
            trained_model_path (str): Path to the trained models used for the supervised pipeline. For internal use only.
            videos (List): List containing the videos used for the experiment. See deepof.data.Project for more information.
            video_resolution (List): List containing the automatically detected resolution of the videos used for the experiment.
            angles (dict): Dictionary containing the angles of the experiment. See deepof.data.Project for more information.
            animal_ids (List): List containing the animal IDs of the experiment. See deepof.data.Project for more information.
            areas (dict): dictionary with areas to compute. By default, it includes head, torso, and back.
            distances (dict): Dictionary containing the distances of the experiment. See deepof.data.Project for more information.
            excluded_bodyparts (list): list of bodyparts to exclude from analysis.
            exp_conditions (dict): Dictionary containing the experimental conditions of the experiment. See deepof.data.Project for more information.

        """
        self._project_path = project_path
        self._project_name = project_name
        self._animal_ids = animal_ids
        self._arena = arena
        self._arena_params = arena_params
        self._arena_dims = arena_dims
        self._bodypart_graph = bodypart_graph
        self._excluded = excluded_bodyparts
        self._exp_conditions = exp_conditions
        self._frame_rate = frame_rate
        self._path = path
        self._quality = quality
        self._scales = scales
        self._tables = tables
        self._table_paths = table_paths
        self._trained_model_path = trained_model_path
        self._videos = videos
        self._video_resolution = video_resolution
        self._angles = angles
        self._areas = areas
        self._distances = distances
        self._connectivity = connectivity

    def __str__(self):  # pragma: no cover
        """Print the object to stdout."""
        lens = len(self._videos)
        return "deepof analysis of {} video{}".format(lens, ("s" if lens > 1 else ""))

    def __repr__(self):  # pragma: no cover
        """Print the object to stdout."""
        lens = len(self._videos)
        return "deepof analysis of {} video{}".format(lens, ("s" if lens > 1 else ""))

    def get_coords(
        self,
        center: str = False,
        polar: bool = False,
        speed: int = 0,
        align: str = False,
        align_inplace: bool = True,
        selected_id: str = None,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """Return a table_dict object with the coordinates of each animal as values.

        Args:
            center (str): Name of the body part to which the positions will be centered. If false, the raw data is returned; if 'arena' (default), coordinates are centered in the pitch
            polar (bool) States whether the coordinates should be converted to polar values.
            speed (int): States the derivative of the positions to report. Speed is returned if 1, acceleration if 2, jerk if 3, etc.
            align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).
            align_inplace (bool): Only valid if align is set. Aligns the vector that goes from the origin to the selected body part with the y-axis, for all timepoints (default).
            selected_id (str): Selects a single animal on multi animal settings. Defaults to None (all animals are processed).
            propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label
            propagate_annotations (dict): If a dictionary is provided, supervised annotations are propagated through the training dataset. This can be used for regularising the latent space based on already known traits.

        Returns:
            table_dict: A table_dict object containing the coordinates of each animal as values.

        """
        tabs = deepof.utils.deepcopy(self._tables)
        coord_1, coord_2 = "x", "y"
        scales = self._scales

        if polar:
            coord_1, coord_2 = "rho", "phi"
            scales = deepof.utils.bp2polar(scales).to_numpy()
            for key, tab in tabs.items():
                tabs[key] = deepof.utils.tab2polar(tab)

        if center == "arena":

            for i, (key, value) in enumerate(tabs.items()):

                value.loc[:, (slice("x"), [coord_1])] = (
                    value.loc[:, (slice("x"), [coord_1])] - scales[i][0]
                )

                value.loc[:, (slice("x"), [coord_2])] = (
                    value.loc[:, (slice("x"), [coord_2])] - scales[i][1]
                )

        elif isinstance(center, str) and center != "arena":

            for i, (key, value) in enumerate(tabs.items()):

                # Center each animal independently
                animal_ids = self._animal_ids
                if selected_id is not None:
                    animal_ids = [selected_id]

                for aid in animal_ids:
                    # center on x / rho
                    value.update(
                        value.loc[:, [i for i in value.columns if i[0].startswith(aid)]]
                        .loc[:, (slice("x"), [coord_1])]
                        .subtract(
                            value[aid + ("_" if aid != "" else "") + center][coord_1],
                            axis=0,
                        )
                    )

                    # center on y / phi
                    value.update(
                        value.loc[:, [i for i in value.columns if i[0].startswith(aid)]]
                        .loc[:, (slice("x"), [coord_2])]
                        .subtract(
                            value[aid + ("_" if aid != "" else "") + center][coord_2],
                            axis=0,
                        )
                    )

        if align:

            assert any(
                center in bp for bp in list(tabs.values())[0].columns.levels[0]
            ), "for align to run, center must be set to the name of a bodypart"
            assert any(
                align in bp for bp in list(tabs.values())[0].columns.levels[0]
            ), "align must be set to the name of a bodypart"

            for key, tab in tabs.items():
                # noinspection PyUnboundLocalVariable
                all_index = tab.index
                all_columns = []
                aligned_coordinates = None
                # noinspection PyUnboundLocalVariable
                for aid in animal_ids:
                    # Bring forward the column to align
                    columns = [
                        i
                        for i in tab.columns
                        if not i[0].endswith(align) and i[0].startswith(aid)
                    ]
                    columns = [
                        (
                            aid + ("_" if aid != "" else "") + align,
                            ("phi" if polar else "x"),
                        ),
                        (
                            aid + ("_" if aid != "" else "") + align,
                            ("rho" if polar else "y"),
                        ),
                    ] + columns

                    partial_aligned = tab[columns]
                    all_columns += columns

                    if align_inplace and not polar:
                        partial_aligned = deepof.utils.align_trajectories(
                            np.array(partial_aligned), mode="all"
                        )
                        partial_aligned[np.abs(partial_aligned) < 1e-5] = 0.0
                        partial_aligned = pd.DataFrame(partial_aligned)
                        aligned_coordinates = pd.concat(
                            [aligned_coordinates, partial_aligned], axis=1
                        )

                aligned_coordinates.index = all_index
                aligned_coordinates.columns = pd.MultiIndex.from_tuples(all_columns)
                tabs[key] = aligned_coordinates

        if speed:
            for key, tab in tabs.items():
                vel = deepof.utils.rolling_speed(tab, deriv=speed, center=center)
                tabs[key] = vel

        # Id selected_id was specified, selects coordinates of only one animal for further processing
        if selected_id is not None:
            for key, val in tabs.items():
                tabs[key] = val.loc[
                    :, deepof.utils.filter_columns(val.columns, selected_id)
                ]

        # Set table_dict to NaN if animals are missing
        tabs = deepof.utils.set_missing_animals(self, tabs, self.get_quality())

        if propagate_annotations:
            annotations = list(propagate_annotations.values())[0].columns

            for key, tab in tabs.items():
                for ann in annotations:
                    tab.loc[:, ann] = propagate_annotations[key].loc[:, ann]

        if propagate_labels:
            for key, tab in tabs.items():
                tab.loc[:, "pheno"] = np.repeat(
                    self._exp_conditions[key][propagate_labels].values, tab.shape[0]
                )

        return TableDict(
            tabs,
            "coords",
            animal_ids=self._animal_ids,
            arena=self._arena,
            arena_dims=self._scales,
            center=center,
            connectivity=self._connectivity,
            polar=polar,
            exp_conditions=self._exp_conditions,
            propagate_labels=propagate_labels,
            propagate_annotations=propagate_annotations,
        )

    def get_distances(
        self,
        speed: int = 0,
        selected_id: str = None,
        filter_on_graph: bool = True,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """Return a table_dict object with the distances between body parts animal as values.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            filter_on_graph (bool): If True, only distances between connected nodes in the DeepOF graph representations are kept. Otherwise, all distances between bodyparts are returned.
            propagate_labels (bool): If True, the pheno column will be propagated from the original data.
            propagate_annotations (Dict): A dictionary of annotations to propagate.

        Returns:
            table_dict: A table_dict object with the distances between body parts animal as values.

        """
        tabs = deepof.utils.deepcopy(self._distances)

        if self._distances is not None:

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="dists")
                    tabs[key] = vel

            if selected_id is not None:
                for key, val in tabs.items():
                    tabs[key] = val.loc[
                        :, deepof.utils.filter_columns(val.columns, selected_id)
                    ]

            # Set table_dict to NaN if animals are missing
            tabs = deepof.utils.set_missing_animals(self, tabs, self.get_quality())

            if propagate_labels:
                for key, tab in tabs.items():
                    tab.loc[:, "pheno"] = self._exp_conditions[key]

            if propagate_annotations:
                annotations = list(propagate_annotations.values())[0].columns

                for key, tab in tabs.items():
                    for ann in annotations:
                        tab.loc[:, ann] = propagate_annotations[key].loc[:, ann]

            if filter_on_graph:

                for key, tab in tabs.items():
                    tabs[key] = tab.loc[
                        :,
                        list(
                            set(
                                [
                                    tuple(sorted(e))
                                    for e in deepof.utils.connect_mouse(
                                        animal_ids=self._animal_ids,
                                        graph_preset=self._bodypart_graph,
                                    ).edges
                                ]
                            )
                            & set(tab.columns)
                        ),
                    ]

            return TableDict(
                tabs,
                animal_ids=self._animal_ids,
                connectivity=self._connectivity,
                exp_conditions=self._exp_conditions,
                propagate_labels=propagate_labels,
                propagate_annotations=propagate_annotations,
                typ="dists",
            )

        raise ValueError(
            "Distances not computed. Read the documentation for more details"
        )  # pragma: no cover

    def get_angles(
        self,
        degrees: bool = False,
        speed: int = 0,
        selected_id: str = None,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """Return a table_dict object with the angles between body parts animal as values.

        Args:
            degrees (bool): If True (default), the angles will be in degrees. Otherwise they will be converted to radians.
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            propagate_labels (bool): If True, the pheno column will be propagated from the original data.
            propagate_annotations (Dict): A dictionary of annotations to propagate.

        Returns:
            table_dict: A table_dict object with the angles between body parts animal as values.

        """
        tabs = deepof.utils.deepcopy(self._angles)

        if self._angles is not None:
            if degrees:
                tabs = {key: np.degrees(tab) for key, tab in tabs.items()}

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="angles")
                    tabs[key] = vel

            if selected_id is not None:
                for key, val in tabs.items():
                    tabs[key] = val.loc[
                        :, deepof.utils.filter_columns(val.columns, selected_id)
                    ]

            # Set table_dict to NaN if animals are missing
            tabs = deepof.utils.set_missing_animals(self, tabs, self.get_quality())

            if propagate_labels:
                for key, tab in tabs.items():
                    tab["pheno"] = self._exp_conditions[key]

            if propagate_annotations:
                annotations = list(propagate_annotations.values())[0].columns

                for key, tab in tabs.items():
                    for ann in annotations:
                        tab.loc[:, ann] = propagate_annotations[key].loc[:, ann]

            return TableDict(
                tabs,
                animal_ids=self._animal_ids,
                connectivity=self._connectivity,
                exp_conditions=self._exp_conditions,
                propagate_labels=propagate_labels,
                propagate_annotations=propagate_annotations,
                typ="angles",
            )

        raise ValueError(
            "Angles not computed. Read the documentation for more details"
        )  # pragma: no cover

    def get_areas(self, speed: int = 0, selected_id: str = "all") -> table_dict:
        """Return a table_dict object with all relevant areas (head, torso, back, full). Unless specified otherwise, the areas are computed for all animals.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select. "all" (default) computes the areas for all animals. Declared in self._animal_ids.

        Returns:
            table_dict: A table_dict object with the areas of the body parts animal as values.
        """
        tabs = deepof.utils.deepcopy(self._areas)

        if self._areas is not None:

            if selected_id == "all":
                selected_ids = self._animal_ids
            else:
                selected_ids = [selected_id]

            for key, tab in tabs.items():

                exp_table = pd.DataFrame()

                for aid in selected_ids:

                    if aid == "":
                        aid = None

                    # get the current table for the current animal
                    current_table = tab.loc[
                        :, deepof.utils.filter_columns(tab.columns, aid)
                    ]
                    exp_table = pd.concat([exp_table, current_table], axis=1)

                tabs[key] = exp_table

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="angles")
                    tabs[key] = vel

            # Set table_dict to NaN if animals are missing
            tabs = deepof.utils.set_missing_animals(self, tabs, self.get_quality())

            areas = TableDict(
                tabs,
                animal_ids=self._animal_ids,
                connectivity=self._connectivity,
                typ="areas",
                exp_conditions=self._exp_conditions,
            )

            return areas

        raise ValueError(
            "Areas not computed. Read the documentation for more details"
        )  # pragma: no cover

    def get_videos(self, play: bool = False):
        """Returns the videos associated with the dataset as a list."""
        if play:  # pragma: no cover
            raise NotImplementedError

        return self._videos

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

    def get_quality(self):
        """Retrieve a dictionary with the tagging quality per video, as reported by DLC or SLEAP."""
        return TableDict(self._quality, typ="quality", animal_ids=self._animal_ids)

    @property
    def get_arenas(self):
        """Retrieve all available information associated with the arena."""
        return self._arena, [self._arena_dims], self._scales

    def edit_arenas(
        self, videos: list = None, arena_type: str = None, verbose: bool = True
    ):  # pragma: no cover
        """Tag the arena in the videos.

        Args:
            videos (list): A list of videos to reannotate. If None, all videos are loaded.
            arena_type (str): The type of arena to use. Must be one of "polygonal-manual", "circular-manual", or "circular-autodetect". If None (default), the arena type specified when creating the project is used.
            verbose (bool): Whether to print the progress of the annotation.

        """
        if videos is None:
            videos = self._videos
        if arena_type is None:
            arena_type = self._arena

        videos_renamed, vid_idcs = [], []
        for vid_idx, vid_in in enumerate(self._videos):
            for vid_out in videos:
                if vid_out in vid_in:
                    videos_renamed.append(vid_in)
                    vid_idcs.append(vid_idx)

        if verbose:
            print(
                "Editing {} arena{}".format(len(videos), "s" if len(videos) > 1 else "")
            )

        edited_scales, edited_arena_params, _ = deepof.utils.get_arenas(
            coordinates=self,
            tables=self._tables,
            arena=arena_type,
            arena_dims=self._arena_dims,
            project_path=self._project_path,
            project_name=self._project_name,
            segmentation_model_path=None,
            videos=videos_renamed,
        )

        if verbose:
            print("Done!")

        # update the scales and arena parameters
        for vid_idx, vid in enumerate(videos):
            self._scales[vid_idcs[vid_idx]] = edited_scales[vid_idx]
            self._arena_params[vid_idcs[vid_idx]] = edited_arena_params[vid_idx]

        self.save(timestamp=False)

    def save(self, filename: str = None, timestamp: bool = True):
        """Save the current state of the Coordinates object to a pickled file.

        Args:
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
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_graph_dataset(
        self,
        animal_id: str = None,
        precomputed_tab_dict: table_dict = None,
        center: str = False,
        polar: bool = False,
        align: str = None,
        preprocess: bool = True,
        **kwargs,
    ) -> table_dict:
        """Generate a dataset with all specified features.

        Args:
            animal_id (str): Name of the animal to process. If None (default) all animals are included in a multi-animal graph.
            precomputed_tab_dict (table_dict): table_dict object for further graph processing. None (default) builds it on the spot.
            center (str): Name of the body part to which the positions will be centered. If false, raw data is returned; if 'arena' (default), coordinates are centered on the pitch.
            polar (bool) States whether the coordinates should be converted to polar values.
            align (str): Selects the body part to which later processes will align the frames with (see preprocess in table_dict documentation).
            preprocess (bool): whether to preprocess the data to pass to autoencoders. If False, node features and distance-weighted adjacency matrices on the raw data are returned.

        Returns:
            merged_features: A graph-based dataset.

        """
        # Get all relevant features
        coords = self.get_coords(
            selected_id=animal_id, center=center, align=align, polar=polar
        )
        speeds = self.get_coords(selected_id=animal_id, speed=1)
        dists = self.get_distances(selected_id=animal_id)

        # Merge and extract names
        tab_dict = coords.merge(speeds, dists)

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

        tab_dict._connectivity = graph
        edge_feature_names = list(list(dists.values())[0].columns)
        feature_names = pd.Index([i for i in list(tab_dict.values())[0].columns])
        node_feature_names = (
            [(i, "x") for i in list(graph.nodes())]
            + [(i, "y") for i in list(graph.nodes())]
            + list(graph.nodes())
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

        # Create graph datasets
        if preprocess:
            to_preprocess, global_scaler = tab_dict.preprocess(**kwargs)
            dataset = [
                to_preprocess[0][:, :, ~feature_names.isin(edge_feature_names)][
                    :, :, node_sorting_indices
                ],
                to_preprocess[0][:, :, feature_names.isin(edge_feature_names)][
                    :, :, edge_sorting_indices
                ],
                to_preprocess[1],
            ]
            try:
                dataset += [
                    to_preprocess[2][:, :, ~feature_names.isin(edge_feature_names)][
                        :, :, node_sorting_indices
                    ],
                    to_preprocess[2][:, :, feature_names.isin(edge_feature_names)][
                        :, :, edge_sorting_indices
                    ],
                    to_preprocess[3],
                ]
            except IndexError:
                dataset += [to_preprocess[2], to_preprocess[2], to_preprocess[3]]

        else:  # pragma: no cover
            to_preprocess = np.concatenate(list(tab_dict.values()))

            # Split node features (positions, speeds) from edge features (distances)
            dataset = (
                to_preprocess[:, ~feature_names.isin(edge_feature_names)][
                    :, node_sorting_indices
                ].reshape([to_preprocess.shape[0], len(graph.nodes()), -1], order="F"),
                deepof.utils.edges_to_weighted_adj(
                    nx.adj_matrix(graph).todense(),
                    to_preprocess[:, feature_names.isin(edge_feature_names)][
                        :, edge_sorting_indices
                    ],
                ),
            )

        try:
            return (
                tuple(dataset),
                nx.adjacency_matrix(graph).todense(),
                tab_dict,
                global_scaler,
            )
        except UnboundLocalError:
            return tuple(dataset), nx.adjacency_matrix(graph).todense(), tab_dict

    # noinspection PyDefaultArgument
    def supervised_annotation(
        self,
        params: Dict = {},
        center: str = "Center",
        align: str = "Spine_1",
        video_output: bool = False,
        frame_limit: int = np.inf,
        debug: bool = False,
        n_jobs: int = 1,
        propagate_labels: bool = False,
    ) -> table_dict:
        """Annotates coordinates with behavioral traits using a supervised pipeline.

        Args:
            params (Dict): A dictionary with the parameters to use for the pipeline. If unsure, leave empty.
            center (str): Body part to center coordinates on. "Center" by default.
            align (str): Body part to rotationally align the body parts with. "Spine_1" by default.
            video_output (bool): It outputs a fully annotated video for each experiment indicated in a list. If set to "all", it will output all videos. False by default.
            frame_limit (int): Only applies if video_output is not False. Indicates the maximum number of frames per video to output.
            debug (bool): Only applies if video_output is not False. If True, all videos will include debug information, such as the detected arena and the preprocessed tracking tags.
            n_jobs (int): Number of jobs to use for parallel processing.
            propagate_labels (bool): If True, the pheno column will be propagated from the original data.

        Returns:
            table_dict: A table_dict object with all supervised annotations per experiment as values.

        """
        tag_dict = {}
        params = deepof.annotation_utils.get_hparameters(params)
        raw_coords = self.get_coords(center=None)

        try:
            coords = self.get_coords(center=center, align=align)
        except AssertionError:

            try:
                coords = self.get_coords(center="Center", align="Spine_1")
            except AssertionError:
                coords = self.get_coords(center="Center", align="Nose")

        dists = self.get_distances()
        speeds = self.get_coords(speed=1)
        if len(self._animal_ids) <= 1:
            features_dict = (
                deepof.post_hoc.align_deepof_kinematics_with_unsupervised_labels(
                    self, center=center, align=align, include_angles=False
                )
            )
        else:  # pragma: no cover
            features_dict = {
                _id: deepof.post_hoc.align_deepof_kinematics_with_unsupervised_labels(
                    self,
                    center=center,
                    align=align,
                    animal_id=_id,
                    include_angles=False,
                )
                for _id in self._animal_ids
            }

        # noinspection PyTypeChecker
        for key in tqdm(self._tables.keys()):

            # Remove indices and add at the very end, to avoid conflicts if
            # frame_rate is specified in project
            tag_index = raw_coords[key].index
            self._trained_model_path = resource_filename(__name__, "trained_models")

            supervised_tags = deepof.annotation_utils.supervised_tagging(
                self,
                raw_coords=raw_coords,
                coords=coords,
                dists=dists,
                full_features=features_dict,
                speeds=speeds,
                video=get_close_matches(
                    key,
                    [vid for vid in self._videos if vid.startswith(key)],
                    cutoff=0.1,
                    n=1,
                )[0],
                trained_model_path=self._trained_model_path,
                center=center,
                params=params,
            )

            supervised_tags.index = tag_index
            tag_dict[key] = supervised_tags

        if propagate_labels:  # pragma: no cover
            for key, tab in tag_dict.items():
                tab["pheno"] = self._exp_conditions[key]

        if video_output:  # pragma: no cover

            deepof.annotation_utils.tagged_video_output(
                self,
                tag_dict,
                video_output=video_output,
                frame_limit=frame_limit,
                debug=debug,
                n_jobs=n_jobs,
                params=params,
            )

        # Set table_dict to NaN if animals are missing
        tag_dict = deepof.utils.set_missing_animals(
            self,
            tag_dict,
            self.get_quality(),
            animal_ids=self._animal_ids + ["supervised"],
        )

        # Add missing tags to all animals
        presence_masks = deepof.utils.compute_animal_presence_mask(self.get_quality())
        for tag in tag_dict:
            for animal in self._animal_ids:
                tag_dict[tag][
                    "{}missing".format(("{}_".format(animal) if animal else ""))
                ] = (1 - presence_masks[tag][animal].values)

        return TableDict(
            tag_dict,
            typ="supervised",
            animal_ids=self._animal_ids,
            arena=self._arena,
            arena_dims=self._arena_dims,
            connectivity=self._connectivity,
            exp_conditions=self._exp_conditions,
            propagate_labels=propagate_labels,
        )

    def deep_unsupervised_embedding(
        self,
        preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        adjacency_matrix: np.ndarray = None,
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
            trained_models = deepof.model_utils.embedding_model_fitting(
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
                pretrained=pretrained,
                save_checkpoints=save_checkpoints,
                save_weights=save_weights,
                input_type=input_type,
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
        arena: str = None,
        arena_dims: np.array = None,
        animal_ids: List = tuple([""]),
        center: str = None,
        connectivity: nx.Graph = None,
        polar: bool = None,
        exp_conditions: dict = None,
        propagate_labels: bool = False,
        propagate_annotations: Union[Dict, bool] = False,
    ):
        """Store single datasets as dictionaries with individuals as keys and pandas.DataFrames as values.

        Includes methods for generating training and testing datasets for the autoencoders.

        Args:
            tabs (Dict): Dictionary of pandas.DataFrames with individual experiments as keys.
            typ (str): Type of the dataset. Examples are "coords", "dists", and "angles". For logging purposes only.
            arena (str): Type of the arena. Must be one of "circular-autodetect", "circular-manual", or "polygon-manual". Handled internally.
            arena_dims (np.array): Dimensions of the arena in mm.
            animal_ids (list): list of animal ids.
            center (str): Type of the center. Handled internally.
            polar (bool): Whether the dataset is in polar coordinates. Handled internally.
            exp_conditions (dict): dictionary with experiment IDs as keys and experimental conditions as values.
            propagate_labels (bool): Whether to propagate phenotypic labels from the original experiments to the transformed dataset.
            propagate_annotations (Dict): Dictionary of annotations to propagate. If provided, the supervised annotations of the individual experiments are propagated to the dataset.

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
        self._propagate_labels = propagate_labels
        self._propagate_annotations = propagate_annotations

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

        return TableDict(
            {k: value for k, value in table.items() if k in keys},
            self._type,
            connectivity=self._connectivity,
            propagate_labels=self._propagate_labels,
            propagate_annotations=self._propagate_annotations,
            exp_conditions=self._exp_conditions,
        )

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
                connectivity=self._connectivity,
                propagate_labels=self._propagate_labels,
                propagate_annotations=self._propagate_annotations,
                exp_conditions={
                    k: value
                    for k, value in self._exp_conditions.items()
                    if k in filtered_table.keys()
                },
            )

        return table

    def _prepare_projection(self) -> np.ndarray:
        """Return a numpy ndarray from the preprocessing of the table_dict object, ready for projection into a lower dimensional space."""
        labels = None

        # Takes care of propagated labels if present
        if self._propagate_labels:
            labels = {k: v.iloc[0, -1] for k, v in self.items()}
            labels = np.array([val for val in labels.values()])

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
            perplexity (int): Perplexity parameter for the t-SNE algorithm. Default is 30.

        Returns:
            tuple: Tuple containing projected data and projection type.

        """
        return self._projection(
            "umap",
            n_components=n_components,
        )

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
            columns_to_keep = deepof.utils.filter_columns(val.columns, selected_id)
            tabs[key] = val.loc[
                :, [bpa for bpa in val.columns if bpa in columns_to_keep]
            ]

        return TableDict(
            tabs,
            typ=self._type,
            connectivity=self._connectivity,
            propagate_labels=self._propagate_labels,
            propagate_annotations=self._propagate_annotations,
            exp_conditions=self._exp_conditions,
        )

    def merge(self, *args, ignore_index=False):
        """Take a number of table_dict objects and merges them to the current one.

        Returns a table_dict object of type 'merged'.
        Only annotations of the first table_dict object are kept.

        Args:
            *args (table_dict): table_dict objects to be merged.
            ignore_index (bool): ignore index when merging. Defaults to False.

        Returns:
            table_dict: Merged table_dict object.
        """
        args = [copy.deepcopy(self)] + list(args)
        merged_dict = defaultdict(list)
        for tabdict in args:
            for key, val in tabdict.items():
                merged_dict[key].append(val)

        propagate_labels = any(
            [self._propagate_labels] + [tabdict._propagate_labels for tabdict in args]
        )

        merged_tables = TableDict(
            {
                key: pd.concat(val, axis=1, ignore_index=ignore_index, join="inner")
                for key, val in merged_dict.items()
            },
            typ="merged",
            connectivity=self._connectivity,
            propagate_labels=propagate_labels,
            propagate_annotations=self._propagate_annotations,
        )

        # If there are labels passed, keep only one and append it as the last column
        for key, tab in merged_tables.items():

            pheno_cols = [col for col in tab.columns if "pheno" in str(col)]
            if len(pheno_cols) > 0:

                pheno_col = (
                    pheno_cols[0] if len(pheno_cols[0]) == 1 else [pheno_cols[0]]
                )
                labels = tab.loc[:, pheno_col].iloc[:, 0]

                merged_tables[key] = tab.drop(pheno_cols, axis=1)
                merged_tables[key]["pheno"] = labels

        # Retake original table dict properties
        merged_tables._animal_ids = self._animal_ids

        return merged_tables

    def get_training_set(
        self, current_table_dict: table_dict, test_videos: int = 0
    ) -> tuple:
        """Generate training and test sets as numpy.array objects for model training.

        Intended for internal usage only.

        Args:
            current_table_dict (table_dict): table_dict object containing the data to be used for training.
            test_videos (int): Number of videos to be used for testing. Defaults to 0.

        Returns:
            tuple: Tuple containing training data, training labels (if any), test data, and test labels (if any).
        """
        raw_data = current_table_dict.values()

        # Padding of videos with slightly different lengths
        # Making sure that the training and test sets end up balanced in terms of labels
        test_index = np.array([], dtype=int)

        raw_data = np.array([v.values for v in raw_data], dtype=object)
        if self._propagate_labels:
            concat_raw = np.concatenate(raw_data, axis=0)

            for label in set(list(concat_raw[:, -1])):
                label_index = np.random.choice(
                    [i for i in range(len(raw_data)) if raw_data[i][0, -1] == label],
                    test_videos,
                    replace=False,
                )
                test_index = np.concatenate([test_index, label_index])
        else:
            test_index = np.random.choice(
                range(len(raw_data)), test_videos, replace=False
            )

        y_train, X_test, y_test = np.array([]), np.array([]), np.array([])
        if test_videos > 0:
            try:
                X_test = np.concatenate(raw_data[test_index])
                X_train = np.concatenate(np.delete(raw_data, test_index, axis=0))
            except ValueError:  # pragma: no cover
                test_index = np.array([], dtype=int)
                X_train = np.concatenate(list(raw_data))
                warnings.warn(
                    "Could not find more than one sample for at least one condition. "
                    "Partition between training and test set was not possible."
                )

        else:
            X_train = np.concatenate(list(raw_data))

        if self._propagate_labels:
            le = LabelEncoder()
            X_train, y_train = X_train[:, :-1], X_train[:, -1][:, np.newaxis]
            y_train[:, 0] = le.fit_transform(y_train[:, 0])
            try:
                X_test, y_test = X_test[:, :-1], X_test[:, -1][:, np.newaxis]
                y_test[:, 0] = le.transform(y_test[:, 0])
            except IndexError:
                pass

        if self._propagate_annotations:  # pragma: no cover
            n_annot = list(self._propagate_annotations.values())[0].shape[1]

            try:
                X_train, y_train = (
                    X_train[:, :-n_annot],
                    np.concatenate([y_train, X_train[:, -n_annot:]], axis=1),
                )
            except ValueError:
                X_train, y_train = X_train[:, :-n_annot], X_train[:, -n_annot:]

            try:
                try:
                    X_test, y_test = (
                        X_test[:, :-n_annot],
                        np.concatenate([y_test, X_test[:, -n_annot:]]),
                    )
                except ValueError:
                    X_test, y_test = X_test[:, :-n_annot], X_test[:, -n_annot:]

            except IndexError:
                pass

        return (
            X_train.astype(float),
            y_train.astype(float),
            X_test.astype(float),
            y_test.astype(float),
            test_index,
        )

    # noinspection PyTypeChecker,PyGlobalUndefined
    def preprocess(
        self,
        automatic_changepoints=False,
        handle_ids: str = "concat",
        window_size: int = 25,
        window_step: int = 1,
        scale: str = "standard",
        pretrained_scaler: Any = None,
        test_videos: int = 0,
        verbose: int = 0,
        shuffle: bool = False,
        filter_low_variance: bool = False,
        interpolate_normalized: int = 10,
        precomputed_breaks: dict = None,
    ) -> np.ndarray:
        """Preprocess the loaded dataset before feeding to unsupervised embedding models.

        Capable of returning training and test sets ready for model training.

        Args:
            automatic_changepoints (str): specifies the changepoint detection kernel to use to rupture the data across time using Pelt. Can be set to "rbf" (default), or "linear". If False, fixed-length ruptures are appiled.
            handle_ids (str): indicates the default action to handle multiple animals in the TableDict object. Must be one of "concat" (body parts from different animals are treated as features) and "split" (different sliding windows are created for each animal).
            window_size (int): Minimum size of the applied ruptures. If automatic_changepoints is False, specifies the size of the sliding window to pass through the data to generate training instances.
            window_step (int): Specifies the minimum jump for the rupture algorithms. If automatic_changepoints is False, specifies the step to take when sliding the aforementioned window. In this case, a value of 1 indicates a true sliding window, and a value equal to window_size splits the data into non-overlapping chunks.
            scale (str): Data scaling method. Must be one of 'standard', 'robust' (default; recommended) and 'minmax'.
            pretrained_scaler (Any): Pre-fit global scaler, trained on the whole dataset. Useful to process single videos.
            test_videos (int): Number of videos to use for testing. If 0, no test set is generated.
            verbose (int): Verbosity level. 0 (default) is silent, 1 prints progress, 2 prints debug information.
            shuffle (bool): Whether to shuffle the data before preprocessing. Defaults to False.
            filter_low_variance (float): remove features with variance lower than the specified threshold. Useful to get rid of the x axis of the body part used for alignment (which would introduce noise after standardization).
            interpolate_normalized(int): if not 0, it specifies the number of standard deviations beyond which values will be interpolated after normalization. Only used if scale is set to "standard".
            precomputed_breaks (dict): If provided, changepoint detection is prevented, and provided breaks are used instead.

        Returns:
            X_train (np.ndarray): 3D dataset with shape (instances, sliding_window_size, features) generated from all training videos.
            y_train (np.ndarray): 3D dataset with shape (instances, sliding_window_size, labels) generated from all training videos. Note that no labels are use by default in the fully unsupervised pipeline (in which case this is an empty array).
            X_test (np.ndarray): 3D dataset with shape (instances, sliding_window_size, features) generated from all test videos (0 by default).
            y_test (np.ndarray): 3D dataset with shape (instances, sliding_window_size, labels) generated from all test videos. Note that no labels are use by default in the fully unsupervised pipeline (in which case this is an empty array).
        """
        # Create a temporary copy of the current TableDict object,
        # to avoid modifying it in place
        table_temp = copy.deepcopy(self)

        assert handle_ids in [
            "concat",
            "split",
        ], "handle IDs should be one of 'concat', and 'split'. See documentation for more details."

        if filter_low_variance:

            # Remove body parts with extremely low variance (usually the result of vertical alignment).
            for key, tab in table_temp.items():
                table_temp[key] = tab.iloc[
                    :,
                    list(np.where(tab.var(axis=0) > filter_low_variance)[0])
                    + list(np.where(["pheno" in str(col) for col in tab.columns])[0]),
                ]

        if scale:
            if verbose:
                print("Scaling data...")

            if scale not in ["robust", "standard", "minmax"]:
                raise ValueError(
                    "Invalid scaler. Select one of standard, minmax or robust"
                )  # pragma: no cover

            # Scale each experiment independently, to control for animal size
            for key, tab in table_temp.items():

                current_tab = deepof.utils.scale_table(
                    coordinates=self,
                    feature_array=tab,
                    scale=scale,
                    global_scaler=None,
                )

                table_temp[key] = pd.DataFrame(
                    current_tab,
                    columns=tab.columns,
                    index=tab.index,
                ).apply(lambda x: pd.to_numeric(x, errors="ignore"), axis=0)

            # Scale all experiments together, to control for differential stats
            if scale == "standard":
                global_scaler = StandardScaler()
            elif scale == "minmax":
                global_scaler = MinMaxScaler()
            else:
                global_scaler = RobustScaler()

            if pretrained_scaler is None:
                concat_data = pd.concat(list(table_temp.values()))
                global_scaler.fit(
                    concat_data.loc[:, concat_data.dtypes == float].values
                )
            else:
                global_scaler = pretrained_scaler

            for key, tab in table_temp.items():

                current_tab = deepof.utils.scale_table(
                    coordinates=self,
                    feature_array=tab,
                    scale=scale,
                    global_scaler=global_scaler,
                )

                table_temp[key] = pd.DataFrame(
                    current_tab, columns=tab.columns, index=tab.index
                )

        else:
            global_scaler = None

        if scale == "standard" and interpolate_normalized:

            # Interpolate outliers after preprocessing
            to_interpolate = copy.deepcopy(table_temp)
            for key, tab in to_interpolate.items():
                cur_tab = copy.deepcopy(tab.values)

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

                to_interpolate[key] = (
                    pd.DataFrame(cur_tab, index=tab.index, columns=tab.columns)
                    .apply(lambda x: pd.to_numeric(x, errors="ignore"))
                    .interpolate(limit_direction="both")
                )

            table_temp = to_interpolate

        # Split videos and generate training and test sets
        X_train, y_train, X_test, y_test, test_index = self.get_training_set(
            table_temp, test_videos
        )

        if verbose:
            print("Breaking time series...")

        # Apply rupture method to each train experiment independently
        X_train, train_breaks = deepof.utils.rupture_per_experiment(
            table_dict=table_temp,
            to_rupture=X_train,
            rupture_indices=[i for i in range(len(table_temp)) if i not in test_index],
            automatic_changepoints=automatic_changepoints,
            window_size=window_size,
            window_step=window_step,
            precomputed_breaks=precomputed_breaks,
        )

        # Print rupture information to screen
        if verbose > 1 and automatic_changepoints:
            rpt_lengths = np.all(X_train != 0, axis=2).sum(axis=1)
            print(
                "average rupture length: {}, standard deviation: {}".format(
                    rpt_lengths.mean(), rpt_lengths.std()
                )
            )
            print("minimum rupture length: {}".format(rpt_lengths.min()))
            print("maximum rupture length: {}".format(rpt_lengths.max()))

        if self._propagate_labels or self._propagate_annotations:

            if train_breaks is None:
                y_train, _ = deepof.utils.rupture_per_experiment(
                    table_dict=table_temp,
                    to_rupture=y_train,
                    rupture_indices=[
                        i for i in range(len(table_temp)) if i not in test_index
                    ],
                    automatic_changepoints=False,
                    window_size=window_size,
                    window_step=window_step,
                    precomputed_breaks=precomputed_breaks,
                )

            else:
                y_train = deepof.utils.split_with_breakpoints(y_train, train_breaks)

            y_train = y_train.mean(axis=1)

        if test_videos and len(test_index) > 0:

            # Apply rupture method to each test experiment independently
            X_test, test_breaks = deepof.utils.rupture_per_experiment(
                table_dict=table_temp,
                to_rupture=X_test,
                rupture_indices=test_index,
                automatic_changepoints=automatic_changepoints,
                window_size=window_size,
                window_step=window_step,
                precomputed_breaks=precomputed_breaks,
            )

            if self._propagate_labels or self._propagate_annotations:
                if test_breaks is None:
                    y_test, _ = deepof.utils.rupture_per_experiment(
                        table_dict=table_temp,
                        to_rupture=y_test,
                        rupture_indices=[
                            i for i in range(len(table_temp)) if i in test_index
                        ],
                        automatic_changepoints=False,
                        window_size=window_size,
                        window_step=window_step,
                        precomputed_breaks=precomputed_breaks,
                    )
                else:
                    y_test = deepof.utils.split_with_breakpoints(y_test, test_breaks)
                    y_test = y_test.mean(axis=1)

            if shuffle:
                shuffle_test = np.random.choice(
                    X_test.shape[0], X_test.shape[0], replace=False
                )
                X_test = X_test[shuffle_test]

                if self._propagate_labels:
                    y_test = y_test[shuffle_test]

        if shuffle:
            shuffle_train = np.random.choice(
                X_train.shape[0], X_train.shape[0], replace=False
            )
            X_train = X_train[shuffle_train]

            if self._propagate_labels:
                y_train = y_train[shuffle_train]

        X_test, y_test = np.array(X_test), np.array(y_test)

        # If automatic changepoints are anabled, train and test can have different seq lengths.
        # To remove that issue, pad the shortest set to match the longest one.
        if (
            test_videos
            and automatic_changepoints
            and len(X_test.shape) > 0
            and X_train.shape[1] != X_test.shape[1]
        ):
            max_seqlength = np.maximum(X_train.shape[1], X_test.shape[1])
            if X_train.shape[1] < max_seqlength:
                X_train = np.pad(
                    X_train,
                    ((0, 0), (0, max_seqlength - X_train.shape[1]), (0, 0)),
                    constant_values=0.0,
                )
            else:
                X_test = np.pad(
                    X_test,
                    ((0, 0), (0, max_seqlength - X_test.shape[1]), (0, 0)),
                    constant_values=0.0,
                )

        if verbose:
            print("Done!")

        if y_train.shape != (0,):
            assert (
                X_train.shape[0] == y_train.shape[0]
            ), "training set ({}) and labels ({}) do not have the same shape".format(
                X_train.shape[0], y_train.shape[0]
            )
        if y_test.shape != (0,):
            assert (
                X_test.shape[0] == y_test.shape[0]
            ), "training set and labels do not have the same shape"

        # If indicated and there are more than one animal in the dataset, split all animals as separate inputs
        if len(self._animal_ids) > 1 and handle_ids == "split":  # pragma: no cover
            X_train_split, X_test_split = [], []
            for aid in self._animal_ids:
                X_train_split.append(
                    X_train[
                        :,
                        :,
                        np.where(
                            [
                                np.array([i]).flatten()[0].startswith(aid)
                                for i in list(table_temp.values())[0].columns
                            ]
                        ),
                    ]
                )
                X_test_split.append(
                    X_test[
                        :,
                        :,
                        np.where(
                            [
                                np.array([i]).flatten()[0].startswith(aid)
                                for i in list(table_temp.values())[0].columns
                            ]
                        ),
                    ]
                )

            X_train, X_test = np.squeeze(np.concatenate(X_train_split)), np.squeeze(
                np.concatenate(X_test_split)
            )

        return (X_train, y_train, X_test, y_test), global_scaler


if __name__ == "__main__":
    # Remove excessive logging from tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
