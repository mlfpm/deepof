# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Data structures for preprocessing and wrangling of DLC output data. This is the main module handled by the user.
There are three main data structures to pay attention to:
- :class:`~deepof.data.Project`, which serves as a configuration hub for the whole pipeline
- :class:`~deepof.data.Coordinates`, which acts as an intermediary between project configuration and data, and contains
a plethora of processing methods to apply, and
- :class:`~deepof.data.TableDict`, which is the main data structure to store the data, having experiment IDs as keys
and processed time-series as values in a dictionary-like object.

For a detailed tutorial on how to use this module, see the advanced tutorials in the main section.

"""

import copy
import os
import warnings
from collections import defaultdict
from typing import Dict
from typing import Tuple, Any, List, NewType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import delayed, Parallel, parallel_backend
from pkg_resources import resource_filename
from sklearn import random_projection
from sklearn.decomposition import KernelPCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tqdm import tqdm

import deepof.models
import deepof.pose_utils
import deepof.train_utils
import deepof.utils
import deepof.visuals

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# CLASSES FOR PREPROCESSING AND DATA WRANGLING


class Project:
    """

    Class for loading and preprocessing DLC data of individual and multiple animals. All main computations are
    handled from here.

    """

    def __init__(
        self,
        arena_dims: int,
        animal_ids: List = tuple([""]),
        arena: str = "circular",
        arena_detection: str = "rule-based",
        enable_iterative_imputation: bool = True,
        exclude_bodyparts: List = tuple([""]),
        exp_conditions: dict = None,
        high_fidelity_arena: bool = False,
        interpolate_outliers: bool = True,
        interpolation_limit: int = 2,
        interpolation_std: int = 3,
        likelihood_tol: float = 0.85,
        model: str = "mouse_topview",
        path: str = deepof.utils.os.path.join("."),
        smooth_alpha: float = 2,
        table_format: str = "autodetect",
        frame_rate: int = None,
        video_format: str = ".mp4",
    ):
        """

        Initializes a Project object.

        Args:
            arena_dims (int): diameter of the arena in mm (so far, only round arenas are supported).
            animal_ids (list): list of animal ids.
            arena (str): arena type. So far, only 'circular' is supported.
            arena_detection (str): method for detecting the arena (must be either 'rule-based' (default) or 'cnn').
            enable_iterative_imputation (bool): whether to use iterative imputation for occluded body parts. Recommended,
            but slow.
            exclude_bodyparts (list): list of bodyparts to exclude from analysis.
            exp_conditions (dict): dictionary with experiment IDs as keys and experimental conditions as values.
            high_fidelity_arena (bool): whether to use high-fidelity arena detection. Recommended if light conditions
            are uneven across videos.
            interpolate_outliers (bool): whether to interpolate missing data.
            interpolation_limit (int): maximum number of missing frames to interpolate.
            interpolation_std (int): maximum number of standard deviations to interpolate.
            likelihood_tol (float): likelihood threshold for outlier detection.
            model (str): model to use for pose estimation. Defaults to 'mouse_topview' (as described in the documentation).
            path (str): path to the folder containing the DLC output data.
            smooth_alpha (float): smoothing intensity. The higher the value, the more smoothing.
            table_format (str): format of the table. Defaults to 'autodetect', but can be set to "csv" or "h5".
            frame_rate (int): frame rate of the videos. If not specified, it will be inferred from the video files.
            video_format (str): video format. Defaults to '.mp4'.

        """

        # Set working paths
        self.path = path
        self.video_path = os.path.join(self.path, "Videos")
        self.table_path = os.path.join(self.path, "Tables")
        self.trained_path = resource_filename(__name__, "trained_models")

        # Detect files to load from disk
        self.table_format = table_format
        if self.table_format == "autodetect":
            ex = [i for i in os.listdir(self.table_path) if not i.startswith(".")][0]
            if ".h5" in ex:
                self.table_format = ".h5"
            elif ".csv" in ex:
                self.table_format = ".csv"
        self.videos = sorted(
            [
                vid
                for vid in deepof.utils.os.listdir(self.video_path)
                if vid.endswith(video_format) and not vid.startswith(".")
            ]
        )
        self.tables = sorted(
            [
                tab
                for tab in deepof.utils.os.listdir(self.table_path)
                if tab.endswith(self.table_format) and not tab.startswith(".")
            ]
        )
        assert len(self.videos) == len(
            self.tables
        ), "Unequal number of videos and tables. Please check your file structure"

        # Loads arena details and (if needed) detection models
        self.arena = arena
        self.arena_detection = arena_detection
        self.arena_dims = arena_dims
        self.ellipse_detection = None
        if arena == "circular" and arena_detection == "cnn":
            self.ellipse_detection = tf.keras.models.load_model(
                [
                    os.path.join(self.trained_path, i)
                    for i in os.listdir(self.trained_path)
                    if i.startswith("elliptical")
                ][0]
            )

        # Set the rest of the init parameters
        self.angles = True
        self.animal_ids = animal_ids
        self.distances = "all"
        self.ego = False
        self.exp_conditions = exp_conditions
        self.high_fidelity = high_fidelity_arena
        self.interpolate_outliers = interpolate_outliers
        self.interpolation_limit = interpolation_limit
        self.interpolation_std = interpolation_std
        self.likelihood_tolerance = likelihood_tol
        self.smooth_alpha = smooth_alpha
        self.frame_rate = frame_rate
        self.video_format = video_format
        self.enable_iterative_imputation = enable_iterative_imputation

        model_dict = {
            "{}mouse_topview".format(aid): deepof.utils.connect_mouse_topview(aid)
            for aid in self.animal_ids
        }
        self.connectivity = {aid: model_dict[aid + model] for aid in self.animal_ids}

        # Remove specified body parts from the mice graph
        self.exclude_bodyparts = exclude_bodyparts
        if len(self.animal_ids) > 1 and self.exclude_bodyparts != tuple([""]):
            self.exclude_bodyparts = [
                aid + "_" + bp for aid in self.animal_ids for bp in exclude_bodyparts
            ]

        if self.exclude_bodyparts != tuple([""]):
            for aid in self.animal_ids:
                for bp in self.exclude_bodyparts:
                    if bp.startswith(aid):
                        self.connectivity[aid].remove_node(bp)

    def __str__(self):  # pragma: no cover
        if self.exp_conditions:
            return "deepof analysis of {} videos across {} condition{}".format(
                len(self.videos),
                len(set(self.exp_conditions.values())),
                ("s" if len(set(self.exp_conditions.values())) > 1 else ""),
            )
        return "deepof analysis of {} videos".format(len(self.videos))

    def __repr__(self):  # pragma: no cover
        if self.exp_conditions:
            return "deepof analysis of {} videos across {} condition{}".format(
                len(self.videos),
                len(set(self.exp_conditions.values())),
                ("s" if len(set(self.exp_conditions.values())) > 1 else ""),
            )
        return "deepof analysis of {} videos".format(len(self.videos))

    @property
    def distances(self):
        """
        List. If not 'all', sets the body parts among which the
        distances will be computed
        """
        return self._distances

    @property
    def ego(self):
        """
        String, name of a body part. If True, computes only the distances
        between the specified body part and the rest
        """
        return self._ego

    @property
    def angles(self):
        """
        Bool. Toggles angle computation. True by default. If turned off,
        enhances performance for big datasets
        """
        return self._angles

    def get_arena(self, tables) -> np.array:
        """
        Returns the arena as recognised from the videos
        """

        scales = []
        arena_params = []
        video_resolution = []

        if self.arena in ["circular"]:

            for vid_index, _ in enumerate(self.videos):
                ellipse, h, w = deepof.utils.recognize_arena(
                    videos=self.videos,
                    tables=tables,
                    vid_index=vid_index,
                    path=self.video_path,
                    arena_type=self.arena,
                    high_fidelity=self.high_fidelity,
                    detection_mode=self.arena_detection,
                    cnn_model=self.ellipse_detection,
                )

                # scales contains the coordinates of the center of the arena,
                # the absolute diameter measured from the video in pixels, and
                # the provided diameter in mm (1 -default- equals not provided)
                scales.append(
                    list(
                        np.array(
                            [
                                ellipse[0][0],
                                ellipse[0][1],
                                np.mean([ellipse[1][0], ellipse[1][1]]) * 2,
                            ]
                        )
                    )
                    + [self.arena_dims]
                )
                arena_params.append(ellipse)
                video_resolution.append((h, w))

        else:
            raise NotImplementedError("arenas must be set to one of: 'circular'")

        return np.array(scales), arena_params, video_resolution

    def load_tables(self, verbose: bool = False) -> deepof.utils.Tuple:
        """

        Loads videos and tables into dictionaries.

        Args:
            verbose (bool): If True, prints the progress of data loading.

        Returns:
            Tuple: A tuple containing the following a dictionary with all loaded tables per experiment,
            and another dictionary with DLC data quality.

        """

        if self.table_format not in [".h5", ".csv"]:
            raise NotImplementedError(
                "Tracking files must be in either h5 or csv format"
            )

        if verbose:
            print("Loading trajectories...")

        tab_dict = {}

        if self.table_format == ".h5":

            tab_dict = {
                deepof.utils.re.findall("(.*)DLC", tab)[0]: pd.read_hdf(
                    deepof.utils.os.path.join(self.table_path, tab), dtype=float
                )
                for tab in self.tables
            }
        elif self.table_format == ".csv":

            tab_dict = {
                deepof.utils.re.findall("(.*)DLC", tab)[0]: pd.read_csv(
                    deepof.utils.os.path.join(self.table_path, tab),
                    header=[0, 1, 2],
                    index_col=0,
                    dtype=float,
                )
                for tab in self.tables
            }

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

        for key, value in tab_dict.items():
            x = value.xs("x", level="coords", axis=1, drop_level=False)
            y = value.xs("y", level="coords", axis=1, drop_level=False)
            lik = value.xs("likelihood", level="coords", axis=1, drop_level=True)

            tab_dict[key] = pd.concat([x, y], axis=1).sort_index(axis=1)
            lik_dict[key] = lik.droplevel("scorer", axis=1)

        if self.smooth_alpha:

            if verbose:
                print("Smoothing trajectories...")

            for key, tab in tab_dict.items():
                cur_idx = tab.index
                cur_cols = tab.columns
                smooth = pd.DataFrame(
                    deepof.utils.smooth_mult_trajectory(
                        np.array(tab),
                        alpha=self.smooth_alpha,
                        w_length=15,
                    )
                ).reset_index(drop=True)
                smooth.columns = cur_cols
                smooth.index = cur_idx
                tab_dict[key] = smooth

        # Remove scorer header
        for key, tab in tab_dict.items():
            tab_dict[key] = tab.loc[:, tab.columns.levels[0][0]]

        if self.exclude_bodyparts != tuple([""]):

            for k, value in tab_dict.items():
                temp = value.drop(self.exclude_bodyparts, axis=1, level="bodyparts")
                temp.sort_index(axis=1, inplace=True)
                temp.columns = pd.MultiIndex.from_product(
                    [sorted(list(set([i[j] for i in temp.columns]))) for j in range(2)]
                )
                tab_dict[k] = temp.sort_index(axis=1)

        if self.interpolate_outliers:

            if verbose:
                print("Interpolating outliers...")

            for k, value in tab_dict.items():

                tab_dict[k] = deepof.utils.interpolate_outliers(
                    value,
                    lik_dict[k],
                    likelihood_tolerance=self.likelihood_tolerance,
                    mode="or",
                    limit=self.interpolation_limit,
                    n_std=self.interpolation_std,
                )

        if self.enable_iterative_imputation:

            if verbose:
                print("Iterative imputation of ocluded bodyparts...")

            for k, value in tab_dict.items():
                imputed = IterativeImputer(
                    skip_complete=True,
                    max_iter=10,
                    n_nearest_features=value.shape[1] // len(self.animal_ids) - 1,
                    tol=1e-1,
                ).fit_transform(value)
                tab_dict[k] = pd.DataFrame(
                    imputed, index=value.index, columns=value.columns
                )

        return tab_dict, lik_dict

    def get_distances(self, tab_dict: dict, verbose: bool = False) -> dict:
        """

        Computes the distances between all selected body parts over time. If ego is provided, it only returns
        distances to a specified bodypart.

        Args:
            tab_dict (dict): Dictionary of pandas DataFrames containing the trajectories of all bodyparts.
            verbose (bool): If True, prints progress. Defaults to False.

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
            key: deepof.utils.bpart_distance(
                tab,
                scales[i, 1],
                scales[i, 0],
            )
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

    def get_angles(self, tab_dict: dict, verbose: bool = False) -> dict:
        """

        Computes all the angles between adjacent bodypart trios per video and per frame in the data.

        Args:
            tab_dict (dict): Dictionary of pandas DataFrames containing the trajectories of all bodyparts.
            verbose (bool): If True, prints progress. Defaults to False.

        Returns:
            dict: Dictionary of pandas DataFrames containing the distances between all bodyparts.

        """

        if verbose:
            print("Computing angles...")

        # Add all three-element cliques on each mouse
        cliques = []
        for i in self.animal_ids:
            cliques += deepof.utils.nx.enumerate_all_cliques(self.connectivity[i])
        cliques = [i for i in cliques if len(i) == 3]

        angle_dict = {}
        for key, tab in tab_dict.items():

            dats = []
            for clique in cliques:
                dat = pd.DataFrame(
                    deepof.utils.angle_trio(
                        np.array(tab[clique]).reshape([3, tab.shape[0], 2])
                    )
                ).T

                orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2]]
                dat.columns = [tuple(clique[i] for i in order) for order in orders]

                dats.append(dat)

            dats = pd.concat(dats, axis=1)

            angle_dict[key] = dats

        # Restore original index
        for key in angle_dict.keys():
            angle_dict[key].index = tab_dict[key].index

        return angle_dict

    def run(self, verbose: bool = True) -> coordinates:
        """

        Generates a deepof.Coordinates dataset using all the options specified during initialization.

        Args:
            verbose (bool): If True, prints progress. Defaults to True.

        Returns:
            coordinates: Deepof.Coordinates object containing the trajectories of all bodyparts.

        """

        tables, quality = self.load_tables(verbose)
        if self.exp_conditions is not None:
            assert (
                tables.keys() == self.exp_conditions.keys()
            ), "experimental IDs in exp_conditions do not match"

        distances = None
        angles = None

        self.scales, self.arena_params, self.video_resolution = self.get_arena(
            tables=tables
        )

        if self.distances:
            distances = self.get_distances(tables, verbose)

        if self.angles:
            angles = self.get_angles(tables, verbose)

        if verbose:
            print("Done!")

        return Coordinates(
            angles=angles,
            animal_ids=self.animal_ids,
            arena=self.arena,
            arena_detection=self.arena_detection,
            arena_dims=self.arena_dims,
            distances=distances,
            exp_conditions=self.exp_conditions,
            path=self.path,
            quality=quality,
            scales=self.scales,
            arena_params=self.arena_params,
            tables=tables,
            trained_model_path=self.trained_path,
            videos=self.videos,
            video_resolution=self.video_resolution,
        )

    @distances.setter
    def distances(self, value):
        self._distances = value

    @ego.setter
    def ego(self, value):
        self._ego = value

    @angles.setter
    def angles(self, value):
        self._angles = value


class Coordinates:
    """
    Class for storing the results of a ran project. Methods are mostly setters and getters in charge of tidying up
    the generated tables.
    """

    def __init__(
        self,
        arena: str,
        arena_detection: str,
        arena_dims: np.array,
        path: str,
        quality: dict,
        scales: np.ndarray,
        arena_params: List,
        tables: dict,
        trained_model_path: str,
        videos: List,
        video_resolution: List,
        angles: dict = None,
        animal_ids: List = tuple([""]),
        distances: dict = None,
        exp_conditions: dict = None,
    ):
        """

        Class for storing the results of a ran project. Methods are mostly setters and getters in charge of tidying up
        the generated tables.

        Args:
            arena (str): Type of arena used for the experiment. See deepof.data.Project for more information.
            arena_detection (str): Type of arena detection used for the experiment. See deepof.data.Project for more
            information.
            arena_dims (np.array): Dimensions of the arena. See deepof.data.Project for more information.
            path (str): Path to the folder containing the results of the experiment.
            quality (dict): Dictionary containing the quality of the experiment. See deepof.data.Project for more information.
            scales (np.ndarray): Scales used for the experiment. See deepof.data.Project for more information.
            arena_params (List): List containing the parameters of the arena. See deepof.data.Project for more information.
            tables (dict): Dictionary containing the tables of the experiment. See deepof.data.Project for more information.
            trained_model_path (str): Path to the trained models used for the supervised pipeline. For internal use only.
            videos (List): List containing the videos used for the experiment. See deepof.data.Project for more information.
            video_resolution (List): List containing the automatically detected resolution of the videos used for the experiment.
            angles (dict): Dictionary containing the angles of the experiment. See deepof.data.Project for more information.
            animal_ids (List): List containing the animal IDs of the experiment. See deepof.data.Project for more information.
            distances (dict): Dictionary containing the distances of the experiment. See deepof.data.Project for more information.
            exp_conditions (dict): Dictionary containing the experimental conditions of the experiment.
            See deepof.data.Project for more information.

        """

        self._animal_ids = animal_ids
        self._arena = arena
        self._arena_detection = arena_detection
        self._arena_params = arena_params
        self._arena_dims = arena_dims
        self._exp_conditions = exp_conditions
        self._path = path
        self._quality = quality
        self._scales = scales
        self._tables = tables
        self._trained_model_path = trained_model_path
        self._videos = videos
        self._video_resolution = video_resolution
        self.angles = angles
        self.distances = distances

    def __str__(self):  # pragma: no cover
        if self._exp_conditions:
            return "deepof coordinates of {} videos across {} condition{}".format(
                len(self._videos),
                len(set(self._exp_conditions.values())),
                ("s" if len(set(self._exp_conditions.values())) > 1 else ""),
            )
        return "deepof analysis of {} videos".format(len(self._videos))

    def __repr__(self):  # pragma: no cover
        if self._exp_conditions:
            return "deepof coordinates of {} videos across {} condition{}".format(
                len(self._videos),
                len(set(self._exp_conditions.values())),
                ("s" if len(set(self._exp_conditions.values())) > 1 else ""),
            )
        return "deepof analysis of {} videos".format(len(self._videos))

    def get_coords(
        self,
        center: str = "arena",
        polar: bool = False,
        speed: int = 0,
        align: str = False,
        align_inplace: bool = True,
        selected_id: str = None,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """

        Returns a table_dict object with the coordinates of each animal as values.

        Args:
            center (str): Name of the body part to which the positions will be centered. If false,
            the raw data is returned; if 'arena' (default), coordinates are centered in the pitch
            polar (bool) States whether the coordinates should be converted to polar values.
            speed (int): States the derivative of the positions to report. Speed is returned if 1, acceleration if 2,
            jerk if 3, etc.
            align (str): Selects the body part to which later processes will align the frames with
            (see preprocess in table_dict documentation).
            align_inplace (bool): Only valid if align is set. Aligns the vector that goes from the origin to the
            selected body part with the y axis, for all time points (default).
            selected_id (str): Selects a single animal on multi animal settings. Defaults to None (all animals are processed).
            propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label
            propagate_annotations (dict): If a dictionary is provided, supervised annotations are propagated through
            the training dataset. This can be used for regularising the latent space based on already known traits.

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
            if self._arena == "circular":

                for i, (key, value) in enumerate(tabs.items()):

                    value.loc[:, (slice("coords"), [coord_1])] = (
                        value.loc[:, (slice("coords"), [coord_1])] - scales[i][0]
                    )

                    value.loc[:, (slice("coords"), [coord_2])] = (
                        value.loc[:, (slice("coords"), [coord_2])] - scales[i][1]
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
                        .loc[:, (slice("coords"), [coord_1])]
                        .subtract(
                            value[aid + ("_" if aid != "" else "") + center][coord_1],
                            axis=0,
                        )
                    )

                    # center on y / phi
                    value.update(
                        value.loc[:, [i for i in value.columns if i[0].startswith(aid)]]
                        .loc[:, (slice("coords"), [coord_2])]
                        .subtract(
                            value[aid + ("_" if aid != "" else "") + center][coord_2],
                            axis=0,
                        )
                    )

                # noinspection PyUnboundLocalVariable
                tabs[key] = value.loc[
                    :,
                    [tab for tab in value.columns if center not in tab[0]],
                ]

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

                        partial_aligned = pd.DataFrame(
                            deepof.utils.align_trajectories(
                                np.array(partial_aligned), mode="all"
                            )
                        )
                        aligned_coordinates = pd.concat(
                            [aligned_coordinates, partial_aligned], axis=1
                        )

                aligned_coordinates.index = all_index
                aligned_coordinates.columns = pd.MultiIndex.from_tuples(all_columns)
                tabs[key] = aligned_coordinates

        if propagate_annotations:
            annotations = list(propagate_annotations.values())[0].columns

            for key, tab in tabs.items():
                for ann in annotations:
                    tab.loc[:, ann] = propagate_annotations[key].loc[:, ann]

        if propagate_labels:
            for key, tab in tabs.items():
                tab.loc[:, "pheno"] = self._exp_conditions[key]

        return TableDict(
            tabs,
            "coords",
            arena=self._arena,
            arena_dims=self._scales,
            center=center,
            polar=polar,
            propagate_labels=propagate_labels,
            propagate_annotations=propagate_annotations,
        )

    def get_distances(
        self,
        speed: int = 0,
        selected_id: str = None,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """

        Returns a table_dict object with the distances between body parts animal as values.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            propagate_labels (bool): If True, the pheno column will be propagated from the original data.
            propagate_annotations (Dict): A dictionary of annotations to propagate.

        Returns:
            table_dict: A table_dict object with the distances between body parts animal as values.

        """

        tabs = deepof.utils.deepcopy(self.distances)

        if self.distances is not None:

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="dists")
                    tabs[key] = vel

            if selected_id is not None:
                for key, val in tabs.items():
                    tabs[key] = val.loc[
                        :,
                        deepof.utils.filter_columns(val.columns, selected_id),
                    ]

            if propagate_labels:
                for key, tab in tabs.items():
                    tab.loc[:, "pheno"] = self._exp_conditions[key]

            if propagate_annotations:
                annotations = list(propagate_annotations.values())[0].columns

                for key, tab in tabs.items():
                    for ann in annotations:
                        tab.loc[:, ann] = propagate_annotations[key].loc[:, ann]

            return TableDict(
                tabs,
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
        """

        Returns a table_dict object with the angles between body parts animal as values.

        Args:
            degrees (bool): If True (default), the angles will be in degrees. Otherwise they will be converted to radians.
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            propagate_labels (bool): If True, the pheno column will be propagated from the original data.
            propagate_annotations (Dict): A dictionary of annotations to propagate.

        Returns:
            table_dict: A table_dict object with the angles between body parts animal as values.

        """

        tabs = deepof.utils.deepcopy(self.angles)

        if self.angles is not None:
            if degrees:
                tabs = {key: np.degrees(tab) for key, tab in tabs.items()}

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="angles")
                    tabs[key] = vel

            if selected_id is not None:
                for key, val in tabs.items():
                    tabs[key] = val.loc[
                        :,
                        deepof.utils.filter_columns(val.columns, selected_id),
                    ]

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
                propagate_labels=propagate_labels,
                propagate_annotations=propagate_annotations,
                typ="angles",
            )

        raise ValueError(
            "Angles not computed. Read the documentation for more details"
        )  # pragma: no cover

    def get_videos(self, play: bool = False):
        """
        Retuens the videos associated with the dataset as a list.
        """

        if play:  # pragma: no cover
            raise NotImplementedError

        return self._videos

    @property
    def get_exp_conditions(self):
        """
        Returns the stored dictionary with experimental conditions per subject
        """

        return self._exp_conditions

    def get_quality(self):
        """
        Retrieves a dictionary with the tagging quality per video, as reported by DLC
        """

        return self._quality

    @property
    def get_arenas(self):
        """
        Retrieves all available information associated with the arena
        """

        return self._arena, [self._arena_dims], self._scales

    # noinspection PyDefaultArgument
    def supervised_annotation(
        self,
        params: Dict = {},
        video_output: bool = False,
        frame_limit: int = np.inf,
        debug: bool = False,
        n_jobs: int = 1,
        propagate_labels: bool = False,
    ) -> table_dict:
        """

        Annotates coordinates with behavioral traits using a supervised pipeline.

        Args:
            params (Dict): A dictionary with the parameters to use for the pipeline. If unsure, leave empty.
            video_output (bool): It outputs a fully annotated video for each experiment indicated in a list. If set to
            "all", it will output all videos. False by default.
            frame_limit (int): Only applies if video_output is not False. Indicates the maximum number of frames per
            video to output.
            debug (bool): Only applies if video_output is not False. If True, all videos will include debug information,
            such as the detected arena and the preprocessed tracking tags.
            n_jobs (int): Number of jobs to use for parallel processing.
            propagate_labels (bool): If True, the pheno column will be propagated from the original data.

        Returns:
            table_dict: A table_dict object with all supervised annotations per experiment as values.

        """

        tag_dict = {}
        raw_coords = self.get_coords(center=None)
        coords = self.get_coords(center="Center", align="Spine_1")
        dists = self.get_distances()
        angs = self.get_angles()
        speeds = self.get_coords(speed=1)

        # noinspection PyTypeChecker
        for key in tqdm(self._tables.keys()):
            # Remove indices and add at the very end, to avoid conflicts if
            # frame_rate is specified in project
            tag_index = raw_coords[key].index
            supervised_tags = deepof.pose_utils.supervised_tagging(
                self,
                raw_coords=raw_coords,
                coords=coords,
                dists=dists,
                angs=angs,
                speeds=speeds,
                video=[vid for vid in self._videos if key + "DLC" in vid][0],
                trained_model_path=self._trained_model_path,
                params=params,
            )
            supervised_tags.index = tag_index
            tag_dict[key] = supervised_tags

        if propagate_labels:
            for key, tab in tag_dict.items():
                tab["pheno"] = self._exp_conditions[key]

        if video_output:  # pragma: no cover

            def output_video(idx):
                """
                Outputs a single annotated video. Enclosed in a function to enable parallelization
                """

                deepof.pose_utils.annotate_video(
                    self,
                    tag_dict=tag_dict[idx],
                    vid_index=list(self._tables.keys()).index(idx),
                    debug=debug,
                    frame_limit=frame_limit,
                    params=params,
                )
                pbar.update(1)

            if isinstance(video_output, list):
                vid_idxs = video_output
            elif video_output == "all":
                vid_idxs = list(self._tables.keys())
            else:
                raise AttributeError(
                    "Video output must be either 'all' or a list with the names of the videos to render"
                )

            pbar = tqdm(total=len(vid_idxs))
            with parallel_backend("threading", n_jobs=n_jobs):
                Parallel()(delayed(output_video)(key) for key in vid_idxs)
            pbar.close()

        return TableDict(
            tag_dict,
            typ="supervised",
            arena=self._arena,
            arena_dims=self._arena_dims,
            propagate_labels=propagate_labels,
        )

    @staticmethod
    def deep_unsupervised_embedding(
        preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        embedding_model: str = "GMVAE",
        batch_size: int = 256,
        encoding_size: int = 4,
        epochs: int = 50,
        hparams: dict = None,
        kl_annealing_mode: str = "linear",
        kl_warmup: int = 0,
        log_history: bool = True,
        log_hparams: bool = False,
        loss: str = "ELBO",
        mmd_annealing_mode: str = "linear",
        mmd_warmup: int = 0,
        montecarlo_kl: int = 10,
        n_components: int = 10,
        overlap_loss: float = 0,
        output_path: str = "unsupervised_trained_models",
        next_sequence_prediction: float = 0,
        phenotype_prediction: float = 0,
        supervised_prediction: float = 0,
        pretrained: str = False,
        save_checkpoints: bool = False,
        save_weights: bool = True,
        reg_cat_clusters: bool = False,
        reg_cluster_variance: bool = False,
        entropy_knn: int = 100,
        input_type: str = False,
        run: int = 0,
        strategy: tf.distribute.Strategy = "one_device",
    ) -> Tuple:
        """

        Annotates coordinates using a deep unsupervised autoencoder.

        Args:
            preprocessed_object (tuple): Tuple containing a preprocessed object (X_train, y_train, X_test, y_test).
            embedding_model (str): Model to use for unsupervised embedding. Must be one of "VQVAE" and "GMVAE".
            batch_size (int): Batch size for training.
            encoding_size (int): Dimention size of the latent space.
            epochs (int): Maximum number of epochs to train the model. Actual training might be shorter, as the model
            will stop training when validation loss stops decreasing.
            hparams (dict): Hyperparameters for the model. See deepof.models for more details.
            kl_annealing_mode (str): Mode for annealing the KL divergence. Must be one of "linear" and "sigmoid".
            kl_warmup (int): Number of epochs to warm up the KL divergence.
            log_history (bool): Whether to log the history of the model to TensorBoard.
            log_hparams (bool): Whether to log the hyperparameters of the model to TensorBoard.
            loss (str): Loss function to use. Must be one of "VQ", "ELBO", "MMD", and "ELBO+MMD".
            mmd_annealing_mode (str): Mode for annealing the MMD loss. Must be one of "linear" and "sigmoid".
            mmd_warmup (int): Number of epochs to warm up the MMD loss.
            montecarlo_kl (int): Number of Monte Carlo samples to take for the Monte-Carlo KL divergence.
            n_components (int): Number of latent clusters for the embedding model to use.
            overlap_loss (float): Weight of the overlap loss, which adds a regularization term to GMVAE models which
            penalizes the overlap between the clusters.
            output_path (str): Path to save the trained model and all log files.
            next_sequence_prediction (float): Weight of the next sequence prediction loss, used as a regularizer.
            phenotype_prediction (float): Weight of the phenotype prediction loss, used to guide training towards using
            traits that are differentially expressed across phenotypes.
            supervised_prediction (float): Weight of the supervised prediction loss, used to guide training towards
            clusters that correlate with the supervised annotations.
            pretrained (str): Whether to load a pretrained model. If False, model is trained from scratch. If not,
            must be the path to a saved model.
            save_checkpoints (bool): Whether to save checkpoints of the model during training. Defaults to False.
            save_weights (bool): Whether to save the weights of the model during training. Defaults to True.
            reg_cat_clusters (bool): Whether to regularize the latent space of GMVAE models based on the number of
            categories in the categorical features.
            reg_cluster_variance (bool): Whether to penalize the latent space of GMVAE models based on variance
            differences across clusters.
            entropy_knn (int): Number of nearest neighbors to use for the entropy computation in overlap_loss.
            input_type (str): Type of the preprocessed_object passed as the first parameter. See deepof.data.TableDict
            for more details.
            run (int): Run number for the model. Used to save the model and log files. Optional.
            strategy (tf.distribute.Strategy): Distributed strategy for TensorFloe to use. Must be one of "one_device",
            or "mirrored_strategy" (capable of handling more than one GPU, ideal for big experiments). If unsure, leave
            as "one_device".

        Returns:
            Tuple: Tuple containing all trained models. See specific model documentation under deepof.models for details.

        """

        trained_models = deepof.train_utils.autoencoder_fitting(
            preprocessed_object=preprocessed_object,
            embedding_model=embedding_model,
            batch_size=batch_size,
            encoding_size=encoding_size,
            epochs=epochs,
            hparams=hparams,
            kl_annealing_mode=kl_annealing_mode,
            kl_warmup=kl_warmup,
            log_history=log_history,
            log_hparams=log_hparams,
            loss=loss,
            mmd_annealing_mode=mmd_annealing_mode,
            mmd_warmup=mmd_warmup,
            montecarlo_kl=montecarlo_kl,
            n_components=n_components,
            overlap_loss=overlap_loss,
            output_path=output_path,
            next_sequence_prediction=next_sequence_prediction,
            phenotype_prediction=phenotype_prediction,
            supervised_prediction=supervised_prediction,
            pretrained=pretrained,
            save_checkpoints=save_checkpoints,
            save_weights=save_weights,
            reg_cat_clusters=reg_cat_clusters,
            reg_cluster_variance=reg_cluster_variance,
            entropy_knn=entropy_knn,
            input_type=input_type,
            run=run,
            strategy=strategy,
        )

        # returns a list of trained tensorflow models
        return trained_models


class TableDict(dict):
    """
    Main class for storing a single dataset as a dictionary with individuals as keys and pandas.DataFrames as values.
    Includes methods for generating training and testing datasets for the autoencoders.
    """

    def __init__(
        self,
        tabs: Dict,
        typ: str,
        arena: str = None,
        arena_dims: np.array = None,
        center: str = None,
        polar: bool = None,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ):
        """

        Main class for storing a single dataset as a dictionary with individuals as keys and pandas.DataFrames as values.
        Includes methods for generating training and testing datasets for the autoencoders.

        Args:
            tabs (Dict): Dictionary of pandas.DataFrames with individual experiments as keys.
            typ (str): Type of the dataset. Examples are "coords", "dists", and "angles". For logging purposes only.
            arena (str): Type of the arena. Currently, only "circular" is supported. Handled internally.
            arena_dims (np.array): Dimensions of the arena in mm.
            center (str): Type of the center. Handled internally.
            polar (bool): Whether the dataset is in polar coordinates. Handled internally.
            propagate_labels (bool): Whether to propagate phenotypic labels from the original experiments to the
            transformed dataset.
            propagate_annotations (Dict): Dictionary of annotations to propagate. If provided, the supervised annotations
            of the individual experiments are propagated to the dataset.

        """

        super().__init__(tabs)
        self._type = typ
        self._center = center
        self._polar = polar
        self._arena = arena
        self._arena_dims = arena_dims
        self._propagate_labels = propagate_labels
        self._propagate_annotations = propagate_annotations

    def filter_videos(self, keys: list) -> table_dict:
        """

        Returns a subset of the original table_dict object, containing only the specified keys. Useful, for example,
        for selecting data coming from videos of a specified condition.

        Args:
            keys (list): List of keys to keep.

        Returns:
            TableDict: Subset of the original table_dict object, containing only the specified keys.

        """

        table = deepof.utils.deepcopy(self)
        assert np.all([k in table.keys() for k in keys]), "Invalid keys selected"

        return TableDict(
            {k: value for k, value in table.items() if k in keys}, self._type
        )

    # noinspection PyTypeChecker
    def plot_heatmaps(
        self,
        bodyparts: list,
        xlim: float = None,
        ylim: float = None,
        save: bool = False,
        i: int = 0,
        dpi: int = 100,
    ) -> plt.figure:
        """

        Plots heatmaps of the specified body parts (bodyparts) of the specified animal (i).

        Args:
            bodyparts (list): list of body parts to plot.
            xlim (float): x-axis limits.
            ylim (float): y-axis limits.
            save (str):  if provided, the figure is saved to the specified path.
            i (int): index of the animal to plot.
            dpi (int): resolution of the figure.

        Returns:
            plt.figure: Figure object containing the heatmaps.

        """

        if self._type != "coords" or self._polar:
            raise NotImplementedError(
                "Heatmaps only available for cartesian coordinates. "
                "Set polar to False in get_coordinates and try again"
            )  # pragma: no cover

        if not self._center:  # pragma: no cover
            warnings.warn("Heatmaps look better if you center the data")

        if self._arena == "circular":
            heatmaps = deepof.visuals.plot_heatmap(
                list(self.values())[i],
                bodyparts,
                xlim=xlim,
                ylim=ylim,
                save=save,
                dpi=dpi,
            )

            return heatmaps

    def _prepare_projection(self) -> np.ndarray:
        """
        Returns a numpy ndarray from the preprocessing of the table_dict object,
        ready for projection into a lower dimensional space
        """

        labels = None

        # Takes care of propagated labels if present
        if self._propagate_labels:
            labels = {k: v.iloc[0, -1] for k, v in self.items()}
            labels = np.array([val for val in labels.values()])

        X = {k: np.mean(v, axis=0) for k, v in self.items()}
        X = np.concatenate(
            [np.array(exp)[:, np.newaxis] for exp in X.values()],
            axis=1,
        ).T

        return X, labels

    def _projection(
        self,
        projection_type: str,
        n_components: int = 2,
        kernel: str = None,
        perplexity: int = None,
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """

        Returns a training set generated from the 2D original data (time x features) and a specified projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons. For internal usage only.

        Args:
            projection_type (str): Projection to be used.
            n_components: Number of components to project to.
            kernel: Kernel to be used for the random and PCA algorithms.
            perplexity: Perplexity parameter for the t-SNE algorithm.

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
        elif projection_type == "tsne":
            projection_type = TSNE(n_components=n_components, perplexity=perplexity)

        X = projection_type.fit_transform(X)

        if labels is not None:
            return X, labels, projection_type

        return X, projection_type

    def random_projection(
        self, n_components: int = 2, kernel: str = "linear"
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """

        Returns a training set generated from the 2D original data (time x features) and a random projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons.

        Args:
            n_components (int): Number of components to project to. Default is 2.
            kernel (str): Kernel to be used for projections. Defaults to linear.

        Returns:
            tuple: Tuple containing projected data and projection type.

        """

        return self._projection("random", n_components=n_components, kernel=kernel)

    def pca(
        self, n_components: int = 2, kernel: str = "linear"
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """

        Returns a training set generated from the 2D original data (time x features) and a PCA projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons.

        Args:
            n_components (int): Number of components to project to. Default is 2.
            kernel (str): Kernel to be used for projections. Defaults to linear.

        Returns:
            tuple: Tuple containing projected data and projection type.

        """

        return self._projection("pca", n_components=n_components, kernel=kernel)

    def tsne(
        self, n_components: int = 2, perplexity: int = 30
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """

        Returns a training set generated from the 2D original data (time x features) and a PCA projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons.

        Args:
            n_components (int): Number of components to project to. Default is 2.
            perplexity (int): Perplexity parameter for the t-SNE algorithm. Default is 30.

        Returns:
            tuple: Tuple containing projected data and projection type.

        """

        return self._projection(
            "tsne", n_components=n_components, perplexity=perplexity
        )

    def filter_id(self, selected_id: str = None) -> table_dict:
        """

        Filters a TableDict object to keep only those columns related to the selected id.

        Args:
            selected_id (str): select a single animal on multi animal settings. Defaults to None
            (all animals are processed).

        Returns:
            table_dict: Filtered TableDict object, keeping only the selected animal.

        """

        tabs = self.copy()
        for key, val in tabs.items():
            columns_to_keep = deepof.utils.filter_columns(val.columns, selected_id)
            tabs[key] = val.loc[
                :,
                [bpa for bpa in val.columns if bpa in columns_to_keep],
            ]

        return TableDict(tabs, typ=self._type)

    def merge(self, *args, ignore_index=False):
        """

        Takes a number of table_dict objects and merges them to the current one.
        Returns a table_dict object of type 'merged'.

        Args:
            *args (table_dict): table_dict objects to be merged.
            ignore_index (bool): ignore index when merging. Defaults to False.


        Returns:
            table_dict: Merged table_dict object.

        """
        args = [self.copy()] + list(args)
        merged_dict = defaultdict(list)
        for tabdict in args:
            for key, val in tabdict.items():
                merged_dict[key].append(val)

        merged_tables = TableDict(
            {
                key: pd.concat(val, axis=1, ignore_index=ignore_index, join="inner")
                for key, val in merged_dict.items()
            },
            typ="merged",
        )

        return merged_tables

    def get_training_set(
        self,
        current_table_dict: table_dict,
        test_videos: int = 0,
    ) -> tuple:
        """

        Generates training and test sets as numpy.array objects for model training. Intended for internal usage only.


        Args:
            current_table_dict (table_dict): table_dict object containing the data to be used for training.
            test_videos (int): Number of videos to be used for testing. Defaults to 0.

        Returns:
            tuple: Tuple containing training data, training labels (if any), test data, and test labels (if any).

        """

        raw_data = current_table_dict.values()

        # Padding of videos with slightly different lengths
        # Making sure that the training and test sets end up balanced
        test_index = np.array([], dtype=int)
        raw_data = np.array([np.array(v) for v in raw_data], dtype=object)
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
            except ValueError:
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

        if self._propagate_annotations:
            n_annot = list(self._propagate_annotations.values())[0].shape[1]

            try:
                X_train, y_train = X_train[:, :-n_annot], np.concatenate(
                    [y_train, X_train[:, -n_annot:]], axis=1
                )
            except ValueError:
                X_train, y_train = X_train[:, :-n_annot], X_train[:, -n_annot:]

            # Convert speed to a boolean value. Is the animal moving?
            y_train[:, -1] = (
                y_train[:, -1] > deepof.pose_utils.get_hparameters()["huddle_speed"]
            )

            try:
                try:
                    X_test, y_test = X_test[:, :-n_annot], np.concatenate(
                        [y_test, X_test[:, -n_annot:]]
                    )
                except ValueError:
                    X_test, y_test = X_test[:, :-n_annot], X_test[:, -n_annot:]

                # Convert speed to a boolean value. Is the animal moving?
                y_test[:, -1] = (
                    y_test[:, -1] > deepof.pose_utils.get_hparameters()["huddle_speed"]
                )

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
        automatic_changepoints="rbf",
        window_size: int = 15,
        window_step: int = 1,
        scale: str = "standard",
        test_videos: int = 0,
        verbose: int = 0,
        conv_filter: bool = None,
        sigma: float = 1.0,
        shift: float = 0.0,
        shuffle: bool = False,
    ) -> np.ndarray:
        """

        Main method for preprocessing the loaded dataset before feeding to unsupervised embedding models.
        Capable of returning training and test sets ready for model training.

        Args:
            automatic_changepoints (str): specifies the changepoint detection kernel to use to rupture the
            data across time using Pelt. Can be set to "rbf" (default), or "linear". If False, fixed-length ruptures are
            appiled.
            window_size (int): Minimum size of the applied ruptures. If automatic_changepoints is False,
            specifies the size of the sliding window to pass through the data to generate training instances.
            window_step (int): Specifies the minimum jump for the rupture algorithms. If automatic_changepoints is False,
            specifies the step to take when sliding the aforementioned window. In this case, a value of 1 indicates
            a true sliding window, and a value equal to to window_size splits the data into non-overlapping chunks.
            scale (str): Data scaling method. Must be one of 'standard' (default; recommended) and 'minmax'.
            test_videos (int): Number of videos to use for testing. If 0, no test set is generated.
            verbose (int): Verbosity level. 0 (default) is silent, 1 prints progress, 2 prints debug information.
            conv_filter (bool): Whether to apply a convolutional filter to the data.
            sigma (float): Sigma parameter for the Gaussian kernel used in the convolutional filter.
            shift (float): Shift parameter for the Gaussian kernel used in the convolutional filter.
            shuffle (bool): Whether to shuffle the data before preprocessing. Defaults to False.

        Returns:
            X_train (np.ndarray): 3D dataset with shape (instances, sliding_window_size, features)
            generated from all training videos.
            y_train (np.ndarray): 3D dataset with shape (instances, sliding_window_size, labels)
            generated from all training videos. Note that no labels are use by default in the fully
            unsupervised pipeline (in which case this is an empty array).
            X_test (np.ndarray): 3D dataset with shape (instances, sliding_window_size, features)
            generated from all test videos (0 by default).
            y_test (np.ndarray): 3D dataset with shape (instances, sliding_window_size, labels)
            generated from all test videos. Note that no labels are use by default in the fully
            unsupervised pipeline (in which case this is an empty array).

        """

        # Create a temporary copy of the current TableDict object,
        # to avoid modifying it in place
        table_temp = copy.deepcopy(self)

        if scale:
            if verbose:
                print("Scaling data...")

            # Scale each experiment independently
            for key, tab in table_temp.items():
                if scale == "standard":
                    current_scaler = StandardScaler()
                elif scale == "minmax":
                    current_scaler = MinMaxScaler()
                else:
                    raise ValueError(
                        "Invalid scaler. Select one of standard, minmax or None"
                    )  # pragma: no cover

                exp_temp = tab.to_numpy()
                if self._propagate_labels:
                    exp_temp = exp_temp[:, :-1]

                if self._propagate_annotations:
                    exp_temp = exp_temp[
                        :, : -list(self._propagate_annotations.values())[0].shape[1]
                    ]

                exp_flat = exp_temp.reshape(-1, exp_temp.shape[-1])
                exp_flat = current_scaler.fit_transform(exp_flat)

                if scale == "standard":
                    assert np.all(np.nan_to_num(np.mean(exp_flat), nan=0) < 0.01)
                    assert np.all(np.nan_to_num(np.std(exp_flat), nan=1) > 0.99)

                current_tab = np.concatenate(
                    [
                        exp_flat.reshape(exp_temp.shape),
                        tab.copy().to_numpy()[:, exp_temp.shape[1] :],
                    ],
                    axis=1,
                )

                table_temp[key] = pd.DataFrame(
                    current_tab,
                    columns=tab.columns,
                    index=tab.index,
                )

        # Split videos and generate training and test sets
        X_train, y_train, X_test, y_test, test_index = self.get_training_set(
            table_temp,
            test_videos,
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
                )
            else:
                y_train = deepof.utils.split_with_breakpoints(y_train, train_breaks)
            y_train = y_train.mean(axis=1)

        if conv_filter == "gaussian" and not automatic_changepoints:
            r = range(-int(window_size / 2), int(window_size / 2) + 1)
            r = [i - shift for i in r]
            g = np.array(
                [
                    1
                    / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-float(x) ** 2 / (2 * sigma ** 2))
                    for x in r
                ]
            )
            g /= np.max(g)
            X_train = X_train * g.reshape([1, window_size, 1])

        if test_videos and len(test_index) > 0:

            # Apply rupture method to each test experiment independently
            X_test, test_breaks = deepof.utils.rupture_per_experiment(
                table_dict=table_temp,
                to_rupture=X_test,
                rupture_indices=test_index,
                automatic_changepoints=automatic_changepoints,
                window_size=window_size,
                window_step=window_step,
            )

            if self._propagate_labels or self._propagate_annotations:
                if test_breaks is None:
                    y_test, _ = deepof.utils.rupture_per_experiment(
                        table_dict=table_temp,
                        to_rupture=y_test,
                        rupture_indices=[
                            i for i in range(len(table_temp)) if i not in test_index
                        ],
                        automatic_changepoints=False,
                        window_size=window_size,
                        window_step=window_step,
                    )
                else:
                    y_test = deepof.utils.split_with_breakpoints(y_test, test_breaks)
                    y_test = y_test.mean(axis=1)

            if conv_filter == "gaussian" and not automatic_changepoints:
                # noinspection PyUnboundLocalVariable
                X_test = X_test * g.reshape([1, window_size, 1])

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

        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Remove excessive logging from tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# TODO: Fix issues and add supervised parameters (time in zone, etc).
# TODO: Label more data for supervised model training
# TODO: Finish visualization pipeline (Projections and time-wise analyses)
# TODO: Add options to compare cohorts in the unsupervised pipeline (cluster population - cluster dinamycs)
