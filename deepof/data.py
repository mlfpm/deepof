# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Data structures for preprocessing and wrangling of DLC output data.

- project: initial structure for specifying the characteristics of the project.
- coordinates: result of running the project. In charge of calling all relevant
computations for getting the data into the desired shape
- table_dict: python dict subclass for storing experimental instances as pandas.DataFrames.
Contains methods for generating training and test sets ready for model training.

"""

import os
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Union

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

# Remove excessive logging from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# DEFINE CUSTOM ANNOTATED TYPES #
coordinates = deepof.utils.NewType("coordinates", deepof.utils.Any)
table_dict = deepof.utils.NewType("table_dict", deepof.utils.Any)


# CLASSES FOR PREPROCESSING AND DATA WRANGLING


class Project:
    """

    Class for loading and preprocessing DLC data of individual and multiple animals. All main computations are called
    here.

    """

    def __init__(
        self,
        arena_dims: int,
        animal_ids: List = tuple([""]),
        arena: str = "circular",
        arena_detection: str = "rule-based",
        enable_iterative_imputation: bool = None,  # This will impute the position of ocluded body parts,
        # which might not be desirable and it's computationally expensive. As an alternative, models should
        # be resistant to NaN values (ie with masking). Add this explanation to the documentation.
        exclude_bodyparts: List = tuple([""]),
        exp_conditions: dict = None,
        high_fidelity_arena: bool = False,
        interpolate_outliers: bool = True,
        interpolation_limit: int = 5,
        interpolation_std: int = 5,
        likelihood_tol: float = 0.5,
        model: str = "mouse_topview",
        path: str = deepof.utils.os.path.join("."),
        smooth_alpha: float = 0,
        table_format: str = "autodetect",
        frame_rate: int = None,
        video_format: str = ".mp4",
    ):

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
        self.subset_condition = None
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
        if len(self.animal_ids) > 1 and len(self.exclude_bodyparts) > 1:
            self.exclude_bodyparts = [
                aid + "_" + bp for aid in self.animal_ids for bp in exclude_bodyparts
            ]

        if self.exclude_bodyparts != tuple([""]):
            for aid in self.animal_ids:
                for bp in self.exclude_bodyparts:
                    if bp.startswith(aid):
                        self.connectivity[aid].remove_node(bp)

    def __str__(self):
        if self.exp_conditions:
            return "deepof analysis of {} videos across {} conditions".format(
                len(self.videos), len(set(self.exp_conditions.values()))
            )
        return "deepof analysis of {} videos".format(len(self.videos))

    @property
    def subset_condition(self):
        """Sets a subset condition for the videos to load. If set,
        only the videos with the included pattern will be loaded"""
        return self._subset_condition

    @property
    def distances(self):
        """List. If not 'all', sets the body parts among which the
        distances will be computed"""
        return self._distances

    @property
    def ego(self):
        """String, name of a body part. If True, computes only the distances
        between the specified body part and the rest"""
        return self._ego

    @property
    def angles(self):
        """Bool. Toggles angle computation. True by default. If turned off,
        enhances performance for big datasets"""
        return self._angles

    def get_arena(self, tables) -> np.array:
        """Returns the arena as recognised from the videos"""

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
        """Loads videos and tables into dictionaries"""

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
                )

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
                cols = tab.columns
                smooth = pd.DataFrame(
                    deepof.utils.smooth_mult_trajectory(
                        np.array(tab), alpha=self.smooth_alpha
                    )
                )
                smooth.columns = cols
                tab_dict[key] = smooth.reset_index(drop=True)

        for key, tab in tab_dict.items():
            tab_dict[key] = tab.loc[:, tab.columns.levels[0][0]]

        if self.subset_condition:
            for key, value in tab_dict.items():
                lablist = [
                    b
                    for b in value.columns.levels[0]
                    if not b.startswith(self.subset_condition)
                ]

                tabcols = value.drop(
                    lablist, axis=1, level=0
                ).T.index.remove_unused_levels()

                tab = value.loc[
                    :, [i for i in value.columns.levels[0] if i not in lablist]
                ]

                tab.columns = tabcols

                tab_dict[key] = tab

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
                    max_iter=5, skip_complete=True
                ).fit_transform(value)
                tab_dict[k] = pd.DataFrame(
                    imputed, index=value.index, columns=value.columns
                )

        return tab_dict, lik_dict

    def get_distances(self, tab_dict: dict, verbose: bool = False) -> dict:
        """Computes the distances between all selected body parts over time.
        If ego is provided, it only returns distances to a specified bodypart"""

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
        Parameters (from self):
            connectivity (dictionary): dict stating to which bodyparts each bodypart is connected;
            table_dict (dict of dataframes): tables loaded from the data;

        Output:
            angle_dict (dictionary): dict containing angle dataframes per vido

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
        """Generates a dataset using all the options specified during initialization"""

        tables, quality = self.load_tables(verbose)
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
            videos=self.videos,
            video_resolution=self.video_resolution,
        )

    @subset_condition.setter
    def subset_condition(self, value):
        self._subset_condition = value

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
    the generated tables. For internal usage only.

    """

    def __init__(
        self,
        arena: str,
        arena_detection: str,
        arena_dims: np.array,
        path: str,
        quality: dict,
        scales: np.array,
        arena_params: List,
        tables: dict,
        videos: List,
        video_resolution: List,
        angles: dict = None,
        animal_ids: List = tuple([""]),
        distances: dict = None,
        exp_conditions: dict = None,
    ):
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
        self._videos = videos
        self._video_resolution = video_resolution
        self.angles = angles
        self.distances = distances

    def __str__(self):
        if self._exp_conditions:
            return "Coordinates of {} videos across {} conditions".format(
                len(self._videos), len(set(self._exp_conditions.values()))
            )
        return "deepof analysis of {} videos".format(len(self._videos))

    def get_coords(
        self,
        center: str = "arena",
        polar: bool = False,
        speed: int = 0,
        align: bool = False,
        align_inplace: bool = True,
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """
        Returns a table_dict object with the coordinates of each animal as values.

            Parameters:
                - center (str): name of the body part to which the positions will be centered.
                If false, the raw data is returned; if 'arena' (default), coordinates are
                centered in the pitch
                - polar (bool): states whether the coordinates should be converted to polar values
                - speed (int): states the derivative of the positions to report. Speed is returned if 1,
                acceleration if 2, jerk if 3, etc.
                - align (bool): selects the body part to which later processes will align the frames with
                (see preprocess in table_dict documentation).
                - align_inplace (bool): Only valid if align is set. Aligns the vector that goes from the origin to
                the selected body part with the y axis, for all time points.
                - propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label
                - propagate_annotations (Dict): if a dictionary is provided, rule based annotations
                are propagated through the training dataset. This can be used for regularising the latent space based
                on already known traits.

            Returns:
                tab_dict (Table_dict): table_dict object containing all the computed information
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
                for aid in self._animal_ids:

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

        if align:

            assert np.any(
                align in bp for bp in list(tabs.values())[0].columns.levels[0]
            ), "align must be set to the name of a bodypart"

            aligned_full = None
            for key, tab in tabs.items():
                for aid in self._animal_ids:
                    # Bring forward the column to align
                    columns = [
                        i
                        for i in tab.columns
                        if align not in i and i[0].startswith(aid)
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

                    aligned_partial = tab[columns]
                    if align_inplace and polar is False:
                        columns = aligned_partial.columns
                        index = aligned_partial.index
                        aligned_partial = pd.DataFrame(
                            deepof.utils.align_trajectories(
                                np.array(aligned_partial), mode="all"
                            )
                        )
                        aligned_partial.columns = columns
                        aligned_partial.index = index

                    aligned_full = pd.concat([aligned_full, aligned_partial], axis=1)

                tabs[key] = aligned_full

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
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """
        Returns a table_dict object with the distances between body parts animal as values.

            Parameters:
                - speed (int): states the derivative of the positions to report. Speed is returned if 1,
                acceleration if 2, jerk if 3, etc.
                - propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label
                - propagate_annotations (Dict): if a dictionary is provided, rule based annotations
                are propagated through the training dataset. This can be used for regularising the latent space based
                on already known traits.

            Returns:
                tab_dict (Table_dict): table_dict object containing all the computed information
        """

        tabs = deepof.utils.deepcopy(self.distances)

        if self.distances is not None:

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="dists")
                    tabs[key] = vel

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
        propagate_labels: bool = False,
        propagate_annotations: Dict = False,
    ) -> table_dict:
        """
        Returns a table_dict object with the angles between body parts animal as values.

            Parameters:
                - angles (bool): if True, returns the angles in degrees. Radians (default) are returned otherwise.
                - speed (int): states the derivative of the positions to report. Speed is returned if 1,
                acceleration if 2, jerk if 3, etc.
                - propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label
                - propagate_annotations (Dict): if a dictionary is provided, rule based annotations
                are propagated through the training dataset. This can be used for regularising the latent space based
                on already known traits.

            Returns:
                tab_dict (Table_dict): table_dict object containing all the computed information
        """

        tabs = deepof.utils.deepcopy(self.angles)

        if self.angles is not None:
            if degrees:
                tabs = {key: np.degrees(tab) for key, tab in tabs.items()}

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="angles")
                    tabs[key] = vel

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
        """Retuens the videos associated with the dataset as a list."""

        if play:  # pragma: no cover
            raise NotImplementedError

        return self._videos

    @property
    def get_exp_conditions(self):
        """Returns the stored dictionary with experimental conditions per subject"""

        return self._exp_conditions

    def get_quality(self):
        """Retrieves a dictionary with the tagging quality per video, as reported by DLC"""

        return self._quality

    @property
    def get_arenas(self):
        """Retrieves all available information associated with the arena"""

        return self._arena, [self._arena_dims], self._scales

    # noinspection PyDefaultArgument
    def rule_based_annotation(
        self,
        params: Dict = {},
        video_output: bool = False,
        frame_limit: int = np.inf,
        debug: bool = False,
        n_jobs: int = 1,
        propagate_labels: bool = False,
    ) -> table_dict:
        """Annotates coordinates using a simple rule-based pipeline"""

        tag_dict = {}
        coords = self.get_coords(center=False)
        dists = self.get_distances()
        speeds = self.get_coords(speed=1)

        # noinspection PyTypeChecker
        for key in tqdm(self._tables.keys()):
            tag_dict[key] = deepof.pose_utils.rule_based_tagging(
                self,
                coords=coords,
                dists=dists,
                speeds=speeds,
                video=[vid for vid in self._videos if key + "DLC" in vid][0],
                params=params,
            )

        if propagate_labels:
            for key, tab in tag_dict.items():
                tab["pheno"] = self._exp_conditions[key]

        if video_output:  # pragma: no cover

            def output_video(idx):
                """Outputs a single annotated video. Enclosed in a function to enable parallelization"""

                deepof.pose_utils.rule_based_video(
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
            typ="rule-based",
            arena=self._arena,
            arena_dims=self._arena_dims,
            propagate_labels=propagate_labels,
        )

    @staticmethod
    def deep_unsupervised_embedding(
        preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
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
        n_components: int = 25,
        overlap_loss: float = 0,
        output_path: str = ".",
        next_sequence_prediction: float = 0,
        phenotype_prediction: float = 0,
        rule_based_prediction: float = 0,
        pretrained: str = False,
        save_checkpoints: bool = False,
        save_weights: bool = True,
        reg_cat_clusters: bool = False,
        reg_cluster_variance: bool = False,
        entropy_knn: int = 100,
        input_type: str = False,
        run: int = 0,
        strategy: tf.distribute.Strategy = tf.distribute.MirroredStrategy(),
    ) -> Tuple:
        """
        Annotates coordinates using an unsupervised autoencoder.
        Full implementation in deepof.train_utils.deep_unsupervised_embedding

        Parameters:
            - preprocessed_object (Tuple[np.ndarray]): tuple containing a preprocessed object (X_train,
            y_train, X_test, y_test)
            - encoding_size (int): number of dimensions in the latent space of the autoencoder
            - epochs (int): epochs during which to train the models
            - batch_size (int): training batch size
            - save_checkpoints (bool): if True, training checkpoints are saved to disk. Useful for debugging,
            but can make training significantly slower
            - hparams (dict): dictionary to change architecture hyperparameters of the autoencoders
            (see documentation for details)
            - kl_warmup (int): number of epochs over which to increase KL weight linearly
            (default is number of epochs // 4)
            - loss (str): Loss function to use. Currently, 'ELBO', 'MMD' and 'ELBO+MMD' are supported.
            - mmd_warmup (int): number of epochs over which to increase MMD weight linearly
            (default is number of epochs // 4)
            - montecarlo_kl (int): Number of Montecarlo samples used to estimate the KL between latent space and prior
            - n_components (int): Number of components of the Gaussian Mixture in the latent space
            - outpath (str): Path where to save the training loggings
            - phenotype_class (float): weight assigned to phenotype classification. If > 0,
            a classification neural network is appended to the latent space,
            aiming to enforce structure from a set of labels in the encoding.
            - predictor (float): weight assigned to a predictor branch. If > 0, a regression neural network
            is appended to the latent space,
            aiming to predict what happens immediately next in the sequence, which can help with regularization.
            - pretrained (bool): If True, a pretrained set of weights is expected.

        Returns:
            - return_list (tuple): List containing all relevant trained models for unsupervised prediction.

        """

        trained_models = deepof.train_utils.autoencoder_fitting(
            preprocessed_object=preprocessed_object,
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
            rule_based_prediction=rule_based_prediction,
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
        super().__init__(tabs)
        self._type = typ
        self._center = center
        self._polar = polar
        self._arena = arena
        self._arena_dims = arena_dims
        self._propagate_labels = propagate_labels
        self._propagate_annotations = propagate_annotations
        self._scaler = None

    def filter_videos(self, keys: list) -> table_dict:
        """Returns a subset of the original table_dict object, containing only the specified keys. Useful, for example,
        for selecting data coming from videos of a specified condition."""

        assert np.all([k in self.keys() for k in keys]), "Invalid keys selected"

        return TableDict(
            {k: value for k, value in self.items() if k in keys}, self._type
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
        """Plots heatmaps of the specified body parts (bodyparts) of the specified animal (i)"""

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

    def get_training_set(
        self,
        test_videos: int = 0,
        encode_labels: bool = True,
    ) -> Tuple[np.ndarray, list, Union[np.ndarray, list], list]:
        """Generates training and test sets as numpy.array objects for model training"""

        # Padding of videos with slightly different lengths
        # Making sure that the training and test sets end up balanced
        raw_data = np.array([np.array(v) for v in self.values()], dtype=object)
        if self._propagate_labels:
            concat_raw = np.concatenate(raw_data, axis=0)
            test_index = np.array([], dtype=int)
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
            X_test = np.concatenate(raw_data[test_index])
            X_train = np.concatenate(np.delete(raw_data, test_index, axis=0))

        else:
            X_train = np.concatenate(list(raw_data))

        if self._propagate_labels:
            X_train, y_train = X_train[:, :-1], X_train[:, -1][:, np.newaxis]
            try:
                X_test, y_test = X_test[:, :-1], X_test[:, -1][:, np.newaxis]
            except IndexError:
                pass

        if self._propagate_annotations:
            n_annot = list(self._propagate_annotations.values())[0].shape[1]

            try:
                X_train, y_train = X_train[:, :-n_annot], np.concatenate(
                    [y_train, X_train[:, -n_annot:]]
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

        if self._propagate_labels and encode_labels:
            le = LabelEncoder()
            y_train[:, 0] = le.fit_transform(y_train[:, 0])
            try:
                y_test[:, 0] = le.transform(y_test[:, 0])
            except IndexError:
                pass

        return (
            X_train.astype(float),
            y_train.astype(float),
            X_test.astype(float),
            y_test.astype(float),
        )

    # noinspection PyTypeChecker,PyGlobalUndefined
    def preprocess(
        self,
        window_size: int = 1,
        window_step: int = 1,
        scale: str = "standard",
        test_videos: int = 0,
        verbose: bool = False,
        conv_filter: bool = None,
        sigma: float = 1.0,
        shift: float = 0.0,
        shuffle: bool = False,
        align: str = False,
    ) -> np.ndarray:
        """

        Main method for preprocessing the loaded dataset. Capable of returning training
        and test sets ready for model training.

            Parameters:
                - window_size (int): Size of the sliding window to pass through the data to generate training instances
                - window_step (int): Step to take when sliding the window. If 1, a true sliding window is used;
                if equal to window_size, the data is split into non-overlapping chunks.
                - scale (str): Data scaling method. Must be one of 'standard' (default; recommended) and 'minmax'.
                - test_videos (int): Number of videos to use when generating the test set.
                If 0, no test set is generated (not recommended).
                - verbose (bool): prints job information if True
                - conv_filter (bool): must be one of None, 'gaussian'. If not None, convolves each instance
                with the specified kernel.
                - sigma (float): usable only if conv_filter is 'gaussian'. Standard deviation of the kernel to use.
                - shift (float): usable only if conv_filter is 'gaussian'. Shift from mean zero of the kernel to use.
                - shuffle (bool): Shuffles the data instances if True. In most use cases, it should be True for training
                and False for prediction.
                - align (bool): If "all", rotates all data instances to align the center -> align (selected before
                when calling get_coords) axis with the y-axis of the cartesian plane. If 'center', rotates all instances
                using the angle of the central frame of the sliding window. This way rotations of the animal are caught
                as well. It doesn't do anything if False.

            Returns:
                - X_train (np.ndarray): 3d dataset with shape (instances, sliding_window_size, features)
                generated from all training videos
                - X_test (np.ndarray): 3d dataset with shape (instances, sliding_window_size, features)
                generated from all test videos (if test_videos > 0)
                - y_train (np.ndarray): 2d dataset with a shape dependent in the type of labels the model uses
                (phenotypes, rule-based tags).
                - y_test (np.ndarray): 2d dataset with a shape dependent in the type of labels the model uses
                (phenotypes, rule-based tags).

        """

        X_train, y_train, X_test, y_test = self.get_training_set(test_videos)

        if scale:
            if verbose:
                print("Scaling data...")

            if scale == "standard":
                self._scaler = StandardScaler()

            elif scale == "minmax":
                self._scaler = MinMaxScaler()
            else:
                raise ValueError(
                    "Invalid scaler. Select one of standard, minmax or None"
                )  # pragma: no cover

            X_train_flat = X_train.reshape(-1, X_train.shape[-1])

            self._scaler.fit(X_train_flat)

            X_train = self._scaler.transform(X_train_flat).reshape(X_train.shape)

            if scale == "standard":
                assert np.all(np.nan_to_num(np.mean(X_train), nan=0) < 0.1)
                assert np.all(np.nan_to_num(np.std(X_train), nan=1) > 0.9)

            if test_videos:
                X_test = self._scaler.transform(
                    X_test.reshape(-1, X_test.shape[-1])
                ).reshape(X_test.shape)

            if verbose:
                print("Done!")

        if align == "all":
            X_train = deepof.utils.align_trajectories(X_train, align)

        X_train = deepof.utils.rolling_window(X_train, window_size, window_step)
        if self._propagate_labels or self._propagate_annotations:
            y_train = deepof.utils.rolling_window(y_train, window_size, window_step)
            y_train = y_train.mean(axis=1)

        if align == "center":
            X_train = deepof.utils.align_trajectories(X_train, align)

        if conv_filter == "gaussian":
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

        if test_videos:

            if align == "all":
                X_test = deepof.utils.align_trajectories(X_test, align)

            X_test = deepof.utils.rolling_window(X_test, window_size, window_step)

            if self._propagate_labels or self._propagate_annotations:
                y_test = deepof.utils.rolling_window(y_test, window_size, window_step)
                y_test = y_test.mean(axis=1)

            if align == "center":
                X_test = deepof.utils.align_trajectories(X_test, align)

            if conv_filter == "gaussian":
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

        return X_train, y_train, np.array(X_test), np.array(y_test)

    def _prepare_projection(self) -> np.ndarray:
        """Returns a numpy ndarray from the preprocessing of the table_dict object,
        ready for projection into a lower dimensional space"""

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

    def _project(
        self,
        proj,
        n_components: int = 2,
        kernel: str = None,
        perplexity: int = None,
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a specified projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        X, labels = self._prepare_projection()

        if proj == "random":
            proj = random_projection.GaussianRandomProjection(n_components=n_components)
        elif proj == "pca":
            proj = KernelPCA(n_components=n_components, kernel=kernel)
        elif proj == "tsne":
            proj = TSNE(n_components=n_components, perplexity=perplexity)

        X = proj.fit_transform(X)

        if labels is not None:
            return X, labels, proj

        return X, proj

    def random_projection(
        self, n_components: int = 2, kernel: str = "linear"
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a random projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        return self._project("random", n_components=n_components, kernel=kernel)

    def pca(
        self, n_components: int = 2, sample: int = 1000, kernel: str = "linear"
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a PCA projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        return self._project("pca", n_components=n_components, kernel=kernel)

    def tsne(
        self, n_components: int = 2, sample: int = 1000, perplexity: int = 30
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a PCA projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        return self._project("tsne", n_components=n_components, perplexity=perplexity)


def merge_tables(*args):
    """

    Takes a number of table_dict objects and merges them
    Returns a table_dict object of type 'merged'

    """
    merged_dict = {key: [] for key in args[0].keys()}
    for tabdict in args:
        for key, val in tabdict.items():
            merged_dict[key].append(val)

    merged_tables = TableDict(
        {
            key: pd.concat(val, axis=1, ignore_index=True)
            for key, val in merged_dict.items()
        },
        typ="merged",
    )

    return merged_tables


# TODO:
#   Add __str__ method for all three major classes!
#   Explore preprocessing in ragged (masked) tensors using change point detection!
#   While some operations (mainly alignment) should be carried out before merging, others require
#   the whole dataset to function properly.
#   - For now, propagate_annotations is optional and requires the user to actively pass a data frame with traits.
#   - If this gives good results, we'll make it default or give a boolean option that requires less effort
