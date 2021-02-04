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

from collections import defaultdict
from joblib import delayed, Parallel, parallel_backend
from typing import Dict, List, Tuple, Union
from multiprocessing import cpu_count
from sklearn import random_projection
from sklearn.decomposition import KernelPCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from tqdm import tqdm
import deepof.pose_utils
import deepof.utils
import deepof.visuals
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import warnings

# DEFINE CUSTOM ANNOTATED TYPES #

Coordinates = deepof.utils.NewType("Coordinates", deepof.utils.Any)
Table_dict = deepof.utils.NewType("Table_dict", deepof.utils.Any)

# CLASSES FOR PREPROCESSING AND DATA WRANGLING


class project:
    """

    Class for loading and preprocessing DLC data of individual and multiple animals. All main computations are called
    here.

    """

    def __init__(
        self,
        animal_ids: List = tuple([""]),
        arena: str = "circular",
        arena_dims: tuple = (1,),
        exclude_bodyparts: List = tuple([""]),
        exp_conditions: dict = None,
        interpolate_outliers: bool = True,
        interpolation_limit: int = 5,
        interpolation_std: int = 5,
        likelihood_tol: float = 0.75,
        model: str = "mouse_topview",
        path: str = deepof.utils.os.path.join("."),
        smooth_alpha: float = 0.99,
        table_format: str = "autodetect",
        video_format: str = ".mp4",
    ):

        self.path = path
        self.video_path = os.path.join(self.path, "Videos")
        self.table_path = os.path.join(self.path, "Tables")

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
        self.angles = True
        self.animal_ids = animal_ids
        self.arena = arena
        self.arena_dims = arena_dims
        self.distances = "all"
        self.ego = False
        self.exp_conditions = exp_conditions
        self.interpolate_outliers = interpolate_outliers
        self.interpolation_limit = interpolation_limit
        self.interpolation_std = interpolation_std
        self.likelihood_tolerance = likelihood_tol
        self.scales = self.get_scale
        self.smooth_alpha = smooth_alpha
        self.subset_condition = None
        self.video_format = video_format

        model_dict = {
            "mouse_topview": deepof.utils.connect_mouse_topview(animal_ids[0])
        }
        self.connectivity = model_dict[model]
        self.exclude_bodyparts = exclude_bodyparts
        if self.exclude_bodyparts != tuple([""]):
            for bp in exclude_bodyparts:
                self.connectivity.remove_node(bp)

    def __str__(self):
        if self.exp_conditions:
            return "deepof analysis of {} videos across {} conditions".format(
                len(self.videos), len(set(self.exp_conditions.values()))
            )
        else:
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

    @property
    def get_scale(self) -> np.array:
        """Returns the arena as recognised from the videos"""

        if self.arena in ["circular"]:

            scales = []
            for vid_index, _ in enumerate(self.videos):

                ellipse = deepof.utils.recognize_arena(
                    self.videos,
                    vid_index,
                    path=self.video_path,
                    arena_type=self.arena,
                )[0]

                scales.append(
                    list(np.array([ellipse[0][0], ellipse[0][1], ellipse[1][1]]) * 2)
                    + list(self.arena_dims)
                )

        else:
            raise NotImplementedError("arenas must be set to one of: 'circular'")

        return np.array(scales)

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
                tab_dict[key] = smooth.iloc[1:, :].reset_index(drop=True)

        for key, tab in tab_dict.items():
            tab_dict[key] = tab[tab.columns.levels[0][0]]

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

        cliques = deepof.utils.nx.enumerate_all_cliques(self.connectivity)
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

        return angle_dict

    def run(self, verbose: bool = True) -> Coordinates:
        """Generates a dataset using all the options specified during initialization"""

        tables, quality = self.load_tables(verbose)
        distances = None
        angles = None

        if self.distances:
            distances = self.get_distances(tables, verbose)

        if self.angles:
            angles = self.get_angles(tables, verbose)

        if verbose:
            print("Done!")

        return coordinates(
            angles=angles,
            animal_ids=self.animal_ids,
            arena=self.arena,
            arena_dims=self.arena_dims,
            distances=distances,
            exp_conditions=self.exp_conditions,
            path=self.path,
            quality=quality,
            scales=self.scales,
            tables=tables,
            videos=self.videos,
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


class coordinates:
    """

    Class for storing the results of a ran project. Methods are mostly setters and getters in charge of tidying up
    the generated tables. For internal usage only.

    """

    def __init__(
        self,
        arena: str,
        arena_dims: np.array,
        path: str,
        quality: dict,
        scales: np.array,
        tables: dict,
        videos: list,
        angles: dict = None,
        animal_ids: List = tuple([""]),
        distances: dict = None,
        exp_conditions: dict = None,
    ):
        self._animal_ids = animal_ids
        self._arena = arena
        self._arena_dims = arena_dims
        self._exp_conditions = exp_conditions
        self._path = path
        self._quality = quality
        self._scales = scales
        self._tables = tables
        self._videos = videos
        self.angles = angles
        self.distances = distances

    def __str__(self):
        if self._exp_conditions:
            return "Coordinates of {} videos across {} conditions".format(
                len(self._videos), len(set(self._exp_conditions.values()))
            )
        else:
            return "deepof analysis of {} videos".format(len(self._videos))

    def get_coords(
        self,
        center: str = "arena",
        polar: bool = False,
        speed: int = 0,
        length: str = None,
        align: bool = False,
        align_inplace: bool = False,
        propagate_labels: bool = False,
    ) -> Table_dict:
        """
        Returns a table_dict object with the coordinates of each animal as values.

            Parameters:
                - center (str): name of the body part to which the positions will be centered.
                If false, the raw data is returned; if 'arena' (default), coordinates are
                centered in the pitch
                - polar (bool): states whether the coordinates should be converted to polar values
                - speed (int): states the derivative of the positions to report. Speed is returned if 1,
                acceleration if 2, jerk if 3, etc.
                - length (str): length of the video in a datetime compatible format (hh::mm:ss). If stated, the index
                of the stored dataframes will reflect the actual timing in the video.
                - align (bool): selects the body part to which later processes will align the frames with
                (see preprocess in table_dict documentation).
                - align_inplace (bool): Only valid if align is set. Aligns the vector that goes from the origin to
                the selected body part with the y axis, for all time points.
                - propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label

            Returns:
                tab_dict (Table_dict): table_dict object containing all the computed information
        """

        tabs = deepof.utils.deepcopy(self._tables)

        if polar:
            for key, tab in tabs.items():
                tabs[key] = deepof.utils.tab2polar(tab)

        if center == "arena":
            if self._arena == "circular":

                for i, (key, value) in enumerate(tabs.items()):

                    try:
                        value.loc[:, (slice("coords"), ["x"])] = (
                            value.loc[:, (slice("coords"), ["x"])]
                            - self._scales[i][0] / 2
                        )

                        value.loc[:, (slice("coords"), ["y"])] = (
                            value.loc[:, (slice("coords"), ["y"])]
                            - self._scales[i][1] / 2
                        )
                    except KeyError:
                        value.loc[:, (slice("coords"), ["rho"])] = (
                            value.loc[:, (slice("coords"), ["rho"])]
                            - self._scales[i][0] / 2
                        )

                        value.loc[:, (slice("coords"), ["phi"])] = (
                            value.loc[:, (slice("coords"), ["phi"])]
                            - self._scales[i][1] / 2
                        )

        elif type(center) == str and center != "arena":

            for i, (key, value) in enumerate(tabs.items()):

                try:
                    value.loc[:, (slice("coords"), ["x"])] = value.loc[
                        :, (slice("coords"), ["x"])
                    ].subtract(value[center]["x"], axis=0)

                    value.loc[:, (slice("coords"), ["y"])] = value.loc[
                        :, (slice("coords"), ["y"])
                    ].subtract(value[center]["y"], axis=0)
                except KeyError:
                    value.loc[:, (slice("coords"), ["rho"])] = value.loc[
                        :, (slice("coords"), ["rho"])
                    ].subtract(value[center]["rho"], axis=0)

                    value.loc[:, (slice("coords"), ["phi"])] = value.loc[
                        :, (slice("coords"), ["phi"])
                    ].subtract(value[center]["phi"], axis=0)

                tabs[key] = value.loc[
                    :, [tab for tab in value.columns if tab[0] != center]
                ]

        if speed:
            for key, tab in tabs.items():
                vel = deepof.utils.rolling_speed(tab, deriv=speed, center=center)
                tabs[key] = vel

        if length:
            for key, tab in tabs.items():
                tabs[key].index = pd.timedelta_range(
                    "00:00:00", length, periods=tab.shape[0] + 1, closed="left"
                ).astype("timedelta64[s]")

        if align:
            assert (
                align in list(tabs.values())[0].columns.levels[0]
            ), "align must be set to the name of a bodypart"

            for key, tab in tabs.items():
                # Bring forward the column to align
                columns = [i for i in tab.columns if align not in i]
                columns = [
                    (align, ("phi" if polar else "x")),
                    (align, ("rho" if polar else "y")),
                ] + columns
                tab = tab[columns]
                tabs[key] = tab

                if align_inplace and polar is False:
                    index = tab.columns
                    tab = pd.DataFrame(
                        deepof.utils.align_trajectories(np.array(tab), mode="all")
                    )
                    tab.columns = index
                    tabs[key] = tab

        if propagate_labels:
            for key, tab in tabs.items():
                tab["pheno"] = self._exp_conditions[key]

        return table_dict(
            tabs,
            "coords",
            arena=self._arena,
            arena_dims=self._scales,
            center=center,
            polar=polar,
            propagate_labels=propagate_labels,
        )

    def get_distances(
        self, speed: int = 0, length: str = None, propagate_labels: bool = False
    ) -> Table_dict:
        """
        Returns a table_dict object with the distances between body parts animal as values.

            Parameters:
                - speed (int): states the derivative of the positions to report. Speed is returned if 1,
                acceleration if 2, jerk if 3, etc.
                - length (str): length of the video in a datetime compatible format (hh::mm:ss). If stated, the index
                of the stored dataframes will reflect the actual timing in the video.
                - propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label

            Returns:
                tab_dict (Table_dict): table_dict object containing all the computed information
        """

        tabs = deepof.utils.deepcopy(self.distances)

        if self.distances is not None:

            if speed:
                for key, tab in tabs.items():
                    vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="dists")
                    tabs[key] = vel

            if length:
                for key, tab in tabs.items():
                    tabs[key].index = pd.timedelta_range(
                        "00:00:00", length, periods=tab.shape[0] + 1, closed="left"
                    ).astype("timedelta64[s]")

            if propagate_labels:
                for key, tab in tabs.items():
                    tab["pheno"] = self._exp_conditions[key]

            return table_dict(tabs, propagate_labels=propagate_labels, typ="dists")

        raise ValueError(
            "Distances not computed. Read the documentation for more details"
        )  # pragma: no cover

    def get_angles(
        self,
        degrees: bool = False,
        speed: int = 0,
        length: str = None,
        propagate_labels: bool = False,
    ) -> Table_dict:
        """
        Returns a table_dict object with the angles between body parts animal as values.

            Parameters:
                - angles (bool): if True, returns the angles in degrees. Radians (default) are returned otherwise.
                - speed (int): states the derivative of the positions to report. Speed is returned if 1,
                acceleration if 2, jerk if 3, etc.
                - length (str): length of the video in a datetime compatible format (hh::mm:ss). If stated, the index
                of the stored dataframes will reflect the actual timing in the video.
                - propagate_labels (bool): If True, adds an extra feature for each video containing its phenotypic label

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

            if length:
                for key, tab in tabs.items():
                    tabs[key].index = pd.timedelta_range(
                        "00:00:00", length, periods=tab.shape[0] + 1, closed="left"
                    ).astype("timedelta64[s]")

            if propagate_labels:
                for key, tab in tabs.items():
                    tab["pheno"] = self._exp_conditions[key]

            return table_dict(tabs, propagate_labels=propagate_labels, typ="angles")

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

        return self._arena, self._arena_dims, self._scales

    # noinspection PyDefaultArgument
    def rule_based_annotation(
        self,
        params: Dict = {},
        video_output: bool = False,
        frame_limit: int = np.inf,
        debug: bool = False,
    ) -> Table_dict:
        """Annotates coordinates using a simple rule-based pipeline"""

        tag_dict = {}
        # noinspection PyTypeChecker
        coords = self.get_coords(center=False)
        dists = self.get_distances()
        speeds = self.get_coords(speed=1)

        for key in tqdm(self._tables.keys()):

            video = [vid for vid in self._videos if key + "DLC" in vid][0]
            tag_dict[key] = deepof.pose_utils.rule_based_tagging(
                list(self._tables.keys()),
                self._videos,
                self,
                coords,
                dists,
                speeds,
                self._videos.index(video),
                arena_type=self._arena,
                recog_limit=1,
                path=os.path.join(self._path, "Videos"),
                params=params,
            )

        if video_output:  # pragma: no cover

            def output_video(idx):
                """Outputs a single annotated video. Enclosed in a function to enable parallelization"""

                deepof.pose_utils.rule_based_video(
                    self,
                    list(self._tables.keys()),
                    self._videos,
                    list(self._tables.keys()).index(idx),
                    tag_dict[idx],
                    debug=debug,
                    frame_limit=frame_limit,
                    recog_limit=1,
                    path=os.path.join(self._path, "Videos"),
                    params=params,
                )
                pbar.update(1)

            if type(video_output) == list:
                vid_idxs = video_output
            elif video_output == "all":
                vid_idxs = list(self._tables.keys())
            else:
                raise AttributeError(
                    "Video output must be either 'all' or a list with the names of the videos to render"
                )

            njobs = cpu_count() // 2
            pbar = tqdm(total=len(vid_idxs))
            with parallel_backend("threading", n_jobs=njobs):
                Parallel()(delayed(output_video)(key) for key in vid_idxs)
            pbar.close()

        return table_dict(
            tag_dict, typ="rule-based", arena=self._arena, arena_dims=self._arena_dims
        )

    def gmvae_embedding(self):
        pass


class table_dict(dict):
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
    ):
        super().__init__(tabs)
        self._type = typ
        self._center = center
        self._polar = polar
        self._arena = arena
        self._arena_dims = arena_dims
        self._propagate_labels = propagate_labels

    def filter_videos(self, keys: list) -> Table_dict:
        """Returns a subset of the original table_dict object, containing only the specified keys. Useful, for example,
        for selecting data coming from videos of a specified condition."""

        assert np.all([k in self.keys() for k in keys]), "Invalid keys selected"

        return table_dict(
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

        y_train, X_test, y_test = [], [], []
        if test_videos > 0:
            X_test = np.concatenate(raw_data[test_index])
            X_train = np.concatenate(np.delete(raw_data, test_index, axis=0))

        else:
            X_train = np.concatenate(list(raw_data))

        if self._propagate_labels:
            X_train, y_train = X_train[:, :-1], X_train[:, -1]
            try:
                X_test, y_test = X_test[:, :-1], X_test[:, -1]
            except TypeError:
                pass

        if encode_labels:
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

        return X_train, y_train, X_test, y_test

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
                - propagate_labels (bool): If True, returns a label vector acompaigning each training instance

            Returns:
                - X_train (np.ndarray): 3d dataset with shape (instances, sliding_window_size, features)
                generated from all training videos
                - X_test (np.ndarray): 3d dataset with shape (instances, sliding_window_size, features)
                generated from all test videos (if test_videos > 0)

        """

        global g
        X_train, y_train, X_test, y_test = self.get_training_set(test_videos)

        if scale:
            if verbose:
                print("Scaling data...")

            if scale == "standard":
                scaler = StandardScaler()
            elif scale == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(
                    "Invalid scaler. Select one of standard, minmax or None"
                )  # pragma: no cover

            X_train = scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)

            if scale == "standard":
                assert np.allclose(np.nan_to_num(np.mean(X_train), nan=0), 0)
                assert np.allclose(np.nan_to_num(np.std(X_train), nan=1), 1)

            if test_videos:
                X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
                    X_test.shape
                )

            if verbose:
                print("Done!")

        if align == "all":
            X_train = deepof.utils.align_trajectories(X_train, align)

        X_train = deepof.utils.rolling_window(X_train, window_size, window_step)
        if self._propagate_labels:
            y_train = y_train[::window_step][: X_train.shape[0]]

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
            if self._propagate_labels:
                y_test = y_test[::window_step][: X_test.shape[0]]

            if align == "center":
                X_test = deepof.utils.align_trajectories(X_test, align)

            if conv_filter == "gaussian":
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

        return X_train, y_train, X_test, y_test

    def random_projection(
        self, n_components: int = None, sample: int = 1000
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a random projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        X = self.get_training_set()[0]

        # Takes care of propagated labels if present
        if self._propagate_labels:
            X = X[:, :-1]

        # noinspection PyUnresolvedReferences
        X = X[np.random.choice(X.shape[0], sample, replace=False), :]
        X = SimpleImputer(strategy="median").fit_transform(X)

        rproj = random_projection.GaussianRandomProjection(n_components=n_components)
        X = rproj.fit_transform(X)

        return X, rproj

    def pca(
        self, n_components: int = None, sample: int = 1000, kernel: str = "linear"
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a PCA projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        X = self.get_training_set()[0]
        X = SimpleImputer(strategy="median").fit_transform(X)

        # Takes care of propagated labels if present
        if self._propagate_labels:
            X = X[:, :-1]

        # noinspection PyUnresolvedReferences
        X = X[np.random.choice(X.shape[0], sample, replace=False), :]

        pca = KernelPCA(n_components=n_components, kernel=kernel)
        X = pca.fit_transform(X)

        return X, pca

    def tsne(
        self, n_components: int = None, sample: int = 1000, perplexity: int = 30
    ) -> deepof.utils.Tuple[deepof.utils.Any, deepof.utils.Any]:
        """Returns a training set generated from the 2D original data (time x features) and a PCA projection
        to a n_components space. The sample parameter allows the user to randomly pick a subset of the data for
        performance or visualization reasons"""

        X = self.get_training_set()[0]
        X = SimpleImputer(strategy="median").fit_transform(X)

        # Takes care of propagated labels if present
        if self._propagate_labels:
            X = X[:, :-1]

        # noinspection PyUnresolvedReferences
        X = X[np.random.choice(X.shape[0], sample, replace=False), :]

        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        X = tsne.fit_transform(X)

        return X, tsne


def merge_tables(*args):
    """

    Takes a number of table_dict objects and merges them
    Returns a table_dict object of type 'merged'

    """
    merged_dict = {key: [] for key in args[0].keys()}
    for tabdict in args:
        for key, val in tabdict.items():
            merged_dict[key].append(val)

    merged_tables = table_dict(
        {
            key: pd.concat(val, axis=1, ignore_index=True)
            for key, val in merged_dict.items()
        },
        typ="merged",
    )

    return merged_tables


# TODO:
#   - Generate ragged training array using a metric (acceleration, maybe?)
#   - Use something like Dynamic Time Warping to put all instances in the same length
#   - with the current implementation, preprocess can't fully work on merged table_dict instances.
#   While some operations (mainly alignment) should be carried out before merging, others require
#   the whole dataset to function properly.
#   - Understand how keras handles NA values. Decide whether to do nothing, to mask them or to impute them using
#   a clear outlier (e.g. -9999)
