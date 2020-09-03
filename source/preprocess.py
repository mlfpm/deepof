# @author lucasmiranda42

from collections import defaultdict
from copy import deepcopy
from pandas_profiling import ProfileReport
from sklearn import random_projection
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import warnings
import networkx as nx

from source.utils import *


class project:
    """

    Class for loading and preprocessing DLC data of individual and social mice.

    """

    def __init__(
        self,
        video_format=".mp4",
        table_format=".h5",
        path=".",
        exp_conditions=False,
        subset_condition=None,
        arena="circular",
        smooth_alpha=0.1,
        arena_dims=[1],
        distances="All",
        ego=False,
        angles=True,
        connectivity=None,
    ):

        self.path = path
        self.video_path = self.path + "/Videos/"
        self.table_path = self.path + "/Tables/"
        self.videos = sorted(
            [vid for vid in os.listdir(self.video_path) if vid.endswith(video_format)]
        )
        self.tables = sorted(
            [tab for tab in os.listdir(self.table_path) if tab.endswith(table_format)]
        )
        self.exp_conditions = exp_conditions
        self.subset_condition = subset_condition
        self.table_format = table_format
        self.video_format = video_format
        self.arena = arena
        self.arena_dims = arena_dims
        self.smooth_alpha = smooth_alpha
        self.distances = distances
        self.ego = ego
        self.angles = angles
        self.connectivity = connectivity
        self.scales = self.get_scale

        # assert [re.findall("(.*)_", vid)[0] for vid in self.videos] == [
        #     re.findall("(.*)\.", tab)[0] for tab in self.tables
        # ], "Video files should match table files"

    def __str__(self):
        if self.exp_conditions:
            return "DLC analysis of {} videos across {} conditions".format(
                len(self.videos), len(self.exp_conditions)
            )
        else:
            return "DLC analysis of {} videos".format(len(self.videos))

    def load_tables(self, verbose):
        """Loads videos and tables into dictionaries"""

        if self.table_format not in [".h5", ".csv"]:
            raise NotImplementedError(
                "Tracking files must be in either h5 or csv format"
            )

        if verbose:
            print("Loading trajectories...")

        if self.table_format == ".h5":
            table_dict = {
                re.findall("(.*?)_", tab)[0]: pd.read_hdf(
                    self.table_path + tab, dtype=float
                )
                for tab in self.tables
            }
        elif self.table_format == ".csv":
            table_dict = {
                re.findall("(.*?)_", tab)[0]: pd.read_csv(
                    self.table_path + tab, dtype=float
                )
                for tab in self.tables
            }

        lik_dict = defaultdict()

        for key, value in table_dict.items():
            x = value.xs("x", level="coords", axis=1, drop_level=False)
            y = value.xs("y", level="coords", axis=1, drop_level=False)
            lik: pd.DataFrame = value.xs(
                "likelihood", level="coords", axis=1, drop_level=True
            )

            table_dict[key] = pd.concat([x, y], axis=1).sort_index(axis=1)
            lik_dict[key] = lik

        if self.smooth_alpha:

            if verbose:
                print("Smoothing trajectories...")

            for key, tab in table_dict.items():
                cols = tab.columns
                smooth = pd.DataFrame(
                    smooth_mult_trajectory(np.array(tab), alpha=self.smooth_alpha)
                )
                smooth.columns = cols
                table_dict[key] = smooth

        for key, tab in table_dict.items():
            table_dict[key] = tab[tab.columns.levels[0][0]]

        if self.subset_condition:
            for key, value in table_dict.items():
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

                table_dict[key] = tab

        return table_dict, lik_dict

    @property
    def get_scale(self):
        """Returns the arena as recognised from the videos"""

        if self.arena in ["circular"]:

            scales = []
            for vid_index, _ in enumerate(self.videos):
                scales.append(
                    list(
                        recognize_arena(
                            self.videos,
                            vid_index,
                            path=self.video_path,
                            arena_type=self.arena,
                        )
                        * 2
                    )
                    + self.arena_dims
                )

        else:
            raise NotImplementedError("arenas must be set to one of: 'circular'")

        return np.array(scales)

    def get_distances(self, table_dict, verbose):
        """Computes the distances between all selected bodyparts over time.
           If ego is provided, it only returns distances to a specified bodypart"""

        if verbose:
            print("Computing distances...")

        nodes = self.distances
        if nodes == "All":
            nodes = table_dict[list(table_dict.keys())[0]].columns.levels[0]

        assert [
            i in list(table_dict.values())[0].columns.levels[0] for i in nodes
        ], "Nodes should correspond to existent bodyparts"

        scales = self.scales[:, 2:]

        distance_dict = {
            key: bpart_distance(tab, scales[i, 1], scales[i, 0],)
            for i, (key, tab) in enumerate(table_dict.items())
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

    def get_angles(self, table_dict, verbose):
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

        bp_net = nx.Graph(self.connectivity)
        cliques = nx.enumerate_all_cliques(bp_net)
        cliques = [i for i in cliques if len(i) == 3]

        angle_dict = {}
        for key, tab in table_dict.items():

            dats = []
            for clique in cliques:
                dat = pd.DataFrame(
                    angle_trio(np.array(tab[clique]).reshape(3, tab.shape[0], 2))
                ).T

                orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2]]
                dat.columns = [tuple(clique[i] for i in order) for order in orders]

                dats.append(dat)

            dats = pd.concat(dats, axis=1)

            angle_dict[key] = dats

        return angle_dict

    def run(self, verbose=False):
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
            tables=tables,
            videos=self.videos,
            arena=self.arena,
            arena_dims=self.arena_dims,
            scales=self.scales,
            quality=quality,
            exp_conditions=self.exp_conditions,
            distances=distances,
            angles=angles,
        )


class coordinates:
    def __init__(
        self,
        tables,
        videos,
        arena,
        arena_dims,
        scales,
        quality,
        exp_conditions=None,
        distances=None,
        angles=None,
    ):
        self._tables = tables
        self.distances = distances
        self.angles = angles
        self._videos = videos
        self._exp_conditions = exp_conditions
        self._arena = arena
        self._arena_dims = arena_dims
        self._scales = scales
        self._quality = quality

    def __str__(self):
        if self._exp_conditions:
            return "Coordinates of {} videos across {} conditions".format(
                len(self._videos), len(self._exp_conditions)
            )
        else:
            return "DLC analysis of {} videos".format(len(self._videos))

    def get_coords(
        self, center="arena", polar=False, speed=0, length=None, align=False
    ):
        tabs = deepcopy(self._tables)

        if polar:
            for key, tab in tabs.items():
                tabs[key] = tab2polar(tab)

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
            for order in range(speed):
                for key, tab in tabs.items():
                    try:
                        cols = tab.columns.levels[0]
                    except AttributeError:
                        cols = tab.columns
                    vel = rolling_speed(tab, typ="coords", order=order + 1)
                    vel.columns = cols
                    tabs[key] = vel

        if length:
            for key, tab in tabs.items():
                tabs[key].index = pd.timedelta_range(
                    "00:00:00", length, periods=tab.shape[0] + 1, closed="left"
                )

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
                tabs[key] = tab[columns]

        return table_dict(
            tabs,
            "coords",
            arena=self._arena,
            arena_dims=self._scales,
            center=center,
            polar=polar,
        )

    def get_distances(self, speed=0, length=None):

        tabs = deepcopy(self.distances)

        if self.distances is not None:

            if speed:
                for order in range(speed):
                    for key, tab in tabs.items():
                        try:
                            cols = tab.columns.levels[0]
                        except AttributeError:
                            cols = tab.columns
                        vel = rolling_speed(tab, typ="dists", order=order + 1)
                        vel.columns = cols
                        tabs[key] = vel

            if length:
                for key, tab in tabs.items():
                    tabs[key].index = pd.timedelta_range(
                        "00:00:00", length, periods=tab.shape[0] + 1, closed="left"
                    )

            return table_dict(tabs, typ="dists")

        raise ValueError(
            "Distances not computed. Read the documentation for more details"
        )

    def get_angles(self, degrees=False, speed=0, length=None):

        tabs = deepcopy(self.angles)

        if self.angles is not None:
            if degrees:
                tabs = {key: np.degrees(tab) for key, tab in tabs.items()}

            if speed:
                for order in range(speed):
                    for key, tab in tabs.items():
                        try:
                            cols = tab.columns.levels[0]
                        except AttributeError:
                            cols = tab.columns
                        vel = rolling_speed(tab, typ="dists", order=order + 1)
                        vel.columns = cols
                        tabs[key] = vel

            if length:
                for key, tab in tabs.items():
                    tabs[key].index = pd.timedelta_range(
                        "00:00:00", length, periods=tab.shape[0] + 1, closed="left"
                    )

            return table_dict(tabs, typ="angles")

        raise ValueError("Angles not computed. Read the documentation for more details")

    def get_videos(self, play=False):
        if play:
            raise NotImplementedError

        return self._videos

    @property
    def get_exp_conditions(self):
        return self._exp_conditions

    def get_quality(self, report=False):
        if report:
            profile = ProfileReport(
                self._quality[report],
                title="Quality Report, {}".format(report),
                html={"style": {"full_width": True}},
            )
            return profile
        return self._quality

    @property
    def get_arenas(self):
        return self._arena, self._arena_dims, self._scales


class table_dict(dict):
    def __init__(self, tabs, typ, arena=None, arena_dims=None, center=None, polar=None):
        super().__init__(tabs)
        self._type = typ
        self._center = center
        self._polar = polar
        self._arena = arena
        self._arena_dims = arena_dims

    def filter(self, keys):
        """Returns a subset of the original table_dict object, containing only the specified keys. Useful, for example,
         for selecting data coming from videos of a specified condition."""

        assert np.all([k in self.keys() for k in keys]), "Invalid keys selected"

        return table_dict(
            {k: value for k, value in self.items() if k in keys}, self._type
        )

    def plot_heatmaps(self, bodyparts, save=False, i=0):

        if self._type != "coords" or self._polar:
            raise NotImplementedError(
                "Heatmaps only available for cartesian coordinates. Set polar to False in get_coordinates and try again"
            )

        if not self._center:
            warnings.warn("Heatmaps look better if you center the data")

        if self._arena == "circular":
            x_lim = (
                [-self._arena_dims[i][2] / 2, self._arena_dims[i][2] / 2]
                if self._center
                else [0, self._arena_dims[i][0]]
            )
            y_lim = (
                [-self._arena_dims[i][2] / 2, self._arena_dims[i][2] / 2]
                if self._center
                else [0, self._arena_dims[i][1]]
            )

            plot_heatmap(
                list(self.values())[i], bodyparts, xlim=x_lim, ylim=y_lim, save=save,
            )

    def get_training_set(self, test_videos=0):
        rmax = max([i.shape[0] for i in self.values()])
        raw_data = np.array(
            [np.pad(v, ((0, rmax - v.shape[0]), (0, 0))) for v in self.values()]
        )
        test_index = np.random.choice(range(len(raw_data)), test_videos, replace=False)

        X_test = []
        if test_videos > 0:
            X_test = np.concatenate(list(raw_data[test_index]))
            X_train = np.concatenate(list(np.delete(raw_data, test_index, axis=0)))

        else:
            X_train = np.concatenate(list(raw_data))

        return X_train, X_test

    def preprocess(
        self,
        window_size=1,
        window_step=1,
        scale="standard",
        test_videos=0,
        verbose=False,
        filter=None,
        sigma=None,
        shift=0,
        shuffle=False,
        align=False,
    ):
        """Builds a sliding window. If specified, splits train and test and
           Z-scores the data using sklearn's standard scaler"""

        X_train, X_test = self.get_training_set(test_videos)

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
                )

            X_train = scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)

            if scale == "standard":
                assert np.allclose(np.mean(X_train), 0)
                assert np.allclose(np.std(X_train, ddof=1), 1)

            if test_videos:
                X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(
                    X_test.shape
                )

            if verbose:
                print("Done!")

        if align == "all":
            X_train = align_trajectories(X_train, align)

        X_train = rolling_window(X_train, window_size, window_step)

        if align == "center":
            X_train = align_trajectories(X_train, align)

        if filter == "gaussian":
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
            X_train = X_train * g.reshape(1, window_size, 1)

        if test_videos:

            if align == "all":
                X_test = align_trajectories(X_test, align)

            X_test = rolling_window(X_test, window_size, window_step)

            if align == "center":
                X_test = align_trajectories(X_test, align)

            if filter == "gaussian":
                X_test = X_test * g.reshape(1, window_size, 1)

            if shuffle:
                X_train = X_train[
                    np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
                ]
                X_test = X_test[
                    np.random.choice(X_test.shape[0], X_test.shape[0], replace=False)
                ]

            return X_train, X_test

        if shuffle:
            X_train = X_train[
                np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)
            ]

        return X_train

    def random_projection(self, n_components=None, sample=1000):

        X = self.get_training_set()
        X = X[np.random.choice(X.shape[0], sample, replace=False), :]

        rproj = random_projection.GaussianRandomProjection(n_components=n_components)
        X = rproj.fit_transform(X)

        return X, rproj

    def pca(self, n_components=None, sample=1000, kernel="linear"):

        X = self.get_training_set()
        X = X[np.random.choice(X.shape[0], sample, replace=False), :]

        pca = KernelPCA(n_components=n_components, kernel=kernel)
        X = pca.fit_transform(X)

        return X, pca

    def tsne(self, n_components=None, sample=1000, perplexity=30):

        X = self.get_training_set()
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
