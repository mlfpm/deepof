import os, re
import numpy as np
import pandas as pd
from collections import defaultdict
from pandarallel import pandarallel
from tqdm import tqdm
from DLC_analysis_additional_functions import *


class get_coordinates:
    """ Class for loading and preprocessing DLC data of individual and social mice. """

    def __init__(
        self,
        video_format=".mp4",
        table_format=".h5",
        path=".",
        exp_conditions=False,
        arena="circular",
        smooth_alpha=0.1,
        p=1,
        verbose=True,
        distances=False,
    ):
        self.path = path
        self.video_path = self.path + "Videos/"
        self.table_path = self.path + "Tables/"
        self.videos = sorted(
            [vid for vid in os.listdir(self.video_path) if vid.endswith(video_format)]
        )
        self.tables = sorted(
            [tab for tab in os.listdir(self.table_path) if tab.endswith(table_format)]
        )
        self.exp_conditions = exp_conditions
        self.table_format = table_format
        self.video_format = video_format
        self.arena = arena
        self.smooth_alpha = smooth_alpha
        self.p = p
        self.verbose = verbose
        self.distances = distances

        assert [re.findall("(.*)_", vid)[0] for vid in self.videos] == [
            re.findall("(.*)\.", tab)[0] for tab in self.tables
        ], "Video files should match table files"

    def __str__(self):
        if self.exp_conditions:
            return "DLC analysis of {} videos across {} conditions".format(
                len(self.videos), self.exp_conditions.shape[1]
            )
        else:
            return "DLC analysis of {} videos".format(len(self.videos))

    def load_tables(self):
        """Loads videos and tables into dictionaries"""

        if self.verbose:
            print("Loading and smoothing trajectories...")

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

        if self.smooth_alpha:

            for dframe in tqdm(table_dict.keys()):
                table_dict[dframe] = table_dict[dframe].apply(
                    lambda x: smooth_mult_trajectory(x, alpha=self.smooth_alpha), axis=0
                )

        for key, tab in table_dict.items():
            table_dict[key] = tab[tab.columns.levels[0][0]]

        return table_dict

    def get_distances(self):
        """Computes the distances between all selected bodyparts over time.
           If ego is provided, it only returns distances to a specified bodypart"""

        table_dict = self.load_tables()

        if self.verbose:
            print("Computing distance based coordinates...")

        distance_dict = defaultdict()
        pandarallel.initialize(nb_workers=self.p, verbose=1)

        nodes = self.distances
        if nodes == "All":
            nodes = table_dict[list(table_dict.keys())[0]].columns.levels[0]

        assert [
            i in list(table_dict.values())[0].columns.levels[0] for i in nodes
        ], "Nodes should correspond to existent bodyparts"

        for key in tqdm(table_dict.keys()):

            distance_dict[key] = table_dict[key][nodes].parallel_apply(
                lambda x: bpart_distance(x, nodes, 1, 1), axis=1,
            )

        return distance_dict

    def run(self):
        """Generates a dataset using all the options specified during initialization"""

        if self.distances == False:
            self.load_tables()
        else:
            self.get_distances()

        if self.verbose == 1:
            print("Done!")
