import os, re
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm
from DLC_analysis_additional_functions import *


class DLC_analysis:
    """ Main class for loading and analysing DLC data of individual and social mice. """

    def __init__(
        self,
        video_format=".mp4",
        table_format=".h5",
        path=".",
        exp_conditions=False,
        arena="circular",
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

    def load_tables(self, smooth_alpha=0.25, p=1):
        """Loads videos and tables into dictionaries"""

        pandarallel.initialize(nb_workers=p, verbose=0)

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

        if smooth_alpha:
            for dframe in tqdm(table_dict.keys()):
                table_dict[dframe] = table_dict[dframe].parallel_apply(
                    smooth_mult_trajectory, axis=0
                )

        for key, tab in table_dict.items():
            table_dict[key] = tab[tab.columns.levels[0][0]]

        return table_dict
