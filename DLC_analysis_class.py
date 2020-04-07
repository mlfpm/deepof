import os
from DLC_analysis_additional_functions import *


class DLC_analysis(
    self,
    video_format=".mp4",
    table_format=".h5",
    path=".",
    exp_conditions=None,
    arena="circular",
):
    """ Main class for loading and analysing DLC data of individual and social mice. """

    def __init__(self):
        self.videos = [vid for vid in path if vid.endswith(video_format)]
        self.tables = [tab for tab in path if tab.endswith(table_format)]
        self.exp_conditions = exp_conditions
        self.arena = arena

    def __print__(self, verbose=False):
        if verbose == False:
            return "DLC analysis of {} videos across {} conditions".format(
                len(self.videos), self.exp_conditions.shape[1]
            )

        else:
            return False
