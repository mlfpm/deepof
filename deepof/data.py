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
import cv2

#try:
#    cv2.imshow("test",1)
#    cv2.waitKey(1)
#    cv2.destroyAllWindows()
#except:
#    pass


import copy
import os
import pickle
import re
import shutil
import warnings
from shutil import rmtree
from time import time
from typing import Any, Dict, List, NewType, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import psutil
import cv2
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
from deepof.config import PROGRESS_BAR_FIXED_WIDTH
import deepof.model_utils
import deepof.models
import deepof.utils
import deepof.visuals
from deepof.visuals_utils import _preprocess_time_bins
from deepof.data_loading import get_dt, load_dt, save_dt


# SET DEEPOF VERSION
current_deepof_version="0.8.0"

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
        fast_implementations_threshold: int = 50000,) -> coordinates:  # pragma: no cover
    """Load a pre-saved pickled Coordinates object. Will update Coordinate objects from older versions of deepof (down to 0.7) to work with this version

    Args:
        project_path (str): name of the file to load.
        Otherwise same as Project init inputs, will only be used in case of a version upgrade

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
        #if coordinates._table_paths[0].endswith(".analysis.h5"):
        #    table_extension=".analysis.h5"
        #else:
        #    _, table_extension = os.path.splitext(coordinates._table_paths[0])
        #_, video_extension = os.path.splitext(coordinates._videos[0])

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
        exp_conditions: dict = None,
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
        if frames_max > 360000: #roughly 4 hour video at 25 fps
            self.very_large_project = True

        # Init the rest of the parameters
        self.angles = True
        self.animal_ids = animal_ids if animal_ids is not None else [""]
        self.areas = True
        self.bodypart_graph = bodypart_graph
        self.connectivity = None
        self.distances = "all"
        self.ego = False
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

        return deepof.utils.get_arenas(
            self,
            tables,
            self.arena,
            self.arena_dims,
            self.segmentation_path,
            self.video_path,
            self.videos,
            debug,
            test,
        )

    #from memory_profiler import profile
    #@profile
    def preprocess_tables(self) -> table_dict:
        """Load tables, performs multiple preprocessing steps:
            - Trajectory loading,
            - Adjusting table headers
            - Updating the bodypart graph
            - Creating a tiemindex column
            - Smooting the trajectories
            - Removing outliers
            - Missing value imputation
        Then saves the preprocessed tables and retruns them in a table dictionary

        Returns:
            tab_dict (table_dict): A table dictionary containing tables for all experiments

        """
        if self.table_format not in ["h5", "csv", "npy", "slp", "analysis.h5"]:
            raise NotImplementedError(
                "Tracking files must be in h5 (DLC or SLEAP), csv, npy, or slp format"
            )  # pragma: no cover


        lik_dict={}
        tab_dict={}
        N_tables = len(self.tables)
        found_individuals=False
        sum_warn_nans=0

        with tqdm(total=N_tables, desc=f"{'Preprocessing tables':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table") as pbar:
            for i, key in enumerate(self.tables.keys()):
                               
                pbar.set_postfix(step="Loading trajectories")

                loaded_tab = deepof.utils.load_table(
                    self.tables[key],
                    self.source_table_path,
                    self.table_format,
                    self.rename_bodyparts,
                    self.animal_ids,
                )

                #check individuals in header for consitency
                if i>0:
                    assert [("individuals" in loaded_tab.index and found_individuals==True) or
                        (not "individuals" in loaded_tab.index and found_individuals==False), 
                        f"Table {key} has different header formatting for \"individuals\" than the other tables!"]
                
                # Check if the files come from a multi-animal DLC project
                if "individuals" in loaded_tab.index:
                    found_individuals=True


                    self.animal_ids = list(
                        loaded_tab.loc["individuals", :].unique()
                    )
        
                    # Adapt each table to work with the downstream pipeline
                    loaded_tab.loc["bodyparts"] = (
                        loaded_tab.loc["individuals"] + "_" + loaded_tab.loc["bodyparts"]
                    )
                    loaded_tab.drop("individuals", axis=0, inplace=True)

                pbar.set_postfix(step="Adjusting headers")

                # Convert the first rows of each dataframe to a multi-index
                tab_copy = loaded_tab.copy()

                tab_copy.columns = pd.MultiIndex.from_arrays(
                    [loaded_tab.iloc[i] for i in range(2)]
                )
                tab_copy = tab_copy.iloc[2:].astype(float)
                loaded_tab = tab_copy.reset_index(drop=True)

                pbar.set_postfix(step="Updating bodypart graphs")

                # reinstate "vanilla" bodyparts without animal ids in case animal ids were already fused with the bp list
                reinstated_bodyparts = list(
                    set(
                        [
                            bp
                            if bp[0 : len(aid) + 1]
                            not in [aid + "_" for aid in self.animal_ids]
                            else bp[len(aid) + 1 :]
                            for aid in self.animal_ids
                            for bp in self.exclude_bodyparts
                        ]
                    )
                )

                # Update body part connectivity graph, taking detected or specified body parts into account
                model_dict = {
                    "{}mouse_topview".format(aid): deepof.utils.connect_mouse(
                        aid,
                        exclude_bodyparts=reinstated_bodyparts,
                        graph_preset=self.bodypart_graph,
                    )
                    for aid in self.animal_ids
                }
                self.connectivity = {
                    aid: model_dict[aid + self.model] for aid in self.animal_ids
                }

                # Remove specified body parts from the mice graph
                if len(self.animal_ids) > 1 and reinstated_bodyparts != [""]:
                    self.exclude_bodyparts = [
                        aid + "_" + bp for aid in self.animal_ids for bp in reinstated_bodyparts
                    ]

                # Pass a time-based index, if specified in init
                if self.frame_rate is not None:
                
                    if (pbar.n+1) % 20 != 0:
                        pbar.set_postfix(step="Updating time index")
                    #These two lines of code here are absolutely necessary 
                    else:
                        pbar.set_postfix(step="Planning AI uprising")
                    
                    loaded_tab.index = pd.timedelta_range(
                        "00:00:00",
                        pd.to_timedelta(
                            int(np.round(loaded_tab.shape[0] // self.frame_rate)), unit="sec"
                        ),
                        periods=loaded_tab.shape[0] + 1,
                        closed="left",
                    ).map(lambda t: str(t)[7:])

                x = loaded_tab.xs("x", level="coords", axis=1, drop_level=False)
                y = loaded_tab.xs("y", level="coords", axis=1, drop_level=False)
                lik = loaded_tab.xs("likelihood", level="coords", axis=1, drop_level=True)

                loaded_tab = pd.concat([x, y], axis=1).sort_index(axis=1)
                likely_dict = {key : lik.fillna(0.0)}

                likely_dict = TableDict(likely_dict, 
                                        typ="quality", 
                                        table_path=os.path.join(self.project_path, self.project_name, "Tables"), 
                                        animal_ids=self.animal_ids)

                # Collect irrelevant bodyparts (either got excluded or are not part of the selected connectivity graph)
                all_bodyparts=list(loaded_tab.columns.levels[0])
                relevant_bodyparts=[]
                for aid in self.animal_ids:
                    relevant_bodyparts=relevant_bodyparts+list(self.connectivity[aid].nodes)

                
                relevant_bodyparts = set(relevant_bodyparts) - set(self.exclude_bodyparts)
                irrelevant_bodyparts = list(set(all_bodyparts) - relevant_bodyparts)
                
                # Remove irrelevant bodyparts (this sentence sounds very unsetteling without context)
                if len(irrelevant_bodyparts) > 0:

                    temp = loaded_tab.drop(irrelevant_bodyparts, axis=1, level="bodyparts")
                    temp.sort_index(axis=1, inplace=True)
                    temp.columns = pd.MultiIndex.from_product(
                        [
                            os_sorted(list(set([i[j] for i in temp.columns])))
                            for j in range(2)
                        ]
                    )
                    loaded_tab = temp.sort_index(axis=1)
                
                table_dict={key:loaded_tab}

                if self.smooth_alpha:

                    pbar.set_postfix(step="Smoothing trajectories")

                    cur_idx = loaded_tab.index
                    cur_cols = loaded_tab.columns
                    smooth = pd.DataFrame(
                        deepof.utils.smooth_mult_trajectory(
                            np.array(loaded_tab), alpha=self.smooth_alpha, w_length=15
                        )
                    ).reset_index(drop=True)
                    smooth.columns = cur_cols
                    smooth.index = cur_idx
                    loaded_tab = smooth

                if self.remove_outliers:

                    pbar.set_postfix(step="Removing outliers")

                    
                    for k, table in table_dict.items():
                        table_dict[k], warn_nans = deepof.utils.remove_outliers(
                            table,
                            likely_dict[k],
                            likelihood_tolerance=self.likelihood_tolerance,
                            mode="or",
                            n_std=self.interpolation_std,
                        )
                    sum_warn_nans+=warn_nans


                if self.iterative_imputation:

                    pbar.set_postfix(step="Iterative imputation of ocluded bodyparts")

                    if self.iterative_imputation == "full":
                        table_dict = deepof.utils.iterative_imputation(
                            self, table_dict, likely_dict, full_imputation=True
                        )
                    else:
                        table_dict = deepof.utils.iterative_imputation(
                            self, table_dict, likely_dict, full_imputation=False
                        )

                # Set table_dict to NaN if animals are missing
                table_dict = deepof.utils.set_missing_animals(self, table_dict, likely_dict)

                pbar.set_postfix(step="Saving data")

                #create folder for current data set
                directory = os.path.join(self.project_path, self.project_name, 'Tables', key)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # save paths for tables
                quality_path = os.path.join(directory, key + '_likelyhood')
                table_path = os.path.join(directory, key)
                lik_dict[key] = save_dt(likely_dict[key],quality_path,self.very_large_project)
                tab_dict[key] = save_dt(table_dict[key],table_path,self.very_large_project)

                #cleanup
                del table_dict[key]
                del loaded_tab
                del likely_dict

                pbar.update() 
        
        #warn in case of excessive missing data
        if sum_warn_nans>0:
            warnings.warn("\033[38;5;208m"
                          f"more than 30% of all tracked position values in {sum_warn_nans} out of {N_tables} tables are missing or outliers.\n"
                          "This may be expected if your mice were obscured and not tracked for long intervals\n"
                          "(e.g. when sleeping in an occluded location or leaving the arena)."
                          "\033[0m"
                          )
        
        #update table path to directory with generated tables
        self.table_path = os.path.join(
            self.project_path, self.project_name, "Tables"
        )

        return tab_dict, TableDict(lik_dict, 
                                   typ="quality", 
                                   table_path=os.path.join(self.project_path, self.project_name, "Tables"), 
                                   animal_ids=self.animal_ids)

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

                #scale
                tab=tab*scaling_ratio

                #save scaled table
                distance_path = os.path.join(self.project_path, self.project_name, 'Tables', key, key)
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
            for i, (key, tab) in enumerate(tab_dict.items()):

                #load active table
                tab = get_dt(tab_dict, key)

                #get distances for this table
                distance_tab=self.get_distances_tab(tab,self.scales[key][2:])

                #save distances for active table
                distance_path = os.path.join(self.project_path, self.project_name, 'Tables', key, key + '_dist')
                distance_dict[key] = save_dt(distance_tab,distance_path,self.very_large_project)

                #clean up
                del distance_tab
                pbar.update()

        return distance_dict
    
    def get_distances_tab(self, tab: pd.DataFrame, scale: np.ndarray=None) -> dict:
        """Compute the distances between all selected body parts over time for a single table.

        Args:
            tab (pd.DataFrame): Pandas DataFrame containing the trajectories of all bodyparts.
            scale (np.ndarray): Array that contains info for scaling of this dataset

        Returns:
            distance_tab: Pandas DataFrame containing the distances between all bodyparts.

        """


        if scale is None:
            scale=[1,1]

        nodes = self.distances
        if nodes == "all":
            nodes = tab.columns.levels[0]

        assert [
            i in tab.columns.levels[0] for i in nodes
        ], "Nodes should correspond to existent bodyparts"

        distance_tab = deepof.utils.bpart_distance(tab, scale[1], scale[0])
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
        """Compute all the angles between adjacent bodypart trios per video and per frame in all datasets in teh given table dictionary.

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
                    angle_path = os.path.join(self.project_path, self.project_name, 'Tables', key, key + '_angle')
                    angle_dict[key] = save_dt(dats,angle_path,self.very_large_project)
                    pbar.update()


        except KeyError:
            raise KeyError(
                "Are you using a custom labelling scheme? Our tutorials may help! "
                "In case you're not, are there multiple animals in your single-animal DLC video? Make sure to set the "
                "animal_ids parameter in deepof.data.Project"
            )

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

                area_path = os.path.join(self.project_path, self.project_name, 'Tables', key, key + '_area')
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
        self.scales, self.arena_params, self.video_resolution = self.get_arena(
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
            tables=tables,
            table_paths=self.tables,
            source_table_path=self.source_table_path,
            trained_model_path=self.trained_path,
            videos=self.videos,
            video_path=self.video_path,
            video_resolution=self.video_resolution,
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
            run_numba (bool): Determines if numba versions of functions should be used (run faster but require initial compilation time on first run)
            very_large_project (bool): Decides if memory efficient data loading and saving should be used
            version (str): version of deepof this object was created with

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
            )
            
            # save paths for modified tables
            table_path = os.path.join(self._project_path, self._project_name, 'Tables', key, key + '_' + file_name)
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
            
        Returns:
            tab (pd.DataFrame): A data frame containing the coordinates for the selected key as values.

        """

        coord_1, coord_2 = "x", "y"

        #load table if not already loaded
        tab = deepof.utils.deepcopy(get_dt(self._tables, key))
        #to avoid reloading quality if quality is given
        if quality is None:
            quality=self.get_quality().filter_videos([key])
            quality[key] = get_dt(quality,key)

        if align:

            assert any(
                center in bp for bp in tab.columns.levels[0]
            ), "for align to run, center must be set to the name of a bodypart"
            assert any(
                align in bp for bp in tab.columns.levels[0]
            ), "align must be set to the name of a bodypart"

        if polar:
            coord_1, coord_2 = "rho", "phi"
            scale = deepof.utils.bp2polar(scale).to_numpy().reshape(-1)
            tab = deepof.utils.tab2polar(tab)

        if center == "arena":


            tab.loc[:, (slice("x"), [coord_1])] = (
                tab.loc[:, (slice("x"), [coord_1])] - scale[0]
            )

            tab.loc[:, (slice("x"), [coord_2])] = (
                tab.loc[:, (slice("x"), [coord_2])] - scale[1]
            )

        #undo mm conversion, if video export is required
        if to_video:

            tab.loc[:, (slice("x"), [coord_1])] = (
                tab.loc[:, (slice("x"), [coord_1])] * scale[2]/scale[3]
            )

            tab.loc[:, (slice("x"), [coord_2])] = (
                tab.loc[:, (slice("x"), [coord_2])] * scale[2]/scale[3]
            )

        elif isinstance(center, str) and center != "arena":

            # Center each animal independently
            animal_ids = self._animal_ids
            if selected_id is not None:
                animal_ids = [selected_id]

            for aid in animal_ids:
                # center on x / rho
                tab.update(
                    tab.loc[:, [i for i in tab.columns if i[0].startswith(aid)]]
                    .loc[:, (slice("x"), [coord_1])]
                    .subtract(
                        tab[aid + ("_" if aid != "" else "") + center][coord_1],
                        axis=0,
                    )
                )

                # center on y / phi
                tab.update(
                    tab.loc[:, [i for i in tab.columns if i[0].startswith(aid)]]
                    .loc[:, (slice("x"), [coord_2])]
                    .subtract(
                        tab[aid + ("_" if aid != "" else "") + center][coord_2],
                        axis=0,
                    )
                )

        if align:

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
                        np.array(partial_aligned),
                        mode="all",
                        run_numba=self._run_numba,
                    )
                    partial_aligned[np.abs(partial_aligned) < 1e-5] = 0.0
                    partial_aligned = pd.DataFrame(partial_aligned)
                    aligned_coordinates = pd.concat(
                        [aligned_coordinates, partial_aligned], axis=1
                    )

            aligned_coordinates.index = all_index
            aligned_coordinates.columns = pd.MultiIndex.from_tuples(all_columns)
            tab = aligned_coordinates

        if speed:
            vel = deepof.utils.rolling_speed(
                tab,
                frame_rate=self._frame_rate,

                deriv=speed,
                center=center,
            )
            tab = vel

        # Id selected_id was specified, selects coordinates of only one animal for further processing
        if selected_id is not None:
            tab = tab.loc[
                :, deepof.utils.filter_columns(tab.columns, selected_id)
            ]

        table_dict={key:tab}
        # Set table_dict to NaN if animals are missing
        table_dict = deepof.utils.set_missing_animals(self, table_dict, quality)
        tab = table_dict[key]

        return tab
    
            
    def get_distances(
        self,
        speed: int = 0,
        selected_id: str = None,
        filter_on_graph: bool = True,
        file_name: str = 'got_distances',
        return_path: bool = False,
    ) -> table_dict:
        """Return a table_dict object with the distances between body parts animal as values.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
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

                #get distances as tab dataFrame  
                tab=self.get_distances_at_key(
                    key,
                    speed=speed,
                    selected_id=selected_id,
                    filter_on_graph=filter_on_graph,
                    )

                # save paths for modified tables
                table_path = os.path.join(self._project_path, self._project_name, 'Tables', key, key + '_' + file_name)
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
        filter_on_graph: bool = True,
    ) -> pd.DataFrame:
        """Return a pd.DataFrame with the distances between body parts of one animal as values.

        Args:
            key (str): key for requested distance
            quality: (table_dict): Quality information for current data Frame
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
            filter_on_graph (bool): If True, only distances between connected nodes in the DeepOF graph representations are kept. Otherwise, all distances between bodyparts are returned.

        Returns:
            tab (pd.DataFrame): A pd.DataFrame with the distances between body parts of one animal as values.

        """

        #load table if not already loaded
        tab = deepof.utils.deepcopy(get_dt(self._distances, key))
        #to avoid reloading quality if quality is given
        if quality is None:
            quality=self.get_quality().filter_videos([key])
            quality[key] = get_dt(quality,key)

        if speed:
            tab = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="dists")

        if selected_id is not None:
            tab = tab.loc[
                :, deepof.utils.filter_columns(tab.columns, selected_id)
            ]


        table_dict={key:tab}
        # Set table_dict to NaN if animals are missing
        table_dict = deepof.utils.set_missing_animals(self, table_dict, quality)
        tab = table_dict[key]

        if filter_on_graph:

            tab = tab.loc[
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
        
        return tab

   
    def get_angles(
        self,
        degrees: bool = False,
        speed: int = 0,
        selected_id: str = None,
        file_name: str = 'got_angles',
        return_path: bool = False,
    ) -> table_dict:
        """Return a table_dict object with the angles between body parts animal as values.

        Args:
            degrees (bool): If True (default), the angles will be in degrees. Otherwise they will be converted to radians.
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.
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
                )

                # save paths for modified tables
                table_path = os.path.join(self._project_path, self._project_name, 'Tables', key, key + '_' + file_name)
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
    ) -> pd.DataFrame:
        """Return a Dataframe with the angles between body parts for one animal as values.

        Args:
            key (str): key for requested distance
            quality: (table_dict): Quality information for current data Frame
            degrees (bool): If True (default), the angles will be in degrees. Otherwise they will be converted to radians.
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select.

        Returns:
            tab (pd.DataFrame): A pd.DataFrame with the angles between body parts of one animal as values.

        """  

        #load table if not already loaded
        tab = deepof.utils.deepcopy(get_dt(self._angles, key))
        #to avoid reloading quality if quality is given
        if quality is None:
            quality=self.get_quality().filter_videos([key])
            quality[key] = get_dt(quality,key)

        if degrees:
            tab = np.degrees(tab) 

        if speed:
            vel = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="angles")
            tab = vel

        if selected_id is not None:
            tab = tab.loc[
                :, deepof.utils.filter_columns(tab.columns, selected_id)
            ]

        table_dict={key:tab}
        # Set table_dict to NaN if animals are missing
        table_dict = deepof.utils.set_missing_animals(self, table_dict, quality)
        tab = table_dict[key]

        return tab

  
    def get_areas(
            self, 
            speed: int = 0,
            selected_id: str = "all",
            file_name: str = 'got_areas',
            return_path: bool = False,
            ) -> table_dict:
        """Return a table_dict object with all relevant areas (head, torso, back, full). Unless specified otherwise, the areas are computed for all animals.

        Args:
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select. "all" (default) computes the areas for all animals. Declared in self._animal_ids.
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
                )

                # save paths for modified tables
                table_path = os.path.join(self._project_path, self._project_name, 'Tables', key, key + '_' + file_name)
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
        ) -> table_dict:
        """Return a pd.DataFrame with all relevant areas (head, torso, back, full). Unless specified otherwise, the areas are computed for all animals.

        Args:
            key (str): key for requested distance
            quality: (table_dict): Quality information for current data Frame
            speed (int): The derivative to use for speed.
            selected_id (str): The id of the animal to select. "all" (default) computes the areas for all animals. Declared in self._animal_ids.
            file_name (str): Name of the file for saving
            return_path (bool): if True, Return only the path to the processed table, if false, return the full table. 

        Returns:
            tab (pd.DataFrame): A pd.DataFrame object with the areas of the body parts animal as values.
        """
        
        #load table if not already loaded
        tab = deepof.utils.deepcopy(get_dt(self._areas, key))
        #to avoid reloading quality if quality is given
        if quality is None:
            quality=self.get_quality().filter_videos([key])
            quality[key] = get_dt(quality,key)

        if selected_id == "all":
            selected_ids = self._animal_ids
        else:
            selected_ids = [selected_id]

        exp_table = pd.DataFrame()

        for aid in selected_ids:

            if aid == "":
                aid = None
            
            if tab is None:
                tab = pd.DataFrame()

            # get the current table for the current animal
            current_table = tab.loc[
                :, deepof.utils.filter_columns(tab.columns, aid)
            ]
            exp_table = pd.concat([exp_table, current_table], axis=1)

        tab = exp_table

        if speed:
            tab = deepof.utils.rolling_speed(tab, deriv=speed + 1, typ="areas")

        table_dict={key:tab}
        # Set table_dict to NaN if animals are missing
        table_dict = deepof.utils.set_missing_animals(self, table_dict, quality)
        tab = table_dict[key]
    
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

        edited_scales, edited_arena_params, _ = deepof.utils.get_arenas(
            coordinates=self,
            tables=self._tables,
            arena=arena_type,
            arena_dims=self._arena_dims,
            segmentation_model_path=None,
            video_path=self._video_path,
            videos=videos_to_update,
        )

        # update the scales and arena parameters
        for key in video_keys:
            self._scales[key] = edited_scales[key]
            self._arena_params[key] = edited_arena_params[key]

        self.save(timestamp=False)

        if verbose:
            print("Done!")

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

            tab_dict._connectivity = graph

            #read table metadata
            if type(list(dists.values())[0]) == str:
                edge_feature_names = get_dt(dists,list(dists.keys())[0], only_metainfo=True)['columns']
            else:
                edge_feature_names = list(list(dists.values())[0].columns)

            if type(list(tab_dict.values())[0]) == str:
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

                        tab, table_path = get_dt(to_preprocess[k], key, return_path=True) 

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
            "sniff_arena_tol": 12.5,                           # Noses needs to be 12.5 mm apart from teh arena edge or closer
            "min_immobility": int(self._frame_rate),           # Min Time interval the mouse needs to be immobile to be counted as immobility, 1 second 
            #"max_immobility": 120*int(self._frame_rate),       # Max Time interval the mouse needs to be immobile to be counted as immobility, 2 minutes (anything longer is counted as "sleeping")                              
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
    def supervised_annotation(
        self,
        center: str = "Center",
        align: str = "Spine_1",
        video_output: bool = False,
        frame_limit: int = np.inf,
        debug: bool = False,
        n_jobs: int = 1,
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
    
        N_preprocessing_steps=4+len(self._animal_ids)
        N_processing_steps=len(self._tables.keys())
        
        with tqdm(total=N_preprocessing_steps, desc=f"{'data preprocessing':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="step") as pbar:

            pbar.set_postfix(step="Loading raw coords")

            tag_dict = {}
            params = self.get_supervised_parameters()

            #get all kinds of tables
            raw_coords = self.get_coords(center=None, file_name='raw', return_path=self._very_large_project)
            pbar.update()
            pbar.set_postfix(step="Loading coords")
 
            try:
                coords = self.get_coords(center=center, align=align, return_path=self._very_large_project)
            except AssertionError: # pragma: no cover

                try:
                    coords = self.get_coords(center="Center", align="Spine_1", return_path=self._very_large_project)
                except AssertionError:
                    coords = self.get_coords(center="Center", align="Nose", return_path=self._very_large_project)
            pbar.update() 

            speeds = self.get_coords(speed=1, file_name='speeds', return_path=self._very_large_project)
            pbar.update()
            pbar.set_postfix(step="Loading speeds") 

            dists = self.get_distances(return_path=self._very_large_project)
            angles = self.get_angles(return_path=self._very_large_project)
            pbar.update() 
            pbar.set_postfix(step="Loading distances")


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
                if isinstance(raw_coords[key], str):
                    tag_index = get_dt(raw_coords,key, only_metainfo=True, load_index=True)['index_column']
                else:
                    tag_index = raw_coords[key].index
                self._trained_model_path = resource_filename(__name__, "trained_models")

                supervised_tags = deepof.annotation_utils.supervised_tagging(
                    self,
                    raw_coords=raw_coords,
                    coords=coords,
                    dists=dists,
                    angles=angles,
                    full_features=features_dict,
                    speeds=speeds,
                    key=key, 
                    trained_model_path=self._trained_model_path,
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
                table_path = os.path.join(coords._table_path, key, key + '_' + "supervised_annotations")
                tag_dict[key] = save_dt(supervised_tags,table_path,self._very_large_project) 

                pbar.update() 

        del features_dict, dists, speeds, coords, raw_coords

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
        self.save(filename="supervised_annotation_instance")

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

            tabs[key] = deepof.utils.filter_animal_id_in_table(val, selected_id)

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
            table_path = os.path.join(self._table_path, key, key + '_' + file_name)
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
            X_train (np.ndarray): Table dict with 3D datasets with shape (instances, sliding_window_size, features) generated from all training videos.
            X_test (np.ndarray): Table dict with 3D datasets 3D dataset with shape (instances, sliding_window_size, features) generated from all test videos (0 by default).
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

                if scale:
                    if verbose:
                        print("Scaling data...")

                    if scale not in ["robust", "standard", "minmax"]:
                        raise ValueError(
                            "Invalid scaler. Select one of standard, minmax or robust"
                        )  # pragma: no cover

                    # Scale each experiment independently, to control for animal size
                    current_tab = deepof.utils.scale_table(
                        coordinates=self,
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
                table_path = os.path.join(self._table_path, key, key + '_' + file_name)
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
                        coordinates=self,
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
                    table_path = os.path.join(self._table_path, key, key + '_' + file_name)
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


    def sample_windows_from_data(self, bin_info: dict={}, N_windows_tab: int=10000, return_edges: bool=False, no_nans: bool=False):
        """
        Sample a set of windows / rows from all entries of table dict to avoid breaking memory.

        Args:
            bin_info (dict): Dictionary containing integer arrays for each experiment, denoting the indices of the tables to sample
            N_windows_tab (int): Maximum number of windows / rows to include from each recording. Will be skipped if bin_info is given
            return_edges (bool): Return second sampled set corresponding to edges [only relevant for internal use] (default: False)
            no_nans (bool): only sample windows / rows that do not contain Nans. Notice: Turning on this option will result in the sampled windows not being completely successive anyomre (default: False). Will be skipped if bin_info is given

        Returns:
            X_data (np.array): Main dataset
            a_data (np.array): Edges dataset (if return_edges is True)
        """
        X_data, a_data = [], []
        use_input_samples=False
        if len(bin_info) > 0 and set(self.keys()).issubset(bin_info.keys()):
            use_input_samples=True 

        for key in self.keys():
            # Load table tuple
            tab_tuple = get_dt(self, key)

            # Check if only one dataset or two (e.g., training and testing data) was given
            if isinstance(tab_tuple, tuple):
                tab = tab_tuple[0]
            else:
                tab = tab_tuple

            #Quick block to process if input samples are given
            if use_input_samples:
                if isinstance(tab, np.ndarray):
                    X_data.append(tab[bin_info[key]])
                    if return_edges and isinstance(tab_tuple, tuple):
                        a_data.append(tab_tuple[1][bin_info[key]])
                    elif return_edges:
                        a_frame = np.zeros(X_data[-1].shape)
                        a_data.append(a_frame)

                elif isinstance(tab, pd.DataFrame):
                    X_data.append(tab.iloc[bin_info[key]])
                    if return_edges and isinstance(tab_tuple, tuple):
                        a_data.append(tab_tuple[1].iloc[bin_info[key]])
                    elif return_edges:
                        a_frame = np.zeros(X_data[-1].shape)
                        a_frame = pd.DataFrame(a_frame)
                        a_data.append(a_frame)

                continue

            if no_nans and isinstance(tab, np.ndarray):
                possible_idcs = ~np.isnan(tab).any(axis=tuple(range(1,tab.ndim)))
                tab=tab[possible_idcs]
            elif no_nans and isinstance(tab, pd.DataFrame):
                possible_idcs = ~np.isnan(tab).any(axis=1)
                tab=tab[possible_idcs]
            else:
                possible_idcs = np.ones(len(tab), dtype=bool)

            # Determine last possible start position based on the number of windows being extracted
            start_max = tab.shape[0] - N_windows_tab

            # Get first window (0 if table length is below N_windows_tab)
            start = np.random.randint(low=0, high=np.max([1, start_max + 1]))

            # Collect data
            if isinstance(tab, np.ndarray):
                X_data.append(tab[start:start + N_windows_tab, :])
            elif isinstance(tab, pd.DataFrame):
                X_data.append(tab.iloc[start:start + N_windows_tab, :])

            #collect idcs 
            bin_info[key]=np.where(possible_idcs)[0][start:start + N_windows_tab]

            # Handle test dataset, if existent
            if return_edges and isinstance(tab_tuple, tuple):
                if isinstance(tab_tuple[1], np.ndarray):
                    a_data.append(tab_tuple[1][start:start + N_windows_tab, :])
                elif isinstance(tab_tuple[1], pd.DataFrame):
                    a_data.append(tab_tuple[1].iloc[start:start + N_windows_tab, :])
            elif return_edges:
                a_frame = np.zeros(X_data[-1].shape)
                if isinstance(tab, pd.DataFrame):
                    a_frame = pd.DataFrame(a_frame)
                a_data.append(a_frame)

        X_data = np.concatenate(X_data, axis=0)

        if return_edges:
            a_data = np.concatenate(a_data, axis=0)
            return X_data, a_data, bin_info
        else:
            return X_data, bin_info


if __name__ == "__main__":
    # Remove excessive logging from tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
