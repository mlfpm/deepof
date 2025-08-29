# @author NoCreativeIdeaForGoodusername
# encoding: utf-8
# module deepof

"""Plotting utility functions for the deepof package."""
import calendar
import copy
import os
import re
import time
import warnings
import itertools
from pathlib import Path
from typing import Any, List, NewType, Tuple, Union, Optional, NamedTuple
import dataclasses
from dataclasses import dataclass, field
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
from matplotlib.patches import Ellipse
import deepof.visuals_utils
from natsort import os_sorted

import deepof.post_hoc
import deepof.utils
from deepof.data_loading import get_dt
from deepof.config import PROGRESS_BAR_FIXED_WIDTH, BODYPART_COLORS
from deepof.visuals_utils import hex_to_BGR, BGR_to_hex, RGB_to_hex



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


@dataclass
class VideoExportConfig:
    """Configuration for video annotations."""
    display_behavior_names: bool = True
    display_video_name: bool = False
    display_time: bool = False
    display_counter: bool = False
    display_arena: bool = False
    display_markers: bool = False
    display_mouse_labels: bool = False
    display_loading_bar: bool = True


@dataclass
class VideoExportProps:
    """Parameters for drawing text and shapes on the video frame."""
    font: int = cv2.FONT_HERSHEY_DUPLEX
    font_scale: float = 0.5
    thickness: int = 1
    padding: int = 5
    text_color: Tuple[int, int, int] = (255, 255, 255)
    outline_color: Tuple[int, int, int] = (0, 0, 0)
    arena_color: Tuple[int, int, int] = (40, 86, 236)
    arena_thickness: int = 3
    marker_radius: int = 3


def _filter_videos_by_condition(
    coordinates: coordinates,
    experiment_ids: List[str],
    conditions: dict[str, Any],
) -> List[str]:
    """Return a list of experiment IDs that match provided conditions."""
    if not conditions:
        return experiment_ids
    all_exp_conditions = coordinates.get_exp_conditions()
    return [
        exp_id for exp_id in experiment_ids
        if all(
            all_exp_conditions.get(exp_id, {}).get(cond) == state
            for cond, state in conditions.items()
        )
    ]

def _determine_behaviors_to_process(
    behavior_dict: dict,
    behaviors: Union[str, List[str]],
    behaviors_renamed: Union[str, List[str]],
) -> List[str]:
    """Determine the final lists of available, selected and renamed behaviors to generate videos for."""
    if isinstance(behaviors, str):
        return [behaviors]
    meta_info = get_dt(behavior_dict, list(behavior_dict.keys())[0], only_metainfo=True)
    if meta_info.get('columns') is None:
        available_behaviors = ["Cluster_"+ str(k) for k in range(meta_info.get('num_cols'))]
    else:
        available_behaviors = meta_info.get('columns')
    if behaviors is None:
        return available_behaviors, available_behaviors, available_behaviors
    return (
        available_behaviors, 
        [b for b in behaviors if b in available_behaviors], # all behaviors that occur in available_behaviors
        [b for i, b in enumerate(behaviors_renamed) if behaviors[i] in available_behaviors] # all renamed_behaviors that correspond to behaviors occuring in available_behaviors
    )

def _get_behavior_mask_and_confidence(
    tab: Union[pd.DataFrame, np.ndarray],
    behavior: str,
    behavior_names: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """Generates a boolean mask and a confidence series for a given behavior."""
    if isinstance(tab, np.ndarray):
        df = pd.DataFrame(tab, columns=behavior_names)
        mask = (df.idxmax(axis=1) == behavior)
        confidence = df[behavior]
    else:
        df = tab.copy()
        if df.columns.tolist() != list(behavior_names):
            df.columns = behavior_names
        mask = df[behavior] > 0.1
        confidence = df[behavior]
    return mask, confidence


def output_videos_per_cluster(
    coordinates: coordinates,
    exp_conditions: dict,
    behavior_dict: dict,
    behaviors: Union[str, list],
    behaviors_renamed: list,
    frame_limit_per_video: int = float('inf'),
    bin_info: dict = None,
    roi_number: int = None,
    animals_in_roi: list = None,
    single_output_resolution: tuple = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    config: VideoExportConfig = VideoExportConfig(),
    out_path: str = ".",
    roi_mode: str = "mousewise",
):
    """
    Generates one consolidated video per behavior, compiled from multiple experiments.
    """
    # Define the manual progress bar locally as it's a specific workaround
    def _loading_basic(current: int, total: int, bar_length: int = 68):
        if total == 0: return
        progress = (current + 1) / total
        filled_length = int(bar_length * progress)
        arrow = '>' if filled_length < bar_length else ''
        bar = f"[{'=' * filled_length}{arrow}{' ' * (bar_length - filled_length - len(arrow))}]"
        percent = f' {progress:.0%}'
        print(bar + percent, end='\r')
        if current == total - 1:
            print()

    output_path = Path(out_path)
    output_path.mkdir(parents=True, exist_ok=True)

    video_paths = coordinates.get_videos(full_paths=True)
    frame_rate = coordinates._frame_rate

    exp_ids_to_process = _filter_videos_by_condition(
        coordinates, list(behavior_dict.keys()), exp_conditions
    )
    available_behaviors, behaviors_to_process, behaviors_renamed_to_process  = _determine_behaviors_to_process(behavior_dict, behaviors, behaviors_renamed)
    
    cluster_export_config = dataclasses.replace(
        config, display_behavior_names=False, display_counter=False,
        display_video_name=True, display_loading_bar=False,
    )

    for behavior in tqdm(behaviors_to_process, desc=f"{'Exporting behavior videos':<{PROGRESS_BAR_FIXED_WIDTH}}", unit=" video"):

        behavior_renamed = behaviors_renamed_to_process[behaviors_to_process.index(behavior)]        

        video_out_path = output_path / f"Behavior={behavior_renamed}_threshold={min_confidence}_{int(time.time())}.mp4"
        out = cv2.VideoWriter(
            str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate, single_output_resolution,
        )

        try:
            print("") # Newline to separate tqdm from manual bar
            
            total_exps = len(exp_ids_to_process)
            for i, exp_id in enumerate(exp_ids_to_process):
                _loading_basic(i, total_exps)
                
                cur_tab = get_dt(behavior_dict, exp_id)
                if type(cur_tab)==np.ndarray:
                    cur_tab=pd.DataFrame(cur_tab,columns=available_behaviors)  

                behavior_mask, confidence = _get_behavior_mask_and_confidence(
                    cur_tab, behavior, available_behaviors
                )

                confidence_indices = np.ones(len(behavior_mask), dtype=bool)
                confidence_indices = deepof.utils.filter_short_bouts(
                    behavior_mask.astype(float), confidence, confidence_indices,
                    min_confidence, min_bout_duration
                )
                
                frames_passing_confidence = np.where(behavior_mask & confidence_indices)[0]

                if bin_info is not None and roi_number is not None:
                    behavior_for_roi = behavior if roi_mode == "behaviorwise" else None
                    frames_in_roi = deepof.visuals_utils.get_behavior_frames_in_roi(
                        behavior=behavior_for_roi, local_bin_info=bin_info[exp_id],
                        animal_ids=animals_in_roi,
                    )
                    selected_frames = np.intersect1d(
                        frames_passing_confidence, frames_in_roi, assume_unique=True
                    )
                else:
                    selected_frames = frames_passing_confidence
                
                if len(selected_frames) > 0:
                    cap = cv2.VideoCapture(video_paths[exp_id])
                    output_annotated_video(
                        coordinates=coordinates, experiment_id=exp_id, tab=cur_tab,
                        behaviors=[behavior], config=cluster_export_config,
                        frames=selected_frames, cap=cap, out=out,
                        v_width=single_output_resolution[0], v_height=single_output_resolution[1],
                        frame_limit=frame_limit_per_video, out_path=output_path,
                        behaviors_renamed=[behavior_renamed],
                    )
        finally:
            out.release()
        
        clear_output()
    
    if hasattr(deepof.visuals_utils.get_behavior_frames_in_roi, '_warning_issued'):
        deepof.visuals_utils.get_behavior_frames_in_roi._warning_issued = False


def output_videos_per_cluster_old(
    coordinates: coordinates,
    exp_conditions: dict,
    behavior_dict: table_dict,
    behaviors: Union[str,list],
    behavior_names: list,
    frame_limit_per_video: int = np.inf,
    bin_info: dict = None,
    roi_number: int = None,
    animals_in_roi: list = None,
    single_output_resolution: tuple = None,
    min_confidence: float = 0.0,
    min_bout_duration: int = None,
    config: VideoExportConfig = VideoExportConfig(),
    out_path: str = ".",
    roi_mode: str = "mousewise",
): # pragma: no cover
    """Given a list of videos, and a list of soft counts per video, outputs a video for each cluster.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        exp_conditions (dict): if provided, data coming from a particular condition is used. If not, all conditions are exported. If a dictionary with more than one entry is provided, the intersection of all conditions (i.e. male, stressed) is used.
        behavior_dict: table_dict containing data tables with behavior information (presence or absence of behaviors (columns) for each frame (rows))
        behaviors (Union[str,list]): list of behaviors to annotate
        behavior_names (list): Names of behaviors, potentially renamed by user.
        frame_limit_per_video: number of frames to render per video.
        bin_info (dict): dictionary containing indices to plot for all experiments
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of teh ROI get excluded 
        single_output_resolution: if single_output is provided, this is the resolution of the output video.
        min_confidence: minimum confidence threshold for a frame to be considered part of a cluster.
        min_bout_duration: minimum duration of a bout to be considered.
        display_time (bool): Displays current time in top left corner of the video frame
        display_arena (bool): Displays arena for each video.
        display_markers (bool): Displays mouse body parts on top of the mice. 
        display_mouse_labels (bool): Displays identities of the mice
        out_path: path to the output directory.
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
    """

    #manual laoding bar for inner loop
    def _loading_basic(current, total, bar_length=68):
        progress = (current + 1) / total
        filled_length = int(bar_length * progress)
        arrow = '>' if filled_length < bar_length else ''
        bar = '[' + '=' * filled_length + arrow + ' ' * (bar_length - filled_length - len(arrow)) + ']'
        percent = f' {progress:.0%}'
        print(bar + percent, end='\r')
        if current == total - 1:
            print()  # Newline when complete

    def filter_experimental_conditions(
        coordinates: coordinates, videos: list, conditions: list
    ):
        """Return a list of videos that match the provided experimental conditions."""
        filtered_videos = videos

        for condition, state in conditions.items():

            filtered_videos = [
                video
                for video in filtered_videos
                if state
                == np.array(
                    coordinates.get_exp_conditions[re.findall("(.+)DLC", video)[0]][
                        condition
                    ]
                )
            ]

        return filtered_videos

    video_paths = filter_experimental_conditions(
        coordinates, coordinates.get_videos(full_paths=True), exp_conditions
    )
    frame_rate=coordinates._frame_rate

    meta_info=get_dt(behavior_dict,list(behavior_dict.keys())[0],only_metainfo=True)
    if isinstance(behaviors, str):
        behaviors = [behaviors]
    elif meta_info.get('columns') is not None and behaviors is None:
        behaviors = meta_info['columns']
    elif meta_info.get('columns') is not None:
        behaviors =[behavior for behavior in behaviors if behavior in meta_info['columns']]
    elif behaviors is not None:
        behaviors = [behavior for behavior in behaviors if behavior in behavior_names] 
    else:
        behaviors=behavior_names

    # Iterate over all clusters, and output a masked video for each
    for cur_behavior in tqdm(behaviors, desc=f"{'Exporting behavior videos':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="video"):
    
        #creates a new line to ensure that the outer loading bar does not get overwritten by the inner one
        print("")

        out = cv2.VideoWriter(
            os.path.join(
                out_path,
                "Behavior={}_threshold={}_{}.mp4".format(
                    cur_behavior, min_confidence, calendar.timegm(time.gmtime())
                ),
            ),
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            single_output_resolution,
        )
      
        bar_len=len(behavior_dict.keys())
        #this loop uses a manual loading bar as tqdm does not work here for some reason
        for i, key in enumerate(behavior_dict.keys()):

            _loading_basic(i, bar_len)
            
            cur_tab=copy.deepcopy(get_dt(behavior_dict, key))

            # Turn numpy arrays (in case of unsupervised data) into dataframes
            if type(cur_tab)==np.ndarray:
                cur_tab=pd.DataFrame(cur_tab,columns=behavior_names) 

                # Get positions at which the current cluster is teh most likely one 
                max_entry_columns = cur_tab.idxmax(axis=1)
                behavior_mask = (cur_behavior == max_entry_columns)
                idx = (cur_behavior == max_entry_columns)
                confidence = cur_tab[cur_behavior]    
        
            else:
                cur_tab.columns = behavior_names

                # Get positions at which teh current behaviro occurs
                behavior_mask = cur_tab[cur_behavior]>0.1
                idx = cur_tab[cur_behavior]>0.1
                confidence = cur_tab[cur_behavior]  

            # Convert mask to contain either selected behavior or nothing for each frame     
            behavior_mask = behavior_mask.astype(str)
            behavior_mask[idx]=str(cur_behavior)
            behavior_mask[~idx]=""
          
            # Get hard counts and confidence estimates per cluster
            confidence_indices = np.ones(behavior_mask.shape[0], dtype=bool)

            # Given a frame mask, output a subset of the given video to disk, corresponding to a particular cluster
            cap = cv2.VideoCapture(video_paths[key])
            v_width, v_height = single_output_resolution

            # Compute confidence mask, filtering out also bouts that are too short
            confidence_indices = deepof.utils.filter_short_bouts(
                idx.astype(float),
                confidence,
                confidence_indices,
                min_confidence,
                min_bout_duration,
            )
            confidence_mask = (behavior_mask == str(cur_behavior)) & confidence_indices

            # get frames for current video
            frames = None
            if bin_info is not None:
                if roi_number is not None:
                    if roi_mode == "behaviorwise":
                        behavior_in=behaviors[0]
                    else:
                        behavior_in=None
                    frames=deepof.visuals_utils.get_behavior_frames_in_roi(behavior=behavior_in, local_bin_info=bin_info[key], animal_ids=animals_in_roi)
                else:
                    frames=bin_info[key]["time"]          

            selected_frames=frames[confidence_mask[frames]]
            
            # Sdjust config for cluster export
            config.display_behavior_names=False
            config.display_counter=False
            config.display_video_name=True
            config.display_loading_bar=False

            if len(selected_frames)>0:
                output_annotated_video(
                    coordinates=coordinates,
                    experiment_id=key,                
                    tab=cur_tab,
                    behaviors=[cur_behavior],
                    config=config,
                    frames=selected_frames,
                    cap=cap,
                    out=out,
                    v_width=v_width,
                    v_height=v_height,
                    frame_limit=frame_limit_per_video,
                    out_path=out_path,
                )
        

        out.release()
        #to not flood the output with loading bars
        clear_output()
    deepof.visuals_utils.get_behavior_frames_in_roi._warning_issued = False


def _prepare_behavior_dataframe(
    tab: pd.DataFrame, behaviors: List[str], behavior_renamed: List[str],
) -> pd.DataFrame:
    """
    Creates a DataFrame where cells contain the behavior name if the condition
    is met (score > 0.1), and an empty string otherwise. This is an efficient
    way to look up the active behavior string for a given frame.
    """
    behavior_df = pd.DataFrame(index=tab.index, columns=behaviors, data="")
    for behavior, behavior_renamed in zip(behaviors, behavior_renamed):
        mask = tab[behavior] > 0.1
        behavior_df.loc[mask, behavior] = behavior_renamed
    return behavior_df


def _draw_arena(
    frame: np.ndarray,
    arena_type: str,
    arena_params: Any,
    params: VideoExportProps
):
    """Draws the arena boundaries on the frame."""
    if arena_type.startswith("circular"):
        cv2.ellipse(
            img=frame,
            center=np.round(arena_params[0]).astype(int),
            axes=np.round(arena_params[1]).astype(int),
            angle=arena_params[2],
            startAngle=0,
            endAngle=360,
            color=params.arena_color,
            thickness=params.arena_thickness,
        )
    elif arena_type.startswith("polygonal"):
        cv2.polylines(
            img=frame,
            pts=[np.array(arena_params, dtype=np.int32)],
            isClosed=True,
            color=params.arena_color,
            thickness=params.arena_thickness,
        )


def _draw_markers(
    frame: np.ndarray,
    coords_at_frame: pd.Series,
    animal_ids: List[str],
    params: VideoExportProps
):
    """Draws body part markers on the frame."""
    for bpart in coords_at_frame.index.get_level_values(0).unique():
        if not np.isnan(coords_at_frame.loc[(bpart, "x")]):
            # Determine color by finding which animal this body part belongs to
            color_idx = [bpart.startswith(id) for id in animal_ids].index(True)
            color = BODYPART_COLORS[color_idx]
            cv2.circle(
                frame,
                (int(coords_at_frame.loc[(bpart, "x")]), int(coords_at_frame.loc[(bpart, "y")])),
                radius=params.marker_radius,
                color=color,
                thickness=-1,  # Filled circle
            )


def _draw_mouse_labels(
    frame: np.ndarray,
    coords_at_frame: pd.Series,
    animal_ids: List[str],
    params: VideoExportProps,
):
    """Draws labels with mouse identities on the frame."""
    for mouse_id in animal_ids:
        center_bpart = f"{mouse_id}_Center"
        if center_bpart in coords_at_frame.index and not np.isnan(coords_at_frame.loc[(center_bpart, "x")]):
            mouse_pos = (
                int(coords_at_frame.loc[(center_bpart, "x")]),
                int(coords_at_frame.loc[(center_bpart, "y")])
            )
            (text_w, text_h), baseline = cv2.getTextSize(
                mouse_id, params.font, params.font_scale, params.thickness
            )
            color_idx = animal_ids.index(mouse_id)
            color = BODYPART_COLORS[color_idx]

            top_left = (mouse_pos[0], mouse_pos[1] - text_h - params.padding)
            bottom_right = (mouse_pos[0] + text_w, mouse_pos[1] + baseline)
            
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
            cv2.putText(
                frame, mouse_id, mouse_pos, params.font,
                params.font_scale, params.text_color, params.thickness
            )


def _draw_behavior_info(
    frame: np.ndarray,
    frame_idx: int,
    v_width: int,
    behavior_df: pd.DataFrame,
    behaviors: List[str],
    behavior_colors: List[str],
    behavior_counters: np.ndarray,
    widest_text_size: Tuple[int, int],
    shift_name_box: bool,
    config: VideoExportConfig,
    params: VideoExportProps,
    frame_rate: int
):
    """Draws the behavior names, background boxes, and counters."""
    text_width, text_height = widest_text_size
    y_start = 10 + text_height
    y_step = 0
    
    for i, behavior in enumerate(behaviors):
        behavior_text = behavior_df.iloc[frame_idx][i]
        if behavior_text:
            box_y = y_start + y_step
            # Draw background rectangle
            top_left = (v_width - text_width - params.padding * 2, box_y - text_height - params.padding)
            bottom_right = (v_width - params.padding, box_y + params.padding)
            cv2.rectangle(frame, top_left, bottom_right, hex_to_BGR(behavior_colors[i]), -1)

            # Update and format text with counter if enabled
            if config.display_counter:
                behavior_counters[i] += 1
                time_str = deepof.visuals_utils.seconds_to_time(behavior_counters[i] / frame_rate, cut_milliseconds=False)[3:11]
                behavior_text += f' {time_str}'
            
            # Draw behavior text
            text_pos = (v_width - text_width - params.padding, box_y)
            cv2.putText(frame, behavior_text, text_pos, params.font, params.font_scale, params.text_color, params.thickness)
            
            if shift_name_box:
                y_step += int(text_height * 2)


def output_annotated_video(
    coordinates: coordinates,
    experiment_id: str,
    tab: np.ndarray,
    behaviors: List[str],
    config: VideoExportConfig = VideoExportConfig(),
    frames: np.array = None,
    cap: Any = None,
    out: Any = None,
    v_width: int = None,
    v_height: int = None,
    frame_limit: int = float('inf'),
    out_path: Path = Path("."),
    behaviors_renamed: List = None,
):
    """
    Generates a video with frames annotated with specified behaviors and other metadata.

    Args:
        coordinates: Coordinates object for the project, used to access video paths and metadata.
        experiment_id: ID of the experiment to export.
        tab: DataFrame with behavior probabilities/scores per frame.
        behaviors: A list of behavior names (columns in `tab`) to annotate.
        config: A dataclass object specifying which annotations to display.
        frames: An array of specific frame indices to include in the output video.
        cap: An existing cv2.VideoCapture object. If None, one will be created.
        out: An existing cv2.VideoWriter object. If None, one will be created.
        v_width: Desired output video width. Defaults to source video width.
        v_height: Desired output video height. Defaults to source video height.
        frame_limit: Maximum number of frames to process.
        out_path: The directory where the output video will be saved.
        behaviors_renamed: List of updated behavior names for display
    """
    video_path = Path(coordinates.get_videos(full_paths=True)[experiment_id])
    video_name_stem = video_path.stem

    # --- Behavior & Frame Preparation ---
    shift_name_box = True
    if behaviors is None:
        # If no behaviors are specified, check if it's a single-behavior-per-frame scenario
        if not (tab.sum(axis=1) > 1.9).any():
            behaviors = list(tab.columns)
            shift_name_box = False  # Display all behaviors simultaneously
        else:
            behaviors = list(tab.columns)
            #raise ValueError("A list of 'behaviors' must be provided for multi-label annotations.")

    if behaviors_renamed is None or len(behaviors_renamed) != len(behaviors):
        behaviors_renamed=behaviors

    behavior_df = _prepare_behavior_dataframe(tab, behaviors, behaviors_renamed)

    
    cur_coords = get_dt(coordinates._tables, experiment_id)

    # Filter frames to ensure they are within bounds and respect the frame limit
    if frames is not None:
        frames = frames[frames < len(behavior_df)]
        if len(frames) > frame_limit:
            frames = frames[:frame_limit]
    else: # If no frames are given, process all frames up to the limit
        total_frames = min(len(behavior_df), frame_limit)
        frames = np.arange(total_frames)

    # --- Video I/O Setup ---
    cap_is_external = cap is not None
    out_is_external = out is not None
    
    if not cap_is_external:
        cap = cv2.VideoCapture(str(video_path))
    
    # Determine video dimensions
    resize_frame = v_width is not None or v_height is not None
    if v_width is None:
        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if v_height is None:
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_rate = coordinates._frame_rate
    if not out_is_external:
        video_out_path = os.path.join(out_path, f"{video_name_stem}_annotated_{int(time.time())}.mp4")
        out = cv2.VideoWriter(
            str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (v_width, v_height)
        )

    # --- Drawing & Annotation Parameter Setup ---
    params = VideoExportProps()
    bg_colors = deepof.visuals_utils.get_behavior_colors(behaviors, tab)
    behavior_counters = np.zeros(len(behaviors))

    # Pre-calculate text size for layout purposes
    widest_text = max(behaviors_renamed, key=len)
    if config.display_counter:
        widest_text += ' 00:00.00'  # Add placeholder for counter time
    (text_w, text_h), _ = cv2.getTextSize(widest_text, params.font, params.font_scale, params.thickness)
    widest_text_size = (text_w, text_h)

    # Pre-calculate scaled coordinates and arena parameters if needed
    scaling_ratio = coordinates._scales[experiment_id][2] / coordinates._scales[experiment_id][3]
    if config.display_arena:
        arena_params = coordinates._arena_params[experiment_id]
        if "polygonal" in coordinates._arena:
            scaled_arena_params = np.array(arena_params) * scaling_ratio
        elif "circular" in coordinates._arena:
            scaled_arena_params = (
                tuple(np.array(arena_params[0]) * scaling_ratio),
                tuple(np.array(arena_params[1]) * scaling_ratio),
                arena_params[2],
            )
    if config.display_markers or config.display_mouse_labels:
        scaled_coords = cur_coords * scaling_ratio
    
    # --- Main Processing Loop ---
    try:
        frame_indices = tqdm(range(len(frames)), desc=f"{'Exporting behavior video':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="Frame") \
                        if config.display_loading_bar else range(len(frames))

        for i in frame_indices:
            frame_idx = frames[i]
            
            # Efficiently seek frames only when necessary (not consecutive)
            if i == 0 or frames[i] - frames[i-1] != 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}. Stopping.")
                break

            # Annotations drawn BEFORE resizing
            if config.display_arena:
                _draw_arena(frame, coordinates._arena, scaled_arena_params, params)
            
            if config.display_markers:
                _draw_markers(frame, scaled_coords.iloc[frame_idx], coordinates._animal_ids, params)
            
            if config.display_mouse_labels:
                _draw_mouse_labels(frame, scaled_coords.iloc[frame_idx], coordinates._animal_ids, params)

            # Resize frame if custom dimensions are provided
            if resize_frame:
                frame = cv2.resize(frame, (v_width, v_height))
            
            # Annotations drawn AFTER resizing
            if config.display_behavior_names:
                _draw_behavior_info(
                    frame, frame_idx, v_width, behavior_df, behaviors_renamed, bg_colors,
                    behavior_counters, widest_text_size, shift_name_box, config, params, frame_rate
                )
            
            if config.display_video_name:
                cv2.putText(frame, video_name_stem, (15, v_height - 15),
                            params.font, 0.75, params.text_color, 2)
            
            if config.display_time:
                time_text = f"time: {deepof.visuals_utils.seconds_to_time(frame_idx / frame_rate)}"
                pos = (params.padding, 10 + text_h)
                cv2.putText(frame, time_text, pos, params.font, params.font_scale * 1.5, params.outline_color, params.thickness + 2)
                cv2.putText(frame, time_text, pos, params.font, params.font_scale * 1.5, params.text_color, params.thickness)

            out.write(frame)

    finally:
        # Ensure all resources are released
        if not cap_is_external:
            cap.release()
        if not out_is_external:
            out.release()
        cv2.destroyAllWindows()


def output_annotated_video_old(
    coordinates: coordinates,
    experiment_id: str,
    tab: np.ndarray,
    behaviors: list,
    frames: np.array = None,
    display_behavior_names: bool = True,
    display_video_name: bool = False,
    display_time: bool = False,
    display_counter: bool = False,
    display_arena: bool = False,
    display_markers: bool = False,
    display_mouse_labels: bool = False,
    display_loading_bar: bool = True,
    cap: Any = None,
    out: Any = None,
    v_width: int = None,
    v_height: int = None,
    frame_limit: int = np.inf,
    out_path: str = ".",
): # pragma: no cover
    """Given a video, and soft_counts per frame, outputs a video with the frames annotated with the cluster they belong to.

    Args:
        coordinates (coordinates): coordinates object for the current project. Used to get video paths.
        experiment_id: id of the experiment to export
        soft_counts: soft cluster assignments for a specific video
        behavior (str): Behavior or Cluster to that gets exported. If none is given, all Clusters get exported for softcounts and only nose2nose gets exported for supervised annotations.
        frames: frames that should be exported.
        display_behavior_names (bool): Display the names of teh respective behaviors
        display_video_name (bool): Display teh name of the video
        display_time (bool): Displays current time in top left corner of the video frame
        display_counter (bool): Displays event counter for each displayed event.   
        display_arena (bool): Displays arena for each video.
        display_markers (bool): Displays mouse body parts on top of the mice.
        display_mouse_labels (bool): Displays identities of the mice
        display_loading_bar (bool): Displays the laoding bar during writing of the video
        cap (Any): video capture object for reading the video, can be provided. Will be created from video at experiment_id otherwise.
        out (Any): video capture object for writing teh video, can be provided.
        v_width (int): video width
        v_height (int): video height    
        frame_limit (int): Maximum number of frames that can be included in a video. No limit per default
        out_path: out_path: path to the output directory.

    """

    video_path=coordinates.get_videos(full_paths=True)[experiment_id]
    # for display
    re_path = re.findall(r".+[/\\]([^/.]+?)(?=\.|DLC)", video_path)[0]

    # if every frame has only one distinct behavior assigned to it, plot all behaviors
    shift_name_box=True
    if behaviors is None and not (np.sum(tab, 1)>1.9).any():
        behaviors = list(tab.columns)
        shift_name_box=False
    elif behaviors is None:
        raise ValueError(
            "Cannot accept no behavior for supervised annotations!"
        )
    # create behavior_df that lists in which frames each behavior occurs
    behavior_df = tab[behaviors]>0.1
    idx = tab[behaviors]>0.1 #OK, I know this looks weird, but it is actually not a bug and works
    behavior_df = behavior_df.astype(str)
    behavior_df[idx]=behaviors
    behavior_df[~idx]=""
    # Else if every frame has only one distinct behavior assigned to it, annotate all behaviors

    cur_coords=get_dt(coordinates._tables,experiment_id)
    
    # Ensure that no frames are requested that are outside of the provided data
    if np.max(frames) >= len(behavior_df):
        frames = np.where(frames<len(behavior_df))[0]
    if len(frames) >= frame_limit:
        frames = frames[0:frame_limit]

    # Given a frame mask, output a subset of the given video to disk, corresponding to a particular cluster
    if cap is None:
        cap = cv2.VideoCapture(video_path)

    # Get width and height of current video
    resize_frame=True    
    if v_width is None and v_height is None:
        resize_frame=False    
    if v_width is None:
        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if v_height is None:  
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  

    video_out = os.path.join(
        out_path,
        os.path.split(video_path)[-1].split(".")[0] 
        + "_annotated_{}.mp4".format(calendar.timegm(time.gmtime())),
    )

    frame_rate=coordinates._frame_rate
    if out is None:
        out = cv2.VideoWriter(
            video_out, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (v_width, v_height)
        )

    # Prepare text
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    thickness = 1
    text_widths=[cv2.getTextSize(behavior, font, font_scale, thickness)[0][0] for behavior in behaviors]
    widest_text=behaviors[np.argmax(text_widths)]
    if display_counter:
        behavior_array=np.zeros(len(behaviors))
        widest_text=widest_text+' 00:00.00'
    if display_arena:
        arena_params = coordinates._arena_params[experiment_id]
        # scale arena_params back o video res
        scaling_ratio = coordinates._scales[experiment_id][2]/coordinates._scales[experiment_id][3]
        if "polygonal" in coordinates._arena:
            arena_params=np.array(arena_params)*scaling_ratio
        elif "circular" in coordinates._arena:
            # scale from mm to original pixel resolution
            arena_params=(tuple(np.array(arena_params[0])*scaling_ratio),tuple(np.array(arena_params[1])*scaling_ratio),arena_params[2])
    if display_markers or display_mouse_labels:
        scaling_ratio = coordinates._scales[experiment_id][2]/coordinates._scales[experiment_id][3]
        cur_coords=cur_coords*scaling_ratio
    (text_width, text_height), baseline = cv2.getTextSize(widest_text, font, font_scale, thickness)
    (text_width_time, text_height_time), baseline = cv2.getTextSize("time: 00:00:00", font, font_scale, thickness)
    x = 10  # 10 pixels from left
    y = 10 + text_height_time  # 10 pixels from top (accounting for text height)
    padding = 5
    bg_color = deepof.visuals_utils.get_behavior_colors(behaviors, tab)

    diff_frames = np.diff(frames)
    for i in (tqdm(range(len(frames)), desc=f"{'Exporting behavior video':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="Frame")) if display_loading_bar else range(len(frames)):

        if i == 0 or diff_frames[i-1] != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()

        if ret == False or cap.isOpened() == False:
            break

        try:

            #####
            # drawn annotations before rescaling
            #####
            if display_arena:
                
                if coordinates._arena.startswith("circular"):
                    # Print arena for debugging
                    cv2.ellipse(
                        img=frame,
                        center=np.round(arena_params[0]).astype(int),
                        axes=np.round(arena_params[1]).astype(int),
                        angle=arena_params[2],
                        startAngle=0,
                        endAngle=360,
                        color=(40, 86, 236),
                        thickness=3,
                    )

                elif coordinates._arena.startswith("polygonal"):

                    # Draw polygon
                    cv2.polylines(
                        img=frame,
                        pts=[np.array(arena_params, dtype=np.int32)],
                        isClosed=True,
                        color=(40, 86, 236),
                        thickness=3,
                    )


            if display_markers:
                # Print body parts for debuging
                for bpart in cur_coords.columns.levels[0]:
                    pass
                    if not np.isnan(cur_coords[bpart]["x"][frames[i]]):
                        cv2.circle(
                            frame,
                            (int(cur_coords[bpart]["x"][frames[i]]), int(cur_coords[bpart]["y"][frames[i]])),
                            radius=3,
                            color=(
                                BODYPART_COLORS[[bpart.startswith(id) for id in coordinates._animal_ids].index(True)] #first index of animal id that fits
                            ),
                            thickness=-1,
                        )
                
            
            if display_mouse_labels:

                for bpart in cur_coords.columns.levels[0]:

                    if bpart.endswith("Center") and not np.isnan(cur_coords[bpart]["x"][frames[i]]):

                        mouse_id=[id for id in coordinates._animal_ids if bpart.startswith(id)][0]
                        mouse_pos=(int(cur_coords[bpart]["x"][frames[i]]), int(cur_coords[bpart]["y"][frames[i]]))
                        (id_text_width, id_text_height), id_baseline = cv2.getTextSize(mouse_id, font, font_scale, thickness)

                        cv2.rectangle(frame, 
                            (mouse_pos[0], mouse_pos[1] - id_text_height - padding),  # Top-left corner
                            (mouse_pos[0] + id_text_width, mouse_pos[1] + id_baseline),  # Bottom-right corner
                            (BODYPART_COLORS[[bpart.startswith(id) for id in coordinates._animal_ids].index(True)]),  # Blue color (BGR format)
                            -1)  # Filled rectangle
                        
                        cv2.putText(frame, mouse_id, mouse_pos, font, font_scale, (255, 255, 255), thickness)



            #####
            #resize frame if resolution is specified, needs to be done after drawn annotations but before written annotations
            #####
            if resize_frame:
                frame = cv2.resize(frame, [v_width, v_height])


            #####
            # written annotations after rescaling
            #####
            if display_behavior_names:
                
                ystep=0
                for z, behavior in enumerate(behaviors):
                    if len(behavior_df[behavior][frames[i]])>0:
                        cv2.rectangle(frame, 
                            (v_width - text_width - x , y - text_height - padding +ystep),  # Top-left corner
                            (v_width - padding, y + baseline +ystep),  # Bottom-right corner
                            hex_to_BGR(bg_color[z]),  # Blue color (BGR format)
                            -1)  # Filled rectangle

                        behavior_text=str(behavior_df[behavior][frames[i]])
                        if display_counter:
                            behavior_array[z]=behavior_array[z]+1
                            behavior_text = behavior_text + ' ' + deepof.visuals_utils.seconds_to_time(behavior_array[z]/frame_rate, cut_milliseconds=False)[3:11]

                        # Draw white main text
                        cv2.putText(frame, str(behavior_text), (v_width - text_width - padding, y+ystep), font, font_scale, (255, 255, 255), thickness)
                    if shift_name_box:
                        ystep=ystep+int(text_height*2)


            if display_video_name:
                cv2.putText(
                    frame,
                    re_path,
                    (int(v_width * 0.3 / 10), int(v_height / 1.05)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )


            if display_time:

                disp_time = "time: " + deepof.visuals_utils.seconds_to_time(frames[i]/frame_rate)
                # Draw black outline
                cv2.putText(frame, disp_time, (x, y), font, font_scale*1.5, (0, 0, 0), thickness + 2)
                # Draw white main text
                cv2.putText(frame, disp_time, (x, y), font, font_scale*1.5, (255, 255, 255), thickness)

            out.write(frame)
        except IndexError:
            ret = False

    cap.release()
    cv2.destroyAllWindows()

    #writevideo = FFMpegWriter(fps=frame_rate)
    #animation.save(save, writer=writevideo)

    return None