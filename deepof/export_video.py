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
    supervised_export: bool = True


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
    all_exp_conditions = coordinates.get_exp_conditions
    assert isinstance(conditions,dict), "Error! To select one experiment condition for export you must enter a dictionary {condition name: experiment condition [optional:, second condition name: second experiment condition, ...]}"
    list(conditions.values())
    filtered_conditions = [
        exp_id for exp_id in experiment_ids if
        all([all_exp_conditions.get(exp_id, {})[cond][0] == state
        for cond, state in conditions.items()])
    ]
    assert len(filtered_conditions)>0, f"No conditions are left after filtering. Make sure that your dictionary keys are among: {list(all_exp_conditions[list(all_exp_conditions.keys())[0]].columns)} and condition values are spelled correctly!" 
    return filtered_conditions

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
    video_export_config: VideoExportConfig = VideoExportConfig(),
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
    
    video_export_config = dataclasses.replace(
        video_export_config, display_behavior_names=False, display_counter=False,
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
                # Data frame conversion as soft counts get safed as numpy arrays
                if type(cur_tab)==np.ndarray:
                    cur_tab=pd.DataFrame(cur_tab,columns=available_behaviors)  

                # Behavior mask will function as behavior dataframe after float conversion
                behavior_mask, confidence = deepof.utils.get_behavior_mask_and_confidence(
                    cur_tab, behavior, video_export_config.supervised_export
                )
                behavior_mask_np=np.array(behavior_mask).squeeze()
                confidence=np.array(confidence).squeeze()

                confidence_indices = np.ones(len(behavior_mask), dtype=bool)
                confidence_indices = deepof.utils.filter_short_bouts(
                   behavior_mask_np, 
                    confidence, 
                    confidence_indices,
                    min_confidence, 
                    min_bout_duration
                )
                
                frames_passing_confidence = np.where(behavior_mask_np & confidence_indices)[0]

                if bin_info is not None:
                    
                    if roi_number is not None:
                        behavior_for_roi = behavior if roi_mode == "behaviorwise" else None
                        frames_in_roi = deepof.utils.get_behavior_frames_in_roi(
                            behavior=behavior_for_roi, local_bin_info=bin_info[exp_id],
                            animal_ids=animals_in_roi,
                        )
                        selected_frames = np.intersect1d(
                            frames_passing_confidence, frames_in_roi, assume_unique=True
                        )
                    else:
                        selected_frames = np.intersect1d(
                            frames_passing_confidence, bin_info[exp_id]["time"], assume_unique=True
                        )

                else:
                    selected_frames = frames_passing_confidence
                
                if len(selected_frames) > 0:
                    cap = cv2.VideoCapture(video_paths[exp_id])
                    output_annotated_video(
                        coordinates=coordinates, experiment_id=exp_id, tab=behavior_mask.astype(float),
                        behaviors=[behavior], video_export_config=video_export_config,
                        frames=selected_frames, cap=cap, out=out,
                        v_width=single_output_resolution[0], v_height=single_output_resolution[1],
                        frame_limit=frame_limit_per_video, out_path=output_path,
                        behaviors_renamed=[behavior_renamed],
                    )
        finally:
            out.release()
        
        clear_output()
    
    if hasattr(deepof.utils.get_behavior_frames_in_roi, '_warning_issued'):
        deepof.utils.get_behavior_frames_in_roi._warning_issued = False


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
    if isinstance(arena_params, Tuple): #Circular (legacy)  arena_type.startswith("circular"):
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
    elif isinstance(arena_params, np.ndarray): #Polygonal
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
    line_step = int(text_height * 2)

    for i, behavior in enumerate(behaviors):
        behavior_text = behavior_df.iloc[frame_idx][i]
        if behavior_text:
            # Fixed vertical position per behavior when shifting is enabled
            box_y = y_start + (i * line_step if shift_name_box else 0)

            # Draw background rectangle
            top_left = (v_width - text_width - params.padding * 2, box_y - text_height - params.padding)
            bottom_right = (v_width - params.padding, box_y + params.padding)
            cv2.rectangle(frame, top_left, bottom_right, hex_to_BGR(behavior_colors[i]), -1)

            # Update and format text with counter if enabled
            if config.display_counter:
                behavior_counters[i] += 1
                time_str = deepof.visuals_utils.seconds_to_time(
                    behavior_counters[i] / frame_rate, cut_milliseconds=False
                )[3:11]
                behavior_text += f' {time_str}'

            # Draw behavior text
            text_pos = (v_width - text_width - params.padding, box_y)
            cv2.putText(frame, behavior_text, text_pos, params.font, params.font_scale, params.text_color, params.thickness)


def output_annotated_video(
    coordinates: coordinates,
    experiment_id: str,
    tab: pd.DataFrame,
    behaviors: List[str],
    video_export_config: VideoExportConfig = VideoExportConfig(),
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
        video_export_config: A dataclass object specifying video export information (what to display, export mode).
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

    # Rename behaviors to user given names 
    behavior_df = _prepare_behavior_dataframe(tab=tab, behaviors=behaviors,behavior_renamed=behaviors_renamed)
    
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
    bg_colors = deepof.visuals_utils.get_behavior_colors(behaviors, behavior_df, coordinates._custom_behaviors)
    behavior_counters = np.zeros(len(behaviors))

    # Pre-calculate text size for layout purposes
    widest_text = max(behaviors_renamed, key=len)
    if video_export_config.display_counter:
        widest_text += ' 00:00.00'  # Add placeholder for counter time
    (text_w, text_h), _ = cv2.getTextSize(widest_text, params.font, params.font_scale, params.thickness)
    widest_text_size = (text_w, text_h)

    # Pre-calculate scaled coordinates and arena parameters if needed
    scaling_ratio = coordinates._scales[experiment_id][2] / coordinates._scales[experiment_id][3]
    if video_export_config.display_arena:
        arena_params = coordinates._arena_params[experiment_id]
        if isinstance(arena_params, np.ndarray): # "polygonal" in coordinates._arena:
            scaled_arena_params = np.array(arena_params) * scaling_ratio
        elif isinstance(arena_params, Tuple):
            scaled_arena_params = (
                tuple(np.array(arena_params[0]) * scaling_ratio),
                tuple(np.array(arena_params[1]) * scaling_ratio),
                arena_params[2],
            )
    if video_export_config.display_markers or video_export_config.display_mouse_labels:
        scaled_coords = cur_coords * scaling_ratio

    # --- Main Processing Loop ---
    try:
        frame_indices = tqdm(range(len(frames)), desc=f"{'Exporting behavior video':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="Frame") \
                        if video_export_config.display_loading_bar else range(len(frames))

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
            if video_export_config.display_arena:
                _draw_arena(frame, coordinates._arena, scaled_arena_params, params)
            
            if video_export_config.display_markers:
                _draw_markers(frame, scaled_coords.iloc[frame_idx], coordinates._animal_ids, params)
            
            if video_export_config.display_mouse_labels:
                _draw_mouse_labels(frame, scaled_coords.iloc[frame_idx], coordinates._animal_ids, params)

            # Resize frame if custom dimensions are provided
            if resize_frame:
                frame = cv2.resize(frame, (v_width, v_height))
            
            # Annotations drawn AFTER resizing
            if video_export_config.display_behavior_names:
                _draw_behavior_info(
                    frame, frame_idx, v_width, behavior_df, behaviors_renamed, bg_colors,
                    behavior_counters, widest_text_size, shift_name_box, video_export_config, params, frame_rate
                )
            
            if video_export_config.display_video_name:
                cv2.putText(frame, video_name_stem, (15, v_height - 15),
                            params.font, 0.75, params.text_color, 2)
            
            if video_export_config.display_time:
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
