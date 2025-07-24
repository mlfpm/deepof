# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Functions and general utilities for the deepof package."""
import copy
import math
import os
from copy import deepcopy
from math import atan2, dist
from typing import Any, List, NewType, Tuple, Union
from dataclasses import dataclass


import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon
from tqdm import tqdm

from deepof.config import PROGRESS_BAR_FIXED_WIDTH, ROI_COLORS, IMG_H_MAX, IMG_W_MAX
from deepof.data_loading import get_dt, save_dt, _suppress_warning
import deepof.data
import deepof.utils



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


# CONNECTIVITY AND GRAPH REPRESENTATIONS


def get_arenas(
    coordinates: coordinates,
    tables: table_dict,
    arena: str,
    arena_dims: int,
    number_of_rois: int,
    segmentation_model_path: str,
    video_path: str,
    videos: list = None,
    debug: bool = False,
    test: bool = False,
):
    """Extract arena parameters from a project or coordinates object.

    Args:
        coordinates (coordinates): Coordinates object.
        tables (table_dict): TableDict object containing tracklets per animal.
        arena (str): Arena type (must be either "polygonal-manual", "circular-manual", "polygonal-autodetect", or "circular-autodetect").
        arena_dims (int): Arena dimensions.
        number_of_rois (int): number of behavior rois,
        segmentation_model_path (str): Path to segmentation model used for automatic arena detection.
        video_path (str): Path to folder with videos.
        videos (dict): Dictionary of videos to extract arena parameters from. Defaults to None (all videos are used).
        debug (bool): If True, a frame per video with the detected arena is saved. Defaults to False.
        test (bool): If True, the function is run in test mode. This means that instead of waiting for user-inputs fixed artifical user-inputs are used. Defaults to False.

    Returns:
        scales (dict): Dictionary of scaling information. Each scales object consists of:
            - x position of the center of arena in mm
            - y position of the center of the arena in mm
            - diameter of the arena (when circular) or length of first edge in pixels
            - diameter of the arena (when circular) or length of first edge in mm
        
        arena_params (dict): Dictionary of arena parameters. Each arena parameter object consists of:
            (when circular)
            - x position of the center of arena in pixel
            - y position of the center of the arena in pixel
            - x axis radii of the arena in pixel
            - y axis radii of the arena in pixel
            - angle of the elipse
            (when polygonal)
            - x and y positions of the polygon vertices in pixel

        video_resolution (dict): Dictionary of video resolutions. Each video resolution object consists of:
            - height of the video in pixel
            - width of the video in pixel

        

    """
    scales = {}
    arena_params = {}
    roi_dicts = {}
    video_resolution = {}
    list_of_rois=list(range(1,number_of_rois+1))

    get_arena = True
    propagate_rois=[]
    arena_dist = None
    


    def get_first_length(arena_corners):
        return math.dist(arena_corners[0], arena_corners[1])
    
        
    #set message for user
    if "polygon" in arena:

        multi_line_message = [
        "Note: The first line you draw will be used for scaling.",
        "This means that this line should correspond to the",
        "\"real world\" length of " + str(arena_dims) + " mm you set during",
        "project creation."
        ]
    else:
        multi_line_message = [
        "Note: The diameter of the arena will be used for scaling.",
        "This means that the diameter should correspond to the",
        "\"real world\" length of " + str(arena_dims) + " mm you set during",
        "project creation."
        ]

    if arena in ["polygonal-manual", "circular-manual"]:  # pragma: no cover

        display_message(multi_line_message)
                
        with tqdm(total=len(videos), desc=f"{'Detecting arenas':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="arena") as pbar:
            for vid_idx, key in enumerate(videos.keys()):
                  
                # let user draw arena
                arena_corners, roi_corners, arena_dist, h, w = extract_polygonal_arena_coordinates(
                    os.path.join(video_path, videos[key]),
                    arena,
                    vid_idx,
                    videos,
                    list_of_rois=list_of_rois,
                    get_arena=get_arena,
                    arena_dims=arena_dims,
                    norm_dist=arena_dist,
                    test=test,
                )

                # if no new arena was detected, skip future detections and set all new dictionary values to the previous entry
                if arena_corners is None:
                    get_arena = False
                    cur_arena_params = arena_params[list(arena_params.keys())[-1]]
                    cur_scales = scales[list(scales.keys())[-1]]
                    
                elif arena == "circular-manual":
                    cur_arena_params = fit_ellipse_to_polygon(arena_corners)
                    cur_scales=list(
                        np.array(
                            [
                                cur_arena_params[0][0]*(arena_dims/arena_dist),
                                cur_arena_params[0][1]*(arena_dims/arena_dist),
                                arena_dist
                            ]
                        )
                    ) + [arena_dims]
                elif arena == "polygonal-manual":
                    cur_arena_params = arena_corners
                    cur_scales=[
                        *(np.mean(cur_arena_params, axis=0)*(arena_dims/arena_dist)),
                        arena_dist,
                        arena_dims,
                    ]
                else:
                    raise(NotImplementedError)

                for roi in copy.copy(list_of_rois):
                    if roi_corners[roi] is None:
                        propagate_rois.append(roi)
                        list_of_rois.remove(roi)
                
                cur_roi_corners={}
                for roi in list(range(1,number_of_rois+1)):
                    if roi in propagate_rois:
                        cur_roi_corners[roi]=roi_dicts[list(roi_dicts.keys())[-1]][roi]
                    else:
                        cur_roi_corners[roi]=roi_corners[roi]                           

                scales[key] = cur_scales
                arena_params[key]=cur_arena_params
                roi_dicts[key]=cur_roi_corners
                video_resolution[key]=(h, w)
                pbar.update()

    elif arena in ["polygonal-autodetect", "circular-autodetect"]:

        # Open GUI for manual labelling of two scaling points in the first video
        arena_reference = None
        if not test:  # pragma: no cover

            #skip arena retival if circular-autodetect was selected
            get_arena=True
            if arena == "circular-autodetect":
                get_arena=False
            else:
                display_message(multi_line_message)

        # Early return in test mode to avoid redundant slow arena detection
        if test:
            get_arena=False
            if "polygonal" in arena:
                scales={'test2': [279.5, 213.5, 420.12, 380], 'test': [279.5, 213.5, 420.12, 380]}
                arena_params={'test2': ((108, 30), (539, 29), (533, 438), (104, 431)), 'test': ((108, 30), (323, 29), (539, 29), (533, 434), (323, 434), (104, 431))}
                video_resolution={'test2': (480, 640), 'test': (480, 640)}
                rois={1: ((106, 230), (533, 230), (533, 438), (104, 431)), 2: ((106, 230), (323, 230), (323, 438), (104, 431))}
                roi_dicts={'test': rois, 'test2': rois} 
                #scale rois and arenas to mm
                arena_params = _scale_arenas_to_mm(arena_params, scales, arena)
                roi_dicts = _scale_rois_to_mm(roi_dicts, scales)

                if test == "detect_arena":
                    arena_reference=np.array([(108, 30), (539, 29), (533, 438), (104, 431)])
                else:
                    return scales, arena_params, roi_dicts, video_resolution
        
            elif "circular" in arena:
                scales={'test2': [300.0, 38.0, 252.0, 380], 'test': [300.0, 38.0, 252.0, 380]}
                arena_params={'test2': ((200, 195), (167, 169), 14.071887016296387), 'test': ((200, 195), (167, 169), 14.071887016296387)}
                video_resolution={'test2': (404, 416), 'test': (404, 416)}
                rois={1: ((145, 130), (145, 255), (260, 255), (260, 130)) , 2: ((145, 190), (145, 255), (260, 255), (260, 190)) }
                roi_dicts={'test': rois, 'test2': rois} 
                #scale rois and arenas to mm
                arena_params = _scale_arenas_to_mm(arena_params, scales, arena)
                roi_dicts = _scale_rois_to_mm(roi_dicts, scales)

                if test == "detect_arena":
                    pass
                else:
                    return scales, arena_params, roi_dicts, video_resolution


        # Load SAM 
        segmentation_model = deepof.utils.load_precompiled_model(
            segmentation_model_path,
            download_path="https://datashare.mpcdf.mpg.de/s/GccLGXXZmw34f8o/download",
            model_path=os.path.join("trained_models", "arena_segmentation","sam_vit_h_4b8939.pth"),
            model_name="Arena segmentation model"
            )                           
        with tqdm(total=len(videos), desc=f"{'Detecting arenas':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="arena") as pbar:
                
            vid_idx = 0
            key = list(videos.keys())[0]
            if not test:
                arena_corners, _, arena_dist, _ , _ = extract_polygonal_arena_coordinates(
                    os.path.join(video_path, videos[key]),
                    arena,
                    vid_idx,
                    videos,
                    list_of_rois=[],
                    get_arena=get_arena,
                    arena_dims=arena_dims,
                    norm_dist=None,
                    test=test,
                )
                get_arena=False

                if arena_corners is not None:
                    arena_reference = arena_corners

            arena_dists={}
            for key in videos.keys():    
                
                arena_parameters, h, w = automatically_recognize_arena(
                    coordinates=coordinates,
                    tables=tables,
                    videos=videos,
                    vid_key=key,
                    path=video_path,
                    arena_type=arena,
                    arena_reference=arena_reference,
                    segmentation_model=segmentation_model,
                    debug=debug,
                )

                if "polygonal" in arena:

                    closest_side_points = closest_side(
                        simplify_polygon(arena_parameters), arena_reference[:2]
                    )

                    arena_dist=dist(*closest_side_points)

                    scales[key]=[
                            *(np.mean(arena_parameters, axis=0)*(arena_dims/arena_dist)),
                            arena_dist,
                            arena_dims,
                        ]
                                    

                elif "circular" in arena:
                    # scales contains the coordinates of the center of the arena,
                    # the absolute diameter measured from the video in pixels, and
                    # the provided diameter in mm (1 -default- equals not provided)

                    arena_dist=np.mean([arena_parameters[1][0], arena_parameters[1][1]])* 2

                    scales[key]=list(
                            np.array(
                                [
                                    arena_parameters[0][0]*(arena_dims/arena_dist),
                                    arena_parameters[0][1]*(arena_dims/arena_dist),
                                    arena_dist,
                                ]
                            )
                        )+ [arena_dims]

                arena_dists[key] = arena_dist                           
                arena_params[key]=arena_parameters
                video_resolution[key]=(h, w)
                pbar.update()

            for vid_idx, key in enumerate(videos.keys()):
                
                if not test:
                    arena_corners, roi_corners, _, _ , _ = extract_polygonal_arena_coordinates(
                        os.path.join(video_path, videos[key]),
                        arena,
                        vid_idx,
                        videos,
                        list_of_rois=list_of_rois,
                        get_arena=False,
                        arena_dims=arena_dims,
                        norm_dist=arena_dists[key],
                        arena_params=arena_params[key],
                        test=test,
                    )

                    for roi in copy.copy(list_of_rois):
                        if roi_corners[roi] is None:
                            propagate_rois.append(roi)
                            list_of_rois.remove(roi)
                
                    cur_roi_corners={}
                    for roi in list(range(1,number_of_rois+1)): # pragma: no cover
                        if roi in propagate_rois:
                            cur_roi_corners[roi]=roi_dicts[list(roi_dicts.keys())[-1]][roi]
                        else:
                            cur_roi_corners[roi]=roi_corners[roi]
                        
                    roi_dicts[key]=cur_roi_corners

                    if arena_corners is not None:
                        arena_reference = arena_corners

    elif not arena:
        return None, None, None, None

    else:  # pragma: no cover
        raise NotImplementedError(
            "arenas must be set to one of: 'polygonal-manual', 'polygonal-autodetect', 'circular-manual', 'circular-autodetect'"
        )
    
    #scale rois and arenas to mm
    arena_params = _scale_arenas_to_mm(arena_params, scales, arena)
    roi_dicts = _scale_rois_to_mm(roi_dicts, scales)

    return scales, arena_params, roi_dicts, video_resolution


def _scale_arenas_to_mm(arena_params, scales, arena):
    """Scales arenas from pixel to mm"""
    for key in arena_params.keys():
        scaling_ratio = scales[key][3]/scales[key][2]
        if "polygonal" in arena:
            arena_params[key]=np.array(arena_params[key])*scaling_ratio
        elif "circular" in arena:
            arena_params[key]=(tuple(np.array(arena_params[key][0])*scaling_ratio),tuple(np.array(arena_params[key][1])*scaling_ratio),arena_params[key][2])
    return arena_params


def _scale_rois_to_mm(roi_dicts, scales):
    """Scales ROIS from pixel to mm"""
    for key in roi_dicts.keys():
        for k, roi in roi_dicts[key].items():
            scaling_ratio = scales[key][3]/scales[key][2]
            roi_dicts[key][k] = np.array(roi)*scaling_ratio
    return roi_dicts


def simplify_polygon(polygon: list, relative_tolerance: float = 0.05):
    """Simplify a polygon using the Ramer-Douglas-Peucker algorithm.

    Args:
        polygon (list): List of polygon coordinates.
        relative_tolerance (float): Relative tolerance for simplification. Defaults to 0.05.

    Returns:
        simplified_poly (list): List of simplified polygon coordinates.

    """
    poly = Polygon(polygon)
    perimeter = poly.length
    tolerance = perimeter * relative_tolerance

    simplified_poly = poly.simplify(tolerance, preserve_topology=False)
    return list(simplified_poly.exterior.coords)[
        :-1
    ]  # Exclude last point (same as first)


def closest_side(polygon: list, reference_side: list):
    """Find the closest side in other polygons to a reference side in the first polygon.

    Args:
        polygon (list): List of polygons.
        reference_side (list): List of coordinates of the reference side.

    Returns:
        closest_side_points (list): List of coordinates of the closest side.

    """

    def angle(p1, p2):
        return atan2(p2[1] - p1[1], p2[0] - p1[0])

    ref_length = dist(*reference_side)
    ref_angle = angle(*reference_side)

    min_difference = float("inf")
    closest_side_points = None

    for i in range(len(polygon)):
        side_points = (polygon[i], polygon[(i + 1) % len(polygon)])
        side_length = dist(*side_points)
        side_angle = angle(*side_points)
        total_difference = abs(side_length - ref_length) + abs(side_angle - ref_angle)

        if total_difference < min_difference:
            min_difference = total_difference
            closest_side_points = list(side_points)

    return closest_side_points


@_suppress_warning(warn_messages=["All-NaN slice encountered"])
def automatically_recognize_arena(
    coordinates: coordinates,
    tables: table_dict,
    videos: dict,
    vid_key: str,
    path: str = ".",
    arena_type: str = "circular-autodetect",
    arena_reference: list = None,
    segmentation_model: torch.nn.Module = None,
    debug: bool = False,
) -> Tuple[np.array, int, int]:
    """Return numpy.ndarray with information about the arena recognised from the first frames of the video.

    WARNING: estimates won't be reliable if the camera moves along the video.

    Args:
        coordinates (coordinates): Coordinates object.
        tables (table_dict): Dictionary of tables per experiment.
        videos (list): Relative paths of the videos to analise.
        vid_key (str): key of video to use.
        path (str): Full path of the directory where the videos are.
        arena_type (string): Arena type; must be one of ['circular-autodetect', 'circular-manual', 'polygon-manual'].
        arena_reference (list): List of coordinates defining the reference arena annotated by the user.
        segmentation_model (torch.nn.Module): Model used for automatic arena detection.
        debug (bool): If True, save a video frame with the arena detected.

    Returns:
        arena (np.ndarray): 1D-array containing information about the arena. If the arena is circular, returns a 3-element-array) -> center, radius, and angle. If arena is polygonal, returns a list with x-y position of each of the n the vertices of the polygon.
        h (int): Height of the video in pixels.
        w (int): Width of the video in pixels.

    """
    # create video capture object and read frame info
    current_video_cap = cv2.VideoCapture(os.path.join(path, videos[vid_key]))
    h = int(current_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(current_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Select the corresponding tracklets
    current_tab = get_dt(tables,vid_key)

    # Get distances of all body parts and timepoints to both center and periphery
    distances_to_center = cdist(
        current_tab.values.reshape(-1, 2), np.array([[w // 2, h // 2]])
    ).reshape(current_tab.shape[0], -1)

    # throws "All-NaN slice encountered" if in at least one frame no body parts could be detected
    possible_frames = np.nanmin(distances_to_center, axis=1) > np.nanpercentile(
        distances_to_center, 5.0
    )

    # save indices of valid frames, shorten distances vector
    possible_indices = np.where(possible_frames)[0]
    possible_distances_to_center = distances_to_center[possible_indices]

    if arena_reference is not None:
        # If a reference is provided manually, avoid frames where the mouse is too close to the edges, which can
        # hinder segmentation
        min_distance_to_arena = cdist(
            current_tab.values.reshape(-1, 2), arena_reference
        ).reshape([distances_to_center.shape[0], -1, len(arena_reference)])

        min_distance_to_arena = min_distance_to_arena[possible_indices]
        frame_index = np.argmax(
            np.nanmin(np.nanmin(min_distance_to_arena, axis=1), axis=1)
        )

    else:
        # If not, use the maximum distance to the center as a proxy
        frame_index = np.argmin(np.nanmax(possible_distances_to_center, axis=1))

    current_frame = possible_indices[frame_index]
    current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    reading_successful, numpy_im = current_video_cap.read()
    current_video_cap.release()

    # Get mask using the segmentation model
    segmentation_model.set_image(numpy_im)

    frame_masks, score, logits = segmentation_model.predict(
        point_coords=np.array([[w // 2, h // 2]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )

    # Get arenas for all retrieved masks, and select that whose area is the closest to the reference
    if arena_reference is not None:
        arenas = [
            arena_parameter_extraction(frame_mask, arena_type)
            for frame_mask in frame_masks
        ]
        arena = arenas[
            np.argmin(
                np.abs(
                    [Polygon(arena_reference).area - Polygon(a).area for a in arenas]
                )
            )
        ]
    else:
        arena = arena_parameter_extraction(frame_masks[np.argmax(score)], arena_type)

    if debug:

        # Save frame with mask and arena detected
        frame_with_arena = np.ascontiguousarray(numpy_im.copy(), dtype=np.uint8)

        if "circular" in arena_type:
            cv2.ellipse(
                img=frame_with_arena,
                center=arena[0],
                axes=arena[1],
                angle=arena[2],
                startAngle=0.0,
                endAngle=360.0,
                color=(40, 86, 236),
                thickness=3,
            )

        elif "polygonal" in arena_type:

            cv2.polylines(
                img=frame_with_arena,
                pts=[arena],
                isClosed=True,
                color=(40, 86, 236),
                thickness=3,
            )

            # Plot scale references
            closest_side_points = closest_side(
                simplify_polygon(arena), arena_reference[:2]
            )

            for point in closest_side_points:
                cv2.circle(
                    frame_with_arena,
                    list(map(int, point)),
                    radius=10,
                    color=(40, 86, 236),
                    thickness=2,
                )

        cv2.imwrite(
            os.path.join(
                coordinates.project_path,
                coordinates.project_name,
                "Arena_detection",
                f"{videos[vid_key][:-4]}_arena_detection.png",
            ),
            frame_with_arena,
        )

    return arena, h, w


def display_message(message: List[str]): # pragma: no cover
    """
    Opens a window that displays a message for the user

    Args:
        message: List of strings containing the message
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)  # White color
    line_type = 2
    
    # Calculate dimensions based on message content
    max_line_length = max(len(line) for line in message)
    line_height = 30  # Height per line of text
    image_height = line_height * len(message) + 20  # Add some padding
    image_width = max(600, max_line_length * 12)  # Minimum width or based on longest line

    # Create a blank image with calculated dimensions
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Initial position for the first line of text
    x, y = 10, line_height
    
    # Loop through each line and put it on the image
    for line in message:
        cv2.putText(image, line, (x, y), font, font_scale, font_color, line_type)
        y += line_height  # Move down for the next line

    window_name = "Arena scaling"
    
    # Display the image in a window
    cv2.imshow(window_name, image)
    
    try:
        # Wait for a key press or until the window is closed
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Exit on 'q' key press
                break
            
            # Check if window is still open
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    except Exception as e:
        print(f"An error occurred: {e}")   # Handle window close exception gracefully

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(window_name)


def extract_polygonal_arena_coordinates(
    video_path: str, 
    arena_type: str, 
    video_index: int, 
    videos: list, 
    list_of_rois: list = 0, 
    get_arena: bool = True, 
    arena_dims: float = 1.0, 
    norm_dist: float = None,
    arena_params: np.ndarray = None,
    test: bool = False, 
):  # pragma: no cover
    """Read a random frame from the selected video, and opens an interactive GUI to let the user delineate the arena manually.

    Args:
        video_path (str): Path to the video file.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygonal-manual".
        video_index (int): Index of the current video in the list of videos.
        videos (list): List of videos to be processed.
        list_of_rois (int): list of roi numbers to draw,
        get_arena (bool): retrieve arena or skip step (default is True)
        arena_dims (float): Distance as taken from video in pixels
        norm_dist (float): Same distance as arena_dims for normalization in mm
        arena_params (np.ndarray): nx2 array containing the x-y coordinates of all n corners of the polygonal arena.
        test (bool): Runs project in test mode and bypasses manual inputs, defaults to false


    Returns:
        arena_corners (np.ndarray): nx2 array containing the x-y coordinates of all n corners of the polygonal arena.
        int: Height of the video.
        int: Width of the video.

    """

    # read random frame from video capture object
    current_video_cap = cv2.VideoCapture(video_path)
    total_frames = int(current_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_number = np.random.choice(total_frames)
    current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    _, numpy_im = current_video_cap.read()
    current_video_cap.release()

    roi_corners = None
    norm_dist_new = None

    # open gui and let user pick corners
    if get_arena:
        arena_corners, norm_dist_new = retrieve_corners_from_image(
            numpy_im,
            arena_type,
            video_index,
            videos,
            current_roi=0,
            arena_dims=arena_dims,
            norm_dist=None,
            test=test,
        )
    elif arena_params is not None:
        arena_corners = extract_corners_from_arena(arena_params)
    else:
        arena_corners = None
    if norm_dist_new is None:
        norm_dist_new = norm_dist

    #let user pick corners for rois
    if len(list_of_rois) > 0:

        roi_corners= {}

        for k in list_of_rois:
            cur_roi_corners, _ = retrieve_corners_from_image(
                numpy_im,
                "polygonal-manual",
                video_index,
                videos,
                current_roi=k,
                arena_dims=arena_dims,
                norm_dist=norm_dist_new,
                arena_corners=arena_corners,
                test=test,
            )
            roi_corners[k] =cur_roi_corners

    return arena_corners, roi_corners, norm_dist_new, numpy_im.shape[0], numpy_im.shape[1]


def fit_ellipse_to_polygon(polygon: list):  # pragma: no cover
    """Fit an ellipse to the provided polygon.

    Args:
        polygon (list): List of (x,y) coordinates of the corners of the polygon.

    Returns:
        center_coordinates (tuple): (x,y) coordinates of the center of the ellipse.
        axes_length (tuple): (a,b) semi-major and semi-minor axes of the ellipse.
        ellipse_angle (float): Angle of the ellipse.

    """
    # Detect the main ellipse containing the arena
    ellipse_params = cv2.fitEllipse(np.array(polygon).astype(np.float32))

    # Parameters to return
    center_coordinates = tuple([int(i) for i in ellipse_params[0]])
    axes_length = tuple([int(i) // 2 for i in ellipse_params[1]])
    ellipse_angle = ellipse_params[2]

    return center_coordinates, axes_length, ellipse_angle

def get_first_length(arena_corners, w_ratio, h_ratio):
    """gets the length of the first edge in arena_corners"""
    return math.dist(
            (arena_corners[0][0]*w_ratio, arena_corners[0][1]*h_ratio),
            (arena_corners[1][0]*w_ratio, arena_corners[1][1]*h_ratio)
        )


def arena_parameter_extraction(
    frame: np.ndarray,
    arena_type: str,
) -> np.array:
    """Return x,y position of the center, the lengths of the major and minor axes, and the angle of the recognised arena.

    Args:
        frame (np.ndarray): numpy.ndarray representing an individual frame of a video
        arena_type (str): Type of arena to be used. Must be either "circular" or "polygonal".

    Returns:
        IF arena_type=="circular":
            center_coordinates (tuple): (x,y) coordinates of the center of the ellipse.
            axes_length (tuple): (a,b) semi-major and semi-minor axes of the ellipse.
            ellipse_angle (float): Angle of the ellipse.
        ELIF arena_type=="polygonal"        
            np.ndarray: (x,y) coordinates of all points of the polygon

    """
    # Obtain contours from the image, and retain the largest one
    cnts, _ = cv2.findContours(
        frame.astype(np.int64), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    main_cnt = np.argmax([len(c) for c in cnts])

    if "circular" in arena_type:
        center_coordinates, axes_length, ellipse_angle = fit_ellipse_to_polygon(
            cnts[main_cnt]
        )
        return center_coordinates, axes_length, ellipse_angle

    elif "polygonal" in arena_type:
        return np.squeeze(cnts[main_cnt])
    

def create_inner_polygon(outer_vertices, target_area_ratio=0.7, tolerance=0.01, max_iterations=100, return_inner=True):
    """
    Creates an inner polygon that covers approximately target_area_ratio Percent of the outer polygon's area.
    Returns either the inner polygon or the difference between outer and inner polygon.
    
    Args:
        outer_vertices (numpy.ndarray): Nx2 array of vertices defining the outer polygon
        target_area_ratio (float): Target ratio of inner to outer polygon area (default: 0.7)
        tolerance (float): Acceptable tolerance for area ratio (default: 0.01)
        max_iterations (int): Maximum number of iterations for binary search (default: 100)
        return_inner (bool): If True, returns inner polygon; if False, returns outer ring (default: True)
    
    Returns:
        vertices (numpy.ndarray): Mx2 array of vertices defining either the inner polygon or outer ring
    """
    
    # Create shapely polygon from vertices
    outer_polygon = Polygon(outer_vertices)
    
    # Ensure the polygon is valid
    if not outer_polygon.is_valid:
        # Try to fix the polygon
        outer_polygon = outer_polygon.buffer(0)
        if not outer_polygon.is_valid:
            raise ValueError("Invalid polygon provided")
    
    if target_area_ratio==0.0 and return_inner==False:
        return outer_vertices
    elif target_area_ratio==0.0 and return_inner:
        return np.array([])
    
    # Get the area of the outer polygon
    outer_area = outer_polygon.area
    target_area = outer_area * target_area_ratio
    
    # Binary search for the correct offset distance
    minx, miny, maxx, maxy = outer_polygon.bounds
    max_dimension = max(maxx - minx, maxy - miny)
    
    # Initial bounds for binary search
    offset_min = 0
    offset_max = max_dimension / 2
    
    best_offset = 0
    best_polygon = None
    best_ratio = 0
    
    for iteration in range(max_iterations):
        offset = (offset_min + offset_max) / 2
        
        # Create offset polygon (negative offset for shrinking)
        inner_polygon = outer_polygon.buffer(-offset, join_style=2, mitre_limit=5.0)
        
        # Handle cases where the polygon might split into multiple parts
        if inner_polygon.is_empty:
            offset_max = offset
            continue
            
        # If the result is a MultiPolygon, take the largest part
        if inner_polygon.geom_type == 'MultiPolygon':
            largest_area = 0
            largest_poly = None
            for poly in inner_polygon.geoms:
                if poly.area > largest_area:
                    largest_area = poly.area
                    largest_poly = poly
            inner_polygon = largest_poly
        
        if inner_polygon is None or inner_polygon.is_empty:
            offset_max = offset
            continue
        
        # Calculate area ratio
        inner_area = inner_polygon.area
        area_ratio = inner_area / outer_area
        
        # Check if we're within tolerance
        if abs(area_ratio - target_area_ratio) < tolerance:
            best_offset = offset
            best_polygon = inner_polygon
            best_ratio = area_ratio
            break
        
        # Update search bounds
        if area_ratio > target_area_ratio:
            offset_min = offset
        else:
            offset_max = offset
        
        # Keep track of best result so far
        if best_polygon is None or abs(area_ratio - target_area_ratio) < abs(best_ratio - target_area_ratio):
            best_offset = offset
            best_polygon = inner_polygon
            best_ratio = area_ratio
    
    if best_polygon is None:
        raise ValueError("Could not create inner ROI polygon with desired area ratio")
        
    if return_inner:
        result_polygon = best_polygon
        vertices = np.array(result_polygon.exterior.coords[:-1])  # Remove duplicate last point
    else:
        # Create a ring polygon by combining outer and inner boundaries
        # First get the outer boundary vertices (clockwise)
        outer_boundary = np.array(outer_polygon.exterior.coords[:-1])
        # Then get the inner boundary vertices (counter-clockwise)
        inner_boundary = np.array(best_polygon.exterior.coords[:-1])[::-1]
        # Combine them with a connecting point to create a valid ring
        vertices = np.vstack([
            outer_boundary,
            outer_boundary[0],  # Connection point
            inner_boundary,
            inner_boundary[0]   # Close the inner boundary
        ])
    
    # Verify the area ratio
    actual_ratio = best_polygon.area / outer_area
    
    return vertices


def extract_corners_from_arena(
    arena_params: Union[tuple, np.ndarray], 
    num_points: int = 100
):
    """
    Extracts polygon corner coordinates from given arena parameters.

    In case of polygonal arenas: Input is returned directly
    In case of circular arenas: Input is converted into a polygon with num_points.

    Args:
        params (Union[Tuple, np.ndarray]):
            - For a circular arena: A tuple containing ((center_x, center_y), (diameter_x, diameter_y), angle_degrees).
            - For a polygonal arena: A NumPy array of shape (N, 2) with vertex coordinates.
        num_points (int): Number of vertices for the ellipse. Defaults to 100.

    Returns:
        polygon (np.ndarray): A NumPy array of shape (M, 2) representing the polygon vertices.

    Raises:
        TypeError: If the input `params` is not a recognized type or format.
    """
    # Case 1: Input is already a polygon array
    if isinstance(arena_params, np.ndarray):
        if len(arena_params.shape) == 2 and arena_params.shape[1] == 2:
            return arena_params
        else:
            raise TypeError(
                f"Input NumPy array must have shape (N, 2), but got {arena_params.shape}"
            )

    # Case 2: Input is an ellipse tuple
    if isinstance(arena_params, tuple):
        # Validate the structure of the ellipse tuple
        try:
            (center, diameters, angle) = arena_params
            if not (isinstance(center, (tuple, list)) and len(center) == 2 and
                    isinstance(diameters, (tuple, list)) and len(diameters) == 2 and
                    isinstance(angle, (float, int))):
                raise ValueError
        except (ValueError, TypeError):
            raise TypeError(
                "Ellipse parameters must be a tuple in the format "
                "((center_x, center_y), (diameter_x, diameter_y), angle_degrees)."
            )

        # Unpack parameters
        center_x, center_y = center
        radius_x, radius_y = diameters
               
        # Convert angle from degrees to radians for numpy's trig functions
        angle_rad = np.deg2rad(angle)

        # Generate points on the ellipse
        # Create an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, num_points)

        # Parametric equation for a standard (non-rotated) ellipse
        x_unrotated = radius_x * np.cos(theta)
        y_unrotated = radius_y * np.sin(theta)

        # Apply rotation matrix
        # x' = x*cos(a) - y*sin(a)
        # y' = x*sin(a) + y*cos(a)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        x_rotated = x_unrotated * cos_a - y_unrotated * sin_a
        y_rotated = x_unrotated * sin_a + y_unrotated * cos_a

        # Translate the ellipse to its center
        x = x_rotated + center_x
        y = y_rotated + center_y

        # Stack the x and y coordinates to form the (N, 2) polygon array
        polygon = np.vstack((x, y)).T
        
        return polygon

    # If input is neither, raise an error
    raise TypeError(
        f"Input must be a NumPy array or a tuple, but got {type(arena_params)}"
    )


##################################################
# Custom GUI elements to avoid having to use PyQt5
##################################################


@dataclass
class DropdownConfig:
    # Position from right edge (will be calculated in init)
    margin_right: int = 10
    margin_top: int = 10
    width: int = 60  # Smaller width
    height: int = 25  # Smaller height
    option_height: int = 25  # Matching height
    font_scale: float = 0.5  # Smaller font
    font_thickness: int = 1
    border_color: Tuple[int, int, int] = (100, 100, 100)
    fill_color: Tuple[int, int, int] = (200, 200, 200)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    main_box_color: Tuple[int, int, int] = (220, 220, 220)  # Light gray background

class DropdownUI:
    def __init__(self, window_name: str, options: List[str], window_width: int, hidden: bool = False, config: DropdownConfig = None):
        self.window_name = window_name
        self.options = options
        self.config = config or DropdownConfig()
        
        # Calculate x position from right edge
        self.x = window_width - self.config.width - self.config.margin_right
        self.y = self.config.margin_top
        
        self.selected_option = options[0]
        self.is_open = False
        self.slider_value = 70
        self.slider_active = False
        self.hidden = hidden

    def _is_point_in_rect(self, point: Tuple[int, int], 
                         rect: Tuple[int, int, int, int]) -> bool:
        x, y = point
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def draw(self, img: np.ndarray) -> None:
        if not self.hidden:
            cfg = self.config
            
            # Draw main box with background
            cv2.rectangle(img, 
                        (self.x, self.y),
                        (self.x + cfg.width, self.y + cfg.height),
                        cfg.main_box_color, -1)  # Filled rectangle
            cv2.rectangle(img, 
                        (self.x, self.y),
                        (self.x + cfg.width, self.y + cfg.height),
                        cfg.border_color, 1)  # Border
            
            # Calculate text size to center it
            text_size = cv2.getTextSize(self.selected_option, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    cfg.font_scale, 
                                    cfg.font_thickness)[0]
            text_x = self.x + (cfg.width - text_size[0]) // 2
            text_y = self.y + (cfg.height + text_size[1]) // 2
            
            cv2.putText(img, self.selected_option,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale, cfg.text_color, cfg.font_thickness)
            
            if self.is_open:
                for i, option in enumerate(self.options):
                    y = self.y + cfg.height * (i + 1)
                    # Background
                    cv2.rectangle(img,
                                (self.x, y),
                                (self.x + cfg.width, y + cfg.option_height),
                                cfg.fill_color, -1)
                    # Border
                    cv2.rectangle(img,
                                (self.x, y),
                                (self.x + cfg.width, y + cfg.option_height),
                                cfg.border_color, 1)
                    # Centered text
                    text_size = cv2.getTextSize(option, 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            cfg.font_scale, 
                                            cfg.font_thickness)[0]
                    text_x = self.x + (cfg.width - text_size[0]) // 2
                    text_y = y + (cfg.option_height + text_size[1]) // 2
                    
                    cv2.putText(img, option,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            cfg.font_scale, cfg.text_color, cfg.font_thickness)

    def handle_mouse(self, event: int, x: int, y: int):
        """Returns the newly selected option if changed, None otherwise"""        
        if event != cv2.EVENT_LBUTTONDOWN or self.hidden:
            return None
            
        # Check main dropdown box click
        if self._is_point_in_rect((x, y), 
                                 (self.x, self.y, self.config.width, self.config.height)):
            self.is_open = not self.is_open
            return "Disable"
            
        if not self.is_open:
            return None
            
        # Check option clicks
        for i, option in enumerate(self.options):
            opt_y = self.y + self.config.height * (i + 1)
            if self._is_point_in_rect((x, y),
                                    (self.x, opt_y, self.config.width, self.config.option_height)):
                old_option = self.selected_option
                self.selected_option = option
                self.is_open = False
                return option if option != old_option else None
                
        self.is_open = False
        return None
    

def retrieve_corners_from_image(
    frame: np.ndarray, arena_type: str, cur_vid: int, videos: list, current_roi: int = 0, arena_dims: float = 1.0, norm_dist: float = None, arena_corners: np.ndarray = None, test: bool = False
):  # pragma: no cover
    """Open a window and waits for the user to click on all corners of the polygonal arena.

    The user should click on the corners in sequential order.

    Args:
        frame (np.ndarray): Frame to display.
        arena_type (str): Type of arena to be used. Must be one of the following: "circular-manual", "polygon-manual".
        cur_vid (int): Index of the current video in the list of videos.
        videos (list): List of videos to be processed.
        current_roi (int): Current ROI to be extracted. 0 is the global arena ROI
        arena_dims (float): Distance as taken from video in pixels
        norm_dist (float): Same distance as arena_dims for normalization in mm
        arena_corners (np.ndarray): Corners of arena, relevant for automatic ROIs
        test (bool): Runs project in test mode and bypasses manual inputs, defaults to false

    Returns:
        corners (np.ndarray): nx2 array containing the x-y coordinates of all n corners.

    """
    corners = []

    roi_colors=ROI_COLORS

    #early return of set of square corners
    if test:
        return [(111, 49), (541, 31), (553, 438), (126, 452)]
    
    if current_roi == 0:
        display_text="deepof - Select polygonal arena corners - (q: exit / d: delete{}) - {}/{} processed".format(
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            )
        color = (40, 86, 236)

    # current roi > 0
    elif current_roi<21:
        display_text="deepof - Select polygonal region of interest corners for roi {} - (q: exit / d: delete{}) - {}/{} processed".format(
                current_roi,
                (" / p: propagate last to all remaining videos" if cur_vid > 0 else ""),
                cur_vid,
                len(videos),
            )
        color = roi_colors[current_roi-1]
    else:
        raise ValueError(
            "only up to 20 ROIs are allowed (what for do you even need so many ROIs?)"
        )


    def click_on_corners(event, x, y, flags, param):
        # Callback function to store the coordinates of the clicked points
        nonlocal corners, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))

    def mouse_callback(event, x, y, flags, param):
        # Callback function to store the coordinates of the clicked points
        nonlocal corners, frame

        image_name = param[0]
        dropdown = param[1]
        new_option = dropdown.handle_mouse(event, x, y)

        if dropdown.selected_option == "Manual" and new_option is None:
            if event == cv2.EVENT_LBUTTONDOWN:
                corners.append((x, y))

        
        
        if new_option is not None:
            # Handle option change
            if new_option == "Inner" or new_option == "Outer":
                cv2.createTrackbar("Approx. Ratio", image_name, dropdown.slider_value, 100, lambda x: None)
                dropdown.slider_active = True
            elif dropdown.slider_active:
                dropdown.slider_value = cv2.getTrackbarPos("Approx. Ratio", image_name)
                cv2.destroyWindow(image_name)
                cv2.namedWindow(image_name)
                cv2.setMouseCallback(image_name, mouse_callback, [image_name, dropdown])
                dropdown.slider_active = False

    # Resize frame to a standard size
    frame = frame.copy() 

    h_ratio = 1
    w_ratio = 1
    if frame.shape[0] > IMG_H_MAX  or frame.shape[1] > IMG_W_MAX:

        if frame.shape[0]/IMG_H_MAX > frame.shape[1]/IMG_W_MAX:
            h_ratio = frame.shape[0]/IMG_H_MAX
            img_w_max = int(frame.shape[1]/h_ratio)
            w_ratio = frame.shape[1]/img_w_max
            frame=cv2.resize(frame, (img_w_max,IMG_H_MAX))

        else:
            w_ratio = frame.shape[1]/IMG_W_MAX
            img_h_max = int(frame.shape[0]/w_ratio)
            h_ratio = frame.shape[0]/img_h_max
            frame=cv2.resize(frame, (IMG_W_MAX,img_h_max))



    options = ["Manual", "Inner", "Outer"]
    
    arena_available = False
    if arena_corners is not None:
        arena_available = True
        # Scale arena corners to frame size
        arena_corners = np.array(arena_corners)  
        arena_corners = np.transpose(np.array([arena_corners[:,0]/w_ratio,arena_corners[:,1]/h_ratio]))


    # Create dropdown
    dropdown = DropdownUI(display_text, options, window_width=frame.shape[1], hidden=not arena_available)

    # Create a window and display the image
    cv2.startWindowThread()

    alpha=0.3
    while True:
        try:
            frame_copy = frame.copy()
            
            cv2.namedWindow(display_text, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(
                display_text,
                mouse_callback, 
                [display_text, dropdown],
            )

            # Draw dropdown
            dropdown.draw(frame_copy)

            # Draw additional content based on selected option
            if (dropdown.selected_option == "Inner" or dropdown.selected_option == "Outer") and dropdown.slider_active:
                return_inner=True
                if dropdown.selected_option == "Outer":
                    return_inner=False
                # remove manual corners
                corners_manual = corners.copy()
                dropdown.slider_value = cv2.getTrackbarPos("Approx. Ratio", display_text)
                corner_array = create_inner_polygon(np.array(arena_corners), dropdown.slider_value/100, return_inner=return_inner).astype(int)
                corners=[tuple([point[0].item(),point[1].item()]) for point in corner_array]

            # Display already selected corners
            if len(corners) > 0:
                for c, corner in enumerate(corners):
                    cv2.circle(frame_copy, (corner[0], corner[1]), 4, color, -1)
                    # Display lines between the corners
                    if len(corners) > 1 and c > 0:
                        if "polygonal" in arena_type or len(corners) < 5:
                            cv2.line(
                                frame_copy,
                                (corners[c - 1][0], corners[c - 1][1]),
                                (corners[c][0], corners[c][1]),
                                color,
                                2,
                            )

            if len(corners) > 1 and "polygonal" in arena_type:
                if norm_dist is None:
                    norm_dist=get_first_length(corners, w_ratio, h_ratio)
                #last distance in unscaled pixles
                cur_dist=math.dist(
                    (corners[-2][0]*w_ratio, corners[-2][1]*h_ratio),
                    (corners[-1][0]*w_ratio, corners[-1][1]*h_ratio)
                )            
                text="last edge in mm: " + str(np.round((cur_dist*(arena_dims/norm_dist))*100)/100)
                cv2.putText(
                    frame_copy,
                    text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Close the polygon
            if len(corners) > 2:
                if "polygonal" in arena_type or len(corners) < 5:
                    cv2.line(
                        frame_copy,
                        (corners[0][0], corners[0][1]),
                        (corners[-1][0], corners[-1][1]),
                        color,
                        2,
                    )

            if len(corners) >= 5 and "circular" in arena_type:
                cv2.ellipse(
                    frame_copy,
                    *fit_ellipse_to_polygon(corners),
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=3,
                )
            
            # Create filled overlay for ROIs (these are always polygonal)
            if current_roi > 0 and len(corners) > 2:         
                overlay = frame_copy.copy()
                pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)           
                # Blend overlay with original frame
                cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)

            cv2.imshow(
                display_text,
                frame_copy,
            )

            key = cv2.waitKey(1) & 0xFF
            # Remove last added coordinate if user presses 'd'
            if key == ord("d"):
                corners = corners[:-1]

            # Exit is user presses 'q'
            if len(corners) > 2:
                if key == ord("q"):
                    break

            # Exit and copy all coordinates if user presses 'c'
            if cur_vid > 0 and key == ord("p"):
                corners = None
                break

        # If user closes the window, recreate the dropdown menu.
        except cv2.error:
            dropdown = DropdownUI(display_text, options, window_width=frame.shape[1], hidden=not arena_available)
            cv2.namedWindow(display_text, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(
                display_text,
                mouse_callback, 
                [display_text, dropdown],
            )
        

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # If no norm dist is given and the arena is circular, take the circle diameter as a norm dist
    if norm_dist is None and corners is not None and len(corners) >= 5 and "circular" in arena_type:
        cur_arena_params = fit_ellipse_to_polygon(corners)
        norm_dist=np.mean([cur_arena_params[1][0]*w_ratio, cur_arena_params[1][1]*h_ratio])* 2

    # Fit ellipse and extract corner points from fitted ellipse (for smoothing)
    if "circular" in arena_type and corners is not None:
        arena_ellipse = fit_ellipse_to_polygon(corners)
        corners = extract_corners_from_arena(arena_ellipse)
    
    # Rescale to original pixel widths
    if corners is not None:
        corners = np.array(corners)
        corners = np.transpose(np.array([corners[:,0]*w_ratio,corners[:,1]*h_ratio]))

    # Return the corners
    return corners, norm_dist