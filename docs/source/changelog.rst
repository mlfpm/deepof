Changelog
=========

[0.8.1] - 2025-06-XX
====================

Added
-------
- File `data_loading.py` with functionality to manage loading of large and small tables. 
- Added `data_manager.py` and `data_explorer.py`
- Utilities to improve experience when working with large data sets.
  - Added more detailed progress bars for various functions
  - Added binning options to `get_graph_dataset`, `preprocess` and `deep_unsupervised_embedding` functions to allow selection of relevant intervals for model training
  - Added `samples_max` input parameter to most plot functions to avoid accidentally plotting hours worth of data at once
- All supervised behaviors were reworked and new behaviors were added 
  - `stat_lookaround` (mouse is standing still and looking around)
  - `stat_active` (mouse is standing still and being active (e.g. is digging)
  - `stat_passive` (mouse is standing still and is inactive)
  - `moving` (mouse is moving)
  - `immobility` (mouse is immobile for at least 1 second)        
- Added real world distance display during arena (or ROI) creation.
- Added new Region of interest (ROI) functionality for most plot functions and data extraction functions
- Added new function for counting behavior events
- Added new functionality for investigating associations between behaviors
- Added automatic saving for supervised annotations after generation
- New project attribute `version` to keep track of version number
- New project attribute `very_large_project` to determine if tables can stay in working memory or need to be saved on the storage drive
- New project attribute `roi_dicts` to contain ROI polygons
- New Tests, among others for polygonal arenas
- Increased Test coverage to 95%
- Added compatibility measures in load_project to be able to load 0.7 projects
- Functions added: `get_dt`,  `save_dt`, `load_dt`, `load_dt_metainfo`, `get_metainfo_from_loaded_dt`, `_init_metainfo`, `sample_windows_from_data`, `extract_windows`, `count_all_events`, `return_transitions`, `preprocess_transitions` and many more

Changed
-------
- Improved data handling
  - Videos and source Tables get no longer copied during project creation. Videos and source Tables are only read but not changed. Processed tables are saved in the database.
  - Table and video paths as well as scaling data for arenas are now saved as dictionaries which makes processing more robust         
- Reworked supervised behaviors
  - The algorithms for all supervised behaviors were updated and many behaviors were changed completely
  - `sniffing` was renamed to `sniff-arena`
  - `climbing` was renamed to `climb-arena`
  - A bug in `lookaround` was fixed (see Fixes) and the behavior was renamed to `sniffing`
  - Threshold values for `nose2nose`, `sidebyside`, `sidereside`, `nose2tail`, `nose2body` and `following` were updated    
- New color maps for supervised behaviors, being consistently applied across functions.
- Experiment conditions can now also be given as a path (and not only as an already loaded dict) during project definition, which will load the experiment conditions automatically
- Replaced old `Kleinberg smoothing` with simpler `median filter` to avoid otherwise occurring merging of distant behavior occurences
- Updated setting of supervised parameters for supervised behaviors to be easier to handle, also added an explanation for this in behavior tutorial.
- Upated `export_annotated_videos` to allow for the export of videos with supervised annotations and to give more options e.g. for the export of multiple behaviors at once 
- Updated outputs of `get_graph_dataset` and `preprocess` to only return concatenated arrays up to a maximum size
- `_preprocess_time_bins` now only returns a single `bin_info` object that is used for all types of processing instead of a variety of different binning object types. 
- More plot inputs are now covered by specific exceptions (e.g. entering a non-existent behavior will now result in in an Exception displaying valid options to choose from)
- Changed digit limit in `time_to_seconds` to 6 for hours, minutes and seconds
- The `plot_Gantt` function now also allows to also compare one behavior across different animals
- Frames are now not classified with a supervised ML-classifier if 10% or more of data in that frame needs to be interpolated
- Reformatted large sections of code

Deprecated
----------
- Currently no removals of features are planned.

Removed
-------
- Unused `breaks` input option from all functions
- Unused `rupture` syntax and functionality
- Unused `propagate labels` and `propagate annotation` functionality
- Several packages that are no longer used after the Rework (see below) 
- Old `huddle` behavior (as it was not sufficiently clearly defined)

Bug Fixes
---------
- **Bug in lookaround behavior that led to lookaround being frequently detected when the mouse was not moving.**
- Bug with open-cv not being able to display the arena selection in Linux systems
- Bug in `plot_heatmaps` which led to the inversion of the y-axis if an axis was already provided as a plot input.
- Bugs related to the `deepof_8` labeling schema
- Bug in table windowing for model training that could lead to start- and end-sections of different tables to get concatenated into one training example
- Bug in `plot_behavior_trends` that led to projects with more than 2 experiment conditions causing an error with this plot 
- Bug in `animate_skeleton` that caused issues if bodyparts were missing
- Minor bug with arena selection display, making the display a lot more responsive
- Minor bug that led to too many warnings getting filtered
- Minor bug in `seconds_to_time` that led to inaccuracies in edge cases
- Added assertion in `preprocess_tables` to ensure that all tables have the same number of animals
- Fixed issue with speed rolling window causing body parts in frames near NaNs being set to 0-speed
- And more minor fixes

Performance
-----------
- Major rework of data loading to allow for the processing of significantly longer videos (videos and tables may cover multiple days of recording)
  - A parallel loading structure was implemented that saves tables as files for large datasets
  - All tables can now be accessed with `get_dt` which automatically loads a given dictionary entry independent of the exact table storage and can return whole tables, specific lines, or only meta info such as the number of rows. 
  - The number of times tables are loaded and saved within the code was greatly reduced to improve performance for large tables
  - Implemented models will generally sample a number of rows from all tables for processing (the functionality remains the same for smaller datasets as in these cases simply all rows are sampled) 
  - Plot functions will sample or cut data automatically to a maximum number of samples (depending on the plot). This limit can be changed and an info message will be displayed to inform the user
- Improved execution speed of some functions by refactoring e.g.
  - `align_deepof_kinematics_with_unsupervised_labels` (ca. 2 times faster)
  - `output_videos_per_cluster` (ca. 10 times faster) 
  - `plot_Gantt` (ca. 100 times faster)
- Improved execution speed of automatic tests (ca. 8 times faster)

Documentation
-------------
- Updated tutorials to contain adjusted functions
- Added new event counting functionality to preprocessing tutorial
- Added explanation of new transition functionality to supervised tutorial
- Added new tutorial explaining the new supervised behaviors with example video snippets and a full explanation of their algorithms
- Added new tutorial for working with large data sets
- Added new tutorial for working with ROIs
- Updated `tutorial_files` for compatibility with deepof 0.8

Dependencies
------------
- Added new dependency library `pyarrow` [version 17.0.0+]
- Added new dependency `duckdb` [version 1.2.2+]
- Added new dependency `xgboost` [version 2.1.4]
- Upgraded several package version requirements
- Removed dependency libraries: `ruptures`, `POT`, `dask`, `dask_image`, `sktime`

Known Issues
------------
- The current imputation method (added in 0.7.0) is sub-optimal and will be replaced in a future update.

Upgrade Notes
-------------
- This current version has compatibility measures added in load_project to be able to load 0.7 projects. However, loading pickled project files with other methods will result in these project files missing attributes that are required for 0.8 and have to be set manually. The project will then be recreated as 0.8 version during loading. 
- This version is a major upgrade from the last released version (`deepof 0.7.2`) and has significant changes in functionality.

Compatibility
-------------
- Limited backwards compatibility with published 0.7 versions. Loading 0.7 projects will automatically recreate them as 0.8 projects.

Additional Information
----------------------
- Release Date: 2024-08-21
- Supported Platforms: Windows, Linux, MacOS
- Download Link: https://pypi.org/project/deepof/0.7.1/
- Full Documentation: https://deepof.readthedocs.io/en/latest/index.html
- Feedback and Bug Reports: https://github.com/mlfpm/deepof/issues

[0.7.1] - 2024-08-27
====================

Updates
-------
- New plot function plot_behavior_trends for plotting of behavioral data for different time bins with polar and line plot options.
- New polar_depiction option for plot_enrichment.

Bug Fixes
---------
- Fixed a bug when extending projects using deepof.data.Coordinates.extend
- Fixed OS compatibility bugs reported in Google colab tutorials.

Known Issues
------------

- Due to a bug the time binning does ignore user bin inputs in this version. This will be fixed in 0.7.2.

Compatibility
-------------
- Full backwards compatibility with published version 0.7.0.

Additional Information
----------------------
- Release Date: 2024-08-21
- Supported Platforms: Windows, Linux, MacOS
- Download Link: https://pypi.org/project/deepof/0.7.1/
- Full Documentation: https://deepof.readthedocs.io/en/latest/index.html
- Feedback and Bug Reports: https://github.com/mlfpm/deepof/issues


[0.7.0] - 2024-08-01
====================

Added
-----
- We now have a changelog.
- Usability features for most plot functions.
- Added time-based binning (start and duration given as “HH:MM:SS.SSS…”).
- Added specific exceptions, displaying correct input options for string-inputs.
- Added exceptions for not supported input argument combinations.
- Added missing input options to some functions for uniformity.
- New project input option `fast_implementations_threshold` (sets the threshold as the minimum number of total frames for which numba functions should get compiled, default is 50,000).
- New `connectivity_dict` option “deepof_11”.
- New user info outputs in case default variables get automatically adjusted (among others in `plot_embeddings`).
- Classes added: `MouseTrackingImputer` with functions: `_initialize_constraints`, `fit_transform`, `_kalman_smoothing`, `_iterative_imputation`.
- Functions added: `point_in_polygon`, `point_in_polygon_numba`, `compute_areas_numba`, `polygon_area_numba`, `kleinberg_core_numba`, `rotate_all_numba`, `rotate_numba`, `get_total_Frames`, `calculate_average_arena`, `seconds_to_time`, `time_to_seconds`, `_preprocess_time_bins`, `_check_enum_inputs`, `rts_smoother_numba`, `enforce_skeleton_constraints_numba`.

Changed
-------
- Updated the data imputation to feature a multi-step process for improved imputation results.
- Removed old drift imputation that could result in jumps of imputed points to the middle of the arena.
- Changed `enable_iterative_imputation` input option for the Project class to `iterative_imputation` that now takes inputs “full” or “partial”.
  - In case of “partial” only a linear imputation is performed that fills small gaps of up to three frames.
  - In case of “full” additionally IterativeImputer and a Kalman filter is run with enforcement of skeleton constraints as a last step.
- The imputation does not change any non-missing values as these are re-added after each step or not changed. However, some values are removed before by the outlier removal step.
- Batching of Kleinberg smoothing can lead to minor deviations in smoothing results.
- In plot functions, set `bin_index` defaults to None for consistency.
- In `plot_heatmaps`, modified arena averaging to be a lot more robust.
- In `plot_gantt`, added time axis units to plot.
- In `plot_enrichment`, changed input option “normalize” to now also normalize the data when supervised annotations are given.
- In `plot_enrichment`, changed `aggregate_experiments` defaults.
- In `plot_enrichment`, changed input argument name “plot_proportions” to “plot_speed” for more intuitive argument naming.
- In `plot_enrichment` changed comparison for speed to “average speed” instead of “sum of all speed”.
- In `plot_embeddings` changed default of `colour_by` to `exp_condition` as this is the only viable coloring option in case of `aggregate_experiments` being given.
- Removed linear imputation in `interpolate_outliers` section and renamed it to “remove_outliers”, all interpolation and imputation related to missing (or removed) data now happens in the iterative imputation-section.

Deprecated
----------
- Currently no removals of features are planned.

Removed
-------
- Input argument “min_confidence” from `plot_enrichment` (because it did nothing).
- Input argument “cluster” in `plot_transitions` (because it did nothing).

Fixed
-----
- Bug in the iterative imputation during project creation that led to unsuitable imputations.
- Nondescript y-axis in `plot_enrichment`.
- Bug due to which `exp_condition` values in plots were not read as strings.
- Bug with correctly handling given axes in `plot_stationary_entropy` and `plot_enrichment`.
- Bug in `plot_gantt` that led to not displaying a behavior if it happened nonstop in the entire observation interval.
- Bug in `export_annotated_video` that resulted in the function never finishing in Windows.
- Minor bug in project in table autodetection.
- Minor bug related to loaded experiment conditions not being saved.
- Minor bug with project loading.
- Minor bug with inconsistent sorting of clusters in `plot_enrichment`.
- Minor bug with inconsistent sorting of colors in `plot_stationary_entropy` and `plot_embeddings`.
- Minor bug in “filter_short_bouts” that led to the display of pointless warning messages.
- Unhandled exception in `plot_stationary_entropy` for extremely short bins.
- Unhandled exception in case of too many drawn samples in `plot_embeddings`.
- Unhandled exception in case of linear dependency between samples in `plot_embeddings`.

Performance
-----------
- Significant performance boost through code optimization and Numba function implementations.
- Achieved up to 200x faster processing in `create()` [speed improvement is smaller if using full imputation option or arena autodetection].
- Achieved up to 40x faster processing in `supervised_annotation()`.
- Various smaller speed improvements in some minor functions.
- New internal “run_numba” switch decides if most numba functions get compiled (i.e., if total frames > threshold).
- Improved memory handling by introducing batching and index-based frame selection.
- Capped Kleinberg smoothing at 50,000 sample batches.
- Drastically reduced overhead in `arena_selection`.
- Functions optimized: `get_areas`, `compute_areas`, `smooth_boolean_array`, `kleinberg`, `automatically_recognize_arena`, `extract_polygonal_arena_coordinates`, `align_trajectories`, `export_annotated_video`.

Documentation
-------------
- Updated tutorials to contain adjusted input arguments for plots.
- Updated `tutorial_files` for compatibility with deepof 0.7.

Dependencies
------------
- Added new dependency library `natsort` [version 8.4.0+].

Known Issues
------------
- The project extension seems to not work properly at the moment, will be fixed in 0.7.1.
- Whilst the new imputation method is better than the previous one, it is by no means perfect and we still plan to work on it and upgrade it further.

Upgrade Notes
-------------
- This current version will not be backwards compatible with older versions. This decision was made for the following reasons:
  - The bug in input sorting was fixed in this version, however, it would not be possible to retrospectively fix the sorting in old projects that were affected by this bug.
  - Deepof 0.7 contains some new functionality (such as the numba compilation option) that would require some additional overhead to ensure compatibility.

Additional Information
----------------------
- Release Date: 2024-08-01
- Supported Platforms: Windows, Linux, MacOS
- Download Link: https://pypi.org/project/deepof/0.7.0/
- Full Documentation: https://deepof.readthedocs.io/en/latest/index.html
- Feedback and Bug Reports: https://github.com/mlfpm/deepof/issues


[0.6.5] - 2024-07-29
====================

Updates
-------
- Minor updates to improve performance and usability.

Bug Fixes
---------
- Major bug in input sorting which, in edge cases, allowed for input lists to get mixed up. Code to test if your old projects may have been affected by this bug is available at the end of this Changelog.
- Fixed OS compatibility bugs reported in previous 0.6.x versions.

Compatibility
-------------
- Full backwards compatibility with published version 0.6.0.

Additional Information
----------------------
- Release Date: 2024-07-29
- Supported Platforms: Windows, Linux, MacOS
- Download Link: https://pypi.org/project/deepof/0.6.5/
- Full Documentation: https://deepof.readthedocs.io/en/latest/index.html
- Feedback and Bug Reports: https://github.com/mlfpm/deepof/issues