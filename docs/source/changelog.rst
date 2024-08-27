Changelog
=========

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