# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Data structures and functions for analyzing supervised and unsupervised model results.

"""

import numpy as np
import ot
import pandas as pd
import tsfresh
import tqdm
from collections import Counter, defaultdict
from joblib import delayed, Parallel
from multiprocessing import cpu_count
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tsfresh.feature_extraction.settings import MinimalFCParameters

import deepof.data


def get_time_on_cluster(
    soft_counts: deepof.data.table_dict,
    breaks: deepof.data.table_dict,
    normalize: bool = True,
    reduce_dim: bool = False,
):
    """

    Given a set of cluster assignments and the corresponding breaks, computes how much each
    animal spent on each cluster.

    Args:
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the
        experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        conditions, and the values are the breaks for each condition.
        normalize (bool): Whether to normalize the time by the total number of frames in
        each condition.
        reduce_dim (bool): Whether to reduce the dimensionality of the embeddings to 2D. If False,
        the embeddings are kept in their original dimensionality.

    Returns:
        A dataframe with the time spent on each cluster for each experiment.

    """

    # Reduce soft counts to hard assignments per video
    hard_counts = {key: np.argmax(value, axis=1) for key, value in soft_counts.items()}

    # Repeat cluster assignments using the break values
    hard_count_counters = {
        key: Counter(np.repeat(value, breaks[key]))
        for key, value in hard_counts.items()
    }

    if normalize:
        # Normalize the above counters
        hard_count_counters = {
            key: {k: v / sum(list(counter.values())) for k, v in counter.items()}
            for key, counter in hard_count_counters.items()
        }

    # Aggregate all videos in a dataframe
    counter_df = pd.DataFrame(hard_count_counters).T.fillna(0)
    counter_df = counter_df[sorted(counter_df.columns)]

    if reduce_dim:

        agg_pipeline = Pipeline(
            [("PCA", PCA(n_components=2)), ("scaler", StandardScaler())]
        )

        counter_df = pd.DataFrame(
            agg_pipeline.fit_transform(counter_df), index=counter_df.index
        )

    return counter_df


def get_aggregated_embedding(
    embedding: deepof.data.table_dict, reduce_dim: bool = False, agg: str = "mean"
):
    """

    Aggregates the embeddings of a set of videos, using the specified aggregation method.
    Instead of an embedding per chunk, the function returns an embedding per experiment.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the
        experimental conditions, and the values are the embeddings for each condition.
        reduce_dim (bool): Whether to reduce the dimensionality of the embeddings to 2D. If False,
        the embeddings are kept in their original dimensionality.
        agg (str): The aggregation method to use. Can be either "mean" or "median".

    Returns:
        A dataframe with the aggregated embeddings for each experiment.

    """

    # aggregate the provided embeddings and cast to a dataframe
    if agg == "mean":
        embedding = pd.DataFrame(
            {key: np.nanmean(value.numpy(), axis=0) for key, value in embedding.items()}
        ).T
    elif agg == "median":
        embedding = pd.DataFrame(
            {
                key: np.nanmedian(value.numpy(), axis=0)
                for key, value in embedding.items()
            }
        ).T

    if reduce_dim:
        agg_pipeline = Pipeline(
            [("PCA", PCA(n_components=2)), ("scaler", StandardScaler())]
        )

        embedding = pd.DataFrame(
            agg_pipeline.fit_transform(embedding), index=embedding.index
        )

    return embedding


def select_time_bin(
    embedding: deepof.data.table_dict,
    soft_counts: deepof.data.table_dict,
    breaks: deepof.data.table_dict,
    bin_size: int,
    bin_index: int,
):
    """

    Selects a time bin and filters all relevant objects (embeddings, soft_counts and breaks).

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the
        experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the
        experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        conditions, and the values are the breaks for each condition.
        bin_size (int): The size of the time bin to select.
        bin_index (int): The index of the time bin to select.

    Returns:
        A tuple of the filtered embeddings, soft counts, and breaks.
    """

    # Get cumulative length of each video using breaks, and mask the cumsum dictionary,
    # to check whether a certain instance falls into the desired bin
    breaks_mask_dict = {
        key: (np.cumsum(value) >= bin_size * bin_index)
        & (np.cumsum(value) < bin_size * (bin_index + 1))
        for key, value in breaks.items()
    }

    # Filter embedding, soft_counts and breaks using the above masks
    embedding = {key: value[breaks_mask_dict[key]] for key, value in embedding.items()}
    soft_counts = {
        key: value[breaks_mask_dict[key]] for key, value in soft_counts.items()
    }
    breaks = {key: value[breaks_mask_dict[key]] for key, value in breaks.items()}

    return embedding, soft_counts, breaks


def condition_distance_binning(
    embedding: deepof.data.table_dict,
    soft_counts: deepof.data.table_dict,
    breaks: deepof.data.table_dict,
    exp_conditions: dict,
    start_bin: int,
    end_bin: int,
    step_bin: int,
    scan_mode: str = "growing-window",
    agg: str = "mean",
    metric: str = "auc",
    n_jobs: int = cpu_count(),
):
    """

    Computes the distance between the embeddings of two conditions, using the specified
    aggregation method.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the
        experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the
        experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the
        names of the experiments, and the values are the names of their corresponding
        experimental conditions.
        start_bin (int): The index of the first bin to compute the distance for.
        end_bin (int): The index of the last bin to compute the distance for.
        step_bin (int): The step size of the bins to compute the distance for.
        scan_mode (str): The mode to use for computing the distance. Can be either "growing-window"
        (used to select optimal binning) or "per-bin" (used to evaluate how discriminability
        evolves in subsequent bins of a specified size).
        agg (str): The aggregation method to use. Can be either "mean", "median", or "time_on_cluster".
        metric (str): The distance metric to use. Can be either "auc" (where the reported 'distance'
        is based on performance of a classifier when separating aggregated embeddings), or
        "wasserstein" (which computes distances based on optimal transport).
        n_jobs (int): The number of jobs to use for parallel processing.

    Returns:
        An array with distances between conditions across the resulting time bins

    """

    # Divide the embeddings in as many corresponding bins, and compute distances
    def embedding_distance(bin_index):

        if scan_mode == "per-bin":
            cur_embedding, cur_soft_counts, cur_breaks = select_time_bin(
                embedding, soft_counts, breaks, step_bin, bin_index
            )

        else:
            cur_embedding, cur_soft_counts, cur_breaks = select_time_bin(
                embedding, soft_counts, breaks, bin_index, 0
            )

        return separation_between_conditions(
            cur_embedding,
            cur_soft_counts,
            cur_breaks,
            exp_conditions,
            agg,
            metric=metric,
        )

    if scan_mode == "per-bin":
        bin_range = range((end_bin // step_bin))
    else:
        bin_range = range(start_bin, end_bin, step_bin)

    exp_condition_distance_array = Parallel(n_jobs=n_jobs)(
        delayed(embedding_distance)(bin_index) for bin_index in tqdm.tqdm(bin_range)
    )

    return np.array(exp_condition_distance_array)


def separation_between_conditions(
    cur_embedding: deepof.data.table_dict,
    cur_soft_counts: deepof.data.table_dict,
    cur_breaks: deepof.data.table_dict,
    exp_conditions: dict,
    agg: str,
    metric: str,
):
    """

    Computes the distance between the embeddings of two conditions, using the specified
    aggregation method.

    Args:
        cur_embedding (TableDict): A dictionary of embeddings, where the keys are the names of the
        experimental conditions, and the values are the embeddings for each condition.
        cur_soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the
        experimental conditions, and the values are the soft counts for each condition.
        cur_breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the
        names of the experiments, and the values are the names of their corresponding
        experimental conditions.
        agg (str): The aggregation method to use. Can be one of "time on cluster", "mean", or "median".
        metric (str): The distance metric to use. Can be either "auc" (where the reported 'distance'
        is based on performance of a classifier when separating aggregated embeddings), or
        "wasserstein" (which computes distances based on optimal transport).

    Returns:
        The distance between the embeddings of the two conditions.

    """

    # Aggregate embeddings and add experimental conditions
    if agg == "time_on_cluster":
        aggregated_embeddings = get_time_on_cluster(
            cur_soft_counts, cur_breaks, reduce_dim=True
        )
    elif agg in ["mean", "median"]:
        aggregated_embeddings = get_aggregated_embedding(
            cur_embedding, agg=agg, reduce_dim=True
        )

    if metric == "auc":

        # Compute AUC of a logistic regression classifying between conditions in the current bin
        y = LabelEncoder().fit_transform(
            aggregated_embeddings.index.map(exp_conditions)
        )

        current_clf = LogisticRegression(penalty="none")
        current_clf.fit(aggregated_embeddings, y)

        current_distance = roc_auc_score(
            y, current_clf.predict_proba(aggregated_embeddings)[:, 1]
        )

    elif metric == "wasserstein":

        aggregated_embeddings["exp_condition"] = aggregated_embeddings.index.map(
            exp_conditions
        )

        # Get arrays to compare, as time on cluster per condition in a list of arrays
        arrays_to_compare = [
            aggregated_embeddings.loc[aggregated_embeddings.exp_condition == cond]
            .drop("exp_condition", axis=1)
            .values
            for cond in set(exp_conditions.values())
        ]

        # Compute Wasserstein distance between conditions in the current bin
        current_distance = ot.sliced_wasserstein_distance(
            *arrays_to_compare, n_projections=10000
        )

    return current_distance


def cluster_enrichment_across_conditions(
    embedding: deepof.data.table_dict,
    soft_counts: deepof.data.table_dict,
    breaks: deepof.data.table_dict,
    exp_conditions: dict,
    bin_size: int = None,
    bin_index: int = None,
    normalize: bool = False,
):
    """

    Computes the population of each cluster across conditions.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the
        experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the
        experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the
        names of the experiments, and the values are the names of their corresponding
        experimental conditions.
        bin_size (int): The size of the time bins to use. If None, the embeddings are not binned.
        bin_index (int): The index of the bin to use. If None, the embeddings are not binned.
        normalize (bool): Whether to normalize the population of each cluster across conditions.

    Returns:
        A long format dataframe with the population of each cluster across conditions.

    """

    # Select time bin and filter all relevant objects
    if bin_size is not None and bin_index is not None:
        embedding, soft_counts, breaks = select_time_bin(
            embedding, soft_counts, breaks, bin_size, bin_index
        )

    assert list(embedding.values())[0].shape[0] > 0

    # Extract time on cluster for all videos and add experimental information
    counter_df = get_time_on_cluster(
        soft_counts, breaks, normalize=normalize, reduce_dim=False
    )
    counter_df["exp condition"] = counter_df.index.map(exp_conditions)

    return counter_df.melt(
        id_vars=["exp condition"], var_name="cluster", value_name="time on cluster"
    )


def get_transitions(state_sequence: list, n_states: int):
    """

    Computes the transitions between states in a state sequence.

    Args:
        state_sequence (list): A list of states.
        n_states (int): The number of states.

    Returns:
        The resulting transition matrix.

    """

    transition_matrix = np.zeros([n_states, n_states])
    for cur_state, next_state in zip(state_sequence[:-1], state_sequence[1:]):
        transition_matrix[cur_state, next_state] += 1

    return transition_matrix


def compute_transition_matrix_per_condition(
    embedding: deepof.data.table_dict,
    soft_counts: deepof.data.table_dict,
    breaks: deepof.data.table_dict,
    exp_conditions: dict,
    bin_size: int = None,
    bin_index: int = None,
    aggregate: str = True,
    normalize: str = True,
):
    """

    Computes the transition matrices specific to each condition.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the
        experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the
        experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the
        names of the experiments, and the values are the names of their corresponding
        bin_size (int): The size of the time bins to use. If None, the embeddings are not binned.
        bin_index (int): The index of the bin to use. If None, the embeddings are not binned.
        aggregate (str): Whether to aggregate the embeddings across time.
        normalize (str): Whether to normalize the population of each cluster across conditions.

    Returns:
        A dictionary of transition matrices, where the keys are the names of the experimental
        conditions, and the values are the transition matrices for each condition.

    """

    # Filter data to get desired subset
    if bin_size is not None and bin_index is not None:
        embedding, soft_counts, breaks = select_time_bin(
            embedding, soft_counts, breaks, bin_size, bin_index
        )

    # Get hard counts per video
    hard_counts = {key: np.argmax(value, axis=1) for key, value in soft_counts.items()}

    # Get transition counts per video
    n_states = list(soft_counts.values())[0].shape[1]
    transitions = {
        key: get_transitions(value, n_states) for key, value in hard_counts.items()
    }

    # Aggregate based on experimental condition if specified
    if aggregate:
        transitions_per_condition = {}
        for exp_cond in set(exp_conditions.values()):
            transitions_per_condition[exp_cond] = np.zeros([n_states, n_states])
            for exp in transitions:
                if exp_conditions[exp] == exp_cond:
                    transitions_per_condition[exp_cond] += transitions[exp]
        transitions = transitions_per_condition

    # Normalize rows if specified
    if normalize:
        transitions = {
            key: np.nan_to_num(value / value.sum(axis=1)[:, np.newaxis])
            for key, value in transitions.items()
        }

    return transitions


def compute_steady_state(
    transition_matrices: dict, return_entropy: bool = False, n_iters: int = 100000
):
    """

    Computes the steady state of each transition matrix provided in a dictionary.

    Args:
        transition_matrices (dict): A dictionary of transition matrices, where the keys are
        the names of the experimental conditions, and the values are the transition matrices for each condition.
        return_entropy (bool): Whether to return the entropy of the steady state. If False, the steady states themselves are
        returned.
        n_iters (int): The number of iterations to use for the Markov chain.

    Returns:
        A dictionary of steady states, where the keys are the names of the experimental conditions, and the values are
        the steady states for each condition. If return_entropy is True, values correspond to the entropy of each
        steady state.

    """

    # Compute steady states by multiplying matrices by themselves n_iters times
    steady_states = {
        key: np.linalg.matrix_power(value, n_iters)
        for key, value in transition_matrices.items()
    }

    # Compute steady state probabilities per state
    steady_states = {
        key: np.nan_to_num(value.sum(axis=0) / value.sum())
        for key, value in steady_states.items()
    }

    # Compute entropy of the steady state distributions if required
    if return_entropy:
        steady_states = {
            key: stats.entropy(value) for key, value in steady_states.items()
        }

    return steady_states


def align_deepof_kinematics_with_unsupervised_labels(
    deepof_project: deepof.data.project,
    kin_derivative: int = 1,
    include_distances: bool = True,
    include_angles: bool = True,
    include_areas: bool = True,
    animal_id: str = None,
):
    """

    In order to annotate time chunks with as many relevant features as possible, this function aligns the kinematics
    of a deepof project (speed and acceleration of body parts, distances, and angles) with the hard cluster assignments
    obtained from the unsupervised pipeline.

    Args:
        deepof_project (Project): A deepof.Project object.
        kin_derivative (int): The order of the derivative to use for the kinematics. 1 = speed, 2 = acceleration, etc.
        include_distances (bool): Whether to include distances in the alignment. kin_derivative is taken into account.
        include_angles (bool): Whether to include angles in the alignment. kin_derivative is taken into account.
        include_areas (bool): Whether to include areas in the alignment. kin_derivative is taken into account.
        animal_id (str): The animal ID to use, in case of multi-animal projects.

    Returns:
        A dictionary of aligned kinematics, where the keys are the names of the experimental conditions, and the
        values are the aligned kinematics for each condition.

    """

    # Compute speeds and accelerations per bodypart
    kinematic_features = defaultdict(pd.DataFrame)

    for der in range(kin_derivative + 1):

        cur_kinematics = deepof_project.get_coords(
            center="Center", align="Spine_1", speed=der
        )

        # If specified, filter on specific animals
        if animal_id is not None:
            cur_kinematics = cur_kinematics.filter_id(animal_id)

        if der == 0:
            cur_kinematics = {key: pd.DataFrame() for key in cur_kinematics.keys()}

        if include_distances:
            cur_distances = deepof_project.get_distances(speed=der)

            # If specified, filter on specific animals
            if animal_id is not None:
                cur_distances = cur_distances.filter_id(animal_id)

            cur_kinematics = {
                key: pd.concat([kin, dist], axis=1)
                for (key, kin), dist in zip(
                    cur_kinematics.items(), cur_distances.values()
                )
            }

        if include_angles:
            cur_angles = deepof_project.get_angles(speed=der)

            # If specified, filter on specific animals
            if animal_id is not None:
                cur_angles = cur_angles.filter_id(animal_id)

            cur_kinematics = {
                key: pd.concat([kin, angle], axis=1)
                for (key, kin), angle in zip(
                    cur_kinematics.items(), cur_angles.values()
                )
            }

        if include_areas:
            cur_areas = deepof_project.get_areas(speed=der, selected_id=animal_id)

            cur_kinematics = {
                key: pd.concat([kin, area], axis=1)
                for (key, kin), area in zip(cur_kinematics.items(), cur_areas.values())
            }

        # Add corresponding suffixes to most common moments
        if der == 0:
            suffix = "_raw"
        elif der == 1:
            suffix = "_speed"
        elif der == 2:
            suffix = "_acceleration"
        else:
            suffix = "_kinematics_{}".format(der)

        for key, kins in cur_kinematics.items():
            kinematic_features[key] = pd.concat(
                [kinematic_features[key], kins.add_suffix(suffix)], axis=1
            )

    # Return aligned kinematics
    return deepof.data.TableDict(kinematic_features, typ="annotations")


def chunk_summary_statistics(chunked_dataset: np.ndarray, body_part_names: list):
    """

    Extracts summary statistics from a chunked dataset using tsfresh.

    Args:
        chunked_dataset (np.ndarray): Preprocessed training set (of shape chunks x time x features),
        where each entry corresponds to a time chunk of data.
        body_part_names (list): A list of the names of the body parts.

    Returns:
        A dataframe of kinematic features, of shape chunks by features.


    """

    # Add index and concatenate
    chunked_processed = []
    for i, chunk in enumerate(chunked_dataset):
        cur_dataset = chunk[~np.all(chunk == 0, axis=1)]
        cur_dataset = np.c_[np.ones(cur_dataset.shape[0]) * i, cur_dataset]

        chunked_processed.append(cur_dataset)

    chunked_processed = pd.DataFrame(np.concatenate(chunked_processed, axis=0))
    chunked_processed.columns = ["id"] + body_part_names

    # Extract time series features with ts-learn and tsfresh
    extracted_features = tsfresh.extract_features(
        chunked_processed,
        column_id="id",
        n_jobs=0,
        default_fc_parameters=MinimalFCParameters(),
    )

    return extracted_features


def annotate_time_chunks(
    deepof_project: deepof.data.project,
    soft_counts: deepof.data.table_dict,
    breaks: deepof.data.table_dict,
    supervised_annotations: deepof.data.table_dict = None,
    animal_id: str = None,
    kin_derivative: int = 1,
    include_distances: bool = True,
    include_angles: bool = True,
    include_areas: bool = True,
    aggregate: str = "tsfresh",
):
    """

    Annotate time chunks produced after change-point detection using the unsupervised pipeline, using a set
    of summary statistics coming from kinematics, distances, angles, and supervised labels when provided.

    Args:
        deepof_project: deepof.data.Project object.
        soft_counts: matrix with soft cluster assignments produced by the unsupervised pipeline.
        breaks: the breaks for each condition.
        supervised_annotations: set of supervised annotations produced by the supervised pipeline withing deepof.
        animal_id: The animal ID to use, in case of multi-animal projects.
        kin_derivative: The order of the derivative to use for the kinematics. 1 = speed, 2 = acceleration, etc.
        include_distances: Whether to include distances in the alignment. kin_derivative is taken into account.
        include_angles: Whether to include angles in the alignment. kin_derivative is taken into account.
        include_areas: Whether to include areas in the alignment. kin_derivative is taken into account.
        aggregate: aggregation mode. Can be either "mean" (computationally cheapest), just use the average per feature,
        or "tsfresh" which runs a thorough feature extraction and selection pipeline on each time series.

    Returns:
        A dataframe of kinematic features, of shape chunks by features.

    """

    # Convert soft_counts to hard labels
    hard_counts = {key: np.argmax(value, axis=1) for key, value in soft_counts.items()}
    hard_counts = pd.Series(
        np.concatenate([value for value in hard_counts.values()], axis=0)
    )

    # Extract (annotated) kinematic features
    comprehensive_features = align_deepof_kinematics_with_unsupervised_labels(
        deepof_project,
        kin_derivative=kin_derivative,
        include_distances=include_distances,
        include_angles=include_angles,
        include_areas=include_areas,
        animal_id=animal_id,
    )

    # Merge supervised labels if provided
    if supervised_annotations is not None:
        comprehensive_features = comprehensive_features.merge(supervised_annotations)

    feature_names = list(list(comprehensive_features.values())[0].columns)

    # Align with breaks per video, by taking averages on the corresponding windows, and concatenate videos
    comprehensive_features = comprehensive_features.preprocess(
        scale=False,
        test_videos=0,
        shuffle=False,
        filter_low_variance=False,
        interpolate_normalized=False,
        precomputed_breaks=breaks,
    )[0]

    # Aggregate summary statistics per chunk, by either taking the average or running ts-fresh
    if aggregate == "mean":
        comprehensive_features[comprehensive_features.sum(axis=2) == 0] = np.nan
        comprehensive_features = np.nanmean(comprehensive_features, axis=1)
        comprehensive_features = pd.DataFrame(
            comprehensive_features, columns=feature_names
        )

    elif aggregate == "tsfresh":

        # Extract all relevant features for each cluster
        comprehensive_features = chunk_summary_statistics(
            comprehensive_features, feature_names
        )

    return comprehensive_features, hard_counts


def chunk_cv_splitter(
    chunk_stats: np.ndarray,
    breaks: np.ndarray,
    n_folds: int = 10,
    qual_filter: np.ndarray = None,
):
    """

    Given a matrix with extracted features per chunk, returns a list containing
    a set of cross-validation folds, grouped by experimental video. This makes
    sure that chunks coming from the same experiment will never be leaked between
    training and testing sets.

    Args:
        chunk_stats: matrix with statistics per chunk, sorted by experiment
        breaks: dictionary containing ruprures per video
        n_folds: number of cross-validation folds to compute
        qual_filter: quality filter to use for the cross-validation. If None, no filter is used.

    Returns:
        list containing a training and testing set per CV fold.

    """

    # Extract number of experiments/folds
    n_experiments = len(breaks)

    # Create a cross-validation loop, with one fold per video
    fold_lengths = np.array([len(value) for value in breaks.values()])

    # Repeat experiment indices across chunks, to generate a valid splitter
    cv_indices = np.repeat(np.arange(n_experiments), fold_lengths)
    if qual_filter is not None:
        cv_indices = cv_indices[qual_filter]

    cv_splitter = GroupKFold(n_splits=n_folds).split(chunk_stats, groups=cv_indices)

    return list(cv_splitter)
