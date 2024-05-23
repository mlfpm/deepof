# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Data structures and functions for analyzing supervised and unsupervised model results."""

from catboost import CatBoostClassifier
from collections import Counter, defaultdict
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from itertools import product
from joblib import delayed, Parallel
from multiprocessing import cpu_count
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM
from scipy import stats
from seglearn import feature_functions
from seglearn.transform import FeatureRep
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Any, NewType, Union
import numpy as np
import os
import ot
import pandas as pd
import pickle
import shap
import tqdm
import umap
import warnings

import deepof.data

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def _fit_hmm_range(concat_embeddings, states, min_states, max_states):
    """Auxiliary function for fitting a range of HMMs with different number of states.

    Args:
        concat_embeddings (np.ndarray): Concatenated embeddings across all animal experiments.
        states (str): Whether to use AIC or BIC to select the number of states.
        min_states (int): Minimum number of states to use for the HMM.
        max_states (int): Maximum number of states to use for the HMM.

    """
    hmm_models = []
    model_selection = []
    for i in tqdm.tqdm(range(min_states, max_states + 1)):

        try:
            model = DenseHMM([Normal() for _ in range(i)])
            model = model.fit(concat_embeddings)
            hmm_models.append(model)

            # Compute AIC and BIC
            n_features = concat_embeddings.shape[2]
            n_params = i * (n_features + n_features * (n_features + 1) / 2) + i * (
                i - 1
            )
            log_likelihood = float(model.log_probability(concat_embeddings).mean())
            if states == "aic":
                model_selection.append(2 * n_params - 2 * log_likelihood)
            elif states == "bic":
                model_selection.append(
                    n_params * np.log(concat_embeddings.shape[0]) - 2 * log_likelihood
                )

        except np.linalg.LinAlgError:
            model_selection.append(np.inf)

    if states in ["aic", "bic"]:
        hmm_model = hmm_models[np.argmin(model_selection)]
    else:
        hmm_model = hmm_models[0]

    return hmm_model, model_selection


def recluster(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict = None,
    min_confidence: float = 0.75,
    states: Union[str, int] = "aic",
    pretrained: Union[bool, str] = False,
    min_states: int = 2,
    max_states: int = 25,
    save: bool = True,
):
    """Recluster the data using a HMM-based approach. If soft_counts is provided, the model will use the soft cluster assignments as priors for a semi-supervised HMM.

    Args:
        coordinates: deepOF project where the data is stored.
        embeddings (table_dict): table dict with neural embeddings per animal experiment across time.
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time.
        min_confidence (float): minimum confidence the model should assign to a data point for the model to avoid resorting to a uniform prior around it.
        states: Number of states to use for the HMM. If "aic" or "bic", the number of states is chosen by minimizing the AIC or BIC criteria (respectively) over a predefined range of states.
        pretrained: Whether to use a pretrained model or not. If True, DeepOF will search for an existing file with the provided parameters. If a string, DeepOF will search for a file with the provided name.
        min_states: Minimum number of states to use for the HMM if automatic search is enabled.
        max_states: Maximum number of states to use for the HMM if automatic search is enabled.
        save: Whether to save the trained model or not.

    Returns:
        soft_counts (table_dict): table dict with soft cluster assignments per animal experiment across time, using the new HMM-based segmentation on the embedding space.

    """

    # Expand dims of each element in the table dict, pad them all to the same length, and concatenate
    model_selection = []
    max_len = max([i.shape[0] for i in embeddings.values()])
    concat_embeddings = np.concatenate(
        [
            np.expand_dims(np.pad(i, ((0, max_len - i.shape[0]), (0, 0))), axis=0)
            for i in embeddings.values()
        ]
    )

    # Load Pretrained model if necessary, or train a new one if not
    if pretrained:  # pragma: no cover
        if isinstance(pretrained, str):
            hmm_model = pickle.load(open(pretrained, "rb"))
        else:
            hmm_model = pickle.load(
                open(
                    os.path.join(
                        coordinates._project_path,
                        coordinates._project_name,
                        "Trained_models",
                        +"hmm_trained_{}.pkl".format(states),
                    ),
                    "rb",
                )
            )

    elif soft_counts is not None:
        concat_soft_counts = np.concatenate(
            [
                np.expand_dims(
                    np.pad(
                        i,
                        ((0, max_len - i.shape[0]), (0, 0)),
                        constant_values=1 / list(soft_counts.values())[0].shape[1],
                    ),
                    axis=0,
                )
                for i in soft_counts.values()
            ]
        )
        if min_confidence is not None:
            for st in concat_soft_counts:
                st[np.where(np.max(st, axis=1) <= min_confidence)[0]] = (
                    1 / list(soft_counts.values())[0].shape[1]
                )

        # Initialize the model
        hmm_model = DenseHMM([Normal() for _ in range(concat_soft_counts.shape[2])])

        # Fit the model
        hmm_model = hmm_model.fit(X=concat_embeddings, priors=concat_soft_counts)

    else:

        if isinstance(states, int):
            min_states = max_states = states

        # Fit a range of HMMs with different number of states
        hmm_model, model_selection = _fit_hmm_range(
            concat_embeddings, states, min_states, max_states
        )

    # Save the best model
    if save:  # pragma: no cover
        pickle.dump(
            hmm_model,
            open(
                os.path.join(
                    coordinates._project_path,
                    coordinates._project_name,
                    "Trained_models",
                    "hmm_trained_{}.pkl".format(states),
                ),
                "wb",
            ),
        )

    # Predict on each animal experiment
    soft_counts = hmm_model.predict_proba(concat_embeddings)
    soft_counts = deepof.data.TableDict(
        {
            key: np.array(soft_counts[i][: embeddings[key].shape[0]])
            for i, key in enumerate(embeddings.keys())
        },
        typ="unsupervised_counts",
        exp_conditions=coordinates.get_exp_conditions,
    )

    if len(model_selection) > 0:
        return soft_counts, model_selection

    return soft_counts


def get_time_on_cluster(
    soft_counts: table_dict,
    breaks: table_dict,
    normalize: bool = True,
    reduce_dim: bool = False,
):
    """Compute how much each animal spends on each cluster.

    Requires a set of cluster assignments and their corresponding breaks.

    Args:
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental conditions, and the values are the breaks for each condition.
        normalize (bool): Whether to normalize the time by the total number of frames in each condition.
        reduce_dim (bool): Whether to reduce the dimensionality of the embeddings to 2D. If False, the embeddings are kept in their original dimensionality.

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
    embedding: np.ndarray, reduce_dim: bool = False, agg: str = "mean"
):
    """Aggregate the embeddings of a set of videos, using the specified aggregation method.

    Instead of an embedding per chunk, the function returns an embedding per experiment.

    Args:
        embedding (np.ndarray): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        reduce_dim (bool): Whether to reduce the dimensionality of the embeddings to 2D. If False, the embeddings are kept in their original dimensionality.
        agg (str): The aggregation method to use. Can be either "mean" or "median".

    Returns:
        A dataframe with the aggregated embeddings for each experiment.

    """
    # aggregate the provided embeddings and cast to a dataframe
    if agg == "mean":
        embedding = pd.DataFrame(
            {key: np.nanmean(value, axis=0) for key, value in embedding.items()}
        ).T
    elif agg == "median":
        embedding = pd.DataFrame(
            {key: np.nanmedian(value, axis=0) for key, value in embedding.items()}
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
    embedding: table_dict = None,
    soft_counts: table_dict = None,
    breaks: table_dict = None,
    supervised_annotations: table_dict = None,
    bin_size: int = 0,
    bin_index: int = 0,
    precomputed: np.ndarray = None,
):
    """Select a time bin and filters all relevant objects (embeddings, soft_counts, breaks, and supervised annotations).

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental conditions, and the values are the breaks for each condition.
        supervised_annotations (TableDict): table dict with supervised annotations per animal experiment across time.
        bin_size (int): The size of the time bin to select.
        bin_index (int): The index of the time bin to select.
        precomputed (np.ndarray): Boolean array. If provided, ignores every othe parameter and just indexes each experiment using the provided mask.

    Returns:
        A tuple of the filtered embeddings, soft counts, and breaks.

    """
    # If precomputed, filter each experiment using the provided boolean array
    if supervised_annotations is None:

        if precomputed is not None:  # pragma: no cover
            breaks_mask_dict = {}

            for key in breaks.keys():
                if embedding[key].shape[0] > len(precomputed):
                    breaks_mask_dict[key] = np.concatenate(
                        [
                            precomputed,
                            [False] * (embedding[key].shape[0] - len(precomputed)),
                        ]
                    ).astype(bool)

                else:
                    breaks_mask_dict[key] = precomputed[: embedding[key].shape[0]]

        else:
            # Get cumulative length of each video using breaks, and mask the cumsum dictionary,
            # to check whether a certain instance falls into the desired bin
            breaks_mask_dict = {
                key: (np.cumsum(value) >= bin_size * bin_index)
                & (np.cumsum(value) < bin_size * (bin_index + 1))
                for key, value in breaks.items()
            }

        # Filter embedding, soft_counts and breaks using the above masks
        embedding = {
            key: value[breaks_mask_dict[key]] for key, value in embedding.items()
        }
        soft_counts = {
            key: value[breaks_mask_dict[key]] for key, value in soft_counts.items()
        }
        breaks = {key: value[breaks_mask_dict[key]] for key, value in breaks.items()}

    else:
        supervised_annotations = {
            key: val.iloc[
                bin_size
                * bin_index : np.minimum(val.shape[0], bin_size * (bin_index + 1))
            ]
            for key, val in supervised_annotations.items()
        }

    return embedding, soft_counts, breaks, supervised_annotations


def condition_distance_binning(
    embedding: table_dict,
    soft_counts: table_dict,
    breaks: table_dict,
    exp_conditions: dict,
    start_bin: int = None,
    end_bin: int = None,
    step_bin: int = None,
    scan_mode: str = "growing_window",
    precomputed_bins: np.ndarray = None,
    agg: str = "mean",
    metric: str = "auc",
    n_jobs: int = cpu_count(),
):
    """Compute the distance between the embeddings of two conditions, using the specified aggregation method.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding experimental conditions.
        start_bin (int): The index of the first bin to compute the distance for.
        end_bin (int): The index of the last bin to compute the distance for.
        step_bin (int): The step size of the bins to compute the distance for.
        scan_mode (str): The mode to use for computing the distance. Can be one of "growing-window" (used to select optimal binning), "per-bin" (used to evaluate how discriminability evolves in subsequent bins of a specified size) or "precomputed", which requires a numpy ndarray with bin IDs to be passed to precomputed_bins.
        precomputed_bins (np.ndarray): numpy array with IDs mapping to different bins, not necessarily having the same size. Difference across conditions for each of these bins will be reported.
        agg (str): The aggregation method to use. Can be either "mean", "median", or "time_on_cluster".
        metric (str): The distance metric to use. Can be either "auc" (where the reported 'distance' is based on performance of a classifier when separating aggregated embeddings), or "wasserstein" (which computes distances based on optimal transport).
        n_jobs (int): The number of jobs to use for parallel processing.

    Returns:
        An array with distances between conditions across the resulting time bins

    """
    # Divide the embeddings in as many corresponding bins, and compute distances
    def embedding_distance(bin_index):

        if scan_mode == "per-bin":

            cur_embedding, cur_soft_counts, cur_breaks, _ = select_time_bin(
                embedding, soft_counts, breaks, bin_size=step_bin, bin_index=bin_index
            )

        elif scan_mode == "growing_window":
            cur_embedding, cur_soft_counts, cur_breaks, _ = select_time_bin(
                embedding, soft_counts, breaks, bin_size=bin_index, bin_index=0
            )

        else:
            assert precomputed_bins is not None, (
                "For precomputed binning, provide a numpy array with bin IDs under "
                "the precomputed_bins parameter"
            )

            cur_embedding, cur_soft_counts, cur_breaks, _ = select_time_bin(
                embedding,
                soft_counts,
                breaks,
                precomputed=(precomputed_bins == bin_index),
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
        bin_range = range(end_bin // step_bin + 1)
    elif scan_mode == "growing_window":
        bin_range = range(start_bin, end_bin, step_bin)
    else:
        bin_range = pd.Series(precomputed_bins).unique()

    exp_condition_distance_array = Parallel(n_jobs=n_jobs)(
        delayed(embedding_distance)(bin_index) for bin_index in bin_range
    )

    return np.array(exp_condition_distance_array)


def separation_between_conditions(
    cur_embedding: table_dict,
    cur_soft_counts: table_dict,
    cur_breaks: table_dict,
    exp_conditions: dict,
    agg: str,
    metric: str,
):
    """Compute the distance between the embeddings of two conditions, using the specified aggregation method.

    Args:
        cur_embedding (TableDict): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        cur_soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        cur_breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding experimental conditions.
        agg (str): The aggregation method to use. Can be one of "time on cluster", "mean", or "median".
        metric (str): The distance metric to use. Can be either "auc" (where the reported 'distance' is based on performance of a classifier when separating aggregated embeddings), or "wasserstein" (which computes distances based on optimal transport).

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

        current_clf = LogisticRegression(penalty=None)
        current_clf.fit(aggregated_embeddings, y)

        current_distance = roc_auc_score(
            y, current_clf.predict_proba(aggregated_embeddings)[:, 1]
        )

    else:

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

        if metric == "wasserstein":
            # Compute Wasserstein distance between conditions in the current bin
            arrays_to_compare = [
                KernelDensity().fit(arr).sample(100, random_state=0)
                for arr in arrays_to_compare
            ]

            current_distance = ot.sliced_wasserstein_distance(
                *arrays_to_compare, n_projections=10000
            )

    return current_distance


def fit_normative_global_model(global_normal_embeddings: pd.DataFrame):
    """Fit a global model to the normal embeddings.

    Args:
        global_normal_embeddings (pd.DataFrame): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.

    Returns:
        A fitted global model.

    """
    # Define the range of bandwidth values to search over
    params = {"bandwidth": np.linspace(0.1, 10, 200)}

    # Create an instance of the KernelDensity estimator
    kde = KernelDensity(kernel="gaussian")

    # Perform a grid search to find the optimal bandwidth value
    grid_search = GridSearchCV(
        kde, params, cv=np.minimum(10, global_normal_embeddings.shape[0])
    )
    grid_search.fit(global_normal_embeddings.values)

    kd_estimation = KernelDensity(
        kernel="gaussian", bandwidth=grid_search.best_params_["bandwidth"]
    ).fit(global_normal_embeddings.values)

    return kd_estimation


def enrichment_across_conditions(
    embedding: table_dict = None,
    soft_counts: table_dict = None,
    breaks: table_dict = None,
    supervised_annotations: table_dict = None,
    exp_conditions: dict = None,
    bin_size: int = None,
    bin_index: int = None,
    precomputed: np.ndarray = None,
    normalize: bool = False,
):
    """Compute the population of each cluster across conditions.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental
        supervised_annotations (tableDict): table dict with supervised annotations per animal experiment across time.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding experimental conditions.
        bin_size (int): The size of the time bins to use. If None, the embeddings are not binned.
        bin_index (int): The index of the bin to use. If None, the embeddings are not binned.
        precomputed (np.ndarray): Boolean array. If provided, ignores every othe parameter and just indexes each experiment using the provided mask.
        normalize (bool): Whether to normalize the population of each cluster across conditions.

    Returns:
        A long format dataframe with the population of each cluster across conditions.

    """
    # Select time bin and filter all relevant objects

    if precomputed is not None:  # pragma: no cover
        embedding, soft_counts, breaks, supervised_annotations = select_time_bin(
            embedding,
            soft_counts,
            breaks,
            supervised_annotations=supervised_annotations,
            precomputed=precomputed,
        )

    elif bin_size is not None and bin_index is not None:
        embedding, soft_counts, breaks, supervised_annotations = select_time_bin(
            embedding, soft_counts, breaks, supervised_annotations, bin_size, bin_index
        )

    if supervised_annotations is None:

        assert list(embedding.values())[0].shape[0] > 0

        # Extract time on cluster for all videos and add experimental information
        counter_df = get_time_on_cluster(
            soft_counts, breaks, normalize=normalize, reduce_dim=False
        )
    else:
        # Extract time on each behaviour for all videos and add experimental information
        counter_df = pd.DataFrame(
            {key: np.sum(val) for key, val in supervised_annotations.items()}
        ).T

    counter_df["exp condition"] = counter_df.index.map(exp_conditions)

    return counter_df.melt(
        id_vars=["exp condition"], var_name="cluster", value_name="time on cluster"
    )


def get_transitions(state_sequence: list, n_states: int):
    """Compute the transitions between states in a state sequence.

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
    embedding: table_dict,
    soft_counts: table_dict,
    breaks: table_dict,
    exp_conditions: dict,
    silence_diagonal: bool = False,
    bin_size: int = None,
    bin_index: int = None,
    aggregate: str = True,
    normalize: str = True,
):
    """Compute the transition matrices specific to each condition.

    Args:
        embedding (TableDict): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        breaks (TableDict): A dictionary of breaks, where the keys are the names of the experimental conditions, and the values are the breaks for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding
        silence_diagonal (bool): If True, diagonal elements on the transition matrix are set to zero.
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
        embedding, soft_counts, breaks, _ = select_time_bin(
            embedding,
            soft_counts,
            breaks,
            bin_size=bin_size,
            bin_index=bin_index,
        )

    # Get hard counts per video
    hard_counts = {key: np.argmax(value, axis=1) for key, value in soft_counts.items()}

    # Get transition counts per video
    n_states = list(soft_counts.values())[0].shape[1]
    transitions = {
        key: get_transitions(value, n_states) for key, value in hard_counts.items()
    }

    if silence_diagonal:
        for key, val in transitions.items():
            np.fill_diagonal(val, 0)
            transitions[key] = val

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
    """Compute the steady state of each transition matrix provided in a dictionary.

    Args:
        transition_matrices (dict): A dictionary of transition matrices, where the keys are the names of the experimental conditions, and the values are the transition matrices for each condition.
        return_entropy (bool): Whether to return the entropy of the steady state. If False, the steady states themselves are returned.
        n_iters (int): The number of iterations to use for the Markov chain.

    Returns:
        A dictionary of steady states, where the keys are the names of the experimental conditions, and the values are the steady states for each condition. If return_entropy is True, values correspond to the entropy of each steady state.

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


def compute_UMAP(embeddings, cluster_assignments):  # pragma: no cover
    """Compute UMAP embeddings for visualization purposes."""
    lda = LinearDiscriminantAnalysis(
        n_components=np.min([embeddings.shape[1], len(set(cluster_assignments)) - 1]),
    )
    concat_embeddings = lda.fit_transform(embeddings, cluster_assignments)

    red = umap.UMAP(
        min_dist=0.99,
        n_components=2,
    ).fit(concat_embeddings)

    return lda, red


def align_deepof_kinematics_with_unsupervised_labels(
    deepof_project: coordinates,
    kin_derivative: int = 1,
    center: str = "Center",
    align: str = "Spine_1",
    include_feature_derivatives: bool = False,
    include_distances: bool = True,
    include_angles: bool = True,
    include_areas: bool = True,
    animal_id: str = None,
):
    """Align kinematics with unsupervised labels.

    In order to annotate time chunks with as many relevant features as possible, this function aligns the kinematics
    of a deepof project (speed and acceleration of body parts, distances, and angles) with the hard cluster assignments
    obtained from the unsupervised pipeline.

    Args:
        deepof_project (coordinates): A deepof.Project object.
        kin_derivative (int): The order of the derivative to use for the kinematics. 1 = speed, 2 = acceleration, etc.
        center (str): Body part to center coordinates on. "Center" by default.
        align (str): Body part to rotationally align the body parts with. "Spine_1" by default.
        include_feature_derivatives (bool): Whether to compute speed on distances, angles, and areas, if they are included.
        include_distances (bool): Whether to include distances in the alignment.
        include_angles (bool): Whether to include angles in the alignment.
        include_areas (bool): Whether to include areas in the alignment.
        animal_id (str): The animal ID to use, in case of multi-animal projects.

    Returns:
        A dictionary of aligned kinematics, where the keys are the names of the experimental conditions, and the
        values are the aligned kinematics for each condition.

    """
    # Compute speeds and accelerations per bodypart
    kinematic_features = defaultdict(pd.DataFrame)

    for der in range(kin_derivative + 1):

        try:
            cur_kinematics = deepof_project.get_coords(
                center=center, align=align, speed=der
            )
        except AssertionError:

            try:
                cur_kinematics = deepof_project.get_coords(
                    center="Center", align="Spine_1"
                )
            except AssertionError:
                cur_kinematics = deepof_project.get_coords(
                    center="Center", align="Nose"
                )

        # If specified, filter on specific animals
        if animal_id is not None:
            cur_kinematics = cur_kinematics.filter_id(animal_id)

        if der == 0:
            cur_kinematics = {key: pd.DataFrame() for key in cur_kinematics.keys()}

        if include_distances:
            if der == 0 or include_feature_derivatives:
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
            if der == 0 or include_feature_derivatives:
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
            if der == 0 or include_feature_derivatives:
                try:
                    cur_areas = deepof_project.get_areas(
                        speed=der, selected_id=animal_id
                    )

                    cur_kinematics = {
                        key: pd.concat([kin, area], axis=1)
                        for (key, kin), area in zip(
                            cur_kinematics.items(), cur_areas.values()
                        )
                    }

                except ValueError:
                    warnings.warn(
                        "No areas found for animal ID {}. Skipping.".format(animal_id)
                    )

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
    """Extract summary statistics from a chunked dataset using seglearn.

    Args:
        chunked_dataset (np.ndarray): Preprocessed training set (of shape chunks x time x features), where each entry corresponds to a time chunk of data.
        body_part_names (list): A list of the names of the body parts.

    Returns:
        A dataframe of kinematic features, of shape chunks by features.

    """
    # Extract time series features with ts-learn and seglearn
    extracted_features = FeatureRep(feature_functions.base_features()).fit_transform(
        chunked_dataset
    )

    # Convert to data frame and add feature names
    extracted_features = pd.DataFrame(extracted_features)
    columns = list(
        product(body_part_names, list(feature_functions.base_features().keys()))
    )
    extracted_features.columns = ["_".join(idx) for idx in columns]

    return extracted_features


def annotate_time_chunks(
    deepof_project: coordinates,
    soft_counts: table_dict,
    breaks: table_dict,
    supervised_annotations: table_dict = None,
    window_size: int = None,
    window_step: int = 1,
    animal_id: str = None,
    samples: int = 10000,
    min_confidence: float = 0.0,
    kin_derivative: int = 1,
    include_distances: bool = True,
    include_angles: bool = True,
    include_areas: bool = True,
    aggregate: str = "mean",
):
    """Annotate time chunks produced after change-point detection using the unsupervised pipeline.

    Uses a set of summary statistics coming from kinematics, distances, angles, and supervised labels when provided.

    Args:
        deepof_project (coordinates): Project object.
        soft_counts (table_dict): matrix with soft cluster assignments produced by the unsupervised pipeline.
        breaks (table_dict): the breaks for each condition.
        supervised_annotations (table_dict): set of supervised annotations produced by the supervised pipeline withing deepof.
        window_size (int): Minimum size of the applied ruptures. If automatic_changepoints is False, specifies the size of the sliding window to pass through the data to generate training instances. None defaults to video frame-rate.
        window_step (int): Specifies the minimum jump for the rupture algorithms. If automatic_changepoints is False, specifies the step to take when sliding the aforementioned window. In this case, a value of 1 indicates a true sliding window, and a value equal to window_size splits the data into non-overlapping chunks.
        animal_id (str): The animal ID to use, in case of multi-animal projects.
        samples (int): Time chunks samples to take to reduce computational time. Defaults to the minimum between 10000 and the number of available chunks.
        min_confidence (float): minimum confidence in cluster assignments used for quality control filtering.
        kin_derivative (int): The order of the derivative to use for the kinematics. 1 = speed, 2 = acceleration, etc.
        include_distances (bool): Whether to include distances in the alignment. kin_derivative is taken into account.
        include_angles (bool): Whether to include angles in the alignment. kin_derivative is taken into account.
        include_areas (bool): Whether to include areas in the alignment. kin_derivative is taken into account.
        aggregate (str): aggregation mode. Can be either "mean" (computationally cheapest), just use the average per feature, or "seglearn" which runs a thorough feature extraction and selection pipeline on each time series.

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
        window_size=(
            window_size if window_size is not None else deepof_project._frame_rate
        ),
        window_step=window_step,
        filter_low_variance=False,
        interpolate_normalized=False,
        automatic_changepoints=False,
        precomputed_breaks=breaks,
    )[0][0]

    # Remove chunks with missing values
    possible_idcs = ~np.isnan(comprehensive_features).any(axis=-1).any(axis=-1)
    comprehensive_features = comprehensive_features[possible_idcs]

    def sample_from_breaks(breaks, idcs):

        # Sample from breaks, keeping each animal's identity
        cumulative_breaks = 0
        subset_breaks = {}
        for key in breaks.keys():
            subset_breaks[key] = breaks[key][
                idcs[
                    (idcs >= cumulative_breaks)
                    & (idcs < cumulative_breaks + breaks[key].shape[0])
                ]
                - cumulative_breaks
            ]
            cumulative_breaks += breaks[key].shape[0]

        return subset_breaks

    # Filter instances with less confidence that specified
    qual_filter = (
        np.concatenate([soft for soft in soft_counts.values()]).max(axis=1)
        > min_confidence
    )[possible_idcs]
    comprehensive_features = comprehensive_features[qual_filter]
    hard_counts = hard_counts[possible_idcs][qual_filter].reset_index(drop=True)
    breaks = sample_from_breaks(breaks, np.where(qual_filter)[0])

    # Sample X and y matrices to increase computational efficiency
    if samples is not None:
        samples = np.minimum(samples, comprehensive_features.shape[0])

        random_idcs = np.random.choice(
            range(comprehensive_features.shape[0]), samples, replace=False
        )

        comprehensive_features = comprehensive_features[random_idcs]
        hard_counts = hard_counts[random_idcs]
        breaks = sample_from_breaks(breaks, random_idcs)

    # Aggregate summary statistics per chunk, by either taking the average or running seglearn
    if aggregate == "mean":
        comprehensive_features[comprehensive_features.sum(axis=2) == 0] = np.nan
        comprehensive_features = np.nanmean(comprehensive_features, axis=1)
        comprehensive_features = pd.DataFrame(
            comprehensive_features, columns=feature_names
        )

    elif aggregate == "seglearn":

        # Extract all relevant features for each cluster
        comprehensive_features = chunk_summary_statistics(
            comprehensive_features, feature_names
        )

    return comprehensive_features, hard_counts, breaks


def chunk_cv_splitter(
    chunk_stats: pd.DataFrame,
    breaks: dict,
    n_folds: int = None,
):
    """Split a dataset into training and testing sets, grouped by video.

    Given a matrix with extracted features per chunk, returns a list containing
    a set of cross-validation folds, grouped by experimental video. This makes
    sure that chunks coming from the same experiment will never be leaked between
    training and testing sets.

    Args:
        chunk_stats (pd.DataFrame): matrix with statistics per chunk, sorted by experiment.
        breaks (dict): dictionary containing ruptures per video.
        n_folds (int): number of cross-validation folds to compute.

    Returns:
        list containing a training and testing set per CV fold.

    """
    # Extract number of experiments/folds
    n_experiments = len(breaks)

    # Create a cross-validation loop, with one fold per video
    fold_lengths = np.array([len(value) for value in breaks.values()])

    # Repeat experiment indices across chunks, to generate a valid splitter
    cv_indices = np.repeat(np.arange(n_experiments), fold_lengths)
    cv_splitter = GroupKFold(
        n_splits=(n_folds if n_folds is not None else n_experiments)
    ).split(chunk_stats, groups=cv_indices)

    return list(cv_splitter)


def train_supervised_cluster_detectors(
    chunk_stats: pd.DataFrame,
    hard_counts: np.ndarray,
    sampled_breaks: dict,
    n_folds: int = None,
    verbose: int = 1,
):  # pragma: no cover
    """Train supervised models to detect clusters from kinematic features.

    Args:
        chunk_stats (pd.DataFrame): table with descriptive statistics for a series of sequences ('chunks').
        hard_counts (np.ndarray): cluster assignments for the corresponding 'chunk_stats' table.
        sampled_breaks (dict): sequence length of each chunk per experiment.
        n_folds (int): number of folds for cross validation. If None (default) leave-one-experiment-out CV is used.
        verbose (int): verbosity level. Must be an integer between 0 (nothing printed) and 3 (all is printed).

    Returns:
        full_cluster_clf (imblearn.pipeline.Pipeline): trained supervised model on the full dataset, mapping chunk stats to cluster assignments. Useful to run the SHAP explainability pipeline.
        cluster_gbm_performance (dict): cross-validated dictionary containing trained estimators and performance metrics.
        groups (list): cross-validation indices. Data from the same animal are never shared between train and test sets.

    """
    groups = chunk_cv_splitter(chunk_stats, sampled_breaks, n_folds=n_folds)

    # Cross-validate GBM training across videos
    cluster_clf = Pipeline(
        [
            ("normalization", StandardScaler()),
            ("oversampling", SMOTE()),
            ("classifier", CatBoostClassifier(verbose=(verbose > 2))),
        ]
    )

    if verbose:
        print("Training cross-validated models for performance estimation...")
    cluster_gbm_performance = cross_validate(
        cluster_clf,
        chunk_stats.values,
        hard_counts.values,
        scoring=[
            "roc_auc_ovo_weighted",
            "roc_auc_ovr_weighted",
        ],
        cv=groups,
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1,
        verbose=(verbose > 1),
    )

    # Train full classifier for explainability testing
    full_cluster_clf = Pipeline(
        [
            ("normalization", StandardScaler()),
            ("oversampling", SMOTE()),
            ("classifier", CatBoostClassifier(verbose=(verbose > 2))),
        ]
    )
    if verbose:
        print("Training on full dataset for feature importance estimation...")
    full_cluster_clf.fit(
        chunk_stats.values,
        hard_counts.values,
    )

    if verbose:
        print("Done!")
    return full_cluster_clf, cluster_gbm_performance, groups


def explain_clusters(
    chunk_stats: pd.DataFrame,
    hard_counts: np.ndarray,
    full_cluster_clf: Pipeline,
    samples: int = 10000,
    n_jobs: int = -1,
):  # pragma: no cover
    """Compute SHAP feature importance for models mapping chunk_stats to cluster assignments.

    Args:
        chunk_stats (pd.DataFrame): matrix with statistics per chunk, sorted by experiment.
        hard_counts (np.ndarray): cluster assignments for the corresponding 'chunk_stats' table.
        full_cluster_clf (imblearn.pipeline.Pipeline): trained supervised model on the full dataset, mapping chunk stats to cluster assignments.
        samples (int): number of samples to draw from the original chunk_stats dataset.
        n_jobs (int): number of parallel jobs to run. If -1 (default), all CPUs are used.

    Returns:
        shap_values (list): shap_values per cluster.
        explainer (shap.explainers._kernel.Kernel): trained SHAP KernelExplainer.

    """
    # Pass the data through the scaler and oversampler before computing SHAP values
    processed_stats = full_cluster_clf.named_steps["normalization"].transform(
        chunk_stats
    )
    processed_stats = full_cluster_clf.named_steps["oversampling"].fit_resample(
        processed_stats, hard_counts
    )[0]
    processed_stats = pd.DataFrame(processed_stats, columns=chunk_stats.columns)

    # Get SHAP values for the given model
    n_clusters = len(np.unique(hard_counts))
    explainer = shap.KernelExplainer(
        full_cluster_clf.named_steps["classifier"].predict_proba,
        data=shap.kmeans(processed_stats, n_clusters),
        normalize=False,
    )
    if samples is not None and samples < chunk_stats.shape[0]:
        processed_stats = processed_stats.sample(samples)
    shap_values = explainer.shap_values(
        processed_stats, nsamples=samples, n_jobs=n_jobs
    )

    return shap_values, explainer, processed_stats
