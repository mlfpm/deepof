# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Data structures and functions for analyzing supervised and unsupervised model results."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import warnings
from collections import Counter, defaultdict
from itertools import product
from multiprocessing import cpu_count
from typing import Any, NewType, Union

import numpy as np
import ot
import pandas as pd
import shap
import tqdm
import umap
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Parallel, delayed
from pomegranate.distributions import Normal
from pomegranate._utils import _update_parameter
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
from typing import Optional, Union

import deepof.data
import deepof.utils
import deepof.visuals_utils
from deepof.data_loading import get_dt, save_dt


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def _fit_hmm_range(
    concat_embeddings, states, min_states, max_states, covariance_type="full"
):

    """Auxiliary function for fitting a range of HMMs with different number of states.

    Args:
        concat_embeddings (np.ndarray): Concatenated embeddings across all animal experiments.
        states (str): Whether to use AIC or BIC to select the number of states.
        min_states (int): Minimum number of states to use for the HMM.
        max_states (int): Maximum number of states to use for the HMM.
        covariance_type: Type of covariance matrix to use for the HMM. Can be either "full", "diag", or "sphere".

    """
    hmm_models = []
    model_selection = []
    for i in tqdm.tqdm(range(min_states, max_states + 1)):

        try:
            try:
                model = DenseHMM(
                    [Normal(covariance_type=covariance_type) for _ in range(i)]
                )
                model = model.fit(concat_embeddings)
            except:
                model = DenseHMM([Normal(covariance_type="diag") for _ in range(i)])
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

        except (np.linalg.LinAlgError, IndexError):
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
    covariance_type: str = "full",
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
        covariance_type: Type of covariance matrix to use for the HMM. Can be either "full", "diag", or "sphere".
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
            hmm_model = pickle.load(open(pretrained, "rb"))[0]
        else:
            hmm_model = pickle.load(
                open(
                    os.path.join(
                        coordinates._project_path,
                        coordinates._project_name,
                        "Trained_models",
                        "hmm_trained_{}.pkl".format(states),
                    ),
                    "rb",
                )
            )[0]

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

        # Initialize and fit the model
        try:
            hmm_model = DenseHMM([Normal() for _ in range(concat_soft_counts.shape[2])])
            hmm_model = hmm_model.fit(X=concat_embeddings, priors=concat_soft_counts)
        except: # pragma: no cover
            hmm_model = DenseHMM(
                [
                    Normal(covariance_type="diag")
                    for _ in range(concat_soft_counts.shape[2])
                ]
            )
            hmm_model = hmm_model.fit(X=concat_embeddings, priors=concat_soft_counts)


    else:

        if isinstance(states, int):
            min_states = max_states = states

        # Fit a range of HMMs with different number of states
        hmm_model, model_selection = _fit_hmm_range(
            concat_embeddings,
            states,
            min_states,
            max_states,
            covariance_type=covariance_type,
        )

    # Save the best model
    if save:  # pragma: no cover
        pickle.dump(
            [hmm_model, model_selection],
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
        table_path=os.path.join(coordinates._project_path, coordinates._project_name, "Tables"),
        exp_conditions=coordinates.get_exp_conditions,
    )

    if len(model_selection) > 0:
        return soft_counts, model_selection

    return soft_counts


def get_time_on_cluster(
    soft_counts: table_dict,
    normalize: bool = True,
    reduce_dim: bool = False,
    bin_info: Union[dict,np.ndarray] = None,
    roi_number: int = None,
    animals_in_roi: list = None,
):
    """Compute how much each animal spends on each cluster.

    Requires a set of cluster assignments.

    Args:
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        normalize (bool): Whether to normalize the time by the total number of frames in each condition.
        reduce_dim (bool): Whether to reduce the dimensionality of the embeddings to 2D. If False, the embeddings are kept in their original dimensionality.
        bin_info (Union[dict,np.ndarray]): A dictionary or single array containing start and end positions of all sections for given embeddings and ROIs
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 

 
    Returns:
        A dataframe with the time spent on each cluster for each experiment.

    """
    hard_count_counters={}
    arr_ranges={}
    for key in soft_counts.keys():
        if isinstance(bin_info, np.ndarray):
            arr_ranges[key] = bin_info 
        elif isinstance(bin_info, dict):
            arr_ranges[key] = bin_info[key]["time"]
        elif bin_info is None:
            arr_ranges[key] = None

    preloaded = {}

    def load_single_key(key,arr_range):
        return key, get_dt(soft_counts, key, load_range=arr_range)

    max_workers = min(32, (cpu_count() or 1) + 4) 

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_key, key, arr_ranges[key]): key for key in soft_counts}
        for future in as_completed(futures):
            key, result = future.result()
            preloaded[key] = result

    for key in soft_counts.keys():
        
        # Determine most likely bin for each frame (N x n_bins) -> (N x 1)
        # Load full dataset (arr_range==None) or section
        #hard_counts = np.argmax(get_dt(soft_counts,key, load_range=arr_range), axis=1)
        hard_counts = np.argmax(preloaded[key], axis=1)

        if roi_number is not None:
            hard_counts = deepof.visuals_utils.get_unsupervised_behaviors_in_roi(hard_counts, bin_info[key], animals_in_roi)
            hard_counts=hard_counts[hard_counts >= 0]
        
        
        # Create dictionary with number of bin_occurences per bin
        hard_count_counters[key] = Counter(hard_counts)

        if normalize:
            hard_count_counters[key]={
                k: v / sum(list(hard_count_counters[key].values())) for k, v in hard_count_counters[key].items()
                }
    #reset function warning
    deepof.visuals_utils.get_unsupervised_behaviors_in_roi._warning_issued = False
            
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


@deepof.data_loading._suppress_warning(
    warn_messages=[
        "Mean of empty slice"
    ]
)
def get_aggregated_embedding(
    embedding: np.ndarray, 
    reduce_dim: bool = False, 
    agg: str = "mean", 
    bin_info: Union[dict,np.ndarray] = None, 
    roi_number:int = None, 
    animals_in_roi: list = None, 
    roi_mode: str = "mousewise"
):
    """Aggregate the embeddings of a set of videos, using the specified aggregation method.

    Instead of an embedding per chunk, the function returns an embedding per experiment.

    Args:
        embedding (np.ndarray): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        reduce_dim (bool): Whether to reduce the dimensionality of the embeddings to 2D. If False, the embeddings are kept in their original dimensionality.
        agg (str): The aggregation method to use. Can be either "mean" or "median".
        bin_info (Union[dict,np.ndarray]): A dictionary or single array containing start and end positions or indices of all sections for given embeddings
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded)       
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded                                                       
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                

    Returns:
        A dataframe with the aggregated embeddings for each experiment.

    """
    
    # Aggregate the provided embeddings and cast to a dataframe
    agg_embedding={}
    preloaded = {}

    def load_single_key(key):
        arr_range = None
        if isinstance(bin_info, dict):
            arr_range = bin_info[key]["time"]
        else:
            arr_range = bin_info
        return key, get_dt(embedding, key, load_range=arr_range)


    max_workers = min(32, (cpu_count() or 1) + 4) 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_key, key): key for key in embedding}
        for future in as_completed(futures):
            key, result = future.result()
            preloaded[key] = result


    for key in embedding.keys():
        
        # Load full dataset (arr_range==None) or section
        #current_embedding=get_dt(embedding,key,load_range=arr_range)
        current_embedding=preloaded[key]
        if roi_number is not None and type(current_embedding)==pd.DataFrame:
            current_embedding=deepof.visuals_utils.get_supervised_behaviors_in_roi(current_embedding, bin_info[key], animals_in_roi, roi_mode)
        elif roi_number is not None and type(current_embedding)==np.ndarray:
            current_embedding=deepof.visuals_utils.get_unsupervised_behaviors_in_roi(current_embedding, bin_info[key], animals_in_roi)

        if agg == "mean":
            agg_embedding[key]=np.nanmean(current_embedding, axis=0)
        elif agg == "median":
            agg_embedding[key]=np.nanmedian(current_embedding, axis=0)
    
    agg_embedding=pd.DataFrame(agg_embedding).T
    n_rows=agg_embedding.shape[0]

    if agg_embedding.isnull().any().any():
        agg_embedding_clean=agg_embedding.dropna()
        assert agg_embedding_clean.shape[0]>0, "agg_embeddings empty after NaN-row removal!"

        warning_message = (
            "\033[38;5;208m\n"  # Set text color to orange
            "Warning! Some rows of aggregated embeddings contained NaNs that were dropped! This can happen if the\n"
            "time bins are short or ROIs are strict, which leads to behaviors never occuring in the set parameters.\n"
            f"In total {np.round((1-agg_embedding_clean.shape[0]/n_rows)*10000)/100} % of all rows were removed."
            "\033[0m"  # Reset text color
        )
        warnings.warn(warning_message)
    else:
        agg_embedding_clean=agg_embedding
    
    if reduce_dim:
        agg_pipeline = Pipeline(
            [("PCA", PCA(n_components=2)), ("scaler", StandardScaler())]
        )

        agg_embedding_clean = pd.DataFrame(
            agg_pipeline.fit_transform(agg_embedding_clean), index=agg_embedding_clean.index
        )
    
    agg_embedding = agg_embedding_clean.reindex(agg_embedding.index)

    deepof.visuals_utils.get_unsupervised_behaviors_in_roi._warning_issued = False

    return agg_embedding


def condition_distance_binning(
    embedding: table_dict,
    soft_counts: table_dict,
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
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding experimental conditions.
        start_bin (int): The index of the first bin to compute the distance for.
        end_bin (int): The index of the last bin to compute the distance for.
        step_bin (int): The step size of the bins to compute the distance for.
        scan_mode (str): The mode to use for computing the distance. Can be one of "growing-window" (used to select optimal binning), "per-bin" (used to evaluate how discriminability evolves in subsequent bins of a specified size) or "precomputed", which requires a numpy ndarray with bin IDs to be passed to precomputed_bins.
        precomputed_bins (np.ndarray): numpy array with integer bin sizes in frames, do not necessarily need to have the same size. Difference across conditions for each of these bins will be reported.
        agg (str): The aggregation method to use. Can be either "mean", "median", or "time_on_cluster".
        metric (str): The distance metric to use. Can be either "auc" (where the reported 'distance' is based on performance of a classifier when separating aggregated embeddings), or "wasserstein" (which computes distances based on optimal transport).
        n_jobs (int): The number of jobs to use for parallel processing.

    Returns:
        An array with distances between conditions across the resulting time bins

    """
    
    # Divide the embeddings in as many corresponding bins, and compute distances
    def embedding_distance(bin_index):

        nonlocal precomputed_cumsums

        if scan_mode == "per-bin":

            bin_info=np.array([bin_index*step_bin,(bin_index+1)*step_bin-1])

        elif scan_mode == "growing_window":
            bin_info=np.array([0,bin_index])

        else:
            assert precomputed_bins is not None, (
                "For precomputed binning, provide a numpy array with bin IDs under "
                "the precomputed_bins parameter"
            )
            
            bin_info=np.array([precomputed_cumsums[bin_index],precomputed_cumsums[bin_index+1]])

        return separation_between_conditions(
            embedding,
            soft_counts,
            bin_info,
            exp_conditions,
            agg,
            metric=metric,
        )

    if scan_mode == "per-bin":
        bin_range = range(end_bin // step_bin)
    elif scan_mode == "growing_window":
        bin_range = range(start_bin, end_bin, step_bin)
    else:
        bin_range = range(len(precomputed_bins))
        precomputed_cumsums=np.insert(np.cumsum(precomputed_bins), 0, 0)

    exp_condition_distance_array = Parallel(n_jobs=n_jobs)(
        delayed(embedding_distance)(bin_index) for bin_index in bin_range
    )

    return np.array(exp_condition_distance_array)


def separation_between_conditions(
    cur_embedding: table_dict,
    cur_soft_counts: table_dict,
    bin_info: Union[dict,np.ndarray],
    exp_conditions: dict,
    agg: str,
    metric: str,
):
    """Compute the distance between the embeddings of two conditions, using the specified aggregation method.

    Args:
        cur_embedding (TableDict): A dictionary of embeddings, where the keys are the names of the experimental conditions, and the values are the embeddings for each condition.
        cur_soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        bin_info (Union[dict,np.ndarray]): A dictionary or single array containing start and end positions or indices of all sections for given embeddings
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding experimental conditions.
        agg (str): The aggregation method to use. Can be one of "time on cluster", "mean", or "median".
        metric (str): The distance metric to use. Can be either "auc" (where the reported 'distance' is based on performance of a classifier when separating aggregated embeddings), or "wasserstein" (which computes distances based on optimal transport).

    Returns:
        The distance between the embeddings of the two conditions.

    """
    # Aggregate embeddings and add experimental conditions
    if agg == "time_on_cluster":
        aggregated_embeddings = get_time_on_cluster(
            cur_soft_counts, reduce_dim=True, bin_info=bin_info,
        )
    elif agg in ["mean", "median"]:
        aggregated_embeddings = get_aggregated_embedding(
            cur_embedding, agg=agg, reduce_dim=True, bin_info=bin_info,
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
    soft_counts: table_dict = None,
    supervised_annotations: table_dict = None,
    exp_conditions: dict = None,
    plot_speed: bool = False,
    bin_info: dict = None,
    roi_number: int = None,
    animals_in_roi: list = None,
    roi_mode: str = "mousewise",
    normalize: bool = False,
):
    """Compute the population of each cluster across conditions.

    Args:
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        supervised_annotations (tableDict): table dict with supervised annotations per animal experiment across time.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding experimental conditions.
        plot_speed (bool): plot "speed" behavior
        bin_info (dict): A dictionary containing start and end positions or indices of all sections for given embeddings and ROIs
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded 
        roi_mode (str): Determines how the rois should be applied to different behaviors. Options are "mousewise" (default, selected mice needs to be inside the ROI) and "behaviorwise" (only mice involved in a behavior need to be inside of the ROI, only for supervised behaviors)                
        normalize (bool): Whether to normalize the population of each cluster across conditions.

    Returns:
        A long format dataframe with the population of each cluster across conditions.

    """

    if supervised_annotations is None:
        keys = soft_counts.keys()
    else:
        keys = supervised_annotations.keys()


    counter_df = pd.DataFrame()


    if supervised_annotations is None:

        #assert list(current_eb.values())[0].shape[0] > 0

        # Extract time on cluster for all videos and add experimental information
        counter_df = get_time_on_cluster(
            soft_counts, normalize=normalize, reduce_dim=False, bin_info=bin_info, roi_number=roi_number, animals_in_roi=animals_in_roi,
        )
    else:
        
        for key in supervised_annotations.keys():
        
            #load and cut current data set
            current_sa=get_dt(supervised_annotations,key).iloc[bin_info[key]["time"]]
            if roi_number is not None:
                current_sa=deepof.visuals_utils.get_supervised_behaviors_in_roi(current_sa, bin_info[key], animals_in_roi,roi_mode)

            #only keep speed column or only drop speed column
            if plot_speed:
                selected_columns = [col for col in current_sa.columns if "speed" in col]
            else:
                selected_columns = [col for col in current_sa.columns if "speed" not in col]

            table = current_sa[selected_columns]

            # Extract time on each behaviour for all videos and add experimental information,
            # normalize to total experiment time if normalization is requested
            if normalize or plot_speed:
                counter_df[key] = (
                    np.sum(table) 
                / len(table)
                )       
            else:
                counter_df[key] = np.sum(table)
            
        counter_df = pd.DataFrame(counter_df).T

    counter_df["exp condition"] = counter_df.index.map(exp_conditions).astype(str)

    enrichment = counter_df.melt(
        id_vars=["exp condition"], var_name="cluster", value_name="time on cluster"
    )
    enrichment["cluster"] = enrichment["cluster"].astype(str)

    return enrichment


def get_transitions(state_sequence: list, n_states: int, index_sequence: list=None):
    """Compute the transitions between states in a state sequence.

    Args:
        state_sequence (list): A list of states.
        n_states (int): The number of states.
        index_sequence (list): An optional list of index positions for the states. Will ensure that state transitions between non-neighboring sequence entries are skipped

    Returns:
        The resulting transition matrix.

    """
    transition_matrix = np.zeros([n_states, n_states])
    if index_sequence is None:
        for cur_state, next_state in zip(state_sequence[:-1], state_sequence[1:]):
            transition_matrix[cur_state, next_state] += 1
    else:
        for k, (cur_state, next_state) in enumerate(zip(state_sequence[:-1], state_sequence[1:])):
            if index_sequence[k+1]-index_sequence[k]==1:
                transition_matrix[cur_state, next_state] += 1
            else:
                continue

    return transition_matrix


def compute_transition_matrix_per_condition(
    soft_counts: table_dict,
    exp_conditions: dict,
    silence_diagonal: bool = False,
    bin_info: dict = None,
    roi_number: int = None,
    animals_in_roi: list = None,
    aggregate: str = True,
    normalize: str = True,
):
    """Compute the transition matrices specific to each condition.

    Args:
        soft_counts (TableDict): A dictionary of soft counts, where the keys are the names of the experimental conditions, and the values are the soft counts for each condition.
        exp_conditions (dict): A dictionary of experimental conditions, where the keys are the names of the experiments, and the values are the names of their corresponding
        silence_diagonal (bool): If True, diagonal elements on the transition matrix are set to zero.
        bin_info (dict): A dictionary containing start and end positions or indices of all sections for given embeddings and ROI information
        roi_number (int): Number of the ROI that should be used for the plot (all behavior that occurs outside of the ROI gets excluded) 
        animals_in_roi (list): List of ids of the animals that need to be inside of the active ROI. All frames in which any of the given animals are not inside of the ROI get excluded        
        aggregate (str): Whether to aggregate the embeddings across time.
        normalize (str): Whether to normalize the population of each cluster across conditions.

    Returns:
        A dictionary of transition matrices, where the keys are the names of the experimental
        conditions, and the values are the transition matrices for each condition.

    """
    
    n_states = get_dt(soft_counts, list(soft_counts.keys())[0], only_metainfo=True)["num_cols"]

    transitions_dict = {}
    if aggregate: 
        for exp_cond in set(exp_conditions.values()):
            transitions_dict[exp_cond] = np.zeros([n_states, n_states])

    for key in soft_counts.keys():

        #Determine load range
        load_range = bin_info[key]["time"]
        if roi_number is not None:
            load_range=deepof.visuals_utils.get_behavior_frames_in_roi(None,bin_info[key],animals_in_roi)

        #load requested range from current soft counts
        current_sc = get_dt(soft_counts, key, load_range=load_range)

        # Get hard counts per video
        hard_counts = np.argmax(current_sc, axis=1)

        # Get transition counts per video
        transitions=get_transitions(hard_counts, n_states, index_sequence=load_range)

        # Exclude transitions accross gaps
         

        if silence_diagonal:
            np.fill_diagonal(transitions, 0)

        # Aggregate based on experimental condition if specified
        if aggregate:
            exp_cond=exp_conditions[key]
            transitions_dict[exp_cond] += transitions
        else:
            transitions_dict[key] = transitions
    # Reset warning
    deepof.visuals_utils.get_behavior_frames_in_roi._warning_issued = False

    # Normalize rows if specified
    if normalize:
        transitions_dict = {
            key: np.nan_to_num(value / value.sum(axis=1)[:, np.newaxis])
            for key, value in transitions_dict.items()
        } 

    return transitions_dict


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
    file_name: str = 'kinematics',
    return_path: bool = False,
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
        file_name (str): Name of table for saving 
        return_path (bool): if True, Return only the path to the processed table, if false, return the full table. 

    Returns:
        A dictionary of aligned kinematics, where the keys are the names of the experimental conditions, and the
        values are the aligned kinematics for each condition.

    """
    # Compute speeds and accelerations per bodypart
    table_keys=deepof_project.get_table_keys()
    kinematic_features={}

    for i, key in enumerate(table_keys):

        #load current quality table for later
        if any([include_distances, include_angles,include_areas]):
            quality=deepof_project.get_quality().filter_videos([key])
            #load table if not already loaded
            quality[key] = get_dt(quality,key)

        kin_features = pd.DataFrame()

        for der in range(kin_derivative + 1):

            if der == 0:
                cur_kin=pd.DataFrame()

            else:
            
                try:
                    cur_kin = deepof_project.get_coords_at_key(
                        key=key, scale=deepof_project._scales[key], quality=quality, center=center, align=align, speed=der
                    )
                except ValueError:

                    try:
                        cur_kin = deepof_project.get_coords_at_key(
                            key=key, scale=deepof_project._scales[key], quality=quality, center="Center", align="Spine_1", speed=der
                        )
                    except ValueError:
                        cur_kin = deepof_project.get_coords_at_key(
                            key=key, scale=deepof_project._scales[key], quality=quality, center="Center", align="Nose", speed=der
                        )


                # If specified, filter on specific animals
                if der != 0 and animal_id is not None:
                    cur_kin=deepof.utils.filter_animal_id_in_table(cur_kin,animal_id)


            if include_distances:
                if der == 0 or include_feature_derivatives:
                    cur_distances = deepof_project.get_distances_at_key(key=key, speed=der, quality=quality)

                    # If specified, filter on specific animals
                    if animal_id is not None:
                        cur_distances = deepof.utils.filter_animal_id_in_table(cur_distances,animal_id)

                    cur_kin = pd.concat([
                        cur_kin, 
                        cur_distances
                        ], axis=1)
                    

            if include_angles:
                if der == 0 or include_feature_derivatives:
                    cur_angles = deepof_project.get_angles_at_key(key=key, speed=der, quality=quality)

                    # If specified, filter on specific animals
                    if animal_id is not None:
                        cur_angles = deepof.utils.filter_animal_id_in_table(cur_angles,animal_id)

                    cur_kin = pd.concat([
                        cur_kin, 
                        cur_angles
                        ], axis=1)

            if include_areas:
                if der == 0 or include_feature_derivatives:
                    try:
                        cur_areas = deepof_project.get_areas_at_key(
                            key=key, speed=der, selected_id=animal_id, quality=quality 
                        )

                        cur_kin = pd.concat([
                            cur_kin, 
                            cur_areas
                        ], axis=1)

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

            kin_features = pd.concat([kin_features, cur_kin.add_suffix(suffix)], axis=1)

        # save paths for modified tables
        table_path = os.path.join(deepof_project._project_path, deepof_project._project_name, 'Tables',key, key + '_' + file_name)
        kinematic_features[key] = save_dt(kin_features,table_path,return_path)

    # Return aligned kinematics
    return deepof.data.TableDict(
        kinematic_features, 
        typ="annotations",
        table_path=os.path.join(deepof_project._project_path, deepof_project._project_name, "Tables"), 
        )


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

    #name for intermediate saving 
    file_name='annot_time_chunks'

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
        comprehensive_features = comprehensive_features.merge(
            supervised_annotations,
            save_as_paths=deepof_project._very_large_project, 
            file_name=file_name,
)

    first_key = list(comprehensive_features.keys())[0]
    feature_names = get_dt(comprehensive_features, first_key, only_metainfo=True)['columns']

    # Do some preprocessing and convert to numpy matrices 
    comprehensive_features = comprehensive_features.preprocess(
        coordinates=deepof_project,
        scale=False,
        test_videos=0,
        window_size=(
            window_size
            if window_size is not None
            else int(np.round(deepof_project._frame_rate))
        ),
        window_step=window_step,
        filter_low_variance=False,
        interpolate_normalized=False,
        save_as_paths=deepof_project._very_large_project,
        file_name=file_name,
    )[0][0]

    # Load sampled features and remove chunks with missing values
    # use up to 200% of requested samples to factor in data reduction by filtering downstream
    N_windows_tab = int(samples*2/len(comprehensive_features))
    sampled_features, bin_info=comprehensive_features.sample_windows_from_data(
        time_bin_info={}, 
        N_windows_tab=N_windows_tab, 
        no_nans=True
        )


    def sample_from_idcs(sampled_idcs_dict, idcs):

        # Sample from idcs_dict, keeping each animal's identity
        cumulative_idcs = 0
        subset_idcs_dict = {}
        for key in sampled_idcs_dict.keys():
            subset_idcs_dict[key] = sampled_idcs_dict[key][
                idcs[
                    (idcs >= cumulative_idcs)
                    & (idcs < cumulative_idcs + sampled_idcs_dict[key].shape[0])
                ]
                - cumulative_idcs
            ]
            cumulative_idcs += sampled_idcs_dict[key].shape[0]

        return subset_idcs_dict


    # Filter instances with less confidence than specified
    sampled_soft_counts, _=soft_counts.sample_windows_from_data(time_bin_info=bin_info)
    sampled_hard_counts=pd.Series(np.argmax(sampled_soft_counts, axis=1))

    qual_filter = (sampled_soft_counts.max(axis=1) > min_confidence)
    sampled_features = sampled_features[qual_filter]
    hard_counts = sampled_hard_counts[qual_filter].reset_index(drop=True)
    bin_info = sample_from_idcs(bin_info, np.where(qual_filter)[0])

    # Sample X and y matrices to increase computational efficiency
    if samples is not None:
        samples = np.minimum(samples, sampled_features.shape[0])

        random_idcs = np.random.choice(
            range(sampled_features.shape[0]), samples, replace=False
        )

        sampled_features = sampled_features[random_idcs]
        hard_counts = hard_counts[random_idcs]
        bin_info = sample_from_idcs(bin_info, random_idcs)

    # Aggregate summary statistics per chunk, by either taking the average or running seglearn
    if aggregate == "mean":
        sampled_features[sampled_features.sum(axis=2) == 0] = np.nan
        sampled_features = np.nanmean(sampled_features, axis=1)
        sampled_features = pd.DataFrame(
            sampled_features, columns=feature_names
        )

    elif aggregate == "seglearn":

        # Extract all relevant features for each cluster
        sampled_features = chunk_summary_statistics(
            sampled_features, feature_names
        )

    return sampled_features, hard_counts, bin_info


def chunk_cv_splitter(
    chunk_stats: pd.DataFrame,
    bin_info: dict,
    n_folds: int = None,
):
    """Split a dataset into training and testing sets, grouped by video.

    Given a matrix with extracted features per chunk, returns a list containing
    a set of cross-validation folds, grouped by experimental video. This makes
    sure that chunks coming from the same experiment will never be leaked between
    training and testing sets.

    Args:
        chunk_stats (pd.DataFrame): matrix with statistics per chunk, sorted by experiment.
        bin_info (dict): A dictionary containing start and end positions or indices of all sections for given embeddings
        n_folds (int): number of cross-validation folds to compute.

    Returns:
        list containing a training and testing set per CV fold.

    """
    # Extract number of experiments/folds
    n_experiments = len(bin_info)

    # Create a cross-validation loop, with one fold per video
    fold_lengths = np.array([len(value) for value in bin_info.values()])

    # Repeat experiment indices across chunks, to generate a valid splitter
    cv_indices = np.repeat(np.arange(n_experiments), fold_lengths)
    cv_splitter = GroupKFold(
        n_splits=(n_folds if n_folds is not None else n_experiments)
    ).split(chunk_stats, groups=cv_indices)

    return list(cv_splitter)


def train_supervised_cluster_detectors(
    chunk_stats: pd.DataFrame,
    hard_counts: np.ndarray,
    bin_info: dict,
    n_folds: int = None,
    verbose: int = 1,
):  # pragma: no cover
    """Train supervised models to detect clusters from kinematic features.

    Args:
        chunk_stats (pd.DataFrame): table with descriptive statistics for a series of sequences ('chunks').
        hard_counts (np.ndarray): cluster assignments for the corresponding 'chunk_stats' table.
        bin_info (dict): A dictionary containing start and end positions or indices of all sections for given embeddings
        n_folds (int): number of folds for cross validation. If None (default) leave-one-experiment-out CV is used.
        verbose (int): verbosity level. Must be an integer between 0 (nothing printed) and 3 (all is printed).

    Returns:
        full_cluster_clf (imblearn.pipeline.Pipeline): trained supervised model on the full dataset, mapping chunk stats to cluster assignments. Useful to run the SHAP explainability pipeline.
        cluster_gbm_performance (dict): cross-validated dictionary containing trained estimators and performance metrics.
        groups (list): cross-validation indices. Data from the same animal are never shared between train and test sets.

    """
    groups = chunk_cv_splitter(chunk_stats, bin_info, n_folds=n_folds)

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


@deepof.data_loading._suppress_warning(
    warn_messages=[
        "The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning"
    ]
)
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
        chunk_stats.values
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
