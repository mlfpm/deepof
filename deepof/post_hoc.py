# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Data structures and functions for analyzing supervised and unsupervised model results."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys
import pickle
import warnings
from collections import Counter, defaultdict
from itertools import product, combinations
from multiprocessing import cpu_count
from typing import Optional, Any, Dict, NewType, Union, Tuple, List
from sklearn.cluster import MiniBatchKMeans
from scipy.ndimage import uniform_filter1d
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
import types

import numpy as np
import ot
import pandas as pd
import shap
import tqdm
import umap
from catboost import CatBoostClassifier


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
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import torch

from deepof.legacy_smote_handling import SimpleSMOTE, ResampledClassifier

from deepof.config import PROGRESS_BAR_FIXED_WIDTH, CONTINUOUS_BEHAVIORS
import deepof.data
import deepof.utils
import deepof.visuals_utils
from deepof.data_loading import get_dt, save_dt


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def _fit_hmm_range(embeddings, states, min_states, max_states, covariance_type="diag"):
    """Auxiliary function for fitting a range of HMMs with different number of states.

    Args:
        concat_embeddings (np.ndarray): Concatenated embeddings across all animal experiments.
        states (str): Whether to use AIC or BIC to select the number of states.
        min_states (int): Minimum number of states to use for the HMM.
        max_states (int): Maximum number of states to use for the HMM.
        covariance_type: Type of covariance matrix to use for the HMM. Can be either "full", "diag", or "sphere".

    """

    # Collect sequences, validate dims, crop to common length
    seq_list = [np.asarray(v) for v in embeddings.values()]
    if not seq_list:
        raise ValueError("No sequences provided.")  # pragma: no cover
    d = seq_list[0].shape[1]
    for s in seq_list:
        if s.ndim != 2 or s.shape[1] != d:
            raise ValueError(f"All sequences must be (T, {d}). Got {s.shape}.")  # pragma: no cover
    min_T = min(s.shape[0] for s in seq_list)
    X = np.stack([s[:min_T].astype(np.float32, copy=False) for s in seq_list], axis=0)  # (N, T, D)
    n_obs = X.shape[0] * X.shape[1]

    def n_params(n_states, n_features, cov):
        cov = (cov or "diag").lower()
        if cov == "full":
            cov_params = n_features * (n_features + 1) // 2
        elif cov == "sphere":
            cov_params = 1
        else:  # diag
            cov_params = n_features
        per_state = n_features + cov_params
        return n_states * per_state + n_states * (n_states - 1)  # transitions only

    model_selection = []
    best_model, best_score = None, np.inf

    for i in tqdm.tqdm(range(min_states, max_states + 1)):
        try:
            used_cov = covariance_type
            try:
                m = DenseHMM([Normal(covariance_type=used_cov) for _ in range(i)]).fit(X)
            except Exception:
                if covariance_type != "diag":
                    used_cov = "diag"
                    m = DenseHMM([Normal(covariance_type="diag") for _ in range(i)]).fit(X)
                else:
                    raise

            ll = m.log_probability(X).numpy()
            total_ll = float(np.sum(ll)) if hasattr(ll, "__len__") else float(ll)

            k = n_params(i, d, used_cov)
            if states == "aic":
                score = 2.0 * k - 2.0 * total_ll
            elif states == "bic":
                score = k * np.log(max(1, n_obs)) - 2.0 * total_ll
            else:
                score = np.nan

            model_selection.append(float(score))

            if states in ("aic", "bic"):
                if score < best_score:
                    best_model, best_score = m, score
                else:
                    del m  # free non-best fit
            else:
                if best_model is None:
                    best_model = m

        except Exception:
            model_selection.append(np.inf)
            continue

    if best_model is None:
        raise RuntimeError("All HMM fits failed across the requested range.")
    return best_model, model_selection
    

def get_contrastive_soft_counts(
    coordinates,
    embeddings: Dict[str, np.ndarray],
    states: Union[str, int] = "bic",
    min_states: int = 2,
    max_states: int = 25,
    # Emissions
    reg_covar: float = 1e-5,
    sample_size: int = 500000,
    random_state: int = 0,
    # Sticky HMM
    p_stay: float = 0.95,
    # Optional priors (if soft_counts are given)
    soft_counts: Optional[Dict[str, np.ndarray]] = None,
    min_confidence: Optional[float] = 0.75,
    prior_weight: float = 1.0,
): # pragma: no cover  #legacy code
    """Extract soft counts for contrastive model.

    If `soft_counts` is provided, it is used as a per-frame prior over states (clusters),
    biasing the forward–backward posteriors (HMM smoothing) without running EM training.

    Notes:
      - If `soft_counts` is provided, K is taken from its second dimension (and AIC/BIC search is skipped).
      - Priors are applied as: log_emiss += prior_weight * log(soft_counts).
      - If min_confidence is not None, frames with max prior <= min_confidence are replaced by uniform priors.
    """
    eps = 1e-12

    # ---- helpers ----
    def _fit_diag_gmm(emb_dict, C, reg, max_n, covariance_type="diag", seed=0,):
        """Fit a diagonal GaussianMixture (sklearn) on a subset of embeddings."""
        Z = emb_dict.sample_windows_from_data(N_windows_tab=int(max_n / len(emb_dict)))[0]
        gm = GaussianMixture(
            n_components=C,
            covariance_type=covariance_type,
            reg_covar=reg,
            max_iter=200,
            tol=1e-3,
            random_state=seed,
            init_params="kmeans",
        )
        gm.fit(Z)
        return gm.means_.astype(np.float32), gm.covariances_.astype(np.float32), gm.weights_.astype(np.float32)

    def _log_emissions_diag(Z: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Compute log p(Z_t | c) under diagonal Gaussians; returns [T, C]."""
        var = np.maximum(var, 1e-10)
        diff = Z[:, None, :] - mu[None, :, :]         # [T, C, D]
        quad = (diff * diff) / var[None, :, :]        # [T, C, D]
        quad = quad.sum(axis=2)                       # [T, C]
        logdet = np.log(var).sum(axis=1)              # [C]
        const = Z.shape[1] * np.log(2.0 * np.pi)
        return -0.5 * (const + logdet[None, :] + quad)

    def _make_sticky_A(pi: np.ndarray, p_stay: float) -> np.ndarray:
        """Construct sticky transition A = p_stay*I + (1-p_stay)*(1*π^T); row-stochastic and positive."""
        C = pi.shape[0]
        A = p_stay * np.eye(C, dtype=np.float64) + (1.0 - p_stay) * (np.ones((C, 1)) @ pi[None, :])
        A = np.maximum(A, 1e-12)
        return A / A.sum(axis=1, keepdims=True)

    def _logsumexp(x: np.ndarray, axis=-1, keepdims=True):
        """Stable log-sum-exp along an axis."""
        m = np.max(x, axis=axis, keepdims=True)
        return np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m

    def _forward_loglik(log_emiss: np.ndarray, log_A: np.ndarray, log_pi0: np.ndarray) -> float:
        """Run forward pass only and return the sequence log-likelihood (sum of per-step normalizers)."""
        T, C = log_emiss.shape
        alpha = log_pi0 + log_emiss[0]
        ll = _logsumexp(alpha, axis=-1, keepdims=True).squeeze(0).item()
        alpha -= _logsumexp(alpha, axis=-1, keepdims=True)
        for t in range(1, T):
            trans = alpha[:, None] + log_A
            alpha = log_emiss[t] + _logsumexp(trans, axis=0, keepdims=True).squeeze(0)
            ct = _logsumexp(alpha, axis=-1, keepdims=True).squeeze(0).item()
            ll += ct
            alpha -= ct
        return float(ll)

    def _forward_backward(log_emiss: np.ndarray, log_A: np.ndarray, log_pi0: np.ndarray) -> np.ndarray:
        """Compute posterior p(c_t | z_1:T) via forward–backward; returns [T, C] (rows sum to 1)."""
        T, C = log_emiss.shape
        alpha = np.empty((T, C), dtype=np.float64)
        beta = np.empty((T, C), dtype=np.float64)

        alpha[0] = log_pi0 + log_emiss[0]
        alpha[0] -= _logsumexp(alpha[0]).squeeze(0)
        for t in range(1, T):
            trans = alpha[t - 1][:, None] + log_A
            alpha[t] = log_emiss[t] + _logsumexp(trans, axis=0).squeeze(0)
            alpha[t] -= _logsumexp(alpha[t]).squeeze(0)

        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            trans = log_A + (log_emiss[t + 1] + beta[t + 1])[None, :]
            beta[t] = _logsumexp(trans, axis=1).squeeze(1)
            beta[t] -= _logsumexp(beta[t]).squeeze(0)

        log_gamma = alpha + beta
        log_gamma -= _logsumexp(log_gamma, axis=1)
        return np.exp(log_gamma).astype(np.float32, copy=False)

    def _get_prior(key: str, T: int, K: int) -> Optional[np.ndarray]:
        """Fetch/align/normalize priors for a given key. Returns (T,K) or None."""
        if soft_counts is None or key not in soft_counts:
            return None

        P = get_dt(soft_counts, key).astype(np.float32, copy=False)  # supports TableDict-like too
        if P.ndim != 2:
            raise ValueError(f"soft_counts[{key}] must be (T,K). Got {P.shape}.")  # pragma: no cover
        if P.shape[1] != K:
            raise ValueError(f"K mismatch for {key}: soft_counts has {P.shape[1]}, expected {K}.")  # pragma: no cover

        # align length to embeddings
        if P.shape[0] < T:
            pad = np.full((T - P.shape[0], K), 1.0 / K, dtype=np.float32)
            P = np.vstack([P, pad])
        elif P.shape[0] > T:
            P = P[:T]

        # normalize + optional confidence gating
        P = np.maximum(P, eps)
        P /= P.sum(axis=1, keepdims=True)

        if min_confidence is not None:
            low = (np.max(P, axis=1) <= float(min_confidence))
            if np.any(low):
                P[low] = 1.0 / K

        return P

    # ---- select K ----
    def _fit_params_for_K(K: int, covariance_type: str = "diag") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _fit_diag_gmm(embeddings, K, reg_covar, sample_size, covariance_type, random_state)

    def _score_K(K: int, criterion: str) -> Tuple[float, float]:
        """caclulates the score for the given number of clusters (K)"""
        mu, var, pi = _fit_params_for_K(K)
        A = _make_sticky_A(pi.astype(np.float64), p_stay=float(p_stay))
        log_A = np.log(np.maximum(A, 1e-12))
        log_pi = np.log(np.maximum(pi.astype(np.float64), 1e-12))

        logL = 0.0
        T_total = 0
        for k in keys:
            cur_embedding = get_dt(embeddings, k)
            Z = cur_embedding.astype(np.float32, copy=False)
            log_emiss = _log_emissions_diag(Z, mu, var).astype(np.float64, copy=False)
            logL += _forward_loglik(log_emiss, log_A, log_pi)
            T_total += Z.shape[0]

        p = 2 * K * D + (K - 1)  # (means+vars) + mixture weights
        if criterion == "aic":
            score = 2 * p - 2 * logL
        elif criterion == "bic":
            score = p * np.log(max(T_total, 1)) - 2 * logL
        else:
            NotImplementedError("invalid states type, try \"aic\", \"bic\" or give a number of states / clusters directly") 
        return float(score), float(logL)

    # ---- prepare ----
    keys = list(embeddings.keys())
    if not keys:
        raise ValueError("Embeddings are empty.")  # pragma: no cover
    D = get_dt(embeddings, keys[0], only_metainfo=True)["shape"][1]

    # ---- determine K (skip selection if priors provided) ----
    model_selection: List[Tuple[int, float, float]] = []

    if soft_counts is not None:
        # infer K from priors (first overlapping key)
        k0 = next((k for k in keys if k in soft_counts), None)
        if k0 is None:
            raise ValueError("soft_counts provided but no keys overlap with embeddings.")  # pragma: no cover
        K_prior = int(get_dt(soft_counts, k0, only_metainfo=True)["shape"][1])

        if isinstance(states, int) and int(states) != K_prior:
            raise ValueError(f"states={states} but soft_counts implies K={K_prior}. They must match.")  # pragma: no cover
        K_best = K_prior

    else:
        # existing behavior
        if isinstance(states, int):
            K_best = int(states)
        else:
            crit = str(states).lower()
            Ks = list(range(max(2, min_states), max(min_states, max_states) + 1))
            best_score, K_best = None, None
            for K in tqdm.tqdm(Ks, desc=f"{'Optimizing number of clusters':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="step"):
                score, logL = _score_K(K, criterion=crit)
                model_selection.append((K, score, logL))
                if (best_score is None) or (score < best_score):
                    best_score, K_best = score, K

    # ---- final decode with best K and full covariance----
    mu, var, pi = _fit_params_for_K(K_best, covariance_type="diag")
    A = _make_sticky_A(pi.astype(np.float64), p_stay=float(p_stay))
    log_A = np.log(np.maximum(A, 1e-12))
    log_pi = np.log(np.maximum(pi.astype(np.float64), 1e-12))

    # ---- generate softcounts ----
    soft_counts_out = {}
    table_path = os.path.join(coordinates._project_path, coordinates._project_name, "Tables")

    for key in tqdm.tqdm(keys, desc=f"{'Extracting soft counts':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table"):
        Z = get_dt(embeddings, key).astype(np.float32, copy=False)
        log_emiss = _log_emissions_diag(Z, mu, var).astype(np.float64, copy=False)

        P = _get_prior(key, T=Z.shape[0], K=K_best)
        if P is not None:
            log_emiss = log_emiss + float(prior_weight) * np.log(np.maximum(P.astype(np.float64), eps))

        cur_soft_counts = _forward_backward(log_emiss, log_A, log_pi)
        table_path_key = os.path.join(table_path, str(key), f"{str(key)}_soft_counts")
        soft_counts_out[str(key)] = deepof.utils.save_dt(cur_soft_counts, table_path_key, coordinates._very_large_project)

    soft_counts_out = deepof.data.TableDict(
        soft_counts_out,
        typ="unsupervised_counts",
        table_path=table_path,
        exp_conditions=coordinates.get_exp_conditions,
    )

    return soft_counts_out



def get_supervised_chaos(
    coordinates,
    quality_threshold: float = 0.75,
    frac_bps_below: float = 0.5,
    chaos_suffix: str = "chaos",
):  
    """Create a supervised-annotations-like table dict containing only quality-based chaos labels.

    Args:
        coordinates (coordinates): deepof.Coordinates object for the project at hand.
        quality_threshold (float): Per-bodypart quality threshold below which a bodypart is counted as low quality.
        frac_bps_below (float): Fraction of bodyparts that need to fall below ``quality_threshold`` for a frame
            to be flagged as chaotic for a given animal.
        chaos_suffix (str): Suffix used for per-animal chaos columns. Resulting columns are of the form
            ``"{animal_id}_{chaos_suffix}"``.

    Returns:
        supervised_chaos (table_dict): table dict with one table per experiment containing
            per-animal chaos columns and one additional ``"any_chaos"`` column.
    """
    quality = coordinates.get_quality()
    if quality is None:
        raise ValueError("Could not obtain quality tables from coordinates.get_quality().")

    animal_ids = coordinates._animal_ids
    if animal_ids is None or animal_ids == "":
        animal_ids = [""]
    else:
        animal_ids = [aid + "_" for aid in animal_ids]
    keys = list(coordinates._tables.keys())
    table_path = os.path.join(coordinates._project_path, coordinates._project_name, "Tables")

    out = {}

    for key in keys:
        q_df = get_dt(quality, key).copy()
        pose = get_dt(coordinates._tables, key)
        chaos_df = pd.DataFrame(index=q_df.index.copy())

        per_animal_chaos = []
        for mid in animal_ids:
            cols = [c for c in q_df.columns if str(c).startswith(f"{mid}")]

            if len(cols) == 0:
                raise ValueError("Provided animal_id is not in quality table!")
            else:
                arr = q_df.loc[:, cols].to_numpy(dtype=np.float32, copy=True)
                bad = (~np.isfinite(arr)) | (arr < float(quality_threshold))
                frac_bad = bad.mean(axis=1)
                chaos = (frac_bad >= float(frac_bps_below)).astype(np.float32)

            chaos_df[f"{mid}{chaos_suffix}"] = chaos
            per_animal_chaos.append(chaos.astype(bool))

        chaos_df["anychaos"] = np.logical_or.reduce(per_animal_chaos).astype(np.float32)

        table_path_key = os.path.join(table_path, key, f"{key}_supervised_chaos")
        out[key] = deepof.utils.save_dt(
            chaos_df,
            table_path_key,
            coordinates._very_large_project,
        )

    return deepof.data.TableDict(
        out,
        typ="supervised_annotation",
        table_path=table_path,
        exp_conditions=coordinates.get_exp_conditions,
    )


def add_chaos_gates(
    coordinates,
    soft_counts_dict,
    soft_counts_chaos_dict,
    supervised_chaos,
    extract_pair,
    window_size: int,
):  
    """Combine regular and chaos-specific soft counts gate-wise.

    Args:
        soft_counts_dict (dict): Dictionary mapping gate -> TableDict with regular soft counts.
        soft_counts_chaos_dict (dict): Dictionary mapping gate -> TableDict with chaos soft counts.
            Typically this contains a single gate generated from ``embedding_gates="any_chaos"``.
        supervised_annotations (table_dict): Table dict with frame-wise chaos annotations. Expected columns are
            ``"{animal_id}_chaos"`` and ``"any_chaos"``.
        extract_pair (list): Tuple of animal ids to extract.
        window_len (int): Window size used to produce the soft counts from frame-wise annotations.

    Returns:
        soft_counts_out (dict): Dictionary mapping gate -> TableDict with regular and chaos-specific
            soft counts concatenated along the feature axis.
    """

    out = {}
    chaos_cols = ["anychaos"]
    table_path = os.path.join(
        coordinates._project_path, coordinates._project_name, "Tables"
    ) 

    for gate, soft_counts_gate in soft_counts_dict.items():

        # get parallel soft_counts for chaos gmm detections
        soft_counts_chaos_gate = soft_counts_chaos_dict['behavior_combinations']

        result_gate = {}

        # add new chaos clusters for each soft_counts table
        for key in soft_counts_gate.keys():
            # Extract info
            ann = get_dt(supervised_chaos, key)
            sc1 = get_dt(soft_counts_gate, key).copy()
            sc2 = get_dt(soft_counts_chaos_gate, key).copy()  
            n_windows = sc1.shape[0]    
            needed_len = n_windows + window_size - 1
            ann_used = ann.iloc[:needed_len]


            # Check shapes
            n_windows = sc1.shape[0]
            if sc2.shape[0] != n_windows or not(ann_used.shape[0] >= n_windows):
                raise ValueError(
                    f"Soft_counts and soft_counts_chaos must have same length, annotations must have same lenght or longer (Error at key{key!r}): "
                    f"{sc1.shape[0]} vs {sc2.shape[0]} vs {ann.shape[0]}"
                )            

            per_signal = []
            for col in chaos_cols:
                raw = ann_used[col].to_numpy(dtype=np.float32, copy=False)
                win = np.convolve(
                    raw,
                    np.ones(window_size, dtype=np.float32),
                    mode="valid",
                ) > 0
                if win.shape[0] != n_windows:
                    raise ValueError(
                        f"Convolved length mismatch for key {key!r}, column {col!r}: "
                        f"{win.shape[0]} vs expected {n_windows}"
                    )
                per_signal.append(win)

            chaos_mask = np.logical_or.reduce(per_signal)

            # regular states only on non-chaotic windows
            sc1[chaos_mask, :] = 0

            # chaos states only on chaotic windows
            sc2[~chaos_mask, :] = 0

            # keep only the chaotic half from the chaos extractor
            n_cols_chaos = sc2.shape[1]
            if n_cols_chaos % 2 != 0:
                raise ValueError(
                    f"Chaos soft counts for key {key!r} have an odd number of columns "
                    f"({n_cols_chaos}); expected two equally sized chaos/non-chaos blocks."
                )

            combined = np.concatenate([sc1, sc2[:, n_cols_chaos // 2 :]], axis=1)

            gate_tag = _gate_to_tag(gate)
            table_path_key = os.path.join(
                table_path, key, f"{key}_soft_counts_combined_{gate_tag}"
            )
            result_gate[key] = deepof.utils.save_dt(
                combined,
                table_path_key,
                coordinates._very_large_project,
            )

        out[gate] = deepof.data.TableDict(
            result_gate,
            typ="unsupervised_counts",
            table_path=table_path,
            exp_conditions=coordinates.get_exp_conditions,
        )

    return out


def _cache_embeddings(
    coordinates,
    embeddings,
    keys: list,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:  
    """Cache embeddings in memory (unless very large project) and return lengths."""
    Z_by_key = (
        {k: np.asarray(get_dt(embeddings, k), dtype=np.float32) for k in keys}
        if not coordinates._very_large_project else {}
    )
    emb_len = {
        k: (
            Z_by_key[k].shape[0]
            if k in Z_by_key
            else get_dt(embeddings, k, only_metainfo=True)["shape"][0]
        )
        for k in keys
    }
    return Z_by_key, emb_len

def _get_Z(Z_by_key: dict, embeddings: dict, key: str) -> np.ndarray: 
    """Retrieve embedding array for a key, loading from disk if not cached."""
    Z = Z_by_key.get(key)
    if Z is None:
        Z = np.asarray(get_dt(embeddings, key), dtype=np.float32)
    return Z


def _mask_to_runs(mask: np.ndarray, min_len: int = 2) -> List[Tuple[int, int]]: 
    mask = np.asarray(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cut = np.where(np.diff(idx) > 1)[0]
    runs = []
    s = idx[0]
    for c in cut:
        e = idx[c] + 1
        if (e - s) >= min_len:
            runs.append((int(s), int(e)))
        s = idx[c + 1]
    e = idx[-1] + 1
    if (e - s) >= min_len:
        runs.append((int(s), int(e)))
    return runs


def _pcca_memberships(msm_model, n_macrostates: int) -> np.ndarray: 
    """Extract PCCA soft memberships (n_active, n_macrostates) from a deeptime MSM."""
    pcca = msm_model.pcca(int(n_macrostates))
    for attr in ("memberships", "chi"):
        if hasattr(pcca, attr):
            return np.asarray(getattr(pcca, attr), dtype=np.float32)
    raise RuntimeError("Could not obtain PCCA memberships. Check deeptime version.")


def _temporal_smooth(P: np.ndarray, win: int) -> np.ndarray: 
    """Apply a uniform moving-average along axis 0 (frames) of a 2-D softcount matrix."""
    if win is None or win <= 1 or P.shape[0] < win:
        return P
    out = uniform_filter1d(P.astype(np.float32), size=win, axis=0, mode="nearest")
    # re-normalise rows
    rs = out.sum(axis=1, keepdims=True)
    rs = np.maximum(rs, 1e-12)
    return out / rs


def _get_gating_series_and_gates(
    coordinates,
    animal_ids,
    window_size: int,
    supervised_annotations=None,
    embedding_gates: Any = "Center",
) -> Tuple[Dict[str, Dict], list]:  
    """Return per-key gating series and the gate labels used downstream."""
    dist_series_dict = get_pairwise_distances(
        coordinates,
        window_size,
        supervised_annotations=supervised_annotations,
        embedding_gates=embedding_gates,
        behavior_combinations=True,
    )

    first_key = list(dist_series_dict.keys())[0]
    gates = list(dist_series_dict[first_key].keys())

    if len(animal_ids) == 1 or len(animal_ids) > 4:
        gates = gates[:1] if gates else [""]

    return dist_series_dict, gates


def compute_gate_edges(
    coordinates,
    animal_ids: list,
    *,
    keys: Optional[list] = None,
    window_size: int = 12,
    supervised_annotations=None,
    M_gates: int = 3,
    embedding_gates: Any = "Center",
    fixed_edges: Optional[list] = None,
) -> Optional[Dict[Any, np.ndarray]]: 
    """
    Precompute bin edges for distance-gated extraction.

    Behavior is intentionally identical to the original in-function logic:
    - supervised gating -> return None
    - fixed_edges provided -> validate and use them
    - otherwise -> compute quantile edges from the full gating series
    """
    if animal_ids is None:
        animal_ids = coordinates._animal_ids

    if not isinstance(embedding_gates, str):
        M_gates = 2 ** len(set(embedding_gates))

    dist_series_dict, gates = _get_gating_series_and_gates(
        coordinates=coordinates,
        animal_ids=animal_ids,
        window_size=window_size,
        supervised_annotations=supervised_annotations,
        embedding_gates=embedding_gates,
    )

    if keys is None:
        keys = list(dist_series_dict.keys())

    if len(animal_ids) == 1 or len(animal_ids) > 4:
        M_gates = 1

    if supervised_annotations is not None:
        return None

    if fixed_edges is not None:
        if len(fixed_edges) != M_gates + 1:
            raise ValueError("fixed_edges must have length \"M_gates\"+1")
        edges = np.asarray(fixed_edges, dtype=np.float64).copy()
        edges[0], edges[-1] = -np.inf, np.inf
        return {gate: edges.copy() for gate in gates}

    qs = np.linspace(0, 1, M_gates + 1)
    gate_edges = {}

    for gate in gates:
        full_g = np.concatenate([dist_series_dict[key][gate] for key in keys])
        edges = np.nanquantile(full_g, qs).astype(np.float64)
        edges[0], edges[-1] = -np.inf, np.inf
        gate_edges[gate] = edges

    return gate_edges


def _build_gate_masks(
    keys: list,
    emb_len: Dict[str, int],
    dist_series_dict: Dict[str, Dict],
    gates: list,
    M_gates: int,
    supervised_annotations=None,
    gate_edges: Optional[Dict[Any, np.ndarray]] = None,
) -> Dict[Any, Dict[int, Dict[str, np.ndarray]]]:  
    """Build per-(gate, bin, key) masks exactly as in the original code."""
    gate_masks: Dict[Any, Dict[int, Dict[str, np.ndarray]]] = {}

    for gate in gates:
        full_g = np.concatenate([dist_series_dict[key][gate] for key in keys])
        gate_masks[gate] = {}

        if supervised_annotations is not None:
            for b in range(M_gates):
                in_bin = (full_g == b)
                gate_masks[gate][b] = {}
                cum = 0
                for key in keys:
                    T = emb_len[key]
                    gate_masks[gate][b][key] = in_bin[cum:cum + T]
                    cum += T
        else:
            if gate_edges is None:
                raise ValueError(
                    "gate_edges must be provided when supervised_annotations is None. "
                    "Use compute_gate_edges() first."
                )

            edges = np.asarray(gate_edges[gate], dtype=np.float64)
            if len(edges) != M_gates + 1:
                raise ValueError(
                    f"gate_edges[{gate!r}] must have length {M_gates + 1}, got {len(edges)}"
                )

            for b in range(M_gates):
                in_bin = (full_g > edges[b]) & (full_g <= edges[b + 1])
                gate_masks[gate][b] = {}
                cum = 0
                for key in keys:
                    T = emb_len[key]
                    gate_masks[gate][b][key] = in_bin[cum:cum + T]
                    cum += T

    return gate_masks


def _reservoir_sample(segments: List[np.ndarray], n: int, seed: int = 0) -> np.ndarray: 
    """Reservoir-sample up to n rows from a list of 2-D arrays without full concat."""
    rng = np.random.default_rng(seed)
    total = sum(s.shape[0] for s in segments)
    if total <= n:
        return np.concatenate(segments, axis=0)

    # pre-allocate reservoir
    D = segments[0].shape[1]
    buf = np.empty((n, D), dtype=np.float32)
    filled = 0
    seen = 0

    for seg in segments:
        for row in seg:
            if filled < n:
                buf[filled] = row
                filled += 1
            else:
                j = int(rng.integers(0, seen + 1))
                if j < n:
                    buf[j] = row
            seen += 1

    return buf[:filled]


def get_pairwise_distances(
    coordinates,
    window_len: int,
    supervised_annotations=None,
    embedding_gates: Any = "Nose",
    behavior_combinations: bool = True,
) -> Dict[str, Dict]: 
    """
    Per-window gating series: pairwise distances OR behavior-combination codes.

    Fixes vs original:
      - deterministic behavior ordering (sorted, not set)
      - guards against all-NaN distance columns
      - reports which behaviors were dropped
      - validates bodypart existence in distance mode
    """
    animal_ids = coordinates._animal_ids
    gating = None

    # ---- decide mode ----
    if (animal_ids and 2 <= len(animal_ids) <= 4
            and supervised_annotations is None
            and isinstance(embedding_gates, str)):
        gating = "distances"
        animal_pairs = list(combinations(list(animal_ids), 2))

    elif animal_ids and supervised_annotations is not None:
        # FIX: sorted list, not set → deterministic bit-coding
        if isinstance(embedding_gates, str):
            embedding_gates = [embedding_gates]
        requested = sorted(set(embedding_gates))

        first_key = list(supervised_annotations.keys())[0]
        available = set(get_dt(supervised_annotations, first_key, only_metainfo=True)["columns"])
        valid = [b for b in requested if b in available]
        dropped = [b for b in requested if b not in available]

        if dropped:
            print(f"[gating] Dropped unavailable behaviors: {dropped}")
        if not valid:
            print("[gating] No valid behaviors remain; falling back to no gating.")
        else:
            gating = "behaviors"
            embedding_gates = valid  # ordered list

    out = {}
    kern = np.ones(window_len, dtype=np.float32)

    for key in coordinates._tables.keys():
        tab = get_dt(coordinates._tables, key)
        out[key] = {}

        if gating == "distances":
            for a_id, b_id in animal_pairs:
                # FIX: validate columns exist
                cx = (f"{a_id}_{embedding_gates}", "x")
                if cx not in tab.columns:
                    raise KeyError(
                        f"Bodypart column {cx} not found in table '{key}'. "
                        f"Available: {[c for c in tab.columns if c[1]=='x'][:10]}..."
                    )

                ax = tab[(f"{a_id}_{embedding_gates}", "x")].to_numpy(np.float64)
                ay = tab[(f"{a_id}_{embedding_gates}", "y")].to_numpy(np.float64)
                bx = tab[(f"{b_id}_{embedding_gates}", "x")].to_numpy(np.float64)
                by = tab[(f"{b_id}_{embedding_gates}", "y")].to_numpy(np.float64)

                dist_raw = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2).astype(np.float32)

                # FIX: guard against all-NaN
                mask = np.isfinite(dist_raw)
                if mask.any():
                    idx = np.arange(dist_raw.size)
                    dist_raw = np.interp(idx, idx[mask], dist_raw[mask]).astype(np.float32)
                else:
                    dist_raw = np.zeros_like(dist_raw)

                out[key][(a_id, b_id)] = np.convolve(dist_raw, kern / window_len, mode="valid")

        elif gating == "behaviors":
            sup = get_dt(supervised_annotations, key)
            cols = []
            for beh in embedding_gates:  # deterministic order
                raw = sup[beh].to_numpy()
                win = (np.convolve(raw, kern, mode="valid") > 0).astype(np.int32)
                if not behavior_combinations:
                    out[key][beh] = win
                else:
                    cols.append(win)

            if behavior_combinations and cols:
                arr = np.array(cols, dtype=np.int32)       # (n_beh, T)
                powers = 2 ** np.arange(len(cols), dtype=np.int32)
                out[key]["behavior_combinations"] = (powers @ arr).astype(np.int32)

        else:
            # no gating
            out[key][""] = np.convolve(
                np.ones(tab.shape[0], dtype=np.float32), kern / window_len, mode="valid"
            )

    return out

def _gate_to_tag(gate: Any) -> str:  
    """Convert a gate key to a filesystem-safe tag."""
    if isinstance(gate, tuple):
        return "_".join(map(str, gate))
    if gate in ("", None):
        return "all"
    return str(gate).replace(os.sep, "-").replace(" ", "_")


def get_contrastive_soft_counts_gmm(
    coordinates,
    embeddings: Dict[str, np.ndarray],
    animal_ids: list,
    window_size: int = 12,
    supervised_annotations=None,
    N_clusters_per_gate: int = 8,
    M_gates: int = 3,
    gate_edges: Optional[Dict[Any, np.ndarray]] = None,
    reg_covar: float = 1e-5,
    sample_size: int = 200000,
    random_state: int = 0,
    embedding_gates: Any = "Center",
    temporal_smooth_win: Optional[int] = 3,
):  
    """
    Distance/behavior-gated GMM decoder.

    Returns:
        Dict[Any, TableDict]: one soft-count TableDict per gate.
        For pairwise distance gating, keys are animal pairs like ("A", "B").
    """
    keys = list(embeddings.keys())
    if not keys:
        raise ValueError("Embeddings are empty.")
    if animal_ids is None:
        animal_ids = coordinates._animal_ids

    # ---- cache embeddings + lengths ----
    Z_by_key, emb_len = _cache_embeddings(coordinates, embeddings, keys)

    if not isinstance(embedding_gates, str):
        M_gates = 2 ** len(sorted(set(embedding_gates)))

    # ---- gating series + masks ----
    dist_series_dict, gates = _get_gating_series_and_gates(
        coordinates=coordinates,
        animal_ids=animal_ids,
        window_size=window_size,
        supervised_annotations=supervised_annotations,
        embedding_gates=embedding_gates,
    )

    if len(animal_ids) == 1 or len(animal_ids) > 4:
        M_gates = 1

    gate_masks = _build_gate_masks(
        keys=keys,
        emb_len=emb_len,
        dist_series_dict=dist_series_dict,
        gates=gates,
        M_gates=M_gates,
        supervised_annotations=supervised_annotations,
        gate_edges=gate_edges,
    )

    # ---- fit GMM per (gate, bin) ----
    models: Dict[Any, List] = {}
    total_steps = len(gates) * M_gates

    with tqdm.tqdm(
        total=total_steps,
        desc=f"{'Fit GMMs':<{PROGRESS_BAR_FIXED_WIDTH}}",
        unit="gmm",
    ) as pbar:
        for gate_idx, gate in enumerate(gates):
            models[gate] = []
            for b in range(M_gates):
                seed_b = int(random_state + 17 * b + 3 * gate_idx)

                bin_segments = []
                n_windows = 0
                for key in keys:
                    Z = _get_Z(Z_by_key, embeddings, key)
                    mask = gate_masks[gate][b][key]
                    idx = np.flatnonzero(mask)
                    n_windows += idx.size
                    if idx.size > 0:
                        bin_segments.append(Z[idx, :])

                if n_windows < max(10, N_clusters_per_gate):
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                X_fit = _reservoir_sample(bin_segments, int(sample_size), seed=seed_b)

                gmm = GaussianMixture(
                    n_components=int(N_clusters_per_gate),
                    covariance_type="full",
                    reg_covar=float(reg_covar),
                    random_state=seed_b,
                    init_params="kmeans",
                    max_iter=200,
                    tol=1e-3,
                ).fit(X_fit)

                models[gate].append({"gmm": gmm})
                pbar.update(1)

    # ---- decode per gate ----
    K_total = M_gates * N_clusters_per_gate
    soft_counts_out_by_gate = {gate: {} for gate in gates}
    table_path = os.path.join(
        coordinates._project_path, coordinates._project_name, "Tables"
    )

    for key in tqdm.tqdm(
        keys,
        desc=f"{'Decode softcounts (GMM)':<{PROGRESS_BAR_FIXED_WIDTH}}",
        unit="table",
    ):
        Z0 = _get_Z(Z_by_key, embeddings, key)

        for gate in gates:
            P = np.full((Z0.shape[0], K_total), float(1e-4), dtype=np.float32)

            for b in range(M_gates):
                model = models[gate][b]
                mask = gate_masks[gate][b][key]
                block = slice(
                    b * N_clusters_per_gate,
                    (b + 1) * N_clusters_per_gate,
                )

                if model is None:
                    if np.any(mask):
                        P[mask, block] = 1.0 / N_clusters_per_gate
                    continue

                idx = np.flatnonzero(mask)
                if idx.size == 0:
                    continue

                Z = Z0[idx, :]
                R = model["gmm"].predict_proba(Z).astype(np.float32, copy=False)
                P[idx, block] = R

            if temporal_smooth_win and temporal_smooth_win > 1:
                P = _temporal_smooth(P, temporal_smooth_win)

            rs = P.sum(axis=1, keepdims=True)
            P = P / np.maximum(rs, 1e-12)

            gate_tag = _gate_to_tag(gate)
            table_path_key = os.path.join(
                table_path, key, f"{key}_soft_counts_gmm_{gate_tag}"
            )
            soft_counts_out_by_gate[gate][key] = deepof.utils.save_dt(
                P, table_path_key, coordinates._very_large_project
            )

    return {
        gate: deepof.data.TableDict(
            soft_counts_out_by_gate[gate],
            typ="unsupervised_counts",
            table_path=table_path,
            exp_conditions=coordinates.get_exp_conditions,
        )
        for gate in gates
    }


def get_contrastive_soft_counts_msm_pcca(
    coordinates,
    embeddings: Dict[str, np.ndarray],
    animal_ids: list,
    window_size: int = 12,
    supervised_annotations=None,
    N_clusters_per_gate: int = 10,
    M_gates: int = 3,
    gate_edges: Optional[Dict[Any, np.ndarray]] = None,
    sample_size: int = 200000,
    random_state: int = 0,
    embedding_gates: Any = "Center",
    temporal_smooth_win: Optional[int] = 3,
    n_micro: int = 400,
    min_micro_per_macro: int = 3,
    lagtime: int = 3,
):  
    """
    Distance/behavior-gated MSM + PCCA with k-means microstates.

    Returns:
        Dict[Any, TableDict]: one soft-count TableDict per gate.
        For pairwise distance gating, keys are animal pairs like ("A", "B").
    """
    keys = list(embeddings.keys())
    if not keys:
        raise ValueError("Embeddings are empty.")
    if animal_ids is None:
        animal_ids = coordinates._animal_ids

    # ---- cache embeddings + lengths ----
    Z_by_key, emb_len = _cache_embeddings(coordinates, embeddings, keys)

    if not isinstance(embedding_gates, str):
        M_gates = 2 ** len(sorted(set(embedding_gates)))

    # ---- gating series + masks ----
    dist_series_dict, gates = _get_gating_series_and_gates(
        coordinates=coordinates,
        animal_ids=animal_ids,
        window_size=window_size,
        supervised_annotations=supervised_annotations,
        embedding_gates=embedding_gates,
    )

    if len(animal_ids) == 1 or len(animal_ids) > 4:
        M_gates = 1

    gate_masks = _build_gate_masks(
        keys=keys,
        emb_len=emb_len,
        dist_series_dict=dist_series_dict,
        gates=gates,
        M_gates=M_gates,
        supervised_annotations=supervised_annotations,
        gate_edges=gate_edges,
    )

    # ---- fit per (gate, bin) ----
    models: Dict[Any, List] = {}
    total_steps = len(gates) * M_gates

    with tqdm.tqdm(
        total=total_steps,
        desc=f"{'Fit MSM/PCCA':<{PROGRESS_BAR_FIXED_WIDTH}}",
        unit="model",
    ) as pbar:
        for gate_idx, gate in enumerate(gates):
            models[gate] = []
            for b in range(M_gates):
                seed_b = int(random_state + 1000 * gate_idx + 17 * b)

                seg_spatial: List[np.ndarray] = []
                seg_temporal: List[np.ndarray] = []
                n_windows = 0

                for key in keys:
                    Z = _get_Z(Z_by_key, embeddings, key)
                    mask = gate_masks[gate][b][key]
                    n_windows += int(mask.sum())

                    for s, e in _mask_to_runs(mask, min_len=2):
                        seg = Z[s:e, :]
                        seg_spatial.append(seg)
                        if seg.shape[0] >= lagtime + 2:
                            seg_temporal.append(seg)

                if not seg_spatial or n_windows < max(50, 5 * N_clusters_per_gate):
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                X_fit = _reservoir_sample(seg_spatial, int(sample_size), seed=seed_b)

                scaler = StandardScaler()
                X_fit = scaler.fit_transform(X_fit)

                n_micro_eff = int(
                    min(
                        n_micro,
                        max(
                            min_micro_per_macro * N_clusters_per_gate,
                            X_fit.shape[0] // 50,
                        ),
                    )
                )
                n_micro_eff = max(n_micro_eff, 2)

                kmeans = MiniBatchKMeans(
                    n_clusters=n_micro_eff,
                    batch_size=4096,
                    max_iter=200,
                    random_state=seed_b,
                    init="k-means++",
                    n_init="auto",
                ).fit(X_fit)

                if not seg_temporal:
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                dtrajs = []
                for seg in seg_temporal:
                    Xs = scaler.transform(seg)
                    dtrajs.append(np.asarray(kmeans.predict(Xs), dtype=np.int32))

                if not dtrajs:
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                counts_est = TransitionCountEstimator(
                    lagtime=int(lagtime),
                    count_mode="sliding",
                ).fit(dtrajs)
                count_model = counts_est.fetch_model()

                msm = MaximumLikelihoodMSM(
                    reversible=True,
                ).fit(count_model).fetch_model()

                n_msm = msm.n_states if hasattr(msm, "n_states") else None
                active_syms = None
                for src in (count_model, msm):
                    for attr in ("active_set", "state_symbols"):
                        cand = getattr(src, attr, None)
                        if (
                            cand is not None
                            and not isinstance(cand, types.MethodType)
                            and not isinstance(cand, types.FunctionType)
                        ):
                            cand = np.asarray(cand, dtype=np.int32)
                            if n_msm is None or cand.shape[0] == n_msm:
                                active_syms = cand
                                break
                    if active_syms is not None:
                        break

                n_active = (
                    active_syms.shape[0]
                    if active_syms is not None
                    else (n_msm or 0)
                )
                if n_active < 2:
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                if active_syms is None:
                    active_syms = np.arange(n_active, dtype=np.int32)

                K_request = int(min(N_clusters_per_gate, n_active))
                if K_request < 2:
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                try:
                    chi_eff = _pcca_memberships(msm, K_request)
                except Exception:
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                chi_eff = np.asarray(chi_eff, dtype=np.float32)
                if chi_eff.shape[0] != n_active:
                    models[gate].append(None)
                    pbar.update(1)
                    continue

                if chi_eff.shape[1] == N_clusters_per_gate:
                    chi = chi_eff
                else:
                    chi = np.zeros(
                        (n_active, N_clusters_per_gate), dtype=np.float32
                    )
                    chi[:, :chi_eff.shape[1]] = chi_eff
                    rs = chi.sum(axis=1, keepdims=True)
                    good = rs.squeeze(-1) > 0
                    chi[good] /= rs[good]

                micro2macro = np.full(
                    (n_micro_eff, N_clusters_per_gate),
                    1.0 / N_clusters_per_gate,
                    dtype=np.float32,
                )
                for i in range(n_active):
                    s = int(active_syms[i])
                    if 0 <= s < n_micro_eff:
                        micro2macro[s, :] = chi[i, :]

                models[gate].append(
                    {
                        "scaler": scaler,
                        "kmeans": kmeans,
                        "micro2macro": micro2macro,
                    }
                )
                pbar.update(1)

    # ---- decode per gate ----
    K_total = M_gates * N_clusters_per_gate
    soft_counts_out_by_gate = {gate: {} for gate in gates}
    table_path = os.path.join(
        coordinates._project_path, coordinates._project_name, "Tables"
    )

    for key in tqdm.tqdm(
        keys,
        desc=f"{'Decode softcounts (MSM/PCCA)':<{PROGRESS_BAR_FIXED_WIDTH}}",
        unit="table",
    ):
        Z0 = _get_Z(Z_by_key, embeddings, key)

        for gate in gates:
            P = np.full((Z0.shape[0], K_total), float(1e-4), dtype=np.float32)

            for b in range(M_gates):
                model = models[gate][b]
                mask = gate_masks[gate][b][key]
                block = slice(
                    b * N_clusters_per_gate,
                    (b + 1) * N_clusters_per_gate,
                )

                if model is None:
                    if np.any(mask):
                        P[mask, block] = 1.0 / N_clusters_per_gate
                    continue

                scaler = model["scaler"]
                kmeans = model["kmeans"]
                m2m = model["micro2macro"]

                for s, e in _mask_to_runs(mask, min_len=2):
                    seg = Z0[s:e, :]
                    Xs = scaler.transform(seg)
                    d = np.asarray(kmeans.predict(Xs), dtype=np.int32)
                    P[s:e, block] = m2m[d, :]

            if temporal_smooth_win and temporal_smooth_win > 1:
                P = _temporal_smooth(P, temporal_smooth_win)

            rs = P.sum(axis=1, keepdims=True)
            P = P / np.maximum(rs, 1e-12)

            gate_tag = _gate_to_tag(gate)
            table_path_key = os.path.join(
                table_path, key, f"{key}_soft_counts_msmpcca_{gate_tag}"
            )
            soft_counts_out_by_gate[gate][key] = deepof.utils.save_dt(
                P, table_path_key, coordinates._very_large_project
            )

    return {
        gate: deepof.data.TableDict(
            soft_counts_out_by_gate[gate],
            typ="unsupervised_counts",
            table_path=table_path,
            exp_conditions=coordinates.get_exp_conditions,
        )
        for gate in gates
    }


def recluster(
    coordinates: coordinates,
    embeddings: table_dict,
    soft_counts: table_dict = None,
    min_confidence: float = 0.75,
    states: Union[str, int] = "aic",
    pretrained: Union[bool, str] = False,
    covariance_type: str = "diag",
    min_states: int = 2,
    max_states: int = 12,
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
        exclude_keys (list): list of keys to exclude
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
            embeddings,
            states,
            min_states,
            max_states,
            covariance_type="diag",
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

    # remove experiment conditions for which potentially no soft_counts were generated
    exp_conds=coordinates.get_exp_conditions
    exp_conds=exp_conds = {key: coordinates.get_exp_conditions[key] for key in embeddings.keys() if key in coordinates.get_exp_conditions}
    if len(exp_conds)==0:
        exp_conds=None

    # Predict on each animal experiment
    soft_counts = hmm_model.predict_proba(concat_embeddings)
    soft_counts = deepof.data.TableDict(
        {
            key: np.array(soft_counts[i][: embeddings[key].shape[0]])
            for i, key in enumerate(embeddings.keys())
        },
        typ="unsupervised_counts",
        table_path=os.path.join(coordinates._project_path, coordinates._project_name, "Tables"),
        exp_conditions=exp_conds,
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
        hard_counts = deepof.utils.row_nanargmax(preloaded[key])

        if roi_number is not None:
            hard_counts = deepof.visuals_utils.get_unsupervised_behaviors_in_roi(hard_counts, bin_info[key], animals_in_roi)
            hard_counts=hard_counts[hard_counts >= 0]
        
        
        # Create dictionary with number of bin_occurences per bin
        hard_count_counters[key] = Counter(hard_counts[~np.isnan(hard_counts)])

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
    redundant_sets = []

    def load_single_key(key):
        arr_range = None
        if isinstance(bin_info, dict):
            arr_range = bin_info[key]["time"]
        else:
            arr_range = bin_info
        return key, get_dt(embedding, key, load_range=arr_range)
    
    # Not yet used
    #def collect_redundant(current_embedding: pd.DataFrame, corr_threshold: float = 0.90, min_obs: int = 10) -> set:
    #    """Collect columns with absolute correlation above threshold."""
    #    corr = current_embedding.corr(min_periods=min_obs).abs()
    #    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    #    return {c for c in upper.columns if any(upper[c] > corr_threshold)}

    max_workers = min(32, (cpu_count() or 1) + 4) 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_key, key): key for key in embedding}
        for future in as_completed(futures):
            key, result = future.result()
            preloaded[key] = result


    for key in embedding.keys():
        
        current_embedding = preloaded[key]
        if roi_number is not None and type(current_embedding) == pd.DataFrame:
            current_embedding = deepof.visuals_utils.get_supervised_behaviors_in_roi(current_embedding, bin_info[key], animals_in_roi, roi_mode)
        elif roi_number is not None and type(current_embedding) == np.ndarray:
            current_embedding = deepof.visuals_utils.get_unsupervised_behaviors_in_roi(current_embedding, bin_info[key], animals_in_roi)

        # currently suboptimal, will be improved in a future version 
        if not type(current_embedding) == pd.DataFrame:  
            current_embedding = pd.DataFrame(current_embedding)

        if agg == "mean":
            agg_embedding[key] = np.nanmean(current_embedding, axis=0)
        elif agg == "median":
            agg_embedding[key] = np.nanmedian(current_embedding, axis=0)
    
    agg_embedding = pd.DataFrame(agg_embedding, index=current_embedding.columns).T

    # Drop columns that were redundant in ALL keys
    if True: #redundant_sets:
        cols_to_drop = [s for s in current_embedding.columns if 'distance' in str(s)] #set.intersection(*redundant_sets) & set(agg_embedding.columns)
        agg_embedding = agg_embedding.drop(columns=cols_to_drop)

    n_rows=agg_embedding.shape[0]

    if agg_embedding.isnull().any().any():
        agg_embedding_clean=agg_embedding.dropna()
        assert agg_embedding_clean.shape[0]>0, "agg_embeddings empty after NaN-row removal!"

        warning_message = (
            "\033[38;5;208m\n"  # Set text color to orange
            "Warning! Some rows of aggregated embeddings contained NaNs that were dropped! This can happen if the\n"
            "time bins are short or ROIs are strict, which leads to behaviors never occuring in some experiments.\n"
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
                selected_columns = [col for col in current_sa.columns if col.endswith("speed")]
            else:
                selected_columns = [col for col in current_sa.columns if not col.endswith(tuple(CONTINUOUS_BEHAVIORS))]

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

    enrichment = (
        counter_df
        .reset_index(names="exp_id")
        .melt(
            id_vars=["exp_id", "exp condition"],
            var_name="cluster",
            value_name="time on cluster",
        )
    )
    if enrichment["cluster"][0]==0:
        enrichment["cluster"] = enrichment["cluster"].astype(float)
    else:    
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
    
    # Check if clusters have collapsed
    assert np.unique(cluster_assignments).size > 1, "LDA could not be computed, as these soft_counts correspond to a collapsed model that only contains a single cluster!"
    
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
            ("classifier", ResampledClassifier(
                estimator=CatBoostClassifier(verbose=(verbose > 2)),
                resampler=SimpleSMOTE(random_state=42),
            )),
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
            ("classifier", ResampledClassifier(
                estimator=CatBoostClassifier(verbose=(verbose > 2)),
                resampler=SimpleSMOTE(random_state=42),
            )),
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
    scaler = full_cluster_clf.named_steps["normalization"]
    clfwrap = full_cluster_clf.named_steps["classifier"]

    X_scaled = scaler.transform(chunk_stats.values)

    resampler = getattr(clfwrap, "resampler_", None) or getattr(clfwrap, "resampler", None)
    if resampler is not None:
        X_scaled, _ = clone(resampler).fit_resample(X_scaled, hard_counts)

    processed_stats = pd.DataFrame(X_scaled, columns=chunk_stats.columns)

    n_clusters = len(np.unique(hard_counts))
    explainer = shap.KernelExplainer(
        clfwrap.predict_proba,  # this expects scaled input
        data=shap.kmeans(processed_stats, n_clusters),
        normalize=False,
    )
    if samples is not None and samples < chunk_stats.shape[0]:
        processed_stats = processed_stats.sample(samples)
    shap_values = explainer.shap_values(
        processed_stats, nsamples=samples, n_jobs=n_jobs
    )

    return shap_values, explainer, processed_stats
