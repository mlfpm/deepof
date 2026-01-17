# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Utility functions for both training autoencoder models in deepof.models and tuning hyperparameters with deepof.hypermodels."""

import os
from datetime import date, datetime
from typing import Any, List, NewType, Tuple, Union, Dict, Callable, Optional
from contextlib import nullcontext
import copy
from dataclasses import dataclass, asdict

from IPython.display import clear_output
import matplotlib.pyplot as plt
import psutil
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Distribution, TransformedDistribution
from torch.distributions.transforms import AffineTransform

from keras_tuner import BayesianOptimization, Hyperband, Objective
from spektral.layers import CensNetConv
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import (
    GRU,
    Bidirectional,
    LayerNormalization,
    TimeDistributed,
)

from deepof.config import PROGRESS_BAR_FIXED_WIDTH
import deepof.data
import deepof.hypermodels
import deepof.models
import deepof.clustering.models_new
import deepof.post_hoc
from deepof.data_loading import get_dt, save_dt
import deepof.clustering.dataset
import warnings
from deepof.clustering.dataset import reorder_and_reshape

tfb = tfp.bijectors
tfd = tfp.distributions
tfpl = tfp.layers

# Ignore warning with no downstream effect
tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


### CONFIGS
@dataclass
class CommonFitCfg:
    # Core identity
    model_name: str = "VaDE"
    encoder_type: str = "recurrent"

    # Training loop
    batch_size: int = 1024
    latent_dim: int = 6
    epochs: int = 10
    n_components: int = 10

    # IO / logging
    output_path: str = "."
    data_path: str = "."
    log_history: bool = True
    pretrained: Optional[str] = None
    save_weights: bool = True
    run: int = 0

    # System
    num_workers: int = 0
    prefetch_factor: int = 0
    use_amp: bool = False

    # Shared regularization knobs
    interaction_regularization: float = 0.0
    kmeans_loss: float = 0.0

    kl_annealing_mode: str = "sigmoid"
    kl_max_weight: float = 1.0
    kl_warmup: int = 5
    kl_end_weight: float = 0.2
    kl_cooldown: int = 5

    # Diagnostics
    diag_max_batches: int = 4


@dataclass
class TurtleTeacherCfg:
    # Teacher on/off + core
    use_turtle_teacher: bool = False
    teacher_gamma: float = 8.0
    teacher_outer_steps: int = 500
    teacher_inner_steps: int = 100
    teacher_normalize_feats: bool = True

    teacher_head_temp: float = 0.35
    teacher_task_temp: float = 0.35
    teacher_alpha_sample_entropy: float = 2.0

    # Distillation (VaDE)
    lambda_distill: float = 4.0
    lambda_decay_start: int = 10
    lambda_end_weight: float = 0.2,
    lambda_cooldown: int = 10,
    distill_sharpen_T: float = 0.5
    distill_conf_weight: bool = False
    distill_conf_thresh: float = 0.3

    # Distillation (generic head for VQVAE/Contrastive)
    generic_lambda_distill: float = 2.0
    generic_distill_sharpen_T: float = 0.5
    generic_distill_conf_weight: bool = True
    generic_distill_conf_thresh: float = 0.6
    generic_distill_warmup_epochs: int = 1

    distill_class_reweight_beta: float = 1.0
    distill_class_reweight_cap: float = 3.0

    # Views
    include_latent_view: bool = True,
    include_edges_view: bool = False
    include_nodes_view: bool = True
    include_angles_view: bool = False
    include_supervised_view: bool = False
    pca_nodes_dim: int = 32
    pca_edges_dim: int = 32
    pca_angles_dim: int = 32

    # Refresh
    teacher_refresh_every: Optional[int] = None
    teacher_freeze_at: Optional[int] = 10
    reinit_gmm_on_refresh: bool = False


@dataclass
class VaDECfg:
    reg_cat_clusters: float = 0.0
    recluster: bool = False
    freeze_gmm_epochs: int = 0
    freeze_decoder_epochs: int = 0
    prior_loss_weight: float = 0.0

    reg_scatter_weight: float = 0.0
    temporal_cohesion_weight: float = 0.0
    reg_scatter_beta: float = 1.0
    repel_weight: float = 0.0
    repel_length_scale: float = 1.0

    tf_cluster_weight: float = 0.0
    nonempty_weight: float = 2e-2


@dataclass
class ContrastiveCfg:
    temperature: float = 0.1
    contrastive_similarity_function: str = "cosine"
    contrastive_loss_function: str = "nce"
    beta: float = 0.1
    tau: float = 0.1


def _append_cfg(lines, title: str, cfg) -> None:
    if cfg is None:
        return

    lines.append(f"[{title}]")
    d = asdict(cfg)  # flat dict
    for k in d.keys(): 
        lines.append(f"{k}: {d[k]}")
    lines.append("")  # spacer


def save_model_info(
    ckpt_path: str,
    *,
    stage: str,
    epoch: Optional[int] = None,
    train_steps: Optional[int] = None,
    val_total: Optional[float] = None,
    score_value: Optional[float] = None,
    extra: Optional[dict] = None,
    # NEW: configs passed in (or you can close over them)
    common_cfg=None,
    teacher_cfg=None,
    vade_cfg=None,
    contrastive_cfg=None,
) -> None:
    info_path = os.path.splitext(ckpt_path)[0] + "_info.txt"
    lines = []
    lines.append(f"stage: {stage}")
    if epoch is not None:
        lines.append(f"epoch: {int(epoch)}")
    if train_steps is not None:
        lines.append(f"train_steps: {int(train_steps)}")
    if val_total is not None:
        lines.append(f"val_total: {float(val_total)}")
    if score_value is not None:
        lines.append(f"score_value: {float(score_value)}")
    lines.append("")

    # Dump configs
    _append_cfg(lines, "common_cfg", common_cfg)
    _append_cfg(lines, "teacher_cfg", teacher_cfg)
    _append_cfg(lines, "vade_cfg", vade_cfg)
    _append_cfg(lines, "contrastive_cfg", contrastive_cfg)

    if extra:
        lines.append("[extra]")
        for k in sorted(extra.keys()):
            lines.append(f"{k}: {extra[k]}")
        lines.append("")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


### CONTRASTIVE LEARNING UTILITIES
def select_contrastive_loss_pt(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: str,
    loss_fn: str = "nce",
    temperature: float = 0.1,
    tau: float = 0.1,
    beta: float = 0.1,
    elimination_topk: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sim_fn = _SIMILARITIES[similarity]

    if loss_fn == "nce":
        return nce_loss_pt(history, future, sim_fn, temperature)
    elif loss_fn == "dcl":
        return dcl_loss_pt(history, future, sim_fn, temperature, debiased=True, tau_plus=tau)
    elif loss_fn == "fc":
        return fc_loss_pt(history, future, sim_fn, temperature, elimination_topk=elimination_topk)
    elif loss_fn == "hard_dcl":
        return hard_loss_pt(history, future, sim_fn, temperature, beta=beta, debiased=True, tau_plus=tau)
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}")
    
def _l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


def _cosine_similarity_pt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x: (N, D), y: (N, D) -> (N, N)
    x1 = x.unsqueeze(1)  # (N, 1, D)
    y1 = y.unsqueeze(0)  # (1, N, D)
    return F.cosine_similarity(x1, y1, dim=2)


def _dot_similarity_pt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x @ y.t()


def _euclidean_similarity_pt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x1 = x.unsqueeze(1)  # (N, 1, D)
    y1 = y.unsqueeze(0)  # (1, N, D)
    d = torch.sqrt(torch.clamp(((x1 - y1) ** 2).sum(dim=2), min=0.0))
    s = 1.0 / (1.0 + d)
    return s


def _edit_similarity_pt(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Matches provided TF code (same as euclidean similarity transform)
    return _euclidean_similarity_pt(x, y)


_SIMILARITIES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cosine": _cosine_similarity_pt,
    "dot": _dot_similarity_pt,
    "euclidean": _euclidean_similarity_pt,
    "edit": _edit_similarity_pt,
}


def _off_diagonal_rows(sim: torch.Tensor) -> torch.Tensor:
    """
    Extract off-diagonal elements row-wise and reshape to (N, N-1),
    mirroring TF's boolean_mask+reshape.
    """
    N = sim.shape[0]
    mask = torch.ones((N, N), dtype=torch.bool, device=sim.device)
    mask.fill_diagonal_(False)
    flat = sim.reshape(-1)
    masked = flat[mask.reshape(-1)]
    return masked.reshape(N, N - 1)


def nce_loss_pt(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact port of provided TF nce_loss (including BCE-with-logits on a positive ratio).
    """
    N = history.shape[0]
    sim = similarity(history, future)  # (N, N)
    pos_sim = torch.exp(torch.diag(sim) / temperature)  # (N,)

    neg = _off_diagonal_rows(sim)  # (N, N-1)
    all_sim = torch.exp(sim / temperature)  # (N, N)

    logits = pos_sim.sum() / all_sim.sum(dim=1)  # (N,)
    labels = torch.ones_like(logits)

    bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
    loss = bce(logits, labels)

    mean_sim = torch.diag(sim).mean()
    mean_neg = neg.mean()
    return loss, mean_sim, mean_neg


def dcl_loss_pt(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    temperature: float = 0.1,
    debiased: bool = True,
    tau_plus: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = history.shape[0]
    sim = similarity(history, future)  # (N, N)
    pos_sim = torch.exp(torch.diag(sim) / temperature)  # (N,)

    neg = _off_diagonal_rows(sim)  # (N, N-1)
    neg_sim = torch.exp(neg / temperature)  # (N, N-1)

    if debiased:
        N_eff = N - 1
        Ng = (-tau_plus * N_eff * pos_sim + neg_sim.sum(dim=-1)) / (1.0 - tau_plus)
        min_clip = N_eff * math.e ** (-1.0 / temperature)
        Ng = torch.clamp(Ng, min=min_clip, max=torch.finfo(history.dtype).max)
    else:
        Ng = neg_sim.sum(dim=-1)

    loss = (-torch.log(pos_sim / (pos_sim + Ng))).mean()

    mean_sim = torch.diag(sim).mean()
    mean_neg = neg.mean()
    return loss, mean_sim, mean_neg


def fc_loss_pt(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    temperature: float = 0.1,
    elimination_topk: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = history.shape[0]
    elim = min(elimination_topk, 0.5)
    k = int(math.ceil(elim * N))

    sim = similarity(history, future) / temperature  # (N, N)
    pos_sim = torch.exp(torch.diag(sim))  # (N,)

    neg_sim_raw = _off_diagonal_rows(sim)  # (N, N-1)
    sorted_sim, _ = torch.sort(neg_sim_raw, dim=1)  # ascending

    if k == 0:
        k = 1
    keep = (N - 1) - k
    trimmed = sorted_sim[:, : max(keep, 0)]  # may be empty

    neg_sim = torch.exp(trimmed).sum(dim=1) if trimmed.numel() > 0 else torch.zeros(
        N, device=sim.device, dtype=sim.dtype
    )

    loss = (-torch.log(pos_sim / (pos_sim + neg_sim))).mean()

    mean_sim = torch.diag(sim).mean() * temperature
    mean_neg = trimmed.mean() * temperature if trimmed.numel() > 0 else torch.tensor(
        0.0, device=sim.device, dtype=sim.dtype
    )
    return loss, mean_sim, mean_neg


def hard_loss_pt(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    temperature: float,
    beta: float = 0.0,
    debiased: bool = True,
    tau_plus: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = history.shape[0]
    sim = similarity(history, future)  # (N, N)

    pos_sim = torch.exp(torch.diag(sim) / temperature)  # (N,)
    neg = _off_diagonal_rows(sim)  # (N, N-1)
    neg_sim = torch.exp(neg / temperature)  # (N, N-1)

    if beta == 0.0:
        reweight = torch.ones_like(neg_sim)
    else:
        denom = neg_sim.mean(dim=1, keepdim=True)
        reweight = (beta * neg_sim) / denom

    if debiased:
        N_eff = N - 1
        Ng = (-tau_plus * N_eff * pos_sim + (reweight * neg_sim).sum(dim=-1)) / (
            1.0 - tau_plus
        )
        min_clip = math.e ** (-1.0 / temperature)
        Ng = torch.clamp(Ng, min=min_clip, max=torch.finfo(history.dtype).max)
    else:
        Ng = neg_sim.sum(dim=-1)

    loss = (-torch.log(pos_sim / (pos_sim + Ng))).mean()

    mean_sim = torch.diag(sim).mean()
    mean_neg = neg.mean()
    return loss, mean_sim, mean_neg


def select_contrastive_loss(
    history,
    future,
    similarity,
    loss_fn="nce",
    temperature=0.1,
    tau=0.1,
    beta=0.1,
    elimination_topk=0.1,
):  # pragma: no cover
    """Select and applies the contrastive loss function to be used in the Contrastive embedding models.

    Args:
        history: Tensor of shape (batch_size, seq_len, embedding_dim).
        future: Tensor of shape (batch_size, seq_len, embedding_dim).
        similarity: Function that computes the similarity between two tensors.
        loss_fn: String indicating the loss function to be used.
        temperature: Float indicating the temperature to be used in the specified loss function.
        tau: Float indicating the tau value to be used if DCL or hard DLC are selected.
        beta: Float indicating the beta value to be used if hard DLC is selected.
        elimination_topk: Float indicating the top-k value to be used if FC is selected.

    """
    similarity_dict = {
        "cosine": _cosine_similarity,
        "dot": _dot_similarity,
        "euclidean": _euclidean_similarity,
        "edit": _edit_similarity,
    }
    similarity = similarity_dict[similarity]

    if loss_fn == "nce":
        loss, pos, neg = nce_loss(history, future, similarity, temperature)
    elif loss_fn == "dcl":
        loss, pos, neg = dcl_loss(
            history, future, similarity, temperature, debiased=True, tau_plus=tau
        )
    elif loss_fn == "fc":
        loss, pos, neg = fc_loss(
            history,
            future,
            similarity,
            temperature,
            elimination_topk=elimination_topk,
        )
    elif loss_fn == "hard_dcl":
        loss, pos, neg = hard_loss(
            history,
            future,
            similarity,
            temperature,
            beta=beta,
            debiased=True,
            tau_plus=tau,
        )

    # noinspection PyUnboundLocalVariable
    return loss, pos, neg


def _cosine_similarity(x, y):  # pragma: no cover
    """Compute the cosine similarity between two tensors."""
    v = tf.keras.losses.CosineSimilarity(
        axis=2, reduction=tf.keras.losses.Reduction.NONE
    )(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return -v


def _dot_similarity(x, y):  # pragma: no cover
    """Compute the dot product between two tensors."""
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)

    return v


def _euclidean_similarity(x, y):  # pragma: no cover
    """Compute the euclidean distance between two tensors."""
    x1 = tf.expand_dims(x, 1)
    y1 = tf.expand_dims(y, 0)
    d = tf.sqrt(tf.reduce_sum(tf.square(x1 - y1), axis=2))
    s = 1 / (1 + d)
    return s


def _edit_similarity(x, y):  # pragma: no cover
    """Compute the edit distance between two tensors."""
    x1 = tf.expand_dims(x, 1)
    y1 = tf.expand_dims(y, 0)
    d = tf.sqrt(tf.reduce_sum(tf.square(x1 - y1), axis=2))
    s = 1 / (1 + d)
    return s


def nce_loss(history, future, similarity, temperature=0.1):  # pragma: no cover
    """Compute the NCE loss function, as described in the paper "A Simple Framework for Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2002.05709)."""
    criterion = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM
    )

    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N**2, dtype=bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    all_sim = K.exp(sim / temperature)

    logits = tf.divide(K.sum(pos_sim), K.sum(all_sim, axis=1))

    lbl = np.ones(history.shape[0])

    # categorical cross entropy
    loss = criterion(y_pred=logits, y_true=lbl)

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def dcl_loss(
    history, future, similarity, temperature=0.1, debiased=True, tau_plus=0.1
):  # pragma: no cover
    """Compute the DCL loss function, as described in the paper "Debiased Contrastive Learning" (https://github.com/chingyaoc/DCL/)."""
    N = history.shape[0]
    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N**2, dtype=bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    neg_sim = K.exp(neg / temperature)

    # estimator g()
    if debiased:
        N = N - 1
        Ng = (-tau_plus * N * pos_sim + K.sum(neg_sim, axis=-1)) / (1 - tau_plus)
        # constraint (optional)
        Ng = tf.clip_by_value(
            Ng,
            clip_value_min=N * np.e ** (-1 / temperature),
            clip_value_max=tf.float32.max,
        )
    else:
        Ng = K.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + Ng)))

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def fc_loss(
    history,
    future,
    similarity,
    temperature=0.1,
    elimination_topk=0.1,
):  # pragma: no cover
    """Compute the FC loss function, as described in the paper "Fully-Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2004.11362)."""
    N = history.shape[0]
    if elimination_topk > 0.5:
        elimination_topk = 0.5
    elimination_topk = np.math.ceil(elimination_topk * N)

    sim = similarity(history, future) / temperature

    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim))

    tri_mask = np.ones(N**2, dtype=bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg_sim = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])

    sorted_sim = tf.sort(neg_sim, axis=1)

    # Top-K cancellation only
    if elimination_topk == 0:
        elimination_topk = 1
    tri_mask = np.ones(N * (N - 1), dtype=bool).reshape(N, N - 1)
    tri_mask[:, -elimination_topk:] = False
    neg = tf.reshape(
        tf.boolean_mask(sorted_sim, tri_mask), [N, N - elimination_topk - 1]
    )
    neg_sim = K.sum(K.exp(neg), axis=1)

    # categorical cross entropy
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + neg_sim)))

    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim)) * temperature
    mean_neg = K.mean(neg) * temperature
    return loss, mean_sim, mean_neg


def hard_loss(
    history, future, similarity, temperature, beta=0.0, debiased=True, tau_plus=0.1
):  # pragma: no cover
    """Compute the Hard loss function, as described in the paper "Contrastive Learning with Hard Negative Samples" (https://arxiv.org/abs/2011.03343)."""
    N = history.shape[0]

    sim = similarity(history, future)
    pos_sim = K.exp(tf.linalg.tensor_diag_part(sim) / temperature)

    tri_mask = np.ones(N**2, dtype=bool).reshape(N, N)
    tri_mask[np.diag_indices(N)] = False
    neg = tf.reshape(tf.boolean_mask(sim, tri_mask), [N, N - 1])
    neg_sim = K.exp(neg / temperature)

    reweight = (beta * neg_sim) / tf.reshape(tf.reduce_mean(neg_sim, axis=1), [-1, 1])
    if beta == 0:
        reweight = 1
    # estimator g()
    if debiased:
        N = N - 1

        Ng = (-tau_plus * N * pos_sim + tf.reduce_sum(reweight * neg_sim, axis=-1)) / (
            1 - tau_plus
        )
        # constraint (optional)
        Ng = tf.clip_by_value(
            Ng, clip_value_min=np.e ** (-1 / temperature), clip_value_max=tf.float32.max
        )
    else:
        Ng = K.sum(neg_sim, axis=-1)

    # contrastive loss
    loss = K.mean(-tf.math.log(pos_sim / (pos_sim + Ng)))
    # similarity of positive pairs (only for debug)
    mean_sim = K.mean(tf.linalg.tensor_diag_part(sim))
    mean_neg = K.mean(neg)
    return loss, mean_sim, mean_neg


def compute_kmeans_loss_pt(latent_means: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Computes a loss based on the singular values of the Gram matrix of the
    latent vectors, encouraging orthogonality.

    Args:
        latent_means: The latent vectors from the model (batch_size, latent_dim).
        weight: The weight to apply to this loss component.

    Returns:
        The calculated scalar loss tensor.
    """
    #Guard for bad inputs
    #z = torch.nan_to_num(latent_means, nan=0.0, posinf=1e4, neginf=-1e4)
    #zc = z - z.mean(dim=0, keepdim=True)
    #std = zc.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    #z_norm = zc / std
    batch_size = float(latent_means.shape[0])
    gram_matrix = (latent_means.T @ latent_means) / batch_size
    
    # Compute singular values, which are the square roots of the eigenvalues for a symmetric matrix
    singular_values = torch.linalg.svdvals(gram_matrix.to(float))
    
    # Clamp to avoid NaN gradients from sqrt(0)
    penalization = torch.sqrt(torch.clamp(singular_values, min=1e-9))

    kmeans_loss = weight * torch.nanmean(penalization[~torch.isinf(penalization)])
    if torch.isnan(kmeans_loss):
        return 0.0
    return kmeans_loss


def compute_kmeans_loss(
    latent_means: tf.Tensor, weight: float = 1.0, batch_size: int = 64
):  # pragma: no cover
    """Add a penalty to the singular values of the Gram matrix of the latent means. It helps disentangle the latent space.

    Based on https://arxiv.org/pdf/1610.04794.pdf, and https://www.biorxiv.org/content/10.1101/2020.05.14.095430v3.

    Args:
        latent_means (tf.Tensor): tensor containing the means of the latent distribution
        weight (float): weight of the Gram loss in the total loss function
        batch_size (int): batch size of the data to compute the kmeans loss for.

    Returns:
        tf.Tensor: kmeans loss

    """
    gram_matrix = (tf.transpose(latent_means) @ latent_means) / tf.cast(
        batch_size, tf.float32
    )
    s = tf.linalg.svd(gram_matrix, compute_uv=False)
    s = tf.sqrt(tf.maximum(s, 1e-9))
    return weight * tf.reduce_mean(s)


@tf.function
def get_k_nearest_neighbors(tensor, k, index):  # pragma: no cover
    """Retrieve indices of the k nearest neighbors in tensor to the vector with the specified index.

    Args:
        tensor (tf.Tensor): tensor to compute the k nearest neighbors for
        k (int): number of nearest neighbors to retrieve
        index (int): index of the vector to compute the k nearest neighbors for

    Returns:
        tf.Tensor: indices of the k nearest neighbors

    """
    query = tf.gather(tensor, index, batch_dims=0)
    distances = tf.norm(tensor - query, axis=1)
    max_distance = tf.gather(tf.sort(distances), k)
    neighbourhood_mask = distances < max_distance
    return tf.squeeze(tf.where(neighbourhood_mask))


@tf.function
def compute_shannon_entropy(tensor):  # pragma: no cover
    """Compute Shannon entropy for a given tensor.

    Args:
        tensor (tf.Tensor): tensor to compute the entropy for

    Returns:
        tf.Tensor: entropy of the tensor

    """
    tensor = tf.cast(tensor, tf.dtypes.int32)
    bins = (
        tf.math.bincount(tensor, dtype=tf.dtypes.float32)
        / tf.cast(tf.shape(tensor), tf.dtypes.float32)[0]
    )
    return -tf.reduce_sum(bins * tf.math.log(bins + 1e-5))


def plot_lr_vs_loss(rates, losses):  # pragma: no cover
    """Plot learning rate versus the loss function of the model.

    Args:
        rates (np.ndarray): array containing the learning rates to plot in the x-axis
        losses (np.ndarray): array containing the losses to plot in the y-axis

    """
    plt.plot(rates, losses)
    plt.gca().set_xscale("log")
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")


def get_angles(pos: int, i: int, d_model: int):
    """Auxiliary function for positional encoding computation.

    Args:
        pos (int): position in the sequence.
        i (int): number of sequences.
        d_model (int): dimensionality of the embeddings.

    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


class RecurrentBlockPT(nn.Module):
    def __init__(self, input_features: int, latent_dim: int, bidirectional_merge: str = "concat"):
        super().__init__()
        self.internal_dim = int(torch.min(torch.tensor([64,latent_dim]))) # Cap maximum internal dimension to avoid tensor size explosion
        self.latent_dim = latent_dim
        if bidirectional_merge != "concat":
            warnings.warn("Bidirectional merge mode defaulting to 'concat'.")

        self.conv1d = nn.Conv1d(
            in_channels=input_features,
            out_channels=2 * self.internal_dim,
            kernel_size=5,
            padding="same",
            bias=False,
        )
        self.gru1 = nn.GRU(
            input_size=2 * self.internal_dim,
            hidden_size=2 * self.internal_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm1 = nn.LayerNorm(4 * self.internal_dim, eps=1e-3)
        self.gru2 = nn.GRU(
            input_size=4 * self.internal_dim,
            hidden_size=self.internal_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.norm2 = nn.LayerNorm(2 * self.internal_dim, eps=1e-3)
        self.projection = nn.Linear(in_features=self.internal_dim*2, out_features=latent_dim*2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x: (B, T, N, F)
        B, T, N, _ = x.shape

        # Force input onto the same device as this block's parameters
        dev = self.conv1d.weight.device
        x = x.to(dev, non_blocking=True)

        # Stage 1: Conv (TimeDistributed over T)
        with torch.amp.autocast(device_type=dev.type, enabled=False):
            x32 = x.float()    
            conv_in = x32.reshape(B * T, N, -1).permute(0, 2, 1)  # (B*T, F, N)
            conv_out = F.relu(self.conv1d(conv_in))             # (B*T, 2*latent, N)
            gru1_in = conv_out.permute(0, 2, 1)                 # (B*T, N, 2*latent)

            # Mask/lengths for packing
            mask = (gru1_in.abs().sum(dim=-1) > 0)              # (B*T, N)
            lengths_cpu = mask.sum(dim=1).to(torch.int64).cpu() # lengths must be CPU for pack_padded_sequence
            valid_idx_cpu = torch.where(lengths_cpu > 0)[0]     # CPU index for CPU lengths
            valid_idx = valid_idx_cpu.to(dev)                   # GPU index for GPU tensors

            # Stage 2: First GRU with packing
            gru1_out_full = torch.zeros(
                B * T, N, 4 * self.internal_dim, device=gru1_in.device, dtype=gru1_in.dtype
            )
            if valid_idx.numel() > 0:
                packed_input = pack_padded_sequence(
                    gru1_in[valid_idx], lengths_cpu[valid_idx_cpu], batch_first=True, enforce_sorted=False
                )
                packed_output, _ = self.gru1(packed_input)
                unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=N)
                # Cast source to buffer dtype to avoid AMP mismatch
                gru1_out_full[valid_idx] = unpacked_output.to(gru1_out_full.dtype)

            # Stage 3: First LayerNorm
            norm1_in = gru1_out_full.reshape(B, T, N, -1)
            norm1_out = self.norm1(norm1_in)

            # Stage 4: Second GRU with packing
            gru2_in = norm1_out.reshape(B * T, N, -1)
            gru2_h_n_full = torch.zeros(
                2, B * T, self.internal_dim, device=gru2_in.device, dtype=gru2_in.dtype
            )
            if valid_idx.numel() > 0:
                packed_input_2 = pack_padded_sequence(
                    gru2_in[valid_idx], lengths_cpu[valid_idx_cpu], batch_first=True, enforce_sorted=False
                )
                _, h_n_2 = self.gru2(packed_input_2)  # (2, B_valid, latent)
                # Cast source to buffer dtype to avoid AMP mismatch
                gru2_h_n_full[:, valid_idx, :] = h_n_2.to(gru2_h_n_full.dtype)

            gru2_final_state = gru2_h_n_full.permute(1, 0, 2).reshape(B * T, -1)

            # Stage 5: Second LayerNorm
            norm2_out = self.norm2(gru2_final_state)  # (B*T, 2*latent)

            final_output = norm2_out.reshape(B, T, -1)  # (B, T, 2*latent)
            if self.internal_dim != self.latent_dim:
                final_output=self.projection(final_output) # restore latent space dependent output shape 
            if torch.isnan(final_output).any():
                print("z issues!")
        return final_output.to(x.dtype)


def get_recurrent_block(
    x: tf.Tensor, latent_dim: int, gru_unroll: bool, bidirectional_merge: str
):
    """Build a recurrent embedding block, using a 1D convolution followed by two bidirectional GRU layers.

    Args:
        x (tf.Tensor): Input tensor.
        latent_dim (int): Number of dimensions of the output tensor.
        gru_unroll (bool): whether to unroll the GRU layers. Defaults to False.
        bidirectional_merge (str): how to merge the forward and backward GRU layers. Defaults to "concat".

    Returns:
        tf.keras.models.Model object with the specified architecture.

    """
    encoder = TimeDistributed(
        tf.keras.layers.Conv1D(
            filters=2 * latent_dim,
            kernel_size=5,
            strides=1,  # Increased strides yield shorter sequences
            padding="same",
            activation="relu",
            kernel_initializer=he_uniform(),
            use_bias=False,
        )
    )(x)
    encoder = tf.keras.layers.Masking(mask_value=0.0)(encoder)
    encoder = TimeDistributed(
        Bidirectional(
            GRU(
                2 * latent_dim,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=True,
                unroll=gru_unroll,
                use_bias=True,
            ),
            merge_mode=bidirectional_merge,
        )
    )(encoder)
    encoder = LayerNormalization()(encoder)
    encoder = TimeDistributed(
        Bidirectional(
            GRU(
                latent_dim,
                activation="tanh",
                recurrent_activation="sigmoid",
                return_sequences=False,
                unroll=gru_unroll,
                use_bias=True,
            ),
            merge_mode=bidirectional_merge,
        )
    )(encoder)
    encoder = LayerNormalization()(encoder)

    return tf.keras.models.Model(x, encoder)


def positional_encoding(position: int, d_model: int):
    """Compute positional encodings, as in https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

    Args:
        position (int): position in the sequence.
        d_model (int): dimensionality of the embeddings.

    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq: tf.Tensor):
    """Create a padding mask, with zeros where data is missing, and ones where data is available.

    Args:
        seq (tf.Tensor): Sequence to compute the mask on

    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.
    return tf.cast(1 - seq[:, tf.newaxis, tf.newaxis, :], tf.float32)


def create_look_ahead_mask(size: int):
    """Create a triangular matrix containing an increasing amount of ones from left to right on each subsequent row.

    Useful for transformer decoder, which allows it to go through the data in a sequential manner, without taking
    the future into account.

    Args:
        size (int): number of time steps in the sequence

    """
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask, tf.float32)


def create_masks(inp: tf.Tensor):
    """Given an input sequence, it creates all necessary masks to pass it through the transformer architecture.

    This includes encoder and decoder padding masks, and a look-ahead mask

    Args:
        inp (tf.Tensor): input sequence to create the masks for.

    """
    # Reduces the dimensionality of the mask to remove the feature dimension
    tar = inp[:, :, 0]
    inp = inp[:, :, 0]

    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def find_learning_rate(
    model, data, epochs=1, batch_size=32, min_rate=10**-8, max_rate=10**-1
):
    """Train the provided model for an epoch with an exponentially increasing learning rate.

    Args:
        model (tf.keras.Model): model to train
        data (tuple): training data
        epochs (int): number of epochs to train the model for
        batch_size (int): batch size to use for training
        min_rate (float): minimum learning rate to consider
        max_rate (float): maximum learning rate to consider

    Returns:
        float: learning rate that resulted in the lowest loss

    """
    init_weights = model.get_weights()
    iterations = len(data)
    factor = K.exp(K.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    model.fit(data, epochs=epochs, batch_size=batch_size, callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses


@tf.function
def get_hard_counts(soft_counts: tf.Tensor):
    """Compute hard counts per cluster in a differentiable way.

    Args:
        soft_counts (tf.Tensor): soft counts per cluster

    """
    max_per_row = tf.expand_dims(tf.reduce_max(soft_counts, axis=1), axis=1)

    mask = tf.cast(soft_counts == max_per_row, tf.float32)
    mask_forward = tf.multiply(soft_counts, mask)
    mask_complement = tf.multiply(1 / soft_counts, mask)

    binary_counts = tf.multiply(mask_forward, mask_complement)

    return tf.math.reduce_sum(binary_counts, axis=0) + 1


@tf.function
def cluster_frequencies_regularizer(
    soft_counts: tf.Tensor, k: int, n_samples: int = 1000
):
    """Compute the KL divergence between the cluster assignment distribution and a uniform prior across clusters.

    While this assumes an equal distribution between clusters, the prior can be tweaked to reflect domain knowledge.

    Args:
        soft_counts (tf.Tensor): soft counts per cluster
        k (int): number of clusters
        n_samples (int): number of samples to draw from the categorical distribution modeling cluster assignments.

    """
    hard_counts = get_hard_counts(soft_counts)

    dist_a = tfd.Categorical(probs=hard_counts / k)
    dist_b = tfd.Categorical(probs=tf.ones(k))

    z = dist_a.sample(n_samples)

    return tf.reduce_mean(dist_a.log_prob(z) - dist_b.log_prob(z))


def get_callbacks(
    embedding_model: str,
    encoder_type: str,
    kmeans_loss: float = 1.0,
    input_type: str = False,
    cp: bool = False,
    logparam: dict = None,
    outpath: str = ".",
    run: int = False,
) -> List[Union[Any]]:
    """Generate callbacks used for model training.

    Args:
        embedding_model (str): name of the embedding model
        encoder_type (str): Architecture used for the encoder. Must be one of "recurrent", "TCN", and "transformer"
        kmeans_loss (float): Weight of the gram loss
        input_type (str): Input type to use for training
        cp (bool): Whether to use checkpointing or not
        logparam (dict): Dictionary containing the hyperparameters to log in tensorboard
        outpath (str): Path to the output directory
        run (int): Run number to use for checkpointing

    Returns:
        List[Union[Any]]: List of callbacks to be used for training

    """
    run_ID = "{}{}{}{}{}{}{}".format(
        "deepof_unsupervised_{}_{}_encodings".format(embedding_model, encoder_type),
        ("_input_type={}".format(input_type if input_type else "coords")),
        ("_kmeans_loss={}".format(kmeans_loss)),
        ("_encoding={}".format(logparam["latent_dim"]) if logparam is not None else ""),
        ("_k={}".format(logparam["n_components"]) if logparam is not None else ""),
        ("_run={}".format(run) if run else ""),
        ("_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")) if not run else ""),
    )

    log_dir = os.path.abspath(os.path.join(outpath, "fit", run_ID))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch=2
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, cooldown=5, min_lr=1e-8
    )

    callbacks = [run_ID, tensorboard_callback, reduce_lr_callback]

    if cp:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(outpath, "checkpoints", run_ID + "/cp-{epoch:04d}.ckpt"),
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        )
        callbacks.append(cp_callback)

    return callbacks


class CustomStopper(tf.keras.callbacks.EarlyStopping):
    """Custom early stopping callback. Prevents the model from stopping before warmup is over."""

    def __init__(self, start_epoch, *args, **kwargs):
        """Initialize the CustomStopper callback.

        Args:
            start_epoch: epoch from which performance will be taken into account when deciding whether to stop training.
            *args: arguments passed to the callback.
            **kwargs: keyword arguments passed to the callback.

        """
        super(CustomStopper, self).__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def get_config(self):  # pragma: no cover
        """Update callback metadata."""
        config = super().get_config().copy()
        config.update({"start_epoch": self.start_epoch})
        return config

    def on_epoch_end(self, epoch, logs=None):
        """Check whether to stop training."""
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class ExponentialLearningRate(tf.keras.callbacks.Callback):
    """Simple class that allows to grow learning rate exponentially during training.

    Used to trigger optimal learning rate search in deepof.train_utils.find_learning_rate.

    """

    def __init__(self, factor: float):
        """Initialize the exponential learning rate callback.

        Args:
            factor(float): factor by which to multiply the learning rate

        """
        super().__init__()
        self.factor = factor
        self.rates = []
        self.losses = []

    # noinspection PyMethodOverriding
    def on_batch_end(self, batch: int, logs: dict):
        """Apply on batch end.

        Args:
            batch: batch number
            logs (dict): dictionary containing the loss for the current batch

        """
        self.rates.append(K.get_value(self.model.optimizer.lr))
        if "total_loss" in logs.keys():
            self.losses.append(logs["total_loss"])
        else:
            self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)


class AffineTransformedDistribution(TransformedDistribution):
    """
    A specific TransformedDistribution for Affine transforms that implements .mean.
    """
    def __init__(self, base_distribution, transform):
        super().__init__(base_distribution, transform)

    @property
    def mean(self):
        """
        Computes the mean of the transformed distribution.
        E[loc + scale * X] = loc + scale * E[X]
        """
        # The transform itself is callable and applies the affine transformation.
        return self.transforms[0](self.base_dist.mean)


class ProbabilisticDecoderPT(nn.Module):
    """
    PyTorch translation of the ProbabilisticDecoder, including scaling transform.
    AMP-safe version: do distribution math in float32, sanitize NaNs/Infs.
    """
    def __init__(self, hidden_dim: int, data_dim: int):
        super().__init__()
        self.loc_projection = nn.Linear(in_features=hidden_dim, out_features=data_dim)

    def forward(self, hidden: torch.Tensor, validity_mask: torch.Tensor) -> AffineTransformedDistribution:
        B, T, _ = hidden.shape

        # Linear projection in float32 for numerical stability under AMP
        hidden_2d = hidden.reshape(B * T, -1).float()
        loc_params = self.loc_projection(hidden_2d).reshape(B, T, -1)

        # Sanitize to avoid NaN/Inf in distribution parameters
        loc_params = torch.nan_to_num(loc_params, nan=0.0, posinf=1e6, neginf=-1e6)

        # Build the distribution in float32 (disable autocast to avoid fp16 validator issues)
        with torch.amp.autocast(device_type=loc_params.device.type, enabled=False):
            loc32 = loc_params  # already float32
            scale32 = torch.ones_like(loc32)  # unit variance

            base_dist = torch.distributions.Normal(loc=loc32, scale=scale32, validate_args=False)
            independent_dist = torch.distributions.Independent(base_dist, 1)

            # Keep transform dtype consistent with distribution dtype
            scale_transform = validity_mask.unsqueeze(-1).to(dtype=loc32.dtype, device=loc32.device)
            transform = AffineTransform(loc=0.0, scale=scale_transform)

            final_dist = AffineTransformedDistribution(independent_dist, transform)

        return final_dist
    

class ProbabilisticDecoder(tf.keras.layers.Layer):
    """Map the reconstruction output of a given decoder to a multivariate normal distribution."""

    def __init__(self, input_shape, **kwargs):
        """Initialize the probabilistic decoder."""
        super().__init__(**kwargs)
        self.time_distributer = tf.keras.layers.Dense(
            tfpl.IndependentNormal.params_size(input_shape[1:]) // 2
        )
        self.probabilistic_decoding = tfpl.DistributionLambda(
            make_distribution_fn=lambda decoded: tfd.Masked(
                tfd.Independent(
                    tfd.Normal(
                        loc=decoded[0],
                        scale=tf.ones_like(decoded[0]),
                        validate_args=False,
                        allow_nan_stats=False,
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                validity_mask=decoded[1],
            ),
            convert_to_tensor_fn="mean",
        )
        self.scaled_probabilistic_decoding = tfpl.DistributionLambda(
            make_distribution_fn=lambda decoded: tfd.Masked(
                tfd.TransformedDistribution(
                    decoded[0],
                    tfb.Scale(tf.cast(tf.expand_dims(decoded[1], axis=2), tf.float32)),
                    name="vae_reconstruction",
                ),
                validity_mask=decoded[1],
            ),
            convert_to_tensor_fn="mean",
        )

    def call(self, inputs):  # pragma: no cover
        """Map the reconstruction output of a given decoder to a multivariate normal distribution.

        Args:
            inputs (tuple): tuple containing the reconstruction output and the validity mask

        Returns:
            tf.Tensor: multivariate normal distribution

        """
        hidden, validity_mask = inputs

        hidden = tf.keras.layers.TimeDistributed(self.time_distributer)(hidden)
        prob_decoded = self.probabilistic_decoding([hidden, validity_mask])
        scaled_prob_decoded = self.scaled_probabilistic_decoding(
            [prob_decoded, validity_mask]
        )

        return scaled_prob_decoded


class ClusterControlPT(nn.Module):
    """
    Calculates clustering metrics. This is a pass-through layer for the main
    latent vector `z`, returning it unmodified alongside a dictionary of metrics.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, z: torch.Tensor, z_cat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates metrics and passes the latent vector `z` through.

        Args:
            z: The latent vector (batch_size, latent_dim).
            z_cat: Cluster probabilities (batch_size, n_components).

        Returns:
            A tuple containing the unmodified `z` and a dictionary of metrics.
        """
        confidence, hard_groups = torch.max(z_cat, dim=1)
        
        # Calculate the number of unique clusters populated in the batch
        num_populated = torch.unique(hard_groups).numel()
        
        metrics = {
            "number_of_populated_clusters": torch.tensor(
                float(num_populated), device=z.device
            ),
            "confidence_in_selected_cluster": torch.mean(confidence),
        }
        
        return z, metrics
    

class ClusterControl(tf.keras.layers.Layer):
    """Identity layer.

    Evaluates different clustering metrics between the components of the latent Gaussian Mixture
    using the entropy of the nearest neighbourhood. If self.loss_weight > 0, it also adds a regularization
    penalty to the loss function which attempts to maximize the number of populated clusters during training.

    """

    def __init__(
        self,
        batch_size: int,
        n_components: int,
        encoding_dim: int,
        k: int = 15,
        *args,
        **kwargs,
    ):
        """Initialize the ClusterControl layer.

        Args:
            batch_size (int): batch size of the model
            n_components (int): number of components in the latent Gaussian Mixture
            encoding_dim (int): dimension of the latent Gaussian Mixture
            k (int): number of nearest components of the latent Gaussian Mixture to consider
            loss_weight (float): weight of the regularization penalty applied to the local entropy of each training instance
            *args: additional positional arguments
            **kwargs: additional keyword arguments

        """
        super(ClusterControl, self).__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.n_components = n_components
        self.enc = encoding_dim
        self.k = k

    def get_config(self):  # pragma: no cover
        """Update Constraint metadata."""
        config = super().get_config().copy()
        config.update({"batch_size": self.batch_size})
        config.update({"n_components": self.n_components})
        config.update({"enc": self.enc})
        config.update({"k": self.k})
        config.update({"loss_weight": self.loss_weight})
        return config

    def call(self, inputs):  # pragma: no cover
        """Update Layer's call method."""
        encodings, categorical = inputs[0], inputs[1]

        hard_groups = tf.math.argmax(categorical, axis=1)
        max_groups = tf.reduce_max(categorical, axis=1)

        n_components = tf.cast(
            tf.shape(
                tf.unique(tf.reshape(tf.cast(hard_groups, tf.dtypes.float32), [-1]))[0]
            )[0],
            tf.dtypes.float32,
        )

        self.add_metric(n_components, name="number_of_populated_clusters")
        self.add_metric(
            max_groups, aggregation="mean", name="confidence_in_selected_cluster"
        )

        return encodings


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformer encoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer."""

    def __init__(self, key_dim, num_heads, dff, rate=0.1):
        """Construct the transformer encoder layer.

        Args:
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers
            dff: dimensionality of the embeddings
            rate: dropout rate

        """
        super(TransformerEncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    dff, activation="relu"
                ),  # (batch_size, seq_len, dff)
                tf.keras.layers.Dense(key_dim),  # (batch_size, seq_len, d_model)
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, return_scores=False):  # pragma: no cover
        """Call the transformer encoder layer."""
        attn_output, attn_scores = self.mha(
            key=x, query=x, value=x, attention_mask=mask, return_attention_scores=True
        )  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        if return_scores:  # pragma: no cover
            return out2, attn_scores

        return out2


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Transformer decoder layer. Based on https://www.tensorflow.org/text/tutorials/transformer."""

    def __init__(self, key_dim, num_heads, dff, rate=0.1):
        """Construct the transformer decoder layer.

        Args:
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers
            dff: dimensionality of the embeddings
            rate: dropout rate

        """
        super(TransformerDecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim
        )

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(key_dim),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(
        self, x, enc_output, training, look_ahead_mask, padding_mask
    ):  # pragma: no cover
        """Call the transformer decoder layer."""
        attn1, attn_weights_block1 = self.mha1(
            key=x,
            query=x,
            value=x,
            attention_mask=look_ahead_mask,
            return_attention_scores=True,
        )  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            key=enc_output,
            query=out1,
            value=enc_output,
            attention_mask=padding_mask,
            return_attention_scores=True,
        )  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# noinspection PyCallingNonCallable
class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder.

    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

    """

    def __init__(
        self,
        num_layers,
        seq_dim,
        key_dim,
        num_heads,
        dff,
        maximum_position_encoding,
        rate=0.1,
    ):
        """Construct the transformer encoder.

        Args:
            num_layers: number of transformer layers to include.
            seq_dim: dimensionality of the sequence embeddings
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers used on the transformer encoder
            dff: dimensionality of the token embeddings
            maximum_position_encoding: maximum time series length
            rate: dropout rate

        """
        super(TransformerEncoder, self).__init__()

        self.rate = rate
        self.key_dim = key_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Conv1D(
            key_dim, kernel_size=1, activation="relu", input_shape=[seq_dim, key_dim]
        )
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.key_dim)
        self.enc_layers = [
            TransformerEncoderLayer(key_dim, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training):  # pragma: no cover
        """Call the transformer encoder."""
        # compute mask on the fly
        mask, _, _ = create_masks(x)
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x


# noinspection PyCallingNonCallable
class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer decoder.

    Based on https://www.tensorflow.org/text/tutorials/transformer.
    Adapted according to https://academic.oup.com/gigascience/article/8/11/giz134/5626377?login=true
    and https://arxiv.org/abs/1711.03905.

    """

    def __init__(
        self,
        num_layers,
        seq_dim,
        key_dim,
        num_heads,
        dff,
        maximum_position_encoding,
        rate=0.1,
    ):
        """Construct the transformer decoder.

        Args:
            num_layers: number of transformer layers to include.
            seq_dim: dimensionality of the sequence embeddings
            key_dim: dimensionality of the time series
            num_heads: number of heads of the multi-head-attention layers used on the transformer encoder
            dff: dimensionality of the token embeddings
            maximum_position_encoding: maximum time series length
            rate: dropout rate

        """
        super(TransformerDecoder, self).__init__()

        self.rate = rate
        self.key_dim = key_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Conv1D(
            key_dim, kernel_size=1, activation="relu", input_shape=[seq_dim, key_dim]
        )
        self.pos_encoding = positional_encoding(maximum_position_encoding, key_dim)
        self.dec_layers = [
            TransformerDecoderLayer(key_dim, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(
        self, x, enc_output, training, look_ahead_mask, padding_mask
    ):  # pragma: no cover
        """Call the transformer decoder."""
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask
            )
            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2

        return x, attention_weights


def log_hyperparameters():
    """Log hyperparameters in tensorboard.

    Blueprint for hyperparameter and metric logging in tensorboard during hyperparameter tuning

    Returns:
        logparams (list): List containing the hyperparameters to log in tensorboard.
        metrics (list): List containing the metrics to log in tensorboard.

    """
    logparams = [
        hp.HParam(
            "latent_dim",
            hp.Discrete([2, 4, 6, 8, 12, 16]),
            display_name="latent_dim",
            description="encoding size dimensionality",
        ),
        hp.HParam(
            "n_components",
            hp.IntInterval(min_value=1, max_value=25),
            display_name="n_components",
            description="latent component number",
        ),
        hp.HParam(
            "kmeans_weight",
            hp.RealInterval(min_value=0.0, max_value=1.0),
            display_name="kmeans_weight",
            description="weight of the kmeans loss",
        ),
    ]

    metrics = [
        hp.Metric(
            "val_number_of_populated_clusters",
            display_name="number of populated clusters",
        ),
        hp.Metric("val_reconstruction_loss", display_name="reconstruction loss"),
        hp.Metric("val_kmeans_loss", display_name="kmeans loss"),
        hp.Metric("val_vq_loss", display_name="vq loss"),
        hp.Metric("val_total_loss", display_name="total loss"),
    ]

    return logparams, metrics



def embedding_model_fitting(
    preprocessed_object: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    adjacency_matrix: np.ndarray,
    embedding_model: str,
    encoder_type: str,
    batch_size: int,
    latent_dim: int,
    epochs: int,
    log_history: bool,
    log_hparams: bool,
    n_components: int,
    output_path: str,
    data_path: str,
    kmeans_loss: float,
    pretrained: str,
    save_checkpoints: bool,
    save_weights: bool,
    input_type: str,
    bin_info: dict,
    # VaDE Model specific parameters
    kl_annealing_mode: str,
    kl_warmup: int,
    reg_cat_clusters: float,
    recluster: bool,
    # Contrastive Model specific parameters
    temperature: float,
    contrastive_similarity_function: str,
    contrastive_loss_function: str,
    beta: float,
    tau: float,
    interaction_regularization: float,
    run: int = 0,
    **kwargs,
):
    """

    Trains the specified embedding model on the preprocessed data.

    Args:
        preprocessed_object (tuple): Tuple containing the preprocessed data.
        adjacency_matrix (np.ndarray): adjacency_matrix (np.ndarray): adjacency matrix of the connectivity graph to use.
        embedding_model (str): Model to use to embed and cluster the data. Must be one of VQVAE (default), VaDE, and contrastive.
        encoder_type (str): Encoder architecture to use. Must be one of "recurrent", "TCN", and "transformer".
        batch_size (int): Batch size to use for training.
        latent_dim (int): Encoding size to use for training.
        epochs (int): Number of epochs to train the autoencoder for.
        log_history (bool): Whether to log the history of the autoencoder.
        log_hparams (bool): Whether to log the hyperparameters used for training.
        n_components (int): Number of components to fit to the data.
        output_path (str): Path to the output directory.
        data_path (str): Path to the directory where intermediate data is saved
        kmeans_loss (float): Weight of the gram loss, which adds a regularization term to VQVAE models which penalizes the correlation between the dimensions in the latent space.
        pretrained (str): Path to the pretrained weights to use for the autoencoder.
        save_checkpoints (bool): Whether to save checkpoints during training.
        save_weights (bool): Whether to save the weights of the autoencoder after training.
        input_type (str): Input type of the TableDict objects used for preprocessing. For logging purposes only.
        bin_info (dict): Dictionary containing numpy integer arrays for each experiment. Each array denotes the samples to be sampled from the respective experiment.

        # VaDE Model specific parameters
        kl_annealing_mode (str): Mode to use for KL annealing. Must be one of "linear" (default), or "sigmoid".
        kl_warmup (int): Number of epochs during which KL is annealed.
        reg_cat_clusters (bool): whether to penalize uneven cluster membership in the latent space, by minimizing the KL divergence between cluster membership and a uniform categorical distribution.
        recluster (bool): Whether to recluster the data after each training using a Gaussian Mixture Model.

        # Contrastive Model specific parameters
        temperature (float): temperature parameter for the contrastive loss functions. Higher values put harsher penalties on negative pair similarity.
        contrastive_similarity_function (str): similarity function between positive and negative pairs. Must be one of 'cosine' (default), 'euclidean', 'dot', and 'edit'.
        contrastive_loss_function (str): contrastive loss function. Must be one of 'nce' (default), 'dcl', 'fc', and 'hard_dcl'. See specific documentation for details.
        beta (float): Beta (concentration) parameter for the hard_dcl contrastive loss. Higher values lead to 'harder' negative samples.
        tau (float): Tau parameter for the dcl and hard_dcl contrastive losses, indicating positive class probability.
        interaction_regularization (float): Weight of the interaction regularization term (L1 penalization to all features not related to interactions).
        run (int): Run number to use for logging.



    Returns:
        List of trained models corresponding to the selected model class. The full trained model is last.

    """

    
    # Select strategy based on available hardware
    if len(tf.config.list_physical_devices("GPU")) > 1:  # pragma: no cover
        strategy = tf.distribute.MirroredStrategy(
            [dev.name for dev in tf.config.list_physical_devices("GPU")]
        )
    elif len(tf.config.list_physical_devices("GPU")) == 1:
        strategy = tf.distribute.OneDeviceStrategy("gpu")
    else:
        strategy = tf.distribute.OneDeviceStrategy("cpu")

    with tf.device("CPU"):

        # Load data
        preprocessed_train, preprocessed_validation= preprocessed_object
        
        # Create two big numpy arrays from tables for node and edge data.
        # Shape may be e.g. (74873, 25, 33), (74873, 25, 11) with (N_samples, L_time_window, N_features)
        X_train, a_train, _ = preprocessed_train.sample_windows_from_data(time_bin_info=bin_info, return_edges=True)
        X_val, a_val, _ = preprocessed_validation.sample_windows_from_data(time_bin_info=bin_info, return_edges=True)


        # Make sure that batch_size is not larger than training set
        if batch_size > X_train.shape[0]:
            batch_size = X_train.shape[0]

        # Set options for tf.data.Datasets
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )

        # Defines hyperparameters to log on tensorboard (useful for keeping track of different models)
        logparam = {
            "latent_dim": latent_dim,
            "n_components": n_components,
            "kmeans_weight": kmeans_loss,
        }

        # Load callbacks
        run_ID, *cbacks = get_callbacks(
            embedding_model=embedding_model,
            encoder_type=encoder_type,
            kmeans_loss=kmeans_loss,
            input_type=input_type,
            cp=save_checkpoints,
            logparam=logparam,
            outpath=output_path,
            run=run,
        )
        if not log_history:
            cbacks = cbacks[1:]

        Xs, ys = X_train, [X_train]
        Xvals, yvals = X_val, [X_val]
        
        train_shape=X_train.shape
        a_train_shape=a_train.shape


        # Cast to float32
        ys = tuple([tf.cast(dat, tf.float32) for dat in ys])
        yvals = tuple([tf.cast(dat, tf.float32) for dat in yvals])

        # Convert data to tf.data.Dataset objects
        train_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (tf.cast(Xs, tf.float32), tf.cast(a_train, tf.float32), tuple(ys))
            )
            .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
            .shuffle(buffer_size=train_shape[0])
            .with_options(options)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            tf.data.Dataset.from_tensor_slices(
                (tf.cast(Xvals, tf.float32), tf.cast(a_val, tf.float32), tuple(yvals))
            )
            .batch(batch_size * strategy.num_replicas_in_sync, drop_remainder=True)
            .with_options(options)
            .prefetch(tf.data.AUTOTUNE)
        )


        embed_x={}
        train_path = os.path.join(data_path, 'embed_x')
        embed_x['embed_x'] = save_dt(Xs,train_path,True)
        embed_a={}
        train_path = os.path.join(data_path, 'embed_a')
        embed_a['embed_a'] = save_dt(a_train,train_path,True) 

        del Xs
        del X_train
        del a_train
        del X_val
        del Xvals
        del a_val
        del ys
        del yvals


    # Build model
    with strategy.scope():

        if embedding_model == "VQVAE":
            ae_full_model = deepof.models.VQVAE(
                input_shape=train_shape,
                edge_feature_shape=a_train_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=not np.all(np.abs(get_dt(embed_a, 'embed_a')) < 10e-10),
                n_components=n_components,
                kmeans_loss=kmeans_loss,
                encoder_type=encoder_type,
                interaction_regularization=interaction_regularization,
            )
            ae_full_model.optimizer = tf.keras.optimizers.Nadam(
                learning_rate=1e-4, clipvalue=0.75
            )

        elif embedding_model == "VaDE":
            ae_full_model = deepof.models.VaDE(
                input_shape=train_shape,
                edge_feature_shape=a_train_shape,
                adjacency_matrix=adjacency_matrix,
                batch_size=batch_size,
                latent_dim=latent_dim,
                use_gnn=not np.all(np.abs(get_dt(embed_a, 'embed_a')) < 10e-10),
                kl_annealing_mode=kl_annealing_mode,
                kl_warmup_epochs=kl_warmup,
                montecarlo_kl=100,
                n_components=n_components,
                reg_cat_clusters=reg_cat_clusters,
                encoder_type=encoder_type,
                interaction_regularization=interaction_regularization,
            )

        elif embedding_model == "Contrastive":
            ae_full_model = deepof.models.Contrastive(
                input_shape=train_shape,
                edge_feature_shape=a_train_shape,
                adjacency_matrix=adjacency_matrix,
                latent_dim=latent_dim,
                use_gnn=not np.all(np.abs(get_dt(embed_a, 'embed_a')) < 10e-10),
                encoder_type=encoder_type,
                temperature=temperature,
                similarity_function=contrastive_similarity_function,
                loss_function=contrastive_loss_function,
                interaction_regularization=interaction_regularization,
                beta=beta,
                tau=tau,
            )

        else:  # pragma: no cover
            raise ValueError(
                "Invalid embedding model. Select one of 'VQVAE', 'VaDE', and 'Contrastive'"
            )

    callbacks_ = cbacks + [
        CustomStopper(
            monitor="val_total_loss",
            mode="min",
            patience=15,
            restore_best_weights=False,
            start_epoch=15,
        )
    ]

    ae_full_model.compile(
        optimizer=ae_full_model.optimizer,
        run_eagerly=False,
    )

    if not pretrained:
        if embedding_model == "VaDE":
            ae_full_model.pretrain(
                train_dataset,
                embed_x=embed_x,
                embed_a=embed_a,
                epochs=(np.minimum(10, epochs) if not pretrained else 0),
                **kwargs,
            )
            ae_full_model.optimizer._iterations.assign(0)


        ae_full_model.fit(
            x=train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks_,
            **kwargs,
        )

        if embedding_model == "VaDE" and recluster == True:  # pragma: no cover
            ae_full_model.pretrain(
                train_dataset, embed_x=embed_x, embed_a=embed_a, epochs=0, **kwargs
            )

    else:  # pragma: no cover
        # If pretrained models are specified, load weights and return
        if embedding_model != "Contrastive":
            ae_full_model.build([train_shape, a_train_shape])
        else:
            ae_full_model.build(
                [
                    (train_shape[0], train_shape[1] // 2, train_shape[2]),
                    (a_train_shape[0], a_train_shape[1] // 2, a_train_shape[2]),
                ]
            )

        ae_full_model.load_weights(pretrained)
        return ae_full_model

    if not os.path.exists(os.path.join(output_path, "trained_weights")):
        os.makedirs(os.path.join(output_path, "trained_weights"))

    if save_weights:
        ae_full_model.save_weights(
            os.path.join(
                "{}".format(output_path),
                "trained_weights",
                "{}_final_weights.h5".format(run_ID),
            )
        )

        # Logs hyperparameters to tensorboard
        if log_hparams:
            logparams, metrics = log_hyperparameters()

            tb_writer = tf.summary.create_file_writer(
                os.path.abspath(os.path.join(output_path, "hparams", run_ID))
            )
            with tb_writer.as_default():
                # Configure hyperparameter logging in tensorboard
                hp.hparams_config(hparams=logparams, metrics=metrics)
                hp.hparams(logparam)  # Log hyperparameters

                # Log metrics
                tf.summary.scalar(
                    "val_total_loss",
                    ae_full_model.history.history["val_total_loss"][-1],
                    step=0,
                )

                if embedding_model != "Contrastive":
                    tf.summary.scalar(
                        "val_reconstruction_loss",
                        ae_full_model.history.history["val_reconstruction_loss"][-1],
                        step=0,
                    )
                    tf.summary.scalar(
                        "val_number_of_populated_clusters",
                        ae_full_model.history.history[
                            "val_number_of_populated_clusters"
                        ][-1],
                        step=0,
                    )
                    tf.summary.scalar(
                        "val_kmeans_loss",
                        ae_full_model.history.history["val_kmeans_loss"][-1],
                        step=0,
                    )

                if embedding_model == "VQVAE":
                    tf.summary.scalar(
                        "val_vq_loss",
                        ae_full_model.history.history["val_vq_loss"][-1],
                        step=0,
                    )

                elif embedding_model == "VaDE":
                    tf.summary.scalar(
                        "val_kl_loss",
                        ae_full_model.history.history["val_kl_divergence"][-1],
                        step=0,
                    )

                elif embedding_model == "Contrastive":
                    tf.summary.scalar(
                        "val_total_loss",
                        ae_full_model.history.history["val_total_loss"][-1],
                        step=0,
                    )

    return ae_full_model


def embedding_per_video(
    coordinates: coordinates,
    to_preprocess: table_dict,
    model: tf.keras.models.Model,
    scale: str = "standard",
    animal_id: str = None,
    global_scaler: Any = None,
	pretrained: bool = False,
	samples_max: int = 227272,
    **kwargs,
):  # pragma: no cover
    """Use a previously trained model to produce embeddings and soft_counts per experiment in table_dict format.

    Args:
        coordinates (coordinates): deepof.Coordinates object for the project at hand.
        to_preprocess (table_dict): dictionary with (merged) features to process.
        model (tf.keras.models.Model): trained deepof unsupervised model to run inference with.
        pretrained (bool): whether to use the specified pretrained model to recluster the data.
        scale (str): The type of scaler to use within animals. Defaults to 'standard', but can be changed to 'minmax', 'robust', or False. Use the same that was used when training the original model.
        animal_id (str): if more than one animal is present, provide the ID(s) of the animal(s) to include.
        global_scaler (Any): trained global scaler produced when processing the original dataset.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.
        **kwargs: additional arguments to pass to coordinates.get_graph_dataset().

    Returns:
        embeddings (table_dict): embeddings per experiment.
        soft_counts (table_dict): soft_counts per experiment.

    """

    # at some point _check_enum_inputs will get moved somewhere else and be reworked to function as a general guard function 
    deepof.visuals_utils._check_enum_inputs(
        coordinates,
        animal_id=animal_id,
    )

    embeddings = {}
    soft_counts = {}
    #interim
    file_name='unsup'


    graph = False
    contrastive = isinstance(model, deepof.clustering.models_new.ContrastivePT)
    if str(model.encoder.spatial_gnn_block) == "CensNetConvPT()":
        graph = True 
    
    keys_to_drop=[]
    window_size = model.window_size
    for key in tqdm.tqdm(to_preprocess.keys(), desc=f"{'Computing embeddings':<{PROGRESS_BAR_FIXED_WIDTH}}", unit="table"):

        dict_to_preprocess = to_preprocess.filter_videos([key])
        #preload datatable in case it is not already, as this will only contain a single table and hence avoid double loading in get_graph_dataset
        dict_to_preprocess[key]=get_dt(dict_to_preprocess,key)
        if dict_to_preprocess[key].isna().all().all():
            keys_to_drop.append(key)
            continue

        #creates a new line to ensure that the outer loading bar does not get overwritten by the inner ones
        print("")

        if graph:
            processed_exp, _, _, _, _ = coordinates.get_graph_dataset(
                animal_id=animal_id,
                precomputed_tab_dict=dict_to_preprocess,
                preprocess=True,
                scale=scale,
                window_size=window_size,
                window_step=1,
                pretrained_scaler=global_scaler,
                samples_max=samples_max,
            )

        else:

            processed_exp, _, _ = dict_to_preprocess.preprocess(
                coordinates=coordinates,
                scale=scale,
                window_size=window_size,
                window_step=1,
                shuffle=False,
                pretrained_scaler=global_scaler,
            )

        tab_tuple=deepof.utils.get_dt(processed_exp[0],key)
        tab_tuple = (reorder_and_reshape(tab_tuple[0]),np.expand_dims(tab_tuple[1],-1))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        x_all = torch.as_tensor(tab_tuple[0], dtype=torch.float32, device=device)
        a_all = torch.as_tensor(tab_tuple[1], dtype=torch.float32, device=device)

        batch_size = 256  # adjust to fit your GPU
        recon_list, emb_list, sc_list = [], [], []

        # Optional AMP for speed/memory on GPU
        if False: #device.type == "cuda":
            try:
                bf16_ok = torch.cuda.is_bf16_supported()
            except Exception:
                major, _ = torch.cuda.get_device_capability()
                bf16_ok = major >= 8  # Ampere+ typically supports bf16
            amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            amp_ctx = nullcontext()

        with torch.inference_mode(), amp_ctx:
            for s in range(0, x_all.size(0), batch_size):
                xb = x_all[s:s + batch_size].to(device, non_blocking=True)
                ab = a_all[s:s + batch_size].to(device, non_blocking=True)

                # Disable attention collection if supported
                if isinstance(model, deepof.clustering.models_new.VaDEPT):
                    _, emb_out, sc_out, _ = model(xb, ab, return_gmm_params=False)
                    sc_list.append(sc_out.detach().cpu())
                elif isinstance(model, deepof.clustering.models_new.VQVAEPT):
                    _, _, _, sc_out, emb_out, _ = model(xb, ab, return_all_outputs=True)
                    sc_list.append(sc_out.detach().cpu())
                elif isinstance(model, deepof.clustering.models_new.ContrastivePT):
                    emb_out = model(xb, ab)
                else:
                    raise RuntimeError("Unexpected model; expected either VADE or VQVAE.")

                emb_list.append(emb_out.detach().cpu())

        # Stitch full outputs
        emb_raw = torch.cat(emb_list, dim=0) if emb_list else None
        print('completed')
        emb = emb_raw.cpu().numpy()

        if not contrastive:
            sc_raw = torch.cat(sc_list, dim=0) if sc_list else None
            sc = sc_raw.cpu().numpy()
            # save paths for modified tables
            table_path = os.path.join(coordinates._project_path, coordinates._project_name, 'Tables',key, key + '_' + file_name + '_softc')
            soft_counts[key] = deepof.utils.save_dt(sc,table_path,coordinates._very_large_project)

        # save paths for modified tables
        table_path = os.path.join(coordinates._project_path, coordinates._project_name, 'Tables',key, key + '_' + file_name + '_embed')
        embeddings[key] = deepof.utils.save_dt(emb,table_path,coordinates._very_large_project) 

        #to not flood the output with loading bars
        clear_output()

    # Notify user about key removal, if applicable 
    exp_conds=copy.copy(coordinates.get_exp_conditions)
    if len(keys_to_drop) > 0:
        for key in keys_to_drop:
            del exp_conds[key]
        print(
            f'\033[33mInfo! Removed keys {str(keys_to_drop)} As table segments contained only NaNs!\033[0m'
        )

    
    table_path=os.path.join(coordinates._project_path, coordinates._project_name, "Tables")
    if isinstance(soft_counts, tuple):
        soft_counts = soft_counts[0]
    embeddings= deepof.data.TableDict(
        embeddings,
        typ="unsupervised_embedding",
        table_path=table_path, 
        exp_conditions=exp_conds,
    )

    if contrastive:

        soft_counts, model_selection = deepof.post_hoc.get_contrastive_soft_counts(
            coordinates, embeddings, 
        )
    else:
        soft_counts=deepof.data.TableDict(
            soft_counts,
            typ="unsupervised_counts",
            table_path=table_path, 
            exp_conditions=exp_conds,
        )

    return (
        embeddings,
        soft_counts,
    )


def tune_search(
    preprocessed_object: tuple,
    adjacency_matrix: np.ndarray,
    encoding_size: int,
    embedding_model: str,
    hypertun_trials: int,
    hpt_type: str,
    k: int,
    project_name: str,
    callbacks: List,
    batch_size: int = 1024,
    n_epochs: int = 30,
    n_replicas: int = 1,
    outpath: str = "unsupervised_tuner_search",
) -> tuple:
    """Define the search space using keras-tuner and hyperband or bayesian optimization.

    Args:
        preprocessed_object (tf.data.Dataset): Dataset object for training and validation.
        adjacency_matrix (np.ndarray): Adjacency matrix for the graph.
        encoding_size (int): Size of the encoding layer.
        embedding_model (str): Model to use to embed and cluster the data. Must be one of VQVAE (default), VaDE, and Contrastive.
        hypertun_trials (int): Number of hypertuning trials to run.
        hpt_type (str): Type of hypertuning to run. Must be one of "hyperband" or "bayesian".
        k (int): Number of clusters on the latent space.
        project_name (str): Name of the project.
        callbacks (List): List of callbacks to use.
        batch_size (int): Batch size to use.
        n_epochs (int): Maximum number of epochs to train for.
        n_replicas (int): Number of replicas to use.
        outpath (str): Path to save the results.

    Returns:
        best_hparams (dict): Dictionary of the best hyperparameters.
        best_run (str): Name of the best run.

    """

    # extract from Tuple
    preprocessed_train, preprocessed_validation= preprocessed_object
    pt_shape=get_dt(preprocessed_train,list(preprocessed_train.keys())[0], only_metainfo=True)['shape']

    #get available memory -10% as buffer
    available_mem=psutil.virtual_memory().available*0.9
    #calculate maximum number of rows that fit in memory based on table info 
    N_windows_max=int(available_mem/((pt_shape[1]+11)*pt_shape[2]*8))

    # Sample up to N_windows_max windows from processed_train and processed_validation
    N_windows_tab=int(N_windows_max/(len(preprocessed_train)+len(preprocessed_validation)))
    
    X_train, a_train, _ = preprocessed_train.sample_windows_from_data(N_windows_tab=N_windows_tab, return_edges=True)
    X_val, a_val, _ = preprocessed_validation.sample_windows_from_data(N_windows_tab=N_windows_tab, return_edges=True)

    # Make sure that batch_size is not larger than training set
    if batch_size > X_train.shape[0]:
        batch_size = X_train.shape[0]

    # Set options for tf.data.Datasets
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )

    Xs, ys = X_train, [X_train]
    Xvals, yvals = X_val, [X_val]

    # Cast to float32
    ys = tuple([tf.cast(dat, tf.float32) for dat in ys])
    yvals = tuple([tf.cast(dat, tf.float32) for dat in yvals])

    # Convert data to tf.data.Dataset objects
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(Xs, tf.float32), tf.cast(a_train, tf.float32), tuple(ys))
        )
        .batch(batch_size, drop_remainder=True)
        .shuffle(buffer_size=X_train.shape[0])
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (tf.cast(Xvals, tf.float32), tf.cast(a_val, tf.float32), tuple(yvals))
        )
        .batch(batch_size, drop_remainder=True)
        .with_options(options)
        .prefetch(tf.data.AUTOTUNE)
    )

    assert hpt_type in ["bayopt", "hyperband"], (
        "Invalid hyperparameter tuning framework. " "Select one of bayopt and hyperband"
    )

    if embedding_model == "VQVAE":
        hypermodel = deepof.hypermodels.VQVAE(
            input_shape=X_train.shape,
            edge_feature_shape=a_train.shape,
            use_gnn=len(preprocessed_object) == 6,
            adjacency_matrix=adjacency_matrix,
            latent_dim=encoding_size,
            n_components=k,
        )
    elif embedding_model == "VaDE":
        hypermodel = deepof.hypermodels.VaDE(
            input_shape=X_train.shape,
            edge_feature_shape=a_train.shape,
            use_gnn=len(preprocessed_object) == 6,
            adjacency_matrix=adjacency_matrix,
            latent_dim=encoding_size,
            n_components=k,
            batch_size=batch_size,
        )
    elif embedding_model == "Contrastive":
        hypermodel = deepof.hypermodels.Contrastive(
            input_shape=X_train.shape,
            edge_feature_shape=a_train.shape,
            use_gnn=len(preprocessed_object) == 6,
            adjacency_matrix=adjacency_matrix,
            latent_dim=encoding_size,
        )

    tuner_objective = "val_total_loss"

    # noinspection PyUnboundLocalVariable
    hpt_params = {
        "hypermodel": hypermodel,
        "executions_per_trial": n_replicas,
        "objective": Objective(tuner_objective, direction="min"),
        "project_name": project_name,
        "tune_new_entries": True,
    }

    if hpt_type == "hyperband":
        tuner = Hyperband(
            directory=os.path.join(
                outpath, "HyperBandx_VQVAE_{}".format(str(date.today()))
            ),
            max_epochs=n_epochs,
            hyperband_iterations=hypertun_trials,
            factor=3,
            **hpt_params,
        )
    else:
        tuner = BayesianOptimization(
            directory=os.path.join(
                outpath, "BayOpt_VQVAE_{}".format(str(date.today()))
            ),
            max_trials=hypertun_trials,
            **hpt_params,
        )

    print(tuner.search_space_summary())

    # Convert data to tf.data.Dataset objects
    tuner.search(
        train_dataset,
        epochs=n_epochs,
        validation_data=val_dataset,
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    best_hparams = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_run = tuner.hypermodel.build(best_hparams)

    print(tuner.results_summary())

    return best_hparams, best_run
