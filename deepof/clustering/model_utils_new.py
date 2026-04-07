# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Utility functions for both training autoencoder models in deepof.models and tuning hyperparameters with deepof.hypermodels."""

import os
from typing import Any, List, NewType, Tuple, Union, Dict, Callable, Optional
import copy
from dataclasses import dataclass, asdict
import tqdm
from contextlib import nullcontext
import math


from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Distribution, TransformedDistribution
from torch.distributions.transforms import AffineTransform

from deepof.config import PROGRESS_BAR_FIXED_WIDTH
from deepof.data_loading import get_dt, save_dt
import deepof.clustering.dataset
import warnings
from deepof.clustering.dataset import reorder_and_reshape

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)

###########################
### CONFIGS
###########################

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

    kl_annealing_mode: str = "tf_sigmoid"
    kl_max_weight: float = 1.0
    kl_warmup: int = 5
    kl_end_weight: float = 0.2
    kl_cooldown: int = 5

    # Diagnostics
    diag_max_batches: int = 4
    seed: int = None


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
    lambda_end_weight: float = 0.2
    lambda_cooldown: int = 10
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
    nonempty_p: float = 2.0
    nonempty_floor_percent: float = 0.05

    kmeans_loss_pretrain: float = 1.0
    repel_weight_pretrain: float = 0.5
    repel_length_scale_pretrain: float = 0.5
    nonempty_weight_pretrain: float = 2e-2
    nonempty_p_pretrain: float = 2.0
    nonempty_floor_percent_pretrain: float = 0.05




@dataclass
class ContrastiveCfg:
    temperature: float = 0.1
    contrastive_similarity_function: str = "cosine"
    contrastive_loss_function: str = "nce"
    beta: float = 0.1
    tau: float = 0.1        
    aug_min_shift: int = 1
    aug_max_shift: int = 6
    aug_p_shift: float = 0.8
    aug_max_rot: int = 30
    aug_n_rot: int = 4
    aug_p_rot: float = 0.8
    aug_max_interp: int = 8
    aug_min_interp: int = 3        
    aug_p_interp: float = 0.3
    aug_noise_sigma: float = 0.03
    aug_p_noise: float = 1.0

def _append_cfg(lines, title: str, cfg) -> None:
    if cfg is None:
        return

    lines.append(f"[{title}]")
    d = asdict(cfg)  # flat dict
    for k in d.keys(): 
        lines.append(f"{k}: {d[k]}")
    lines.append("")  # spacer


def _unwrap_dp(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, torch.nn.DataParallel) else m


def save_model_info(
    ckpt_path: str,
    *,
    stage: str,
    epoch: Optional[int] = None,
    train_steps: Optional[int] = None,
    val_total: Optional[float] = None,
    score_value: Optional[float] = None,
    extra: Optional[dict] = None,
    common_cfg=None,
    teacher_cfg=None,
    vade_cfg=None,
    contrastive_cfg=None,
    model: Optional[nn.Module] = None,
    log_summary: Optional[Dict[str, Any]] = None,
    rebuild_spec: Optional[Dict[str, Any]] = None,
    save_weights: bool = True,
    save_bundle: bool = True,   # if True -> saves dict with state_dict + optional spec/log; else raw state_dict
) -> None:
    """Saves all config and training information for a freshly trained model (+ optionally the model weights)."""
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

    if save_weights and (model is not None):
        lines.append("[checkpoint_format]")
        if save_bundle:
            lines.append("ckpt_contains: bundle")
            keys = ["state_dict"]
            if rebuild_spec is not None: keys.append("rebuild_spec")
            if log_summary is not None: keys.append("log_summary")
            lines.append("bundle_keys: " + ", ".join(keys))
        else:
            lines.append("ckpt_contains: state_dict_only")
        lines.append("")

    # Dump configs (unchanged)
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

    if save_weights and (model is not None):
        m = _unwrap_dp(model)
        if save_bundle:
            payload = {"state_dict": m.state_dict()}
            if rebuild_spec is not None:
                payload["rebuild_spec"] = rebuild_spec
            if log_summary is not None:
                payload["log_summary"] = log_summary
            torch.save(payload, ckpt_path)
        else:
            torch.save(m.state_dict(), ckpt_path)

    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


######################################
### CONTRASTIVE LEARNING UTILITIES
######################################

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
    
def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
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


def nce_loss_pt_old(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    temperature: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the NCE loss function, as described in the paper "A Simple Framework for Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2002.05709).
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

def nce_loss_pt(history, future, similarity, temperature=0.1):
    """
    Standard NCE loss 
    """
    sim = similarity(history, future) / temperature        # (N,N)
    labels = torch.arange(sim.size(0), device=sim.device)
    loss = torch.nn.functional.cross_entropy(sim, labels)  # row-wise softmax

    mean_pos = torch.diag(sim).mean() * temperature
    off = _off_diagonal_rows(sim * temperature)  
    mean_neg = off.mean() if off.numel() else torch.tensor(0., device=sim.device)
    return loss, mean_pos, mean_neg


def dcl_loss_pt(
    history: torch.Tensor,
    future: torch.Tensor,
    similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    temperature: float = 0.1,
    debiased: bool = True,
    tau_plus: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the DCL loss function, as described in the paper "Debiased Contrastive Learning" (https://github.com/chingyaoc/DCL/)."""
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
    """Compute the FC loss function, as described in the paper "Fully-Contrastive Learning of Visual Representations" (https://arxiv.org/abs/2004.11362)."""
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
    """Compute the Hard loss function, as described in the paper "Contrastive Learning with Hard Negative Samples" (https://arxiv.org/abs/2011.03343)."""
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


def compute_kmeans_loss_pt(latent_means: torch.Tensor, weight: float) -> torch.Tensor:
    """
    Computes a loss based on the singular values of the Gram matrix of the
    latent vectors, encouraging orthogonality.     
    Based on https://arxiv.org/pdf/1610.04794.pdf, and https://www.biorxiv.org/content/10.1101/2020.05.14.095430v3.

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
    
def embedding_per_video(
    coordinates: coordinates,
    to_preprocess: table_dict,
    model: str,
    metainfo: dict,
    supervised_annotations: table_dict = None,
    scale: str = "standard",
    animal_id: str = None,
    extract_pair: list = None,
    global_scaler: Any = None,
    softcounts_extraction_method = None,
    embedding_gates: str = "Center",
    states: int = 24,
    quality_threshold: float = 0.75,
    frac_bps_below: float = 0.5,
    samples_max: int = 227272,
):  # pragma: no cover
    """Use a previously trained model to produce embeddings and soft_counts per experiment in table_dict format.

    Args:
        coordinates (coordinates): deepof.Coordinates object for the project at hand.
        to_preprocess (table_dict): dictionary with (merged) features to process.
        model (tf.keras.models.Model): trained deepof unsupervised model to run inference with.
        metainfo (dict): meta_nfo dictionary containing information regarding dataset preprocessing.
        supervised_annotations (table_dict): table dict with supervised annotations per experiment.
        pretrained (bool): whether to use the specified pretrained model to recluster the data.
        scale (str): The type of scaler to use within animals. Defaults to 'standard', but can be changed to 'minmax', 'robust', or False. Use the same that was used when training the original model.
        animal_id (str): if more than one animal is present, provide the ID(s) of the animal(s) to include.
        global_scaler (Any): trained global scaler produced when processing the original dataset.
        softcounts_extraction_method (str): Method used for softcounts extraction, can be None, "gmm", "msm" (for msm-pcca) or "combined" for an approach that applies msm-pcca first, then filters out all samples with high tracking uncertainty and uses a gmm approach to predict separate clusters on the uncertain sampel fraction. If None, decoder of model is used. If model has no decoder, "msm" is used as a default.
        distance_bp (str): The mosue bodypart that will be used for distance binning during softcounts extraction. Only relevant for experiments with 2+ mice that use a not-none softcounts_extraction_method.
        samples_max (int): Maximum number of samples taken for plotting to avoid excessive computation times. If the number of rows in a data set exceeds this number the data is downsampled accordingly.

    Returns:
        embeddings (table_dict): embeddings per experiment.
        soft_counts (table_dict): soft_counts per experiment.

    """

    def _extract_pair_to_gate_key(
        coordinates,
        extract_pair: Optional[list],
    ) -> Any:
        """
        Convert extract_pair list to the gate key used in soft_counts_dict.
        """
        animal_ids = coordinates._animal_ids
        if extract_pair is None:
            if len(animal_ids) == 1:
                return ""
            elif len(animal_ids) >= 2:
                return tuple(sorted([animal_ids[0], animal_ids[1]]))
            else:
                raise AssertionError("No animal IDs found in coordinates._animal_ids.")

        if extract_pair == [""]:
            return ""

        if not isinstance(extract_pair, list) or len(extract_pair) != 2:
            raise AssertionError(
                "extract_pair must be a list with two animal ids or [\"\"] in case of a single mouse!"
            )

        id1, id2 = extract_pair
        if id1 not in animal_ids or id2 not in animal_ids:
            raise AssertionError(
                f"Animal IDs {id1}, {id2} not found in coordinates._animal_ids: {animal_ids}"
            )

        return tuple(sorted([id1, id2]))

    extract_pair = _extract_pair_to_gate_key(coordinates, extract_pair)

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
    # The contrastive model only consists out of an encoder and hence needs additional soft_counts extraction
    contrastive = isinstance(model, deepof.clustering.models_new.ContrastivePT)
    if contrastive and softcounts_extraction_method is None:
        softcounts_extraction_method = "msm"
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
                dist_standardize=metainfo['dist_standardize'],
                speed_standardize=metainfo['speed_standardize'] ,
                coord_standardize=metainfo['coord_standardize'],
            )    

        else:

            processed_exp, _, _ = dict_to_preprocess.preprocess(
                coordinates=coordinates,
                scale=scale,
                window_size=window_size,
                window_step=1,
                shuffle=False,
                pretrained_scaler=global_scaler,
                dist_standardize=metainfo['dist_standardize'],
                speed_standardize=metainfo['speed_standardize'] ,
                coord_standardize=metainfo['coord_standardize'],
            )

        tab_tuple=deepof.utils.get_dt(processed_exp[0],key)
        tab_tuple = (reorder_and_reshape(tab_tuple[0]),np.expand_dims(tab_tuple[1],-1))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        x_all = torch.as_tensor(tab_tuple[0], dtype=torch.float32, device=device)
        a_all = torch.as_tensor(tab_tuple[1], dtype=torch.float32, device=device)

        batch_size = 256  # adjust to fit your GPU
        emb_list, sc_list = [], []
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

    gate_edges = None
    if softcounts_extraction_method in {"gmm", "msm", "combined"}:
        if isinstance(animal_id,str):
            animal_id=[animal_id]
        gate_edges = deepof.post_hoc.compute_gate_edges(
            coordinates=coordinates,
            animal_ids=animal_id,
            keys=list(embeddings.keys()),
            window_size=window_size,
            supervised_annotations=supervised_annotations,
            M_gates=3,
            embedding_gates=embedding_gates,
        )


    if softcounts_extraction_method == "gmm":

        soft_counts_dict = deepof.post_hoc.get_contrastive_soft_counts_gmm(
            coordinates=coordinates,
            embeddings=embeddings,
            window_size=window_size,
            animal_ids=animal_id,
            supervised_annotations=supervised_annotations,
            embedding_gates=embedding_gates,
            temporal_smooth_win=3,
            N_clusters_per_gate=states,
            M_gates=3,
            gate_edges=gate_edges,
        )
        soft_counts = soft_counts_dict[extract_pair]


    elif softcounts_extraction_method == "msm" or softcounts_extraction_method == "combined":

        soft_counts_dict = deepof.post_hoc.get_contrastive_soft_counts_msm_pcca(
            coordinates=coordinates,
            embeddings=embeddings,
            window_size=window_size,
            animal_ids=animal_id,
            supervised_annotations=supervised_annotations,
            embedding_gates=embedding_gates,
            temporal_smooth_win=3,
            N_clusters_per_gate=states,
            M_gates=3,
            gate_edges=gate_edges,
            n_micro=200,  # 400
            lagtime=3,    # 3
        )
        if softcounts_extraction_method == "combined":

            supervised_chaos = deepof.post_hoc.get_supervised_chaos(coordinates, quality_threshold, frac_bps_below)

            soft_counts_chaos_dict = deepof.post_hoc.get_contrastive_soft_counts_gmm(
                coordinates=coordinates,
                embeddings=embeddings,
                window_size=window_size,
                animal_ids=animal_id,
                supervised_annotations=supervised_chaos,
                temporal_smooth_win=3,
                N_clusters_per_gate=states,
                embedding_gates=['anychaos'],
                M_gates=3,
                gate_edges=None,
            )
            soft_counts_dict = deepof.post_hoc.add_chaos_gates(
                coordinates=coordinates, 
                soft_counts_dict=soft_counts_dict, 
                soft_counts_chaos_dict=soft_counts_chaos_dict,
                supervised_chaos=supervised_chaos, 
                extract_pair=extract_pair,
                window_size=window_size)
        
        soft_counts = soft_counts_dict[extract_pair]

    elif softcounts_extraction_method is not None:
        raise ValueError("For \"softcounts_extraction_method\" only \"gmm\", \"msm\" or \"combined\" are supported!")
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


def _slice_time_per_sample(
    x: torch.Tensor,          # (B,T,...)
    start: torch.Tensor,      # (B,)
    length: int,
) -> torch.Tensor:
    """
    Slice a per-sample contiguous window along time dim=1.
    Returns shape (B, length, ...)
    """
    B, T = x.shape[0], x.shape[1]
    t_idx = start[:, None] + torch.arange(length, device=x.device)[None, :]  # (B,L)
    b_idx = torch.arange(B, device=x.device)[:, None]                       # (B,1)
    return x[b_idx, t_idx]  # advanced indexing -> (B,L,...)


@torch.no_grad()
def _materialize_encoder(model, x_shape, a_shape, device):
    """
    Run a tiny encoder forward pass to force lazy modules (CensNetConvPT) to build
    their Parameters so load_state_dict can actually load them.
    """

    T, N, F = x_shape
    T2, E, EF = a_shape
    assert T == T2

    x = torch.zeros((1, T, N, F), device=device, dtype=torch.float32)
    a = torch.zeros((1, T, E, EF), device=device, dtype=torch.float32)

    # Make sure there's at least one non-zero timestep (guards any masking logic)
    x[:, 0, 0, 0] = 1.0
    a[:, 0, 0, 0] = 1.0

    _ = model.encoder(x, a)   # sufficient to build encoder.spatial_gnn_block params


def load_model_from_ckpt(path: str, device=None, strict: bool = False):
    """
    Load a single model checkpoint saved via save_model_info(..., save_bundle=True)
    using only the checkpoint path.
    Returns: model, log_summary, rebuild_spec, load_report
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)  # weights_only=True is NOT compatible with arbitrary dict payloads
    if "state_dict" not in ckpt:
        raise RuntimeError(f"Checkpoint at {path} is not a bundle (missing 'state_dict').")
    if "rebuild_spec" not in ckpt:
        raise RuntimeError(f"Checkpoint at {path} is missing 'rebuild_spec' (cannot rebuild model from path only).")

    spec = ckpt["rebuild_spec"]
    state = ckpt["state_dict"]
    log_summary = ckpt.get("log_summary", {})

    model_name = spec["model_name"].lower()

    # --- rebuild ---
    if model_name == "vqvae":
        from deepof.clustering.models_new import VQVAEPT
        model = VQVAEPT(
            input_shape=tuple(spec["x_shape"]),
            edge_feature_shape=tuple(spec["a_shape"]),
            adjacency_matrix=np.asarray(spec["adjacency_matrix"]),
            latent_dim=int(spec["latent_dim"]),
            n_components=int(spec["n_components"]),
            encoder_type=str(spec["encoder_type"]),
            use_gnn=bool(spec.get("use_gnn", True)),
            interaction_regularization=float(spec.get("interaction_regularization", 0.0)),
            kmeans_loss=float(spec.get("kmeans_loss", 0.0)),
        )

    elif model_name == "contrastive":
        import deepof.clustering.models_new as models_new
        model = models_new.ContrastivePT(
            input_shape=tuple(spec["x_shape"]),
            edge_feature_shape=tuple(spec["a_shape"]),
            adjacency_matrix=np.asarray(spec["adjacency_matrix"]),
            latent_dim=int(spec["latent_dim"]),
            encoder_type=str(spec["encoder_type"]),
            use_gnn=bool(spec.get("use_gnn", True)),
            temperature=float(spec.get("temperature", 0.1)),
            similarity_function=str(spec.get("similarity_function", "cosine")),
            loss_function=str(spec.get("loss_function", "nce")),
            beta=float(spec.get("beta", 0.1)),
            tau=float(spec.get("tau", 0.1)),
            interaction_regularization=float(spec.get("interaction_regularization", 0.0)),
        )

    elif model_name == "vade":
        from deepof.clustering.models_new import VaDEPT
        model = VaDEPT(
            input_shape=tuple(spec["x_shape"]),
            edge_feature_shape=tuple(spec["a_shape"]),
            adjacency_matrix=np.asarray(spec["adjacency_matrix"]),
            latent_dim=int(spec["latent_dim"]),
            n_components=int(spec["n_components"]),
            encoder_type=str(spec["encoder_type"]),
            use_gnn=bool(spec.get("use_gnn", True)),
            kmeans_loss=float(spec.get("kmeans_loss", 1.0)),
            interaction_regularization=float(spec.get("interaction_regularization", 0.0)),
            lens_enabled=bool(spec.get("lens_enabled", False)),
        )

    else:
        raise ValueError(f"Unknown model_name in rebuild_spec: {model_name}")

    model.to(device)
    model.eval()
    _materialize_encoder(model, tuple(spec["x_shape"]), tuple(spec["a_shape"]), device)
    rep = model.load_state_dict(state, strict=strict)
    model.eval()

    load_report = {"missing": rep.missing_keys, "unexpected": rep.unexpected_keys}
    return model, log_summary, spec, load_report