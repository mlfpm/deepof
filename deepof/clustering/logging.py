"""
logging and user-feedback functionality for models
"""
# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

from typing import Any, NewType, Iterable, Tuple, Dict, Optional, Mapping, Callable

import numpy as np
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


import deepof.clustering.model_utils_new
from deepof.data_loading import get_dt
from copy import deepcopy
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

@torch.no_grad()
def get_q_vade(
    model: nn.Module,
    x: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts soft cluster assignments q(c|z) from VaDE.

    Assumes model(x, a, return_gmm_params=True) returns a tuple whose third
    element is the cluster responsibility matrix of shape [B, K].
    """
    outputs = model(x, a, return_gmm_params=True)
    q = outputs[2]

    if not torch.is_tensor(q) or q.ndim != 2: # pragma: no cover
        raise RuntimeError(
            f"Expected VaDE responsibilities at outputs[2] with shape [B, K], got {type(q)}"
        )

    q = q.float().clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


@torch.no_grad()
def get_q_vqvae(
    model: nn.Module,
    x: torch.Tensor,
    a: torch.Tensor,
    *,
    distill_head: nn.Module,
) -> torch.Tensor:
    """
    Extracts soft cluster assignments for VQVAE by applying the distillation head
    to the pre-quantization encoder output.
    """
    _, _, _, _, encoder_output, _ = model(
        x, a, return_losses=True, return_all_outputs=True
    )

    logits = distill_head(encoder_output.float())
    q = F.softmax(logits, dim=-1).clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


@torch.no_grad()
def get_q_contrastive(
    model: nn.Module,
    x_full: torch.Tensor,
    a_full: torch.Tensor,
    *,
    distill_head: nn.Module,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Extracts soft cluster assignments for the contrastive model by reproducing
    the main-window embedding path used for distillation during training.
    """
    del a_full  # Not used; edges are recomputed from x_full exactly as in training.

    a_full = deepof.clustering.model_utils_new.recompute_edges(x_full, edge_index)

    half_len = x_full.shape[1] // 2
    starts = torch.full(
        (x_full.shape[0],),
        fill_value=half_len // 2,
        device=x_full.device,
        dtype=torch.long,
    )

    x = deepof.clustering.model_utils_new.slice_time_per_sample(x_full, starts, half_len)
    a = deepof.clustering.model_utils_new.slice_time_per_sample(a_full, starts, half_len)

    z = model(x, a)
    z = F.normalize(z, dim=1)

    logits = distill_head(z.float())
    q = F.softmax(logits, dim=-1).clamp_min(1e-8)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return q


@torch.no_grad()
def compute_vade_specific_diagnostics(model: nn.Module) -> Dict[str, float]:
    """
    Computes VaDE-specific diagnostics related to the latent GMM.

    Returns an empty dict for models without a latent_space GMM.
    """
    base = deepof.clustering.model_utils_new.unwrap_dp(model)
    latent_space = getattr(base, "latent_space", None)
    if latent_space is None: # pragma: no cover
        return {}

    out: Dict[str, float] = {}

    if hasattr(latent_space, "gmm_log_vars"):
        gmm_log_vars = latent_space.gmm_log_vars.detach()
        out["diag/gmm_logvar_min"] = float(gmm_log_vars.min().item())
        out["diag/gmm_logvar_max"] = float(gmm_log_vars.max().item())

    if hasattr(latent_space, "prior"):
        prior = latent_space.prior
        prior = prior.detach() if torch.is_tensor(prior) else torch.as_tensor(prior)
        prior = prior.clamp_min(1e-9)
        out["diag/prior_entropy"] = float(-(prior * prior.log()).sum().item())

    return out


@torch.no_grad()
def compute_diagnostics(
    model: nn.Module,
    dataloader: DataLoader,
    q_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    n_components: int,
    tau_star: Optional[torch.Tensor] = None,
    distill_sharpen_T: float = 0.5,
    distill_conf_weight: bool = False,
    distill_conf_thresh: float = 0.55,
    max_batches: int = 4,
    extra_stats_fn: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Computes clustering diagnostics and alignment score from model soft assignments.

    The function is model-agnostic: it only requires a q_fn that extracts soft
    assignments q of shape [B, K] from a batch.

    Args:
    model (nn.Module): Model to evaluate.
    dataloader (DataLoader): Validation or analysis loader.
    q_fn (Callable): Function that maps (model, x, a) -> q of shape [B, K].
    device (torch.device): Device for inference.
    n_components (int): Number of clusters/components K.
    tau_star (Optional[torch.Tensor]): Teacher assignments of shape [N, K]. If provided,
        the balance term is based on KL(q_marginal || tau_marginal). If None, the balance
        term falls back to normalized marginal entropy of q.
    distill_sharpen_T (float): Temperature used when computing teacher-confidence diagnostics.
    distill_conf_weight (bool): Whether teacher confidence weighting is enabled.
    distill_conf_thresh (float): Threshold used for teacher confidence weighting diagnostics.
    max_batches (int): Maximum number of dataloader batches to inspect.
    extra_stats_fn (Optional[Callable]): Optional model-specific diagnostics helper.

    Returns:
    Dict[str, float]: Diagnostics dictionary including conf_norm, bal_norm, and alignment_score.
    """
    if n_components < 2: # pragma: no cover
        raise ValueError(f"n_components must be >= 2, got {n_components}")

    was_training = model.training
    model.eval()

    total_samples = 0
    sum_ent = 0.0
    sum_max_p = 0.0
    sum_q = None

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        batch = deepof.clustering.model_utils_new.move_to(batch, device)
        x, a = batch[0], batch[1]

        q = q_fn(model, x, a)

        if not torch.is_tensor(q) or q.ndim != 2 or q.size(1) != n_components: # pragma: no cover
            raise RuntimeError(
                f"q_fn must return a [B, {n_components}] tensor, got {tuple(q.shape)}"
            )

        q = q.float().clamp_min(1e-8)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        ent = -(q * q.log()).sum(dim=-1)  # [B]
        sum_ent += float(ent.sum().item())
        sum_max_p += float(q.max(dim=-1).values.sum().item())
        total_samples += q.size(0)

        q_sum = q.sum(dim=0)  # [K]
        sum_q = q_sum if sum_q is None else (sum_q + q_sum)

    out: Dict[str, float] = {
        "diag/q_mean_entropy": float("nan"),
        "diag/q_marginal_entropy": float("nan"),
        "diag/q_mean_max_prob": float("nan"),
        "diag/teacher_marginal_entropy": float("nan"),
        "diag/teacher_conf_mean": float("nan"),
        "diag/teacher_weight_mean": float("nan"),
        "diag/kl_marg_q_to_tau": float("nan"),
        "conf_norm": float("nan"),
        "bal_norm": float("nan"),
        "alignment_score": float("nan"),
    }

    if total_samples > 0:
        q_marginal = (sum_q / total_samples).clamp_min(1e-9)
        mean_q_entropy = sum_ent / total_samples
        q_marginal_entropy = float(-(q_marginal * q_marginal.log()).sum().item())
        mean_q_max_prob = sum_max_p / total_samples

        logK = math.log(float(n_components))
        conf_norm = _clip01(1.0 - mean_q_entropy / max(1e-9, logK))

        out["diag/q_mean_entropy"] = mean_q_entropy
        out["diag/q_marginal_entropy"] = q_marginal_entropy
        out["diag/q_mean_max_prob"] = mean_q_max_prob

        if tau_star is not None:
            tau = tau_star.detach().to(device=q_marginal.device, dtype=q_marginal.dtype)
            if tau.ndim != 2 or tau.size(1) != n_components: # pragma: no cover
                raise RuntimeError(
                    f"tau_star must have shape [N, {n_components}], got {tuple(tau.shape)}"
                )

            tau_marg = tau.mean(dim=0).clamp_min(1e-9)
            teacher_marginal_entropy = float(-(tau_marg * tau_marg.log()).sum().item())

            kl_marg_q_to_tau = float(
                (q_marginal * (q_marginal.log() - tau_marg.log())).sum().item()
            )
            kl_marg_q_to_tau = max(0.0, kl_marg_q_to_tau)

            bal_norm = _clip01(1.0 - kl_marg_q_to_tau / max(1e-9, logK))

            T = float(distill_sharpen_T)
            if T > 0.0:
                tau_sharp = torch.softmax(tau.clamp_min(1e-8).log() / T, dim=-1)
            else:
                tau_sharp = tau

            conf = tau_sharp.max(dim=1).values
            teacher_conf_mean = float(conf.mean().item())

            if distill_conf_weight:
                thr = float(distill_conf_thresh)
                w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(min=0.0, max=1.0)
                teacher_weight_mean = float(w.mean().item())
            else:
                teacher_weight_mean = 1.0

            out["diag/teacher_marginal_entropy"] = teacher_marginal_entropy
            out["diag/teacher_conf_mean"] = teacher_conf_mean
            out["diag/teacher_weight_mean"] = teacher_weight_mean
            out["diag/kl_marg_q_to_tau"] = kl_marg_q_to_tau

        else:
            bal_norm = _clip01(q_marginal_entropy / max(1e-9, logK))

        alignment_score = conf_norm * bal_norm

        out["conf_norm"] = conf_norm
        out["bal_norm"] = bal_norm
        out["alignment_score"] = alignment_score

    if extra_stats_fn is not None:
        out.update(extra_stats_fn(model))

    if was_training:
        model.train()

    return out


def init_log_summary():
    """Initialize distionary structure of losses to collect"""

    log_summary={}
    for data_type in ['train','val']:
        log_summary[data_type]={}
        log_summary[data_type]['total_loss']=[]
        log_summary[data_type]['reconstruction_loss']=[]
        log_summary[data_type]['kl_divergence']=[]
        log_summary[data_type]['cat_cluster_loss']=[]
        log_summary[data_type]['kmeans_loss']=[]
        log_summary[data_type]['distill_loss']=[]
        log_summary[data_type]['temporal_loss']=[]
        log_summary[data_type]['scatter_loss']=[]
        log_summary[data_type]['nonempty_loss']=[]
        log_summary[data_type]['repel_loss']=[]
        log_summary[data_type]['tf_cluster_loss']=[]
        log_summary[data_type]['prior_loss']=[]
        log_summary[data_type]['activity_l1']=[]
        log_summary[data_type]['pos_similarity']=[]
        log_summary[data_type]['neg_similarity']=[]
        log_summary[data_type]['conf_norm']=[]
        log_summary[data_type]['bal_norm']=[]
        log_summary[data_type]['alignment_score']=[]


    return log_summary


def _update_log_summary(log_summary: dict, train_logs: dict, val_logs: dict):
    """Append the current losses from train and val logs to the log summary."""
    
    for data_type, logs in zip(['train', 'val'], [train_logs, val_logs]):
        # Append scores directly
        if data_type=="train":
            for key in log_summary.keys():
                if key=="train" or key=="val" or key=="test":
                    continue
                value = logs.get(key, np.nan)
                log_summary[key].append(value)
        # Append losses to training / testing dicts
        for key in log_summary[data_type].keys():
            value = logs.get(key, np.nan)
            log_summary[data_type][key].append(value)
    return log_summary


def print_losses(model_name: str,
                  log_summary: dict,
                  epoch: int,
                  n_epochs: int,
                  train_logs: dict,
                  val_logs: dict,
                  klw: float = 0.0,
                  lambda_d: float = 0.0):
    """Print losses neatly aligned and append them to the log summary."""

    # Define consistent field width for alignment
    col_width = 10

    def _fmt_loss(name: str, logs: dict, width: int = col_width, precision: int = 4):
        val = logs.get(name, float("nan"))
        # right-aligned, fixed width, fixed decimals
        return f"{val:<{width}.{precision}f}"

    loss_names = [
        "total_loss", "pos_similarity", "neg_similarity", "reconstruct_loss", "prior_loss", "kl_div",
        "kmeans_loss", "cat_clust_loss", "distill_loss", "temporal_loss", 
        "scatter_loss", "repel_loss", "nonempty_loss", "tf_clust_loss",
    ]

    # Print header line
    header = f"Epoch {epoch+1}/{n_epochs}"
    if model_name == "vade":
        header += f" | KLw={klw:.3f}"
    header += f" | λ_distill={lambda_d:.3f}"
 
    print(header)
    print("Losses:")

    # Helper for print losses oderly
    def _print_phase(phase: str, logs: dict):
        line = f"  {phase:<7}:"
        z=0
        for name in loss_names:
            # Skip non-relevant losses
            if np.isnan(logs.get(name, float("nan"))):
                continue
            key_label = name.replace("_loss", "").replace("_", "")
            key_label = key_label.replace("reconstruct", "recon")
            key_label = key_label.replace("similarity", "-sim")
            line += f" {key_label[:9]:<9}: {_fmt_loss(name, logs)} |"
            if (z + 1) % 5 == 0 and z != len(loss_names) - 1:  # line wrap for long outputs
                line += "\n          "
            z+=1
        print(line.rstrip("|"))

    # Print train and validation sections
    _print_phase("Train", train_logs)
    _print_phase("Val", val_logs)

    print("Scores:")

    # Alignment metrics
    footer = f"  Train total={train_logs.get('total_loss', np.nan):.4f} | Val total={val_logs.get('total_loss', np.nan):.4f}"
    footer += (f" | Align: conf={val_logs.get('conf_norm', np.nan):.3f} | " +
        f"bal={val_logs.get('bal_norm', np.nan):.3f} | " +
        f"score={val_logs.get('alignment_score', np.nan):.3f}")
    print(footer)
        

    # Update summary
    log_summary = _update_log_summary(log_summary, train_logs, val_logs)
    return log_summary


def average_logs(logs_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, Tuple[float, int]] = {}
    for logs in logs_list:
        for k, v in logs.items():
            s, n = out.get(k, (0.0, 0))
            out[k] = (s + float(v), n + 1)
    return {k: s / max(n, 1) for k, (s, n) in out.items()}


def log_epoch_to_tensorboard(
    writer: Optional[object],
    train_logs: Dict[str, float],
    val_logs: Dict[str, float],
    epoch: int,
    score_value: float = float("nan"),
    lambda_d: float = 0.0,
):
    """
    Writes per-epoch training and validation metrics to TensorBoard.

    Args:
        writer (Optional[SummaryWriter]): TensorBoard writer. If None, this is a no-op.
        train_logs (Dict[str, float]): Training metrics for the current epoch.
        val_logs (Dict[str, float]): Validation metrics for the current epoch.
        epoch (int): Current epoch index.
        score_value (float): Current alignment score. Only logged if finite.
        lambda_d (float): Current distillation lambda weight.
    """
    if writer is None:
        return

    for k, v in train_logs.items():
        writer.add_scalar(f"Train/{k}", v, epoch)
    for k, v in val_logs.items():
        writer.add_scalar(f"Val/{k}", v, epoch)
    writer.add_scalar("Distill/lambda", lambda_d, epoch)

    if math.isfinite(score_value):
        writer.add_scalar("Val/alignment_score", score_value, epoch)
        writer.add_scalar("Val/conf_norm", val_logs.get("conf_norm", float("nan")), epoch)
        writer.add_scalar("Val/bal_norm", val_logs.get("bal_norm", float("nan")), epoch)
