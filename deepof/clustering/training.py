"""
Training functionality for all models
"""
# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

from typing import Any, NewType, Tuple, Dict, Optional, List, Callable
from types import SimpleNamespace
from dataclasses import dataclass
from functools import partial

import os
import numpy as np
import math


import torch
import torch.nn as nn
import torch.nn.functional as F


import deepof.clustering.dataset
import deepof.clustering.model_utils_new
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deepof.clustering.models_new import VaDEPT, VQVAEPT, ContrastivePT 
from deepof.clustering.model_utils_new import (
    CommonFitCfg,
    TurtleTeacherCfg,
    VaDECfg, 
    ContrastiveCfg,
    move_to,
    unwrap_dp,
    recompute_edges,
    ckpt_paths,
    save_model_info,
    load_best_checkpoints,
    slice_time_per_sample,
)
from deepof.clustering.logging import (
    init_log_summary, 
    average_logs, 
    print_losses, 
    log_epoch_to_tensorboard, 
    compute_vade_specific_diagnostics, 
    get_q_vade,
    get_q_vqvae,
    get_q_contrastive
)
from deepof.clustering.losses import Dynamic_weight_manager, build_optimizer_generic, build_optimizer_vade, VadeLoss, select_contrastive_loss_pt
from deepof.clustering.teacher_model import DiscriminativeHead, extract_latents, initialize_gmm_from_teacher, run_turtle_teacher_on_views
import deepof.clustering.teacher_model

# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


@dataclass
class StepResult:
    loss: torch.Tensor
    logs: Dict[str, float]


def _progress_wrap(dataloader, show: bool, desc: str, leave=False):
    return tqdm(dataloader, desc=desc, mininterval=0.5, leave=leave) if show else dataloader

def _format_postfix(logs: Dict[str, float], max_items: int = 4) -> Dict[str, str]:
    priority = [
        "total_loss",
        "reconstruct_loss", "enc_rec_loss",
        "kl_div",
        "vq_loss", "kmeans_loss",
        "repel_loss", "nonempty_loss", 
        "pos_similarity", "neg_similarity",
    ]
    out: Dict[str, str] = {}
    for k in priority:
        if k in logs:
            out[k] = f"{logs[k]:.4f}"
            if len(out) >= max_items:
                return out
    for k, v in logs.items():
        if k not in out:
            out[k] = f"{v:.4f}"
            if len(out) >= max_items:
                break
    return out


def train_one_epoch_indexed(
    model: nn.Module,
    model_name: str,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    step_fn: Callable[[nn.Module, Any, SimpleNamespace], StepResult],
    device: torch.device,
    epoch: int,
    num_epochs: int,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = True,
    grad_clip_value: Optional[float] = 0.75,
    ctx: Optional[SimpleNamespace] = None,
    show_progress: bool = True,
    leave: bool=False,
) -> Dict[str, float]:
    model.train()
    if ctx and getattr(ctx, "criterion", None) is not None:
        ctx.criterion.train()

    logs_accum = []
    iterator = _progress_wrap(dataloader, show_progress, desc=f"Train {epoch+1}/{num_epochs}", leave=leave)

    seen = 0
    mean_kl_weight = 0.0
    mean_lambda_weight = 0.0
    for step, batch in enumerate(iterator):
        batch = move_to(batch, device) 
        x, a = batch[0], batch[1]              # <----
        idx = batch[-2]   # assume dataset returns (x, a)
        B = x.size(0)
        #idx = torch.arange(seen, seen + B, device=x.device, dtype=torch.long)
        seen += B

        batch_idx = (x, a, idx)

        if scaler is not None and use_amp and device.type == "cuda":
            with autocast(device_type=device.type, dtype=torch.float16):
                res = step_fn(model, batch_idx, SimpleNamespace(
                    train=True, epoch=epoch, num_epochs=num_epochs, **(ctx.__dict__ if ctx else {})
                ))
            scaler.scale(res.loss).backward()
            
            if grad_clip_value is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            scaler.step(optimizer)
            #if torch.isnan(model.encoder.node_recurrent_block.conv1d.weight).any():
            #    print("z issues!")
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            res = step_fn(model, batch_idx, SimpleNamespace(
                train=True, epoch=epoch, num_epochs=num_epochs, **(ctx.__dict__ if ctx else {})
            ))
            optimizer.zero_grad(set_to_none=True)
            res.loss.backward()
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
            optimizer.step()

        if ctx is not None and model_name=="vade":
            if hasattr(ctx.criterion, "kl_scheduler") and ctx.criterion.kl_scheduler is not None:
                ctx.criterion.kl_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_kl_weight=ctx.criterion.kl_scheduler.get_weight()
            if hasattr(ctx.criterion, "lambda_scheduler") and ctx.criterion.lambda_scheduler is not None:
                ctx.criterion.lambda_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_lambda_weight=ctx.criterion.lambda_scheduler.get_weight()
        elif ctx is not None:
            if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
                ctx.lambda_scheduler.step()
                if(step==int(len(iterator)/2)):
                    mean_lambda_weight=ctx.lambda_scheduler.get_weight()

        logs_accum.append(res.logs)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(_format_postfix(res.logs), refresh=False)        

    return average_logs(logs_accum), mean_kl_weight, mean_lambda_weight


@torch.no_grad()
def validate_one_epoch_indexed(
    model: nn.Module,
    dataloader: DataLoader,
    step_fn: Callable[[nn.Module, Any, SimpleNamespace], StepResult],
    device: torch.device,
    epoch: int,
    num_epochs: int,
    ctx: Optional[SimpleNamespace] = None,
    show_progress: bool = True,
    leave: bool=False,
) -> Dict[str, float]:
    model.eval()
    if ctx and getattr(ctx, "criterion", None) is not None:
        ctx.criterion.eval()

    logs_accum = []
    iterator = _progress_wrap(dataloader, show_progress, desc=f"Val {epoch+1}/{num_epochs}", leave=leave)

    seen = 0
    for batch in iterator:
        batch = move_to(batch, device)
        x, a = batch[0], batch[1]
        idx = batch[-2]
        B = x.size(0)
        #idx = torch.arange(seen, seen + B, device=x.device, dtype=torch.long)
        seen += B

        res = step_fn(model, (x, a, idx), SimpleNamespace(
            train=False, epoch=epoch, num_epochs=num_epochs, **(ctx.__dict__ if ctx else {})
        ))
        logs_accum.append(res.logs)

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(_format_postfix(res.logs), refresh=False)

    return average_logs(logs_accum)

def step_vade(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],  # (x, a, idx or None)
    ctx: SimpleNamespace,
) -> StepResult:
    x, a, idx = batch


    base_m = unwrap_dp(model)
    T_high = float(getattr(ctx, "resp_temp_start", 1.8))
    T_low  = 1.0
    T_resp = T_high
    if hasattr(ctx.criterion, "kl_scheduler") and (ctx.criterion.kl_scheduler is not None):
        w = float(ctx.criterion.kl_scheduler.get_weight())
        w_max = float(getattr(ctx.criterion.kl_scheduler, "max_weight", 1.0))
        prog = 0.0 if w_max <= 0 else min(1.0, max(0.0, w / w_max))
        T_resp = T_low + (T_high - T_low) * (1.0 - prog)
    # Clamp and set
    T_resp = float(max(1.0, min(T_high, T_resp)))
    if hasattr(base_m, "latent_space"):
        setattr(base_m.latent_space, "responsibility_temp", T_resp)


    outputs = model(x, a, return_gmm_params=True)

    if hasattr(ctx, "criterion") and ctx.criterion is not None:
        apply_distill = getattr(ctx, "apply_distill", True)
        batch_indices = idx if apply_distill else None

        if hasattr(ctx.criterion, "kl_scheduler") and ctx.criterion.kl_scheduler is not None:
            ctx.criterion.kl_weight = float(ctx.criterion.kl_scheduler.get_weight())
        elif hasattr(ctx.criterion, "kl_weight"):
            ctx.criterion.kl_weight = float(ctx.criterion.kl_weight)
        if hasattr(ctx, "lambda_scheduler") and ctx.criterion.lambda_scheduler is not None:
            ctx.criterion.lambda_distill = float(ctx.criterion.lambda_scheduler.get_weight())

        loss_dict = ctx.criterion(outputs, x, batch_indices=batch_indices)
        total = loss_dict["total_loss"]
        
        if loss_dict["reconstruct_loss"] > 50000:
            print("Halt!")

        # EMA update of π from current batch responsibilities, currently unused 
        with torch.no_grad(): # pragma: no cover
            beta = float(getattr(ctx, "pi_ema_beta", 0.00))
            if beta > 0.0 and outputs[2] is not None:
                q_batch = outputs[2].detach()                 # (B,C)
                m = q_batch.mean(dim=0)                       # (C,)
                base = unwrap_dp(model)
                if hasattr(base.latent_space, "prior"):
                    pi = base.latent_space.prior
                    if isinstance(pi, torch.nn.Parameter):
                        pi_data = pi.data
                        pi_data.mul_(1.0 - beta).add_(beta * m)
                        pi_data.clamp_(min=1e-6)
                        pi_data.div_(pi_data.sum())
                    else:
                        pi.mul_(1.0 - beta).add_(beta * m)
                        pi.clamp_(min=1e-6)
                        pi.div_(pi.sum())

        logs = {
            "total_loss": float(total.detach().item()),
            "reconstruct_loss": float(loss_dict["reconstruct_loss"].detach().item()),
            "kl_div": float(loss_dict["kl_div"].detach().item()),
            "cat_clust_loss": float(loss_dict["cat_clust_loss"].detach().item()),
            "kmeans_loss": float(loss_dict["kmeans_loss"].detach().item()) if torch.is_tensor(loss_dict["kmeans_loss"]) else float(loss_dict["kmeans_loss"]),
            "activity_l1": float(loss_dict["activity_l1"].detach().item()),
            "prior_loss": float(loss_dict["prior_loss"].detach().item()),
            "distill_loss": float(loss_dict["distill_loss"].detach().item()),
            "tf_clust_loss" : float(loss_dict["tf_clust_loss"].detach().item()),
            "nonempty_loss": float(loss_dict["nonempty_loss"].detach().item()),
            "temporal_loss": float(loss_dict["temporal_loss"].detach().item()),
            "scatter_loss": float(loss_dict["scatter_loss"].detach().item()),
            "repel_loss": float(loss_dict["repel_loss"].detach().item()),
        }
        return StepResult(loss=total, logs=logs)
    else: # pragma: no cover
        raise RuntimeError("VaDE step requires ctx.criterion (VaDELoss) to be provided.")


def step_vqvae_distill(
    model: nn.Module,  # VQVAEPT or DataParallel(VQVAEPT)
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (x, a, idx)
    ctx: SimpleNamespace,
) -> StepResult:
    x, a, idx = batch
    device = x.device
    apply_distill = getattr(ctx, "apply_distill", True)

    # Forward (ask for all to get z_e)
    encoding_recon, recon, quant, soft_counts, encoder_output, vq_losses = model(
        x, a, return_losses=True, return_all_outputs=True
    )

    # Flatten targets to match decoder distribution event shape
    B, T, N, F = x.shape
    x_flat = x.view(B, T, N * F)

    with torch.amp.autocast(device_type=device.type, enabled=False):
        enc_rec_loss = -(encoding_recon.log_prob(x_flat.float())).mean()
        rec_loss     = -(recon.log_prob(x_flat.float())).mean()

    vq_loss     = float(vq_losses.get("vq_loss", 0.0))
    kmeans_loss = float(vq_losses.get("kmeans_loss", 0.0))
    vq_total_t  = torch.as_tensor(vq_loss + kmeans_loss, device=device, dtype=enc_rec_loss.dtype)

    base_total = enc_rec_loss + rec_loss + vq_total_t
    distill_loss = torch.tensor(0.0, device=device, dtype=enc_rec_loss.dtype)

    # Distillation on z_e
    lambda_distill = 0.0
    if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
        lambda_distill = float(ctx.lambda_scheduler.get_weight())

    if apply_distill and hasattr(ctx, "distill_head") and lambda_distill > 0.0:
        z_e = encoder_output
        logits = ctx.distill_head(z_e)

        idx = idx.to(device).long()
        tau_b = ctx.tau_star[idx]

        eps = 1e-8
        T = float(getattr(ctx, "distill_sharpen_T", 0.5))
        if T > 0.0:
            logits_t = (tau_b.clamp_min(eps)).log() / T
            tau_b = torch.softmax(logits_t, dim=-1)

        if getattr(ctx, "distill_conf_weight", False):
            conf = tau_b.max(dim=1).values
            thr = float(getattr(ctx, "distill_conf_thresh", 0.6))
            w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(0.0, 1.0).detach()
            per_sample = _soft_ce_logits(logits, tau_b, reduction="none")
            distill_loss = (w * per_sample).mean()
        else:
            distill_loss = _soft_ce_logits(logits, tau_b, reduction="mean")

        distill_loss = lambda_distill * distill_loss
        total = base_total + distill_loss
    else:
        total = base_total

    # Populated clusters (codes used)
    try:
        populated = soft_counts.argmax(dim=-1).unique().numel()
        populated_f = float(populated)
    except Exception:
        populated_f = float("nan")

    logs = {
        "total_loss": float(total.detach().item()),
        "enc_rec_loss": float(enc_rec_loss.detach().item()),
        "reconstruct_loss": float(rec_loss.detach().item()),
        "vq_loss": vq_loss,
        "kmeans_loss": kmeans_loss,
        "number_of_populated_clusters": populated_f,
        "distill_loss": float(distill_loss.detach().item()) if torch.is_tensor(distill_loss) else float(distill_loss),
    }
    return StepResult(loss=total, logs=logs)


def _soft_ce_logits(logits: torch.Tensor, soft_targets: torch.Tensor, eps: float = 1e-8, reduction: str = "mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    soft_targets = torch.clamp(soft_targets, min=eps, max=1.0)
    per_sample = -(soft_targets * log_probs).sum(dim=-1)
    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    return per_sample



def step_contrastive_distill(
    model: nn.Module,  # ContrastivePT or DataParallel(ContrastivePT)
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # (x, a, idx)
    ctx: SimpleNamespace,
) -> StepResult:
    x_full, a_full, idx = batch
    base = unwrap_dp(model)
    device = x_full.device
    apply_distill = getattr(ctx, "apply_distill", True)
    edge_index = getattr(ctx, "edge_index", None)  
    if edge_index is None: # pragma: no cover
        raise RuntimeError("ctx.edge_index is required for contrastive augmentation!")
    
    contrastive_cfg=getattr(ctx, "contrastive_cfg", None)
    
    a_full = recompute_edges(x_full, edge_index)
    rot_precomp = getattr(ctx, "rot_precomp", None)
    if rot_precomp is None: # pragma: no cover
        raise RuntimeError("ctx.rot_precomp is required (build it once in fit_contrastive).")

    x_aug, a_aug = _make_augmented_view(
        x_full, a_full, edge_index, rot_precomp,
        min_shift = contrastive_cfg.aug_min_shift,
        max_shift = contrastive_cfg.aug_max_shift,
        p_shift = contrastive_cfg.aug_p_shift,
        noise_sigma = contrastive_cfg.aug_noise_sigma,  
        p_noise = contrastive_cfg.aug_p_noise,           
        max_interp = contrastive_cfg.aug_max_interp,
        min_interp = contrastive_cfg.aug_min_interp,         
        p_interp = contrastive_cfg.aug_p_interp, 
        max_rot = contrastive_cfg.aug_max_rot, 
        n_rot = contrastive_cfg.aug_n_rot,
        p_rot = contrastive_cfg.aug_p_rot,         
    )

    # Cut middle section from tensor
    half_len = x_full.shape[1] // 2
    starts=(torch.ones([x_full.shape[0]],device=x_full.device)*half_len // 2).int()

    x = slice_time_per_sample(x_full, starts, half_len)
    a = slice_time_per_sample(a_full, starts, half_len)
        
    # Encode via forward for DP compatibility
    z = model(x, a)
    z_aug = model(x_aug, a_aug)

    # Normalize row-wise
    z = torch.nn.functional.normalize(z, dim=1)
    z_aug = torch.nn.functional.normalize(z_aug, dim=1)

    # Base contrastive loss
    loss, pos_mean, neg_mean = select_contrastive_loss_pt(
        z, z_aug,
        similarity=base.similarity_function,
        loss_fn=base.loss_function,
        temperature=base.temperature,
        tau=base.tau,
        beta=base.beta,
        elimination_topk=0.1,
    )

    distill_loss = torch.tensor(0.0, device=device, dtype=loss.dtype)

    # Distillation on the main window embedding
    lambda_distill = 0.0
    if hasattr(ctx, "lambda_scheduler") and ctx.lambda_scheduler is not None:
        lambda_distill = float(ctx.lambda_scheduler.get_weight())

    if apply_distill and hasattr(ctx, "distill_head") and lambda_distill > 0.0:
        z_main = z
        logits = ctx.distill_head(z_main)

        idx = idx.to(device).long()
        tau_b = ctx.tau_star[idx]

        eps = 1e-8
        T = float(getattr(ctx, "distill_sharpen_T", 0.5))
        if T > 0.0:
            logits_t = (tau_b.clamp_min(eps)).log() / T
            tau_b = torch.softmax(logits_t, dim=-1)

        if getattr(ctx, "distill_conf_weight", False):
            conf = tau_b.max(dim=1).values
            thr = float(getattr(ctx, "distill_conf_thresh", 0.6))
            w = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(0.0, 1.0).detach()
            per_sample = _soft_ce_logits(logits, tau_b, reduction="none")
            distill_loss = (w * per_sample).mean()
        else:
            distill_loss = _soft_ce_logits(logits, tau_b, reduction="mean")

        distill_loss = lambda_distill * distill_loss
        total = loss + distill_loss
    else:
        total = loss

    logs = {
        "total_loss": float(total.detach().item()),
        "pos_similarity": float(pos_mean.detach().item()),
        "neg_similarity": float(neg_mean.detach().item()),
        "distill_loss": float(distill_loss.detach().item()) if torch.is_tensor(distill_loss) else float(distill_loss),
    }
    return StepResult(loss=total, logs=logs)


def embedding_model_fittingPT(
    preprocessed_object: Tuple[dict, dict],
    adjacency_matrix: np.ndarray,
    meta_info: dict,
    encoder_type: str,
    batch_size: int,
    latent_dim: int,
    epochs: int,
    n_components: int,
    # Logging/IO
    output_path: str,
    learning_rate: float = 1e-3,
    log_history: bool = True,
    data_path: str = ".",
    pretrained: Optional[str] = None,
    save_weights: bool = True,
    run: int = 0,
    # VaDE-specific
    reg_cat_clusters: float = 0.0,
    recluster: bool = False,
    freeze_gmm_epochs: int = 0,
    freeze_decoder_epochs: int = 0,
    prior_loss_weight: float = 0.0,
    gmm_learning_rate: float = 1e-3,
    learning_rate_pretrain: float = 1e-3,
    # Regularization knobs
    interaction_regularization: float = 0.0,
    kmeans_loss: float = 0.0,
    # System
    num_workers: int = 0,
    prefetch_factor: int = 0,
    use_amp: bool = False,
    # TURTLE teacher + distillation (VaDE)
    use_turtle_teacher: bool = True,
    teacher_gamma: float = 8.0,
    teacher_outer_steps: int = 500,
    teacher_inner_steps: int = 100,
    teacher_normalize_feats: bool = True,
    lambda_distill: float = 4.0,
    lambda_decay_start: int = 10,
    lambda_end_weight: float = 0.2,
    lambda_cooldown: int = 10,
    teacher_refresh_every: Optional[int] = False,
    teacher_freeze_at: Optional[int] = 10,
    teacher_head_temp: float = 0.5,
    teacher_task_temp: float = 0.5,
    teacher_alpha_sample_entropy: float = 2.0,
    teacher_batch_size: int = 2048,
    # Vade pretrain
    pretrain_epochs: int = 10,
    kmeans_loss_pretrain: float = 1.0,
    repel_weight_pretrain: float = 0.5,
    repel_length_scale_pretrain: float = 0.5,
    nonempty_weight_pretrain: float = 2e-2,
    nonempty_p_pretrain: float = 2.0,
    nonempty_floor_percent_pretrain: float = 0.05,
    # KL cap
    kl_annealing_mode: str = "tf_sigmoid",
    kl_max_weight: float = 1,
    kl_warmup: int = 5,
    kl_end_weight: float = 0.2,
    kl_cooldown: int = 5,
    kl_annealing_mode_pretrain: str = "tf_sigmoid",
    kl_max_weight_pretrain: float = 0.2,
    kl_warmup_pretrain: int = 15,
    kl_end_weight_pretrain: float = 0.2,
    kl_cooldown_pretrain: int = 10,
    reg_scatter_weight: float = 0,
    temporal_cohesion_weight: float = 0,
    reg_scatter_beta: float = 1.0,
    repel_weight: float = 0,
    repel_length_scale: float = 1.0,
    # TF-style cluster term
    tf_cluster_weight: float = 0.0,
    nonempty_weight: float = 2e-2,
    nonempty_floor_percent: float = 0.05,
    nonempty_p: float = 2.0,
    # Distillation weighting (VaDE)
    distill_conf_weight: bool = False,
    distill_conf_thresh: float = 0.3,
    distill_sharpen_T: float = 0.5,
    # Views for teacher
    include_edges_view: bool = False,
    include_nodes_view: bool = True,
    pca_nodes_dim: int = 32,
    pca_edges_dim: int = 32,
    include_angles_view: bool = False,
    pca_angles_dim: int = 32,
    reinit_gmm_on_refresh: bool = False,
    # Diagnostics
    diag_max_batches: int = 4,
    # Model type
    model_name: str = "VaDE",   # "VaDE" (default), "VQVAE", "Contrastive"
    # Distill head (for VQVAE/Contrastive)
    generic_lambda_distill: float = 2.0,
    generic_distill_sharpen_T: float = 0.5,
    generic_distill_conf_weight: bool = True,
    generic_distill_conf_thresh: float = 0.6,
    generic_distill_warmup_epochs: int = 1,
    distill_class_reweight_beta: float = 1,
    distill_class_reweight_cap: float = 3,
    # Contrastive opts
    temperature: float = 0.1,
    contrastive_similarity_function: str = "cosine",
    contrastive_loss_function: str = "nce",
    beta: float = 0.1,
    tau: float = 0.1,
    # Contrastive augmentations
    aug_min_shift: int = 1,
    aug_max_shift: int = 3,
    aug_p_shift: int = 0.4,
    aug_max_rot: int = 30, 
    aug_n_rot: int = 3, 
    aug_p_rot: int = 0.8,
    aug_max_interp: int = 8,
    aug_min_interp: int = 3,         
    aug_p_interp: float = 0.4, 
    aug_noise_sigma: float = 0.03,  
    aug_p_noise: float = 0.4, 
    # Dataset management 
    h5_dataset_folder: Optional[str] = None,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:
    
    # Verify if various model inputs have valid values (TO DO)
    deepof.clustering.model_utils_new.check_model_inputs()

    # Create configs for different models to avoid gigantic function signaturs
    common_cfg = CommonFitCfg(
        model_name=model_name.lower(),
        encoder_type=encoder_type,
        batch_size=batch_size,
        latent_dim=latent_dim,
        epochs=epochs,
        n_components=n_components,
        learning_rate=learning_rate,
        output_path=output_path,
        data_path=data_path,
        log_history=log_history,
        pretrained=pretrained,
        save_weights=save_weights,
        run=run,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        use_amp=use_amp,
        interaction_regularization=interaction_regularization,
        kmeans_loss=kmeans_loss,
        diag_max_batches=diag_max_batches,
        seed=0, 
    )

    teacher_cfg = TurtleTeacherCfg(
        use_turtle_teacher=use_turtle_teacher,
        teacher_gamma=teacher_gamma,
        teacher_outer_steps=teacher_outer_steps,
        teacher_inner_steps=teacher_inner_steps,
        teacher_normalize_feats=teacher_normalize_feats,
        teacher_head_temp=teacher_head_temp,
        teacher_task_temp=teacher_task_temp,
        teacher_alpha_sample_entropy=teacher_alpha_sample_entropy,

        lambda_distill=lambda_distill,
        lambda_decay_start=lambda_decay_start,
        lambda_end_weight=lambda_end_weight,
        lambda_cooldown=lambda_cooldown,
        distill_sharpen_T=distill_sharpen_T,
        distill_conf_weight=distill_conf_weight,
        distill_conf_thresh=distill_conf_thresh,

        generic_lambda_distill=generic_lambda_distill,
        generic_distill_sharpen_T=generic_distill_sharpen_T,
        generic_distill_conf_weight=generic_distill_conf_weight,
        generic_distill_conf_thresh=generic_distill_conf_thresh,
        generic_distill_warmup_epochs=generic_distill_warmup_epochs,
        distill_class_reweight_beta=distill_class_reweight_beta,
        distill_class_reweight_cap=distill_class_reweight_cap,

        include_edges_view=include_edges_view,
        include_nodes_view=include_nodes_view,
        include_angles_view=include_angles_view,
        pca_nodes_dim=pca_nodes_dim,
        pca_edges_dim=pca_edges_dim,
        pca_angles_dim=pca_angles_dim,

        # normalize "False" -> None 
        teacher_refresh_every=(None if teacher_refresh_every is False else teacher_refresh_every),
        teacher_freeze_at=teacher_freeze_at,
        reinit_gmm_on_refresh=reinit_gmm_on_refresh,
        teacher_batch_size=teacher_batch_size,
    )

    vade_cfg = VaDECfg(
        reg_cat_clusters=reg_cat_clusters,
        recluster=recluster,
        freeze_gmm_epochs=freeze_gmm_epochs,
        freeze_decoder_epochs=freeze_decoder_epochs,
        gmm_learning_rate=gmm_learning_rate,
        learning_rate_pretrain=learning_rate_pretrain,
        prior_loss_weight=prior_loss_weight,
        pretrain_epochs=pretrain_epochs,

        reg_scatter_weight=reg_scatter_weight,
        temporal_cohesion_weight=temporal_cohesion_weight,
        reg_scatter_beta=reg_scatter_beta,
        repel_weight=repel_weight,
        repel_length_scale=repel_length_scale,

        tf_cluster_weight=tf_cluster_weight,
        nonempty_weight=nonempty_weight,
        nonempty_floor_percent=nonempty_floor_percent,
        nonempty_p=nonempty_p,

        kmeans_loss_pretrain = kmeans_loss_pretrain,
        repel_weight_pretrain = repel_weight_pretrain,
        repel_length_scale_pretrain = repel_length_scale_pretrain,
        nonempty_weight_pretrain = nonempty_weight_pretrain,
        nonempty_p_pretrain = nonempty_p_pretrain,
        nonempty_floor_percent_pretrain = nonempty_floor_percent_pretrain,

        kl_annealing_mode=kl_annealing_mode,
        kl_max_weight=kl_max_weight,
        kl_warmup=kl_warmup,
        kl_end_weight=kl_end_weight,
        kl_cooldown=kl_cooldown,

        kl_annealing_mode_pretrain=kl_annealing_mode_pretrain,
        kl_max_weight_pretrain=kl_max_weight_pretrain,
        kl_warmup_pretrain=kl_warmup_pretrain,
        kl_end_weight_pretrain=kl_end_weight_pretrain,
        kl_cooldown_pretrain=kl_cooldown_pretrain,
    )

    contrastive_cfg = ContrastiveCfg(
        temperature=temperature,
        contrastive_similarity_function=contrastive_similarity_function,
        contrastive_loss_function=contrastive_loss_function,
        beta=beta,
        tau=tau,
        aug_min_shift=aug_min_shift,
        aug_max_shift=aug_max_shift,
        aug_p_shift=aug_p_shift,
        aug_noise_sigma=aug_noise_sigma,
        aug_p_noise=aug_p_noise,
        aug_min_interp=aug_min_interp,
        aug_max_interp=aug_max_interp,
        aug_p_interp=aug_p_interp,
        aug_max_rot=aug_max_rot,
        aug_n_rot=aug_n_rot,
        aug_p_rot=aug_p_rot,
    )

    return embedding_model_fitting(
        preprocessed_object, 
        adjacency_matrix,
        meta_info,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg, 
        vade_cfg=vade_cfg,
        contrastive_cfg=contrastive_cfg,
        h5_dataset_folder=h5_dataset_folder,
    )

   

def embedding_model_fitting(
    preprocessed_object: Tuple[dict, dict],
    adjacency_matrix: np.ndarray,
    meta_info: dict,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    vade_cfg: VaDECfg,
    contrastive_cfg: ContrastiveCfg,
    h5_dataset_folder: str = None,
    shuffle: bool = True,
    device: str = None,
) -> Tuple[nn.Module, nn.Module, Optional[nn.Module]]:


    # ----------------------------------------------------
    # Prepare device and data
    # ----------------------------------------------------
    model_name = common_cfg.model_name # Name defaults to "vade"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str) and (device=="cpu" or device=="gpu"):
        device = torch.device(device)
    else: # pragma: no cover
        raise ValueError("If a device is given, it needs to be either cpu or gpu!")
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.manual_seed(common_cfg.seed)
    np.random.seed(common_cfg.seed)

    if h5_dataset_folder is None:
        data_path = os.path.join(common_cfg.output_path, "Datasets")
    else:
        data_path = h5_dataset_folder
    preprocessed_train, preprocessed_val = preprocessed_object
    train_dataset = deepof.clustering.dataset.BatchDictDataset(
        preprocessed_train, data_path, "train_", force_rebuild=False,
        h5_chunk_len=common_cfg.batch_size, supervised_dict=None
    )
    val_dataset = deepof.clustering.dataset.BatchDictDataset(
        preprocessed_val, data_path, "val_", force_rebuild=False,
        h5_chunk_len=common_cfg.batch_size, supervised_dict=None
    )

    train_loader = train_dataset.make_loader(
        batch_size=common_cfg.batch_size, shuffle=shuffle, num_workers=common_cfg.num_workers, drop_last=False,
        iterable_for_h5=True, pin_memory=(device.type == 'cuda'), prefetch_factor=common_cfg.prefetch_factor,
        persistent_workers=(common_cfg.num_workers > 0), block_shuffle=shuffle, permute_within_block=False, seed=common_cfg.seed,
    )
    val_loader = val_dataset.make_loader(
        batch_size=common_cfg.batch_size, shuffle=False, num_workers=common_cfg.num_workers, drop_last=False,
        iterable_for_h5=True, pin_memory=(device.type == 'cuda'), prefetch_factor=common_cfg.prefetch_factor,
        persistent_workers=(common_cfg.num_workers > 0), block_shuffle=False, permute_within_block=False,
    )

    writer = None
    if common_cfg.log_history:
        log_dir = os.path.join(common_cfg.output_path, "logs", f"{model_name}_run_{common_cfg.run}")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs -> {log_dir}")


    # ----------------------------------------------------
    # Branch 1: VQVAE
    # ----------------------------------------------------
    if model_name == "vqvae":
        return fit_VQVAE(
            train_loader,
            val_loader,
            preprocessed_train,
            adjacency_matrix,
            common_cfg,
            teacher_cfg,
            writer,
        )

    # ----------------------------------------------------
    # Branch 2: Contrastive
    # ----------------------------------------------------
    elif model_name == "contrastive":
        return fit_contrastive(
            train_loader,
            val_loader,
            preprocessed_train,
            adjacency_matrix,
            meta_info,
            common_cfg,
            teacher_cfg,
            contrastive_cfg,
            writer,
        )

    # ----------------------------------------------------
    # Branch 3: VaDE
    # ----------------------------------------------------
    elif model_name == "vade":
        return fit_VADE(
            train_loader,
            val_loader,
            preprocessed_train,
            adjacency_matrix,
            common_cfg,
            teacher_cfg,
            vade_cfg,
            writer,
        )
    else: # pragma: no cover
        raise ValueError(f"Unsupported model: {model_name}")


def fit_VQVAE(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    writer: SummaryWriter,
):
    
    # Some setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    model_name = "vqvae"
    rebuild_spec={                    
        "model_name": model_name,
        "x_shape": train_loader.dataset.x_shape,
        "a_shape": train_loader.dataset.a_shape,
        "adjacency_matrix": adjacency_matrix.astype("float32"),
        "latent_dim": common_cfg.latent_dim,
        "n_components": common_cfg.n_components,
        "encoder_type": common_cfg.encoder_type,
        "use_gnn": True,
        "interaction_regularization": common_cfg.interaction_regularization,
    }

    # Create model
    model = VQVAEPT(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        n_components=common_cfg.n_components,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        interaction_regularization=common_cfg.interaction_regularization,
        kmeans_loss=common_cfg.kmeans_loss,
    ).to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Create teacher
    teacher_cfg.include_latent_view=False
    teacher, tau_star, teacher_views = deepof.clustering.teacher_model.maybe_build_turtle_teacher(
        teacher_cfg=teacher_cfg,
        common_cfg=common_cfg,
        train_dataset=train_loader.dataset,
        preprocessed_train=preprocessed_train,
        data_path=data_path,
        device=device,
        latent_view=None,
    )
    if tau_star is not None:
        tau_star = tau_star.to(device)

    # Set distillation weights
    apply_distill = (tau_star is not None)
    lambda_scheduler = None
    if apply_distill:
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch,
            mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0,
            at_max_epochs=teacher_cfg.lambda_decay_start,
            max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown,
            end_weight=teacher_cfg.lambda_end_weight,
        )

    distill_head = DiscriminativeHead(common_cfg.latent_dim, common_cfg.n_components).to(device)
    optimizer = build_optimizer_generic(model, distill_head, base_lr=common_cfg.learning_rate, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))

    # Set up best-val and best-score saving
    _, best_path_val, best_path_score, _ = ckpt_paths("vqvae", common_cfg=common_cfg)
    best_val = float("inf")
    best_score = -float("inf")
    best_score_val = float("inf")
    score_value = float("nan")
    score_start_epoch = max(3, math.ceil(0.1 * common_cfg.epochs))
    score_tol = 0.01
    log_summary=init_log_summary()

    print(f"\n--- Training VQVAE for {common_cfg.epochs} epochs ---")
    for epoch in range(common_cfg.epochs):

        # Summarize some variables into namespace
        ctx = SimpleNamespace(
            tau_star=tau_star,
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
            distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
            distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
            apply_distill=apply_distill,
        )

        # Train and validate
        train_logs, _, lam = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_vqvae_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=False,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_vqvae_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs,
            ctx=SimpleNamespace(apply_distill=False), show_progress=True,
        )
        v_total = float(val_logs.get("total_loss", float("inf")))
        score_value = float("nan")

        if apply_distill:
            diag = deepof.clustering.logging.compute_diagnostics(
                model=model,
                dataloader=val_loader,
                q_fn=partial(get_q_vqvae, distill_head=distill_head),
                device=device,
                n_components=common_cfg.n_components,
                tau_star=tau_star,
                distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
                distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
                distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
                max_batches=common_cfg.diag_max_batches,
            )
            val_logs.update(diag)
            score_value = float(val_logs["alignment_score"])
        else:
            val_logs["alignment_score"] = float("nan")
            val_logs["conf_norm"] = float("nan")
            val_logs["bal_norm"] = float("nan")

        # Print training progress            
        log_summary = print_losses(model_name="vqvae", log_summary=log_summary, epoch=epoch, n_epochs=common_cfg.epochs, lambda_d=lam, train_logs=train_logs, val_logs=val_logs)
        log_epoch_to_tensorboard(writer, train_logs, val_logs, epoch, score_value, lam)

        # Save best model based on total validation loss
        if v_total < best_val:
            best_val = v_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=None,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                )
                print(f"  Saved best VAL model -> {best_path_val} (val: {best_val:.4f})")

        # Save best model based on model balance and certainty score
        improved_score = (
            apply_distill and math.isfinite(score_value) and
            (
                (score_value > best_score) or
                (abs(score_value - best_score) <= score_tol and v_total < best_score_val)
            )
        )

        if improved_score and epoch > score_start_epoch:
            best_score = score_value
            best_score_val = v_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                )
                print(f"  Saved best SCORE model -> {best_path_score} (score: {best_score:.6f})")
        

    # Load states of best val and score models
    model_val, model_score = load_best_checkpoints(
        model, best_path_val, best_path_score, device, common_cfg.save_weights
    )

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model_val), unwrap_dp(model_score), None, log_summary


def fit_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    meta_info: dict,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    contrastive_cfg: ContrastiveCfg,
    writer: SummaryWriter,
):
    
    # Some setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    model_name = "contrastive"
    rebuild_spec={                    
        "model_name": model_name,
        "x_shape": train_loader.dataset.x_shape,
        "a_shape": train_loader.dataset.a_shape,
        "adjacency_matrix": adjacency_matrix.astype("float32"),
        "latent_dim": common_cfg.latent_dim,
        "n_components": common_cfg.n_components,
        "encoder_type": common_cfg.encoder_type,
        "use_gnn": True,
        "interaction_regularization": common_cfg.interaction_regularization,
    }

    # Create model
    model = ContrastivePT(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        similarity_function=contrastive_cfg.contrastive_similarity_function,
        loss_function=contrastive_cfg.contrastive_loss_function,
        temperature=contrastive_cfg.temperature,
        beta=contrastive_cfg.beta,
        tau=contrastive_cfg.tau,
    ).to(device, non_blocking=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    n_nodes = train_loader.dataset.x_shape[1]
    edge_index_global, edge_index_local, _ = _build_edge_from_metainfo(
    meta_info=meta_info,
    device=device,
    n_nodes=n_nodes,
    return_local=True,
    )
    rot_precomp = build_rotation_precomp(edge_index=edge_index_local, n_nodes=n_nodes, device=device)

    # Create teacher
    teacher_cfg.include_latent_view=False
    teacher, tau_star, teacher_views = deepof.clustering.teacher_model.maybe_build_turtle_teacher(
        teacher_cfg=teacher_cfg,
        common_cfg=common_cfg,
        train_dataset=train_loader.dataset,
        preprocessed_train=preprocessed_train,
        data_path=data_path,
        device=device,
        latent_view=None,
    )
    if tau_star is not None:
        tau_star = tau_star.to(device)

    # Set distillation weights
    apply_distill = (tau_star is not None)
    lambda_scheduler = None
    if apply_distill:
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch,
            mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0,
            at_max_epochs=teacher_cfg.lambda_decay_start,
            max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown,
            end_weight=teacher_cfg.lambda_end_weight,
        )

    distill_head = DiscriminativeHead(common_cfg.latent_dim, common_cfg.n_components).to(device)
    optimizer = build_optimizer_generic(model, distill_head, base_lr=common_cfg.learning_rate, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))

    # Set up best-val and best-score saving
    _, best_path_val, best_path_score, _ = ckpt_paths("contrastive", common_cfg=common_cfg)
    best_val = float("inf")
    best_score = -float("inf")
    best_score_val = float("inf")
    score_value = float("nan")
    score_start_epoch = max(3, math.ceil(0.1 * common_cfg.epochs))
    score_tol = 0.01
    log_summary=init_log_summary()

    print(f"\n--- Training Contrastive for {common_cfg.epochs} epochs ---")
    for epoch in range(common_cfg.epochs):

        # Summarize some variables into namespace
        ctx = SimpleNamespace(
            tau_star=tau_star,
            distill_head=distill_head,
            lambda_scheduler=lambda_scheduler,
            distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
            distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
            distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
            apply_distill=apply_distill,
            edge_index=edge_index_global,
            edge_index_local=edge_index_local,
            contrastive_cfg=contrastive_cfg,
            rot_precomp=rot_precomp,
        )

        # Train and validate
        train_logs, _, lam = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_contrastive_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=False, 
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_contrastive_distill,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs,
            ctx=SimpleNamespace(apply_distill=False,edge_index=edge_index_global,edge_index_local=edge_index_local,contrastive_cfg=contrastive_cfg, rot_precomp=rot_precomp), show_progress=True,
        )
        v_total = float(val_logs.get("total_loss", float("inf")))
        score_value = float("nan")

        if apply_distill:
            diag = deepof.clustering.logging.compute_diagnostics(
                model=model,
                dataloader=val_loader,
                q_fn=partial(
                    get_q_contrastive,
                    distill_head=distill_head,
                    edge_index=edge_index_global,
                ),
                device=device,
                n_components=common_cfg.n_components,
                tau_star=tau_star,
                distill_sharpen_T=teacher_cfg.generic_distill_sharpen_T,
                distill_conf_weight=teacher_cfg.generic_distill_conf_weight,
                distill_conf_thresh=teacher_cfg.generic_distill_conf_thresh,
                max_batches=common_cfg.diag_max_batches,
            )
            val_logs.update(diag)
            score_value = float(val_logs["alignment_score"])
        else:
            val_logs["alignment_score"] = float("nan")
            val_logs["conf_norm"] = float("nan")
            val_logs["bal_norm"] = float("nan")

        # Print training progress
        log_summary = print_losses(model_name="Contrastive", log_summary=log_summary, epoch=epoch, n_epochs=common_cfg.epochs, lambda_d=lam, train_logs=train_logs, val_logs=val_logs)
        log_epoch_to_tensorboard(writer, train_logs, val_logs, epoch, score_value, lam)

   
        # Save best model based on total validation loss
        if v_total < best_val:
            best_val = v_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=None,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    contrastive_cfg=contrastive_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                )
                print(f"  Saved best VAL model -> {best_path_val} (val: {best_val:.4f})")

        # Save best model based on model balance and certainty score
        improved_score = (
            apply_distill and math.isfinite(score_value) and
            (
                (score_value > best_score) or
                (abs(score_value - best_score) <= score_tol and v_total < best_score_val)
            )
        )

        if improved_score and epoch > score_start_epoch:
            best_score = score_value
            best_score_val = v_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=v_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    contrastive_cfg=contrastive_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                )
                print(f"  Saved best SCORE model -> {best_path_score} (score: {best_score:.6f})")


    # Load states of best val and score models
    model_val, model_score = load_best_checkpoints(
        model, best_path_val, best_path_score, device, common_cfg.save_weights
    )

    if writer:
        writer.flush(); writer.close()

    return unwrap_dp(model_val), unwrap_dp(model_score), None, log_summary   


def fit_VADE(
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocessed_train: dict,
    adjacency_matrix: np.ndarray,
    common_cfg : CommonFitCfg,
    teacher_cfg: TurtleTeacherCfg,
    vade_cfg: VaDECfg,
    writer: SummaryWriter,
    tuning_mode: bool = False,
):
    

    ###############
    # Set up
    ###############

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(common_cfg.output_path, "Datasets")
    n_batches_per_epoch = len(train_loader)

    # Create model and step function
    model = VaDEPT(
        input_shape=train_loader.dataset.x_shape,
        edge_feature_shape=train_loader.dataset.a_shape,
        adjacency_matrix=adjacency_matrix,
        latent_dim=common_cfg.latent_dim,
        n_components=common_cfg.n_components,
        encoder_type=common_cfg.encoder_type,
        use_gnn=True,
        kmeans_loss=vade_cfg.kmeans_loss_pretrain,
        interaction_regularization=common_cfg.interaction_regularization,
    ).to(device, non_blocking=True)
    step_fn = step_vade

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # More setup
    optimizer = build_optimizer_vade(model=model, base_lr=vade_cfg.learning_rate_pretrain, gmm_lr=0.0) #gmm learnign rate is not used in pretraining
    scaler = GradScaler(enabled=(device.type == "cuda" and common_cfg.use_amp))
    n_batches_per_epoch = len(train_loader)

    model_name = "vade"
    rebuild_spec={                    
        "model_name": model_name,
        "x_shape": train_loader.dataset.x_shape,
        "a_shape": train_loader.dataset.a_shape,
        "adjacency_matrix": adjacency_matrix.astype("float32"),
        "latent_dim": common_cfg.latent_dim,
        "n_components": common_cfg.n_components,
        "encoder_type": common_cfg.encoder_type,
        "use_gnn": True,
        "kmeans_loss": common_cfg.kmeans_loss,
        "interaction_regularization": common_cfg.interaction_regularization,
        "lens_enabled": False,
    }

    # Load pretrained model if available + early return
    if common_cfg.pretrained:
        print(f"Loading pretrained weights from {common_cfg.pretrained}")
        model, log_summary, spec, load_report = deepof.clustering.model_utils_new.load_model_from_ckpt(common_cfg.pretrained)
        if writer:
            writer.flush(); writer.close()
        return unwrap_dp(model), None, None


    ###############
    # Define function, will start in pretrain mode
    ###############

    criterion = VadeLoss(
        common_cfg=common_cfg,
        vade_cfg=vade_cfg,
        teacher_cfg=teacher_cfg,
    ).to(device)


    ###############
    # Pretraining
    ###############

    print("\n--- Pretraining (reconstruction and setting up the latent space) ---")
    unwrap_dp(model).set_pretrain_mode(True)
    pre_epochs = vade_cfg.pretrain_epochs
    ctx = SimpleNamespace(criterion=criterion, scheduler=None, scheduler_per_batch=True)
    kl_scheduler = Dynamic_weight_manager(
        n_batches_per_epoch, mode=vade_cfg.kl_annealing_mode_pretrain,
        warmup_epochs=vade_cfg.kl_warmup_pretrain, max_weight=vade_cfg.kl_max_weight_pretrain, cooldown_epochs=vade_cfg.kl_cooldown_pretrain, end_weight=vade_cfg.kl_end_weight_pretrain
    )
    criterion.set_kl_scheduler(kl_scheduler)

    leave = False 
    for ep in range(pre_epochs):
        
        # Leave loading bar in last step (for optics)
        if ep == len(range(pre_epochs))-1:
            leave=True
        pre_logs, _, _ = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_fn,
            device=device, epoch=ep, num_epochs=pre_epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True, leave=leave
        )
        if writer:
            writer.add_scalar("Pretrain/total_loss", pre_logs["total_loss"], ep)

    ###############
    # Main training
    ###############

    # Finish pretraining, reset KL schedule
    unwrap_dp(model).set_pretrain_mode(False)
    criterion.set_mode("main")
    # New KL schedule
    kl_scheduler = Dynamic_weight_manager(
        n_batches_per_epoch, mode=vade_cfg.kl_annealing_mode,
        warmup_epochs=vade_cfg.kl_warmup, max_weight=vade_cfg.kl_max_weight,
        cooldown_epochs=vade_cfg.kl_cooldown, end_weight=vade_cfg.kl_end_weight
    )
    criterion.set_kl_scheduler(kl_scheduler)
    
    optimizer = build_optimizer_vade(model=model, base_lr=common_cfg.learning_rate, gmm_lr=vade_cfg.gmm_learning_rate)

    # VaDE unified checkpoint paths
    _, best_path_val, best_path_score, teacher_init_path = ckpt_paths("vade", common_cfg=common_cfg)

    tau_star = None
    teacher_init_model = None  # returned as 3rd output
    # cached views for refresh
    pca_pos = pca_spd = pca_edges = pca_angles_train = None

    log_summary=init_log_summary()
    if teacher_cfg.use_turtle_teacher:
        print("\n--- Extracting latents for teacher ---")
        z_all = extract_latents(
            unwrap_dp(model), train_loader.dataset, device=device,
            batch_size=2048, num_workers=common_cfg.num_workers
        ).cpu()

        # Set up lambda schedule (distillation weight)
        lambda_scheduler = Dynamic_weight_manager(
            n_batches_per_epoch, mode=common_cfg.kl_annealing_mode,
            warmup_epochs=0, at_max_epochs=teacher_cfg.lambda_decay_start, max_weight=teacher_cfg.lambda_distill,
            cooldown_epochs=teacher_cfg.lambda_cooldown, end_weight=teacher_cfg.lambda_end_weight
        )

        # Build teacher for VADE
        teacher_cfg.include_latent_view=True # Vade has a free and useful latent view due to pretraining
        teacher, tau_star, teacher_views = deepof.clustering.teacher_model.maybe_build_turtle_teacher(
            teacher_cfg=teacher_cfg,    
            common_cfg=common_cfg,        
            train_dataset=train_loader.dataset,
            preprocessed_train=preprocessed_train,
            data_path=data_path,
            device=device,
            latent_view=z_all.to(device),
        )

        # Get views for teacher
        pca_pos = teacher_views.get("pca_pos", None)
        pca_spd = teacher_views.get("pca_spd", None)
        pca_edges = teacher_views.get("pca_edges", None)
        pca_angles_train = teacher_views.get("pca_angles", None)

        print("\n--- Initializing GMM from teacher τ* ---")
        initialize_gmm_from_teacher(model, z_all, tau_star, min_var=0.01)
        criterion.set_teacher(tau_star=tau_star.to(device), lambda_distill=teacher_cfg.lambda_distill, lambda_scheduler=lambda_scheduler)

        # Save teacher-init checkpoint + info (VaDE only)
        teacher_init_model = deepcopy(unwrap_dp(model))
        if common_cfg.save_weights:
            save_model_info(
                teacher_init_path,
                stage="teacher_init",
                epoch=pre_epochs - 1,
                train_steps=pre_epochs * len(train_loader),
                extra={"note": "after pretrain + teacher + GMM init, before main training"},
                common_cfg=common_cfg,
                teacher_cfg=teacher_cfg,
                vade_cfg=vade_cfg,
                model=unwrap_dp(model),
                log_summary=log_summary,
                rebuild_spec=rebuild_spec,
                save_weights=common_cfg.save_weights,
            )
            print(f"  Saved teacher-init model -> {teacher_init_path}")

    else:
        # If there is no teacher, init GMM directly with train_loader
        print("\n--- Initializing GMM from embeddings (sklearn) ---")
        unwrap_dp(model).initialize_gmm_from_data(train_loader)

    # Inits for training
    best_val = -float("inf") #start negative, as Vade first is expected to get worse validation wise, then top out and get better
    best_score = -float("inf")
    max_score = -float("inf")
    best_score_val = float("inf")
    score_value = float("nan")
    score_tol = 0.01
    val_tol = 0.01
    score_start_epoch = max(3, math.ceil(0.1 * common_cfg.epochs))
    val_top_reached=False

    # Epoch dependent weights
    lambda_d = 0.0
    klw = 0.0

    print(f"\n--- Training for {common_cfg.epochs} epochs ---")
    print("NOTE: some losses are intentionally defined negative and function as rewards.")
    for epoch in range(common_cfg.epochs):
               
        base = unwrap_dp(model)

        # Freezing/unfreezing of model parts based on user inputs
        if epoch == 0 and vade_cfg.freeze_gmm_epochs > 0:
            print(f"Freezing GMM for {vade_cfg.freeze_gmm_epochs} epoch(s)")
            base.latent_space.gmm_means.requires_grad = False
            base.latent_space.gmm_log_vars.requires_grad = False
        if epoch == vade_cfg.freeze_gmm_epochs:
            new_base_lr = 5e-4
            new_gmm_lr = 2e-4
            for i, group in enumerate(optimizer.param_groups):
                if i == 0: group["lr"] = new_base_lr
                elif i == 1: group["lr"] = new_gmm_lr
            print("Unfreezing GMM")
            base.latent_space.gmm_means.requires_grad = True
            base.latent_space.gmm_log_vars.requires_grad = True
            #base.latent_space.freeze_lens(False)
        if epoch == 0 and vade_cfg.freeze_decoder_epochs > 0:
            print(f"Freezing decoder for {vade_cfg.freeze_decoder_epochs} epoch(s)")
            for p in base.decoder.parameters():
                p.requires_grad = False
        if epoch == vade_cfg.freeze_decoder_epochs:
            print("Unfreezing decoder")
            for p in base.decoder.parameters():
                p.requires_grad = True

        # Refresh Turtle teacher (i.e. retrain on current model and potentially also reinit GMM)
        if (
            epoch > 0 and teacher_cfg.use_turtle_teacher and (teacher_cfg.teacher_refresh_every is not None) and (teacher_cfg.teacher_refresh_every > 0)
            and (epoch) % teacher_cfg.teacher_refresh_every == 0 and (teacher_cfg.teacher_freeze_at is None or (epoch) <= teacher_cfg.teacher_freeze_at)
        ):
            print(f"\n--- Refresh TURTLE teacher at epoch {epoch+1} ---")
            z_curr = extract_latents(base, train_loader.dataset, device=device, batch_size=2048, num_workers=common_cfg.num_workers).cpu()

            # Get current views
            views_dict = {"z": z_curr.to(device)}
            if teacher_cfg.include_nodes_view and (pca_pos is not None) and (pca_spd is not None):
                views_dict["pca_pos"] = pca_pos.to(device)
                views_dict["pca_spd"] = pca_spd.to(device)
            if teacher_cfg.include_edges_view and (pca_edges is not None):
                views_dict["pca_edges"] = pca_edges.to(device)
            if teacher_cfg.include_angles_view and (pca_angles_train is not None):
                views_dict["pca_angles"] = pca_angles_train.to(device)

            # Rerun teacher
            teacher, tau_star = run_turtle_teacher_on_views(
                views_dict=views_dict, n_components=common_cfg.n_components, gamma=teacher_cfg.teacher_gamma,
                alpha_sample_entropy=teacher_cfg.teacher_alpha_sample_entropy,
                outer_steps=max(200, teacher_cfg.teacher_outer_steps), inner_steps=teacher_cfg.teacher_inner_steps,
                normalize_feats=teacher_cfg.teacher_normalize_feats, verbose=True, device=device,
                head_temp=teacher_cfg.teacher_head_temp, task_temp=teacher_cfg.teacher_task_temp,
                batch_size = teacher_cfg.teacher_batch_size,
            )
            tau_star = tau_star.detach()
            criterion.set_teacher(tau_star=tau_star.to(device), lambda_distill=teacher_cfg.lambda_distill, lambda_scheduler=lambda_scheduler)

            # Optionally reinit GMM
            if teacher_cfg.reinit_gmm_on_refresh:
                initialize_gmm_from_teacher(model, z_curr, tau_star, min_var=1e-4)
                print("  Reinitialized GMM from refreshed τ*.")

        # Actual training of the main model
        ctx = SimpleNamespace(criterion=criterion, scheduler_per_batch=True)
        # klw is updated in every training step, getting the weight for printing at the middle of the epochs
        train_logs, klw, lambda_d = train_one_epoch_indexed(
            model=model, model_name=model_name, dataloader=train_loader, optimizer=optimizer, step_fn=step_fn,
            device=device, epoch=epoch, num_epochs=common_cfg.epochs, scaler=scaler, use_amp=common_cfg.use_amp,
            grad_clip_value=0.75, ctx=ctx, show_progress=True,
        )
        val_logs = validate_one_epoch_indexed(
            model=model, dataloader=val_loader, step_fn=step_fn, device=device,
            epoch=epoch, num_epochs=common_cfg.epochs, ctx=SimpleNamespace(criterion=criterion, apply_distill=False),
            show_progress=True,
        )

        # A ton of diagnostics for printing training progress
        diag = deepof.clustering.logging.compute_diagnostics(
            model=model,
            dataloader=val_loader,
            q_fn=get_q_vade,
            device=device,
            n_components=common_cfg.n_components,
            tau_star=getattr(criterion, "tau_star", None),
            distill_sharpen_T=float(getattr(criterion, "distill_sharpen_T", 0.5)),
            distill_conf_weight=bool(getattr(criterion, "distill_conf_weight", False)),
            distill_conf_thresh=float(getattr(criterion, "distill_conf_thresh", 0.55)),
            max_batches=common_cfg.diag_max_batches,
            extra_stats_fn=compute_vade_specific_diagnostics,
        )
        
        val_logs.update(diag)
        val_total = float(val_logs.get("total_loss", float("inf")))
        score_value = float(val_logs["alignment_score"])

        log_summary = print_losses(model_name="vade", log_summary=log_summary, epoch=epoch, n_epochs=common_cfg.epochs, klw=klw, lambda_d=lambda_d, train_logs=train_logs, val_logs=val_logs)
        log_epoch_to_tensorboard(writer, train_logs, val_logs, epoch, score_value)

        # Deterimine if validation loss and / or balance + certainty score has improved
        improved_val = ((val_total + val_tol) < best_val)

        if not improved_val and not val_top_reached:
            best_val = val_total

        improved_score = (
            math.isfinite(score_value) and (
                (score_value > best_score) or
                (abs(score_value - best_score) <= score_tol and val_total < best_score_val)
            )
        )
        # for tuning
        max_score= score_value if score_value>max_score else max_score

        # Save best model based on total validation loss
        if improved_val:
            val_top_reached=True
            best_val = val_total
            val_tol=0.0
            if common_cfg.save_weights:
                save_model_info(
                    best_path_val,
                    stage="best_val",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=val_total,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    vade_cfg=vade_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                )
                print(f"Saved best VAL model -> {best_path_val} (val={best_val:.4f})")


        # Save best model based on model balance and certainty score
        if improved_score and epoch > score_start_epoch:
            best_score = score_value
            best_score_val = val_total
            if common_cfg.save_weights:
                save_model_info(
                    best_path_score,
                    stage="best_score",
                    epoch=epoch,
                    train_steps=(epoch + 1) * len(train_loader),
                    val_total=val_total,
                    score_value=score_value,
                    common_cfg=common_cfg,
                    teacher_cfg=teacher_cfg,
                    vade_cfg=vade_cfg,
                    model=unwrap_dp(model),
                    log_summary=log_summary,
                    rebuild_spec=rebuild_spec,
                    save_weights=common_cfg.save_weights,
                )
                print(f"Saved best SCORE model -> {best_path_score} (score={best_score:.4f})")


    # Load states of best val and score models
    model_val, model_score = load_best_checkpoints(
        model, best_path_val, best_path_score, device, common_cfg.save_weights
    )

    if writer:
        writer.flush(); writer.close()

    if tuning_mode:
        return unwrap_dp(model_val), unwrap_dp(model_score), teacher_init_model, log_summary, max_score 


    return unwrap_dp(model_val), unwrap_dp(model_score), teacher_init_model, log_summary    


#
####
#
############
#
################################
#CONTRASTIVE MODEL AUGMENTATIONS
################################
#
############
#
####
#


def _build_edge_from_metainfo(
    meta_info: dict,
    device: torch.device,
    n_nodes: int,
    return_local: bool = True,
    return_node_names: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[str]]]:
    """
    Builds:
      - edge_index_global: (E,2) long on device, includes cross-mouse edges (e.g. B_Nose--W_Nose)
      - edge_index_local:  (E_local,2) long on device, excludes cross-mouse edges (within-mouse only)
      - node_names: list[str] length n_nodes in node-axis order (optional)

    Assumes node names are like 'B_Nose', 'W_Tail_base', etc.
    """
    if "node_columns" not in meta_info or "edge_columns" not in meta_info: # pragma: no cover
        raise RuntimeError("meta_info must contain 'node_columns' and 'edge_columns'.")

    node_cols = list(meta_info["node_columns"])
    edge_cols = list(meta_info["edge_columns"])

    # 1) Infer node order from the first n_nodes entries of form (bp, 'x')
    node_names: List[str] = []
    for c in node_cols:
        if isinstance(c, tuple) and len(c) == 2 and c[1] == "x":
            node_names.append(c[0])
            if len(node_names) == n_nodes:
                break

    if len(node_names) != n_nodes: # pragma: no cover
        raise RuntimeError(
            f"Failed to infer {n_nodes} node names from meta_info['node_columns']. Got {len(node_names)}."
        )

    node_to_idx = {name: i for i, name in enumerate(node_names)}

    # 2) Build global edge index 
    pairs = []
    for (u_name, v_name) in edge_cols:
        if u_name not in node_to_idx or v_name not in node_to_idx: # pragma: no cover
            raise RuntimeError(
                f"Edge ({u_name},{v_name}) contains node(s) not found in inferred node list."
            )
        pairs.append((node_to_idx[u_name], node_to_idx[v_name]))

    edge_index_global = torch.tensor(pairs, dtype=torch.long, device=device)

    if not return_local:
        return edge_index_global, None, (node_names if return_node_names else None)

    # 3) Build local/within-mouse edge index by filtering on name prefix ('B_' vs 'W_')
    def animal_key(name: str) -> str:
        # 'B_Right_bhip' -> 'B', 'W_Nose' -> 'W'
        # (split on first underscore)
        if "_" not in name:
            return("")
        return name.split("_", 1)[0]

    keys = [animal_key(nm) for nm in node_names]  # len N
    # map each node index -> animal id (e.g., B->0, W->1)
    uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
    animal_id = torch.tensor([uniq[k] for k in keys], dtype=torch.long, device=device)  # (N,)

    u = edge_index_global[:, 0]
    v = edge_index_global[:, 1]
    same_animal = (animal_id[u] == animal_id[v])
    edge_index_local = edge_index_global[same_animal]

    return edge_index_global, edge_index_local, (node_names if return_node_names else None)


def _plot_augmentation(x_in: torch.Tensor, x_aug: torch.Tensor): # pragma: no cover
    """
    Plots one random batch element as a row of skeletons over time (top: original, bottom: augmented).
    Uses _plot_augmentation._edge_index if available (set by augmentation functions).
    """
    import matplotlib.pyplot as plt

    edge_index = getattr(_plot_augmentation, "_edge_index", None)

    b = int(torch.randint(0, x_in.size(0), (1,)).item())
    xin = x_in[b, ::4, :, 0:2].detach().cpu()   # (T,N,2)
    xau = x_aug[b, ::4, :, 0:2].detach().cpu()  # (T,N,2)
    T = xin.size(0)

    dx = 2.5  # horizontal offset per frame
    fig, ax = plt.subplots(2, 1, figsize=(min(28, 1.2 * T), 6), sharey=True)

    def draw_row(ax_, X, title):
        for t in range(T):
            off = t * dx
            pts = X[t]  # (N,2)
            xs = pts[:, 0].numpy() + off
            ys = pts[:, 1].numpy()
            ax_.plot(xs, ys, "kx", ms=3)

            if edge_index is not None:
                ei = edge_index.detach().cpu().numpy()
                for (i, j) in ei:
                    ax_.plot([xs[i], xs[j]], [ys[i], ys[j]], "k-", lw=0.8, alpha=0.7)

        ax_.set_title(title)
        ax_.axis("off")

    draw_row(ax[0], xin, "original")
    draw_row(ax[1], xau, "augmented")
    range_min=np.min([plt.xlim(),plt.ylim()])
    range_max=np.max([plt.xlim(),plt.ylim()])
    plt.xlim([range_min,range_max])
    plt.ylim([range_min,range_max])
    plt.tight_layout()
    plt.show()


@dataclass
class RotationPrecomp:
    # Triplets (a,b,c) in a tensor for cheap access
    triplets: torch.Tensor            # (M,3) long, on device
    centers: torch.Tensor             # (M,) long, on device

    # Variable-length node lists, stored as list-of-tensors (each on device)
    branches_a: List[torch.Tensor]    # len M, each (Ka,) long
    branches_c: List[torch.Tensor]    # len M, each (Kc,) long

    # 0 => prefer a-branch, 1 => prefer c-branch, 2 => tie (random at runtime)
    prefer_side: torch.Tensor         # (M,) uint8 on device


def build_rotation_precomp(edge_index: torch.Tensor, n_nodes: int, device: torch.device) -> RotationPrecomp:
    """
    Build triplets and per-triplet branch node sets ONCE.

    This runs Python/CPU graph logic once, but stores results as CUDA tensors (if device is CUDA)
    so the augmentation step doesn't do BFS anymore.
    """
    ei_cpu = edge_index.detach().cpu().tolist()

    # adjacency list on CPU
    adj = [[] for _ in range(n_nodes)]
    for u, v in ei_cpu:
        adj[u].append(v)
        adj[v].append(u)

    # triplets (a,b,c): for each center b, all unordered neighbor pairs (a,c)
    triplets_py: List[Tuple[int,int,int]] = []
    for b in range(n_nodes):
        nb = adj[b]
        if len(nb) < 2:
            continue
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                triplets_py.append((nb[i], b, nb[j]))

    # branch BFS (still CPU, but one-time)
    def branch_nodes(center: int, side: int) -> List[int]:
        seen = {side}
        stack = [side]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v == center or v in seen:
                    continue
                seen.add(v)
                stack.append(v)
        return list(seen)  # excludes center by construction

    branches_a: List[torch.Tensor] = []
    branches_c: List[torch.Tensor] = []
    prefer: List[int] = []

    for (a, b, c) in triplets_py:
        ba = branch_nodes(b, a)
        bc = branch_nodes(b, c)

        branches_a.append(torch.tensor(ba, dtype=torch.long, device=device))
        branches_c.append(torch.tensor(bc, dtype=torch.long, device=device))

        prefer.append(2)

    triplets = torch.tensor(triplets_py, dtype=torch.long, device=device)
    centers = triplets[:, 1].contiguous() if triplets.numel() else torch.empty((0,), dtype=torch.long, device=device)
    prefer_side = torch.tensor(prefer, dtype=torch.uint8, device=device)

    return RotationPrecomp(
        triplets=triplets,
        centers=centers,
        branches_a=branches_a,
        branches_c=branches_c,
        prefer_side=prefer_side,
    )


def _augment_time_shift(
    x: torch.Tensor,             # (B,T_full,N,3)
    edge_index: torch.Tensor,    # (E,2) (not used here, kept for signature consistency / plotting)
    min_shift: int = 1,
    max_shift: int = 3,
    p: float = 0.8,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a half-window slice (T_full//2) from the middle of the full window.
    If triggered, shifts the slice start by +/- U[min_shift, max_shift] (per sample).
    Shift is consistent across the whole window (no frame-to-frame jitter).
    """
    B, T = x.shape[0], x.shape[1]
    half_len = T // 2
    base = (T - half_len) // 2  # == T//4 when T even

    # vvvvv sample shifts per sample vvvvv
    apply = (torch.rand(B, device=x.device) < p)

    mag = torch.randint(min_shift, max_shift + 1, (B,), device=x.device)     # (B,)
    sgn = (torch.randint(0, 2, (B,), device=x.device) * 2 - 1)               # (B,) in {-1,+1}
    shift = mag * sgn
    shift = shift * apply.long()  # (B,) zero if not applied

    start = base + shift
    start = start.clamp(0, T - half_len)  # keep valid
    # <^^^^ sample shifts <^^^^

    x_cut = slice_time_per_sample(x, start, half_len)

    if plot: # pragma: no cover
        # show what changed (note: this plots only the cut windows)
        _plot_augmentation._edge_index = edge_index
        _plot_augmentation(slice_time_per_sample(x, (torch.ones([B],device=x.device)*(T - half_len) // 2).int(), half_len), x_cut)

    return x_cut


def _augment_angle_rotations(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # (E,2)
    rot_precomp: RotationPrecomp,
    n_rot: int = 3,
    max_rot: float = 30.0,
    p: float = 0.5,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly apply up to n_rot joint-like rotations (consistent across time per sample).
    Branch selection is done by a graph-cut rule:
      - For a center b and neighbor side s, rotate the connected component reachable from s
        when traversal through b is forbidden.
    This yields more realistic articulated rotations (e.g., center rotating head/tail as a unit).

    Notes on speed:
      - n_rot is small (<=3), so small Python loops are fine.
      - Rotations are vectorized across (B,T,...) with Torch ops.
    """
    B, T, N, _ = x.shape
    M = int(rot_precomp.triplets.shape[0])
    if n_rot <= 0 or max_rot <= 0.0 or p <= 0.0 or M == 0:
        return x

    x_aug = x.clone()
    coords = x_aug[..., 0:2]  # (B,T,N,2)

    apply = (torch.rand(B, device=x.device) < p).to(x.dtype)  # (B,)
    max_rad = float(max_rot) * math.pi / 180.0

    # choose up to n_rot triplets with unique centers (still a tiny Python loop)
    perm = torch.randperm(M, device=x.device)
    chosen_idx = []
    center_count = torch.zeros(N, dtype=torch.int, device=x.device)
    
    # NOTE: this .tolist() is small (M is small for N=11), but you could avoid it later if needed
    for k in perm.tolist():
        b0 = int(rot_precomp.centers[k].item())
        if center_count[b0] >= 2:
            continue
        center_count[b0] += 1
        chosen_idx.append(k)
        if len(chosen_idx) >= n_rot:
            break

    for k in chosen_idx:
        a0, b0, c0 = rot_precomp.triplets[k].tolist()

        pref = int(rot_precomp.prefer_side[k].item())
        if pref == 0:
            rot_nodes = rot_precomp.branches_a[k]
        elif pref == 1:
            rot_nodes = rot_precomp.branches_c[k]
        else:
            # tie
            rot_nodes = rot_precomp.branches_a[k] if torch.rand((), device=x.device) < 0.5 else rot_precomp.branches_c[k]

        if rot_nodes.numel() == 0:
            continue

        theta = (torch.rand(B, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * max_rad
        theta = theta * apply  # (B,)

        cos_t = torch.cos(theta).view(B, 1, 1)
        sin_t = torch.sin(theta).view(B, 1, 1)

        pivot = coords[:, :, b0, :].unsqueeze(2)                  # (B,T,1,2)
        pts = coords.index_select(dim=2, index=rot_nodes)         # (B,T,K,2)
        rel = pts - pivot                                         # (B,T,K,2)

        rx = rel[..., 0] * cos_t - rel[..., 1] * sin_t
        ry = rel[..., 0] * sin_t + rel[..., 1] * cos_t
        new_pts = torch.stack([rx, ry], dim=-1) + pivot           # (B,T,K,2)

        coords[:, :, rot_nodes, :] = new_pts

    x_aug[..., 0:2] = coords

    if plot: # pragma: no cover
        _plot_augmentation._edge_index = edge_index
        _plot_augmentation(x, x_aug)

    return x_aug


def _augment_noise_xys(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # (E,2)
    sigma: float = 0.03,
    p: float = 0.5,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add Gaussian noise per bodypart (per sample), consistent across the window:
      - For each (sample, node), choose axis in {x,y} and add a random offset to that axis.
      - Also add random offset to speed channel.
    Recomputes a from x after augmentation.
    """
    if sigma <= 0.0 or p <= 0.0:
        return x

    B, T, N, F = x.shape
    x_aug = x.clone()

    # vvvvv VECTORIZED per-node offsets (no python loop) vvvvv
    apply = (torch.rand(B, device=x.device) < p).to(x.dtype)          # (B,)
    apply_bn = apply.view(B, 1).expand(B, N)                           # (B,N)

    # For each (B,N) choose whether to perturb x or y
    axis = torch.randint(0, 2, (B, N), device=x.device)                # (B,N), 0=x 1=y

    # One offset per (B,N), constant over time
    offset_xy = sigma * torch.randn((B, N), device=x.device, dtype=x.dtype) * apply_bn  # (B,N)
    dx = offset_xy * (axis == 0).to(x.dtype)                            # (B,N)
    dy = offset_xy * (axis == 1).to(x.dtype)                            # (B,N)

    # Speed offsets per (B,N)
    ds = sigma * torch.randn((B, N), device=x.device, dtype=x.dtype) * apply_bn         # (B,N)

    # Broadcast across time
    x_aug[:, :, :, 0] = x_aug[:, :, :, 0] + dx.view(B, 1, N)            # <----
    x_aug[:, :, :, 1] = x_aug[:, :, :, 1] + dy.view(B, 1, N)            # <----
    x_aug[:, :, :, 2] = x_aug[:, :, :, 2] + ds.view(B, 1, N)            # <----

    if plot: # pragma: no cover
        _plot_augmentation._edge_index = edge_index  # <----
        _plot_augmentation(x, x_aug)                 # <----

    return x_aug


def _augment_linear_interpolate_segments(
    x: torch.Tensor,             # (B,T,N,3)
    edge_index: torch.Tensor,    # (E,2)
    min_len: int = 5,
    max_len: int = 15,
    p: float = 0.3,
    plot: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace one random contiguous segment (length <= max_len) with linear interpolation, per sample with prob p.
    Applies to all node channels (x,y,speed). Recomputes a from x after augmentation.
    """
    if max_len <= 0 or p <= 0.0:
        return x

    B, T = x.size(0), x.size(1)
    if T < 3:
        return x

    x_aug = x.clone()

    # vvvvv VECTORIZED interpolation (no python loops over bs / frames) vvvvv
    device = x.device
    dtype = x_aug.dtype

    apply = (torch.rand(B, device=device) < p)  # (B,)

    # Sample L per sample (even for non-applied; we mask later)
    L = torch.randint(min_len, max_len + 1, (B,), device=device)  # (B,)

    # Need endpoints at (t0-1) and (t0+L) within [0, T-1]
    # -> t0 in [1, T-L-1]  (inclusive) => randint(1, T-L) (exclusive high)
    t0_max = (T - L).clamp_min(2)  # ensure high>=2 so randint(1, high) is valid
    # torch.randint doesn't support per-element highs directly; sample uniform and mod safely:
    # We'll sample from a generous range then clamp.
    t0 = torch.randint(1, T - 1, (B,), device=device)  # provisional
    t0 = torch.minimum(t0, (T - L - 1).clamp_min(1))   # enforce t0 <= T-L-1

    t_start = t0 - 1                 # (B,)
    t_end = t0 + L                   # (B,)

    # Gather endpoints: (B,N,3)
    b_idx = torch.arange(B, device=device)
    start = x_aug[b_idx, t_start]    # (B,N,3)
    end   = x_aug[b_idx, t_end]      # (B,N,3)

    # Build mask over time: frames to replace are t0..t0+L-1
    tt = torch.arange(T, device=device).view(1, T)                    # (1,T)
    t0v = t0.view(B, 1)                                                # (B,1)
    Lv  = L.view(B, 1)                                                 # (B,1)
    mask = (tt >= t0v) & (tt < (t0v + Lv)) & apply.view(B, 1)          # (B,T)

    # Alpha for each t in the segment: alpha = (t - (t0-1)) / (L+1)
    denom = (Lv + 1).to(dtype)                                         # (B,1)
    alpha = ((tt.to(dtype) - (t0v.to(dtype) - 1.0)) / denom)           # (B,T)
    alpha = alpha.clamp(0.0, 1.0)

    # Interpolated frames: (B,T,N,3)
    start_e = start.unsqueeze(1)                                       # (B,1,N,3)
    end_e   = end.unsqueeze(1)                                         # (B,1,N,3)
    alpha_e = alpha.unsqueeze(-1).unsqueeze(-1)                        # (B,T,1,1)
    interp  = (1.0 - alpha_e) * start_e + alpha_e * end_e              # (B,T,N,3)

    # Apply only on masked frames
    x_aug = torch.where(mask.unsqueeze(-1).unsqueeze(-1), interp, x_aug)
    # <^^^^ VECTORIZED interpolation <^^^^

    if plot: # pragma: no cover
        _plot_augmentation._edge_index = edge_index  # <----
        _plot_augmentation(x, x_aug)                 # <----

    return x_aug


def _make_augmented_view(
    x: torch.Tensor,   # (B,T,N,3)
    a: torch.Tensor,   # (B,T,E,1)
    edge_index: torch.Tensor,
    rot_precomp: RotationPrecomp,
    min_shift: int = 1,
    max_shift: int = 6,
    p_shift: float = 0.8,
    n_rot: int =3,
    max_rot: int = 30, 
    p_rot: float = 0.7,
    max_interp: int = 6,
    min_interp: int = 5,
    p_interp: float = 0.6,
    noise_sigma: float = 0.02,
    p_noise: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Produce augmented (x_aug, a_aug). a_aug is recomputed from x_aug, then affine-matched to a.
    """
    x_aug_raw = x

    x_aug = _augment_time_shift(x_aug_raw, edge_index, min_shift=min_shift, max_shift=max_shift, p=p_shift, plot=False)
    #x_aug = _augment_full_rotation(x_aug, edge_index, max_rot=180, p=0.5, plot=False)
    x_aug = _augment_angle_rotations(x_aug, edge_index, rot_precomp, n_rot=n_rot, max_rot=max_rot, p=p_rot, plot=False)
    x_aug = _augment_linear_interpolate_segments(x_aug, edge_index, min_len=min_interp, max_len=max_interp, p=p_interp, plot=False)
    x_aug = _augment_noise_xys(x_aug, edge_index, sigma=noise_sigma, p=p_noise, plot=False)

    a_aug = recompute_edges(x_aug, edge_index) 

    return x_aug, a_aug