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


from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Distribution, TransformedDistribution
from torch.distributions.transforms import AffineTransform

from keras_tuner import BayesianOptimization, Hyperband, Objective

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

    kl_annealing_mode: str = "sigmoid"
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
    nonempty_p: float = 2.0
    nonempty_floor_percent: float = 0.05


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
    common_cfg=None,
    teacher_cfg=None,
    vade_cfg=None,
    contrastive_cfg=None,
) -> None:
    """ Saves all config and training information for a freshly trained model """
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

        soft_counts = deepof.post_hoc.get_contrastive_soft_counts(
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


    