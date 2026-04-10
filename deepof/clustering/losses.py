"""
Model losses
"""
# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

from typing import Any, NewType, Tuple, Dict, Optional, Callable

import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


import deepof.clustering.model_utils_new
from deepof.clustering.model_utils_new import CommonFitCfg, TurtleTeacherCfg, VaDECfg, ContrastiveCfg 



# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


#########################
### CONTRASTIVE LOSSES
#########################


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
    else: # pragma: no cover
        raise ValueError(f"Unknown loss_fn: {loss_fn}")
    

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: #pragma no cover
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


#########################
### VADE LOSSES
#########################


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


class Dynamic_weight_manager:
    """handles KL and lambda weights over epochs"""
    def __init__(
        self,
        n_batches_per_epoch: int,
        mode: str = "sigmoid",
        warmup_epochs: int = 15,
        max_weight: float = 1.0,
        at_max_epochs: int = 0,
        cooldown_epochs: int = 15,
        end_weight: float = 1.0,
    ):
        self.mode = mode
        self.warmup_iters = max(1, warmup_epochs * n_batches_per_epoch)
        self.at_max_iters = max(0, at_max_epochs * n_batches_per_epoch)
        self.cooldown_iters = max(0, cooldown_epochs * n_batches_per_epoch)
        self.total_iters = self.warmup_iters + self.at_max_iters + self.cooldown_iters
        self.current_iteration = 0
        self.max_weight = float(max_weight)
        self.end_weight = float(end_weight)

    def _shape(self, p: float) -> float:
        p = float(max(0.0, min(1.0, p)))
        if self.mode == "linear":
            return p
        elif self.mode == "sigmoid":
            return 1.0 / (1.0 + math.exp(-12.0 * (p - 0.5)))
        elif self.mode == "tf_sigmoid":
            eps = 1e-2
            denom = max(eps, p - p * p)
            arg = (2.0 * p - 1.0) / denom
            return 1.0 / (1.0 + math.exp(-arg))
        else:
            return p

    def get_weight(self) -> float:
        t = self.current_iteration

        # If at or beyond max schedule, return end weight directly
        if t >= self.total_iters:
            return self.end_weight
        
        # If at max schedule, return max
        if self.at_max_iters > 0 and t >= self.warmup_iters and t < self.warmup_iters + self.at_max_iters:
            return self.max_weight

        # Warmup: 0 -> max_weight
        if t <= self.warmup_iters:
            p = t / self.warmup_iters  # [0,1]
            return self.max_weight * self._shape(p)

        # Cooldown: max_weight -> end_weight
        if self.cooldown_iters <= 0:
            return self.max_weight

        tc = t - (self.warmup_iters + self.at_max_iters)
        pc = tc / self.cooldown_iters  # [0,1]
        # Linear blend 
        return (1.0 - pc) * self.max_weight + pc * self.end_weight

    def step(self):
        self.current_iteration += 1


def cluster_frequencies_regularizer(soft_counts: torch.Tensor) -> torch.Tensor:
    mean_freq = torch.mean(soft_counts, dim=0)
    n_components = soft_counts.shape[1]
    uniform_target = torch.ones(n_components, device=soft_counts.device) / n_components
    kld_loss = nn.KLDivLoss(reduction="batchmean")
    return kld_loss(torch.log(mean_freq + 1e-9), uniform_target)


class VadeLoss(nn.Module):
    """
    VaDE loss function combining reconstruction, KL divergence, clustering, and optional teacher distillation terms.

    Supports two training phases (pretrain and main) with separate parameter sets for phase-dependent
    regularization terms. Call set_pretrain_mode() to switch between them.

    Phase-dependent parameters (stored separately for pretrain and main):
    - repel_weight, repel_length_scale
    - nonempty_weight, nonempty_floor, nonempty_p

    Phase-independent parameters are set once at construction and do not change between phases.

    Args:
        common_cfg (CommonFitCfg): Common training configuration providing n_components, latent_dim, and kmeans_loss.
        vade_cfg (VaDECfg): VaDE-specific configuration including both pretrain and main parameter values.
        teacher_cfg (TurtleTeacherCfg): Teacher distillation configuration.
        kl_scheduler (Optional[Dynamic_weight_manager]): KL weight scheduler. Can be replaced later via set_kl_scheduler().
        l1_activity_weight (float): L1 regularization weight on z_log_var. Defaults to 0.1.
        gmm_logvar_clamp (Tuple[float, float]): Min and max clamp values for GMM log-variances. Defaults to (-8.0, 8.0).
    """
    def __init__(
            self,
            common_cfg: "CommonFitCfg",
            vade_cfg: "VaDECfg",
            teacher_cfg: "TurtleTeacherCfg",
            kl_scheduler: Optional["Dynamic_weight_manager"] = None,
            l1_activity_weight: float = 0.1,
            gmm_logvar_clamp: Tuple[float, float] = (-8.0, 8.0),
        ):
        super().__init__()
        # Core dimensions
        self.n_components = common_cfg.n_components
        self.latent_dim = common_cfg.latent_dim

        # Phase-independent loss parameters
        self.l1_activity_weight = float(l1_activity_weight)
        self.tf_cluster_weight = float(vade_cfg.tf_cluster_weight)
        self.reg_cat_clusters_weight = float(vade_cfg.reg_cat_clusters)
        self.temporal_cohesion_weight = float(vade_cfg.temporal_cohesion_weight)
        self.reg_scatter_weight = float(vade_cfg.reg_scatter_weight)
        self.reg_scatter_beta = float(vade_cfg.reg_scatter_beta)
        self.gmm_logvar_clamp = gmm_logvar_clamp

        # Distillation parameters (teacher and lambda set later via set_teacher())
        self.lambda_distill = 0.0
        self.lambda_scheduler = None
        self.tau_star = None
        self.teacher_marginal = None
        self.class_weight = None
        self.distill_sharpen_T = float(teacher_cfg.distill_sharpen_T)
        self.distill_conf_weight = bool(teacher_cfg.distill_conf_weight)
        self.distill_conf_thresh = float(teacher_cfg.distill_conf_thresh)
        self.distill_use_class_reweight = True
        self.distill_class_reweight_beta = float(teacher_cfg.distill_class_reweight_beta)
        self.distill_class_reweight_cap = (
            None if teacher_cfg.distill_class_reweight_cap is None
            else float(teacher_cfg.distill_class_reweight_cap)
        )

        # KL scheduling
        self.kl_scheduler = kl_scheduler

        # Mode-dependent parameters: two complete sets, applied via set_pretrain_mode()
        self.mode_params = {
            "pretrain": {
                "kmeans_loss" : float(vade_cfg.kmeans_loss_pretrain),
                "repel_weight": float(vade_cfg.repel_weight_pretrain),
                "repel_length_scale": float(vade_cfg.repel_length_scale_pretrain),
                "nonempty_weight": float(vade_cfg.nonempty_weight_pretrain),
                "nonempty_floor": max(1e-4, vade_cfg.nonempty_floor_percent_pretrain / common_cfg.n_components),
                "nonempty_p": int(vade_cfg.nonempty_p_pretrain),
            },
            "main": {
                "kmeans_loss" : float(common_cfg.kmeans_loss),
                "repel_weight": float(vade_cfg.repel_weight),
                "repel_length_scale": float(vade_cfg.repel_length_scale),
                "nonempty_weight": float(vade_cfg.nonempty_weight),
                "nonempty_floor": max(1e-4, vade_cfg.nonempty_floor_percent / common_cfg.n_components),
                "nonempty_p": int(vade_cfg.nonempty_p),
            },
        }

        # Start in pretrain mode
        self.set_mode("pretrain")

    def set_mode(self, mode: str):
        """Copies the parameter set for the given phase into the active attributes."""
        self.pretrain_mode = (mode == "pretrain")
        params = self.mode_params[mode]
        self.kmeans_loss_weight = params["kmeans_loss"]
        self.repel_weight = params["repel_weight"]
        self.repel_length_scale = params["repel_length_scale"]
        self.nonempty_weight = params["nonempty_weight"]
        self.nonempty_floor = params["nonempty_floor"]
        self.nonempty_p = params["nonempty_p"]


    def set_teacher(self, tau_star: torch.Tensor, lambda_distill: float = 1.0, lambda_scheduler: Optional["Dynamic_weight_manager"] = None,):
        """
        Sets teacher assignments and distillation parameters.

        Computes inverse-marginal class reweighting from the teacher distribution
        if distill_use_class_reweight is enabled.

        Args:
            tau_star (torch.Tensor): Teacher soft assignments of shape [N, K].
            lambda_distill (float): Distillation loss weight. Defaults to 1.0.
            lambda_scheduler (Optional[Dynamic_weight_manager]): Optional scheduler for lambda_distill.
        """
        self.tau_star = tau_star
        self.lambda_distill = float(lambda_distill)
        self.lambda_scheduler = lambda_scheduler

        #Compute inverse-marginal class weights from teacher τ*
        self.class_weight = None
        if self.distill_use_class_reweight and (tau_star is not None):
            with torch.no_grad():
                eps = 1e-8
                pi = tau_star.mean(dim=0).clamp_min(eps)             # (C,)
                w = pi.pow(-self.distill_class_reweight_beta)        # inverse-marginal
                w = w / w.mean()                                     # normalize to mean 1
                if self.distill_class_reweight_cap is not None:
                    w = w.clamp_max(self.distill_class_reweight_cap) # cap to avoid extreme weights
            self.class_weight = w  # move to device at use-time

        if tau_star is not None:
            with torch.no_grad():
                eps = 1e-8
                self.teacher_marginal = tau_star.mean(dim=0).clamp_min(eps)  # (C,)

    def set_kl_scheduler(self, kl_scheduler: Optional["Dynamic_weight_manager"] = None):
        """Replaces the KL weight scheduler and resets its iteration counter."""
        self.kl_scheduler = kl_scheduler 
        self.kl_scheduler.current_iteration = 0

    @staticmethod
    def _log_normal_diag(x, mean, log_var):
        """Log-probability under a diagonal Gaussian."""
        LOG_2PI = math.log(2.0 * math.pi)
        return -0.5 * torch.sum(
            LOG_2PI + log_var + (x - mean) ** 2 * torch.exp(-log_var), dim=-1
        )

    def _log_mog(self, z, gmm_means, gmm_log_vars, prior, eps=1e-8):
        """Log-probability under a mixture of diagonal Gaussians."""
        S, B, D = z.shape
        C = gmm_means.shape[0]
        gmm_log_vars = torch.clamp(
            gmm_log_vars,
            min=self.gmm_logvar_clamp[0],
            max=self.gmm_logvar_clamp[1],
        )
        log_prior = torch.log(torch.clamp(prior, min=eps))

        z_exp = z.unsqueeze(2)                          # (S, B, 1, D)
        means = gmm_means.view(1, 1, C, D)
        log_vars = gmm_log_vars.view(1, 1, C, D)

        log_p_z_given_c = self._log_normal_diag(z_exp, means, log_vars)  # (S, B, C)
        log_mix = log_prior.view(1, 1, C)
        return torch.logsumexp(log_mix + log_p_z_given_c, dim=-1)        # (S, B)

    def _monte_carlo_kl(self, z_mean, z_log_var, gmm_means, gmm_log_vars, prior,
                        n_samples: int = 32):
        """
        Monte Carlo estimate of KL(q(z|x) || p(z)) where p(z) is the GMM prior.

        This is the key loss term that pulls latent representations toward cluster centers.
        """
        z_log_var = torch.clamp(z_log_var, min=-4.0, max=4.0)
        B, D = z_mean.shape
        scale_q = torch.exp(0.5 * z_log_var)

        eps = torch.randn(n_samples, B, D, device=z_mean.device, dtype=z_mean.dtype)
        z_samples = z_mean.unsqueeze(0) + eps * scale_q.unsqueeze(0)  # (S, B, D)

        log_q = self._log_normal_diag(
            z_samples, z_mean.unsqueeze(0), z_log_var.unsqueeze(0)
        )
        log_p = self._log_mog(z_samples, gmm_means, gmm_log_vars, prior)

        kl = (log_q - log_p).mean()
        return torch.clamp(kl, min=0.0)
    
    def _log_p_z_given_c(self, z: torch.Tensor, gmm_means: torch.Tensor, gmm_log_vars: torch.Tensor) -> torch.Tensor:
        """
        Computes log p(z|c) for each component c using diagonal Gaussians.

        Args:
            z (torch.Tensor): Latent samples of shape [B, D].
            gmm_means (torch.Tensor): Component means of shape [C, D].
            gmm_log_vars (torch.Tensor): Component log-variances of shape [C, D].

        Returns:
            torch.Tensor: Log-likelihoods of shape [B, C].
        """
        # clamp for stability
        gmm_log_vars = torch.clamp(gmm_log_vars, min=self.gmm_logvar_clamp[0], max=self.gmm_logvar_clamp[1])
        scale = torch.exp(0.5 * gmm_log_vars).clamp(min=1e-3)
        dist = Normal(gmm_means.unsqueeze(0), scale.unsqueeze(0))  # (1,C,D)
        logp = dist.log_prob(z.unsqueeze(1)).sum(dim=-1)           # (B,C)
        return logp
    

    def forward(self, model_outputs, x_original, batch_indices: Optional[torch.Tensor] = None):
        """
        Computes the full VaDE loss from model outputs and original inputs.

        Args:
            model_outputs: Tuple of (recon_dist, latent_z, q, kmeans_loss, z_mean, z_log_var, gmm_params).
            x_original (torch.Tensor): Original input tensor of shape [B, T, N, F].
            batch_indices (Optional[torch.Tensor]): Sample indices into tau_star for distillation.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of all individual loss terms and the total loss.
        """
        (recon_dist, latent_z, q, kmeans_loss, z_mean, z_log_var, gmm_params) = model_outputs
        device = z_mean.device
        B, T, N, F = x_original.shape
        x_flat = x_original.view(B, T, N * F)

        # Reconstruction loss
        with torch.amp.autocast(device_type=x_flat.device.type, enabled=False):
            reconstruction_loss = -(recon_dist.log_prob(x_flat.float())).mean()

        # Ensure q is normalized
        if q is not None:
            eps = 1e-8
            q = q.clamp_min(eps)
            q = q / q.sum(dim=-1, keepdim=True)

        # Activity regularizer: average over batch (matches TF effective scaling)
        activity_l1 = self.l1_activity_weight * torch.sum(torch.abs(z_log_var), dim=-1).mean() 

        # KL weight from scheduler
        klw = 0.0
        if self.kl_scheduler is not None:
            klw = float(self.kl_scheduler.get_weight())

        # KL terms
        z_mean32 = z_mean.float()
        z_log_var32 = z_log_var.float().clamp(min=-4.0, max=2.0)

        if self.pretrain_mode:
            # Pretrain: weak standard-VAE KL against N(0, I).
            # This keeps the latent space compact and well-behaved
            # without imposing mixture structure too early.
            kl_vec = 0.5 * (
                z_mean32.pow(2) + z_log_var32.exp() - 1.0 - z_log_var32
            ).sum(dim=-1) / z_log_var32.shape[-1]
            kl_batch = klw * kl_vec.mean()
        else:
            # Main training: proper MC-KL against the GMM prior.
            with torch.amp.autocast(device_type=device.type, enabled=False):
                kl_vec = self._monte_carlo_kl(
                    z_mean32,
                    z_log_var32,
                    gmm_params["means"].float(),
                    gmm_params["log_vars"].float(),
                    gmm_params["prior"].float(),
                )
                kl_batch = klw * kl_vec

        # Init Losses
        tf_cluster = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
        prior_loss = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
        cat_cluster_loss = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
        scatter_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        repel_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        distill_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        temporal_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)
        nonempty_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)


        # ----------------------------------------------------------------
        # Losses used in all modes

        if not torch.is_tensor(kmeans_loss):
            kmeans_loss = torch.as_tensor(kmeans_loss, device=device, dtype=reconstruction_loss.dtype)
        if q is not None:
            kmeans_loss = (self.kmeans_loss_weight * kmeans_loss).to(reconstruction_loss.dtype)

        repel_weight = self.repel_weight
        repel_length_scale = self.repel_length_scale          
        if repel_weight > 0.0:
            with torch.amp.autocast(device_type=x_flat.device.type, enabled=False):
                # Use data-derived centroids instead of (possibly frozen) GMM means
                if q is not None and latent_z is not None:
                    qf = q.float().detach()  # (B, C)
                    zf = latent_z.float()    # (B, D)
                    pi_b = qf.sum(dim=0).clamp_min(1e-8)  # (C,)
                    means = (qf.t() @ zf) / pi_b.unsqueeze(1)  # (C, D) — soft centroids
                else:
                    means = gmm_params["means"].float()
                
                C = means.size(0)
                diffs = means.unsqueeze(1) - means.unsqueeze(0)
                D2 = (diffs * diffs).sum(dim=-1)
                Kmat = torch.exp(-D2 / max(1e-9, 2.0 * (repel_length_scale ** 2)))
                Kmat = Kmat - torch.diag(torch.diag(Kmat))
                denom = float(max(1, C * C - C))
                repel_loss = (repel_weight * (Kmat.sum() / denom)).to(reconstruction_loss.dtype)


        # Non-empty floor on batch marginal q̄(c)
        nonempty_w = self.nonempty_weight
        if (nonempty_w > 0.0) and (q is not None):
            p = self.nonempty_p
            q_marg = q.mean(dim=0)  # (C,)

            base_floor = float(self.nonempty_floor)
            if self.teacher_marginal is not None:
                pi_t = self.teacher_marginal.to(q_marg.device)  # (C,)
                alpha = 0.9
                floor_c = torch.max(base_floor * torch.ones_like(pi_t),
                                    alpha * pi_t)
            else:
                floor_c = base_floor * torch.ones_like(q_marg)

            underuse = (floor_c - q_marg).clamp_min(0.0)
            pen = underuse.pow(p).sum()
            nonempty_loss = (nonempty_w * pen).to(reconstruction_loss.dtype)

        # ----------------------------------------------------------------
        # Losses only used in main training

        if not self.pretrain_mode and q is not None:
            z_for = latent_z
            logp = self._log_p_z_given_c(
                z_for.float(),
                gmm_params["means"].float(),
                gmm_params["log_vars"].float(),
            )
            post_like = torch.softmax(logp, dim=-1)
            tf_cluster = -(q * post_like).sum(dim=-1).mean() * self.tf_cluster_weight

            # Prior match (uniform)
            C = self.n_components
            log_pi = math.log(1.0 / max(1, C))
            prior_loss = -(q * log_pi).sum(dim=-1).mean()

            # Cat balance (optional)
            if self.reg_cat_clusters_weight > 0 and q is not None:
                cat_cluster_loss = self.reg_cat_clusters_weight * cluster_frequencies_regularizer(q)

            # Temporal cohesion on q(c|z)
            rho = self.temporal_cohesion_weight
            if (rho > 0.0) and (q is not None) and (q.size(0) > 1):
                diffs = (q[1:] - q[:-1]).abs().sum(dim=-1).mean()
                temporal_loss = (rho * diffs).to(reconstruction_loss.dtype)

            eta = self.reg_scatter_weight
            beta = self.reg_scatter_beta
            if eta > 0.0 and (q is not None):
                qf = q.float()                        # (B,C)
                z = z_mean.float()                    # (B,D)
                pi_b = qf.sum(dim=0).clamp_min(1e-8)  # (C,)
                mu = (qf.t() @ z) / pi_b.unsqueeze(1) # (C,D)
                diff = z.unsqueeze(1) - mu.unsqueeze(0)              # (B,C,D)
                scat_c = (qf.unsqueeze(-1) * diff.pow(2)).sum(dim=0) / pi_b.unsqueeze(1)  # (C,D)
                w = ((pi_b / pi_b.mean()).pow(-beta)).unsqueeze(1)   # (C,1)
                scatter_loss = eta * (w * scat_c).mean()
            else:
                scatter_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)

        # ----------------------------------------------------------------
        # Distillation
        if (self.lambda_distill > 0.0) and (self.tau_star is not None) and (batch_indices is not None):
            eps = 1e-8
            tau_batch = self.tau_star[batch_indices]  # (B,C)

            if self.distill_sharpen_T is not None and self.distill_sharpen_T > 0.0:
                logits_t = (tau_batch.clamp_min(eps)).log() / float(self.distill_sharpen_T)
                tau_batch = torch.softmax(logits_t, dim=-1)

            per_sample_ce = -(tau_batch * (q.clamp_min(eps).log())).sum(dim=-1)  # (B,)

            if self.distill_conf_weight:
                conf = tau_batch.max(dim=1).values
                thr = float(self.distill_conf_thresh)
                w_conf = ((conf - thr) / max(1e-6, (1.0 - thr))).clamp(min=0.0, max=1.0).detach()
            else:
                w_conf = None

            w_total = None
            if self.distill_use_class_reweight and (self.class_weight is not None):
                w_class = torch.matmul(tau_batch, self.class_weight.to(tau_batch.device))  # (B,)
                w_class = (w_class / w_class.mean().clamp_min(1e-8)).detach()
                w_total = w_class if w_conf is None else (w_class * w_conf)
            else:
                w_total = w_conf

            if w_total is not None:
                distill_loss = (w_total * per_sample_ce).mean()
            else:
                distill_loss = per_sample_ce.mean()

            distill_loss = (self.lambda_distill * distill_loss).to(reconstruction_loss.dtype)
        else:
            distill_loss = torch.tensor(0.0, device=x_original.device, dtype=reconstruction_loss.dtype)


        # Total loss
        total = (
            reconstruction_loss
            + kl_batch
            + cat_cluster_loss
            + temporal_loss
            + nonempty_loss
            + tf_cluster
            + prior_loss
            + kmeans_loss
            + activity_l1
            + scatter_loss
            + repel_loss
            + distill_loss
        )

        return {
            "total_loss": total,
            "reconstruct_loss": reconstruction_loss,
            "kl_surrogate": kl_batch,                               
            "kl_div": kl_batch,                        
            "kl_weight": torch.tensor(klw, device=device, dtype=reconstruction_loss.dtype),
            "tf_clust_loss": tf_cluster,
            "prior_loss": prior_loss,
            "kmeans_loss": kmeans_loss,
            "activity_l1": activity_l1,
            "cat_clust_loss": cat_cluster_loss,
            "distill_loss": distill_loss,
            "nonempty_loss": nonempty_loss,
            "temporal_loss": temporal_loss,
            "scatter_loss": scatter_loss,
            "repel_loss": repel_loss,
        }
    

##############
# Optimization
##############


def build_optimizer_generic(
    model: nn.Module,
    distill_head: Optional[nn.Module] = None,
    base_lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    params = list(deepof.clustering.model_utils_new.unwrap_dp(model).parameters())
    if distill_head is not None:
        params += list(distill_head.parameters())
    return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)


def build_optimizer_vade(
    model: nn.Module,
    base_lr: float = 3e-4,
    gmm_lr: float = 1e-4,    #3e-4 * 0.33       
) -> torch.optim.Optimizer:
    m = deepof.clustering.model_utils_new.unwrap_dp(model)
    gmm_params = [m.latent_space.gmm_means, m.latent_space.gmm_log_vars]
    if hasattr(m.latent_space, "prior") and isinstance(m.latent_space.prior, torch.nn.Parameter):
        gmm_params.append(m.latent_space.prior)
    gmm_ids = {id(p) for p in gmm_params}
    base_params = [p for p in m.parameters() if id(p) not in gmm_ids]
    return torch.optim.Adam(
        [
            {"params": base_params, "lr": base_lr},
            {"params": gmm_params, "lr": gmm_lr},
        ]
    )

