"""
Turtle teacher model for training support
"""
# @author NoCreativeIdeaForGoodUsername
# encoding: utf-8
# module deepof

from typing import Any, NewType, Tuple, Dict, Optional, List

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


import deepof.clustering.model_utils_new
from deepof.clustering.model_utils_new import CommonFitCfg, TurtleTeacherCfg, VaDECfg, ContrastiveCfg 
from deepof.clustering.dataset import BatchDictDataset
from sklearn.decomposition import IncrementalPCA
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset


# DEFINE CUSTOM ANNOTATED TYPES #
project = NewType("deepof_project", Any)
coordinates = NewType("deepof_coordinates", Any)
table_dict = NewType("deepof_table_dict", Any)


def soft_cross_entropy_logits(logits, soft_targets, eps=1e-8, reduction="mean"):
    log_probs = F.log_softmax(logits, dim=-1)
    soft_targets = torch.clamp(soft_targets, min=eps, max=1.0)
    loss = -(soft_targets * log_probs).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
   

class TurtleHeads(nn.Module):
    """Trains each linear head for each view to resemble the target tau (the lienar fit based on all views)
    i.e. tries to get as close as possible to prediciting the separation within the full dataset (at teh current batch) based on each view of the dataset.
    
    Args:
        feature_dims (List[int]): List containing the feature dimensionality of each active view.
        n_components (int): Number of output clusters.
        inner_lr (float): Learning rate used for the per-head inner optimization. Defaults to 0.1.
        M (int): Number of inner optimization steps performed for the heads at each outer step. Defaults to 100.
        weight_decay (float): Weight decay used for the head optimizers. Defaults to 1e-4.
        normalize_feats (bool): If True, L2-normalizes each input feature vector before fitting or inference. Defaults to True.
        temperature (float): Temperature used to scale the logits produced by each head. Defaults to 0.7.
    """
    def __init__(self, feature_dims: List[int], n_components: int,
                 inner_lr: float = 0.1, M: int = 100,
                 weight_decay: float = 1e-4,
                 normalize_feats: bool = True,
                 temperature: float = 0.7,
                 ):
        super().__init__()
        self.M = M
        self.normalize_feats = normalize_feats
        self.temperature = temperature
        self.heads = nn.ModuleList()
        self._optims = []

        for d in feature_dims:
            head = nn.Linear(d, n_components)
            self.heads.append(head)
            self._optims.append(
                torch.optim.SGD(head.parameters(), lr=inner_lr, weight_decay=weight_decay)
            )

    def _maybe_normalize(self, feats_list):
        if not self.normalize_feats:
            return feats_list
        out = []
        for i, f in enumerate(feats_list):
            out.append(F.normalize(f, dim=-1))
        return out

    def inner_fit(self, feats_list, soft_targets):
        """
        Trains each linear head for each view to resemble the target tau (the lienar fit based on all views)

        Args:
            feats_list (List[torch.Tensor]): List of feature tensors, one per view, each of shape [B, D_i].
            soft_targets (torch.Tensor): Soft target assignments of shape [B, K], typically produced by the task encoder.

        Returns:
            None
        """
        feats_list = self._maybe_normalize([f.detach().float() for f in feats_list])
        soft_targets = soft_targets.detach().float()
        
        for _ in range(self.M):
            for head, opt, feats in zip(self.heads, self._optims, feats_list):
                logits = head(feats) / self.temperature
                loss = soft_cross_entropy_logits(logits, soft_targets)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    @torch.no_grad()
    def logits_list(self, feats_list):
        """Computes detached logits for each view-specific head."""
        feats_list = self._maybe_normalize([f.detach().float() for f in feats_list])
        return [(head(feats) / self.temperature).detach() for head, feats in zip(self.heads, feats_list)]

class TaskEncoder(nn.Module):
    """
    Applies linear projections (fully connected layers) to each individual view and sums up the result
    
    Args:
        feature_dims (List[int]): List containing the feature dimensionality of each active view.
        n_components (int): Number of output clusters.
        temperature (float): Temperature used to scale the per-view logits before averaging and softmax. Defaults to 1.0.
    """
    def __init__(self, feature_dims: List[int], n_components: int, 
                 temperature: float = 1.0,
                 ):
        super().__init__()
        self.temperature = temperature

        self.projs = nn.ModuleList()
        for d in feature_dims:
            proj = nn.Linear(d, n_components)
            self.projs.append(proj)

    def forward(self, feats_list):
        """
        Computes soft cluster assignments from the available views.

        Args:
            feats_list (List[torch.Tensor]): List of feature tensors, one per view, each of shape [B, D_i], D_i may vary based on view type.

        Returns:
            torch.Tensor: Soft assignments of shape [B, n_clusters], obtained by averaging projected logits across views and applying softmax.
        """
        logits = None
        for proj, feats in zip(self.projs, feats_list):
            # Ensure float for stability
            out = proj(feats.float()) / self.temperature
            logits = out if logits is None else (logits + out)
        
        logits = logits / max(len(self.projs), 1)
        return F.softmax(logits, dim=-1)
    

class TurtleTeacher(nn.Module):
    """ 
    Teacher model that learns soft cluster assignments τ which are easy to predict
    from each individual view using linear heads, while regularizers encourage
    confident assignments and balanced cluster usage.

    Based on "Let Go of Your Labels with Unsupervised Transfer" by Artyom Gadetsky et al., see https://arxiv.org/abs/2406.07236  

    Args:
        feature_dims (List[int]): List containing the feature dimensionality of each active view.
        n_components (int): Number of output clusters.
        gamma (float): Strength of the marginal entropy penalty that encourages balanced cluster usage. Defaults to 10.0.
        alpha_sample_entropy (float): Weight of the per-sample entropy term that encourages confident assignments. Defaults to 0.1.
        inner_lr (float): Learning rate for the per-view heads during inner optimization. Defaults to 0.1.
        inner_steps (int): Number of inner optimization steps used to fit the per-view heads at each outer step. Defaults to 100.
        head_wd (float): Weight decay used for the per-view head optimizers. Defaults to 1e-4.
        head_temp (float): Temperature used for the per-view head logits. Defaults to 0.5.
        task_temp (float): Temperature used for the task encoder logits. Defaults to 0.5.
        normalize_feats (bool): If True, L2-normalizes features before passing them to the per-view heads. Defaults to True.
        lr_theta (float): Learning rate for the task encoder optimizer. Defaults to 5e-3.
        delta_death_barrier (float): Strength of the penalty discouraging dead or weakly used clusters. Defaults to 40.0.
        device (str): Device on which the teacher should operate. Defaults to "cpu".
    """
    def __init__(self, feature_dims: List[int], n_components: int,
                 gamma: float = 10.0,
                 alpha_sample_entropy: float = 0.1,
                 inner_lr: float = 0.1,
                 inner_steps: int = 100,
                 head_wd: float = 1e-4,
                 head_temp: float = 0.5,
                 task_temp: float = 0.5,
                 normalize_feats: bool = True,
                 lr_theta: float = 5e-3,
                 delta_death_barrier: float = 40.0,
                 device: str = "cpu",                 
                 ):
        super().__init__()
        self.n_components = n_components
        self.gamma = gamma
        self.delta = delta_death_barrier
        self.alpha = alpha_sample_entropy

        # to predict general labeling using individual views
        self.heads = TurtleHeads(
            feature_dims, n_components,
            inner_lr=inner_lr, M=inner_steps,
            weight_decay=head_wd,
            normalize_feats=normalize_feats,
            temperature=head_temp,
        )

        # To generate general labeling using full dataset
        self.task_encoder = TaskEncoder(
            feature_dims, n_components,
            temperature=task_temp,
        )
        self.opt_theta = torch.optim.Adam(self.task_encoder.parameters(), lr=lr_theta)
        self.device = torch.device(device)

    def to(self, device: torch.device):
        self.device = device
        self.heads.to(device)
        self.task_encoder.to(device)
        return self

    # Helper for full dataset prediction
    @torch.no_grad()
    def predict(self, loader) -> torch.Tensor:
        """
        Runs a sequential pass over the data to compute assignments for the whole dataset.
        Used to generate the final tau_star without loading everything into GPU RAM at once.
        Args:
            loader (DataLoader): DataLoader yielding batches of views.

        Returns:
            torch.Tensor: Tensor of shape [N, K] containing the soft assignments for the full dataset.
        """
        self.task_encoder.eval()
        all_taus = []
        for batch_views in loader:
            # Move batch to GPU
            batch_views = [b.to(self.device) for b in batch_views]
            tau = self.task_encoder(batch_views)
            all_taus.append(tau.cpu())
        self.task_encoder.train()
        return torch.cat(all_taus, dim=0)

    # Fits on batches from a DataLoader ---
    def fit(self, loader, outer_steps: int = 200,
            rho: float = 0.04, verbose: bool = True):
        """
        Fits the teacher by alternating between per-view head updates and task encoder updates on mini-batches.

        At each outer step, the task encoder produces a batch of soft assignments tau.
        The per-view heads are then fitted to match tau, after which the task encoder is updated so that tau becomes easier 
        to predict from each view while remaining confident and well-distributed across clusters.

        Args:
        loader (DataLoader): DataLoader yielding batches of feature tensors, one tensor per active view.
        outer_steps (int): Number of outer optimization steps for the task encoder. Defaults to 200.
        rho (float): Weight of the optional batch-local smoothness regularizer between neighboring rows in a batch. Defaults to 0.04.
        verbose (bool): If True, prints optimization statistics during fitting. Defaults to True.

        Returns:
        None
        """
        
        def _entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
            p = p.clamp_min(eps)
            return -(p * p.log()).sum(dim=-1)
        
        # Create an infinite iterator over the loader
        iterator = iter(loader)
        
        # Outer fitting loop 
        for step in range(outer_steps):
            # Fetch next batch
            try:
                feats_list = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                feats_list = next(iterator)
            
            # Move batch to GPU (non_blocking helps if pin_memory=True)
            feats_list = [f.to(self.device, non_blocking=True) for f in feats_list]

            # Generate labeling based on linear layers (will start out as random gibberish)
            tau = self.task_encoder(feats_list)  # [Batch, K]

            # Inner Loop: Update Heads
            # Train linear heads to reflect current labeling based only on their individual views
            self.heads.inner_fit(feats_list, tau)

            # Compute Loss Components
            logits_k = self.heads.logits_list(feats_list)
            # Sum of cross entropy losses all heads have when trying to predicting the current labeling
            # (Minimizing this pushes τ toward labels that each view can predict linearly.)
            ce_term = 0.0           
            for k, logits in enumerate(logits_k):
                ce_term = ce_term + soft_cross_entropy_logits(logits, tau)
            ce_term = ce_term / max(len(logits_k), 1)

            # sample entropy of labelling accross clusters
            # (Less uniform labelling i.e. in one sample containing values for n clusters, one cluster is more prominent than others, 
            # i.e. lower sample entropy, is better)
            sample_entropy = _entropy(tau).mean()
            
            # Estimate Marginal Entropy (Batch Approximation)
            # (Measure for usage of clusters, even usuage of all clusters i.e. higher H(marg) is better)
            marginal = tau.mean(dim=0)
            H_marginal = _entropy(marginal.unsqueeze(0)).mean()
            logK = float(np.log(self.n_components))
            H_target = torch.tensor(1 * logK, device=H_marginal.device) #option for weighting for not compeltely even distribution, currently not applied
            marg_gap = torch.relu(H_target - H_marginal)          
            gamma_t = float(self.gamma) * (1.0 - float(step) / float(max(1, outer_steps)))

            # Dead penalty
            # (punishes dead i.e. empty or nearly empty clusters)
            dead_floor = max(1e-4, 0.1 / self.n_components)
            tau_clamp = tau.clamp_min(1e-8)
            # Using tau**2 makes diffuse assignments count less than confident ones
            gamma_pow = 2.0
            usage = (tau_clamp ** gamma_pow).mean(dim=0)
            dead_pen = torch.relu(dead_floor - usage).sum() / (dead_floor * self.n_components)
            delta_t = self.delta * max(0.5, 0.6 + 0.4 * (1.0 - step / (float(max(1, outer_steps)))))
            
            # Active count metric (for logging)
            K_dim = int(marginal.numel())
            tau_act = 0.02
            active_soft = torch.sigmoid((marginal - dead_floor) / tau_act)
            active_count = active_soft.sum()
            
            loss = (
                ce_term
                + self.alpha * sample_entropy
                + gamma_t * marg_gap
                + delta_t * dead_pen
            )

            # Optional adjacency penalty between neighboring rows in the current batch.
            # If the loader is shuffled, this is not true temporal smoothing;
            # it acts only as a weak local consistency / stability regularizer.
            if (step % 2) != 0 and rho > 0.0:
                diff = tau[1:] - tau[:-1]
                smooth = (diff.abs().sum(dim=-1)).mean()
                loss = loss + rho * smooth

            # Optimization
            self.opt_theta.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_theta.step()

            if verbose and (step % 20 == 0 or step == outer_steps - 1):
                with torch.no_grad():
                    mean_max_p = tau.max(dim=1).values.mean().item()
                    print(f"[Teacher] step {step:03d} | loss {loss.item():.4f} | CE {ce_term.item():.4f} | "
                          f"E[H(τ)] {sample_entropy.item():.4f} | H(marg) {H_marginal.item():.4f} | "
                          f"mean max_p {mean_max_p:.3f} | dead_pen {dead_pen.item():.3f} | "
                          f"active≈{active_count.item():.2f}/{K_dim}")


@torch.no_grad()
def extract_latents(model: nn.Module, dataset: BatchDictDataset, device: torch.device,
                    batch_size: int = 512, num_workers: int = 0) -> torch.Tensor:
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )
    """
        Extracts latent mean vectors using the model encoder.

        Args:
            model (nn.Module): Model providing an encoder and latent-space encoder head.
            dataset (BatchDictDataset): Dataset from which latent representations should be extracted.
            device (torch.device): Device on which the model inference should be performed.
            batch_size (int): Batch size used for latent extraction. Defaults to 512.
            num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.

        Returns:
            torch.Tensor: Tensor of shape [N_samples, latent_dim] containing the extracted latent mean vectors on CPU.
    """
    base = deepof.clustering.model_utils_new.unwrap_dp(model)                                                      
    base.eval()
    zs = []
    for batch in loader:
        x, a, *rest = deepof.clustering.model_utils_new.move_to(batch, device)
        # Use encoder + posterior head to get z_mean in encoder space (latent_dim) 
        enc = base.encoder(x, a)                                                 
        z_mean, _ = base.latent_space._encode(enc)                             
        zs.append(z_mean.detach().cpu())
    z_all = torch.cat(zs, dim=0)
    return z_all  # CPU


def initialize_gmm_from_teacher(model: nn.Module,
                                z_all: torch.Tensor,        # [N, D] z_mean for train set (CPU or device)
                                tau_star: torch.Tensor,     # [N, C] teacher labels (on same device you ran teacher)
                                min_var: float = 1e-4,
                                min_mass: float = 1e-6) -> None:
    """
    Compute GMM parameters from teacher assignments:
      μ_c = sum_i τ*_ic u_i / sum_i τ*_ic
      σ^2_c = sum_i τ*_ic (u_i - μ_c)^2 / sum_i τ*_ic
      π_c = sum_i τ*_ic / N

    Writes directly into model.latent_space.{gmm_means, gmm_log_vars, prior}.

    Args:
        model (nn.Module): Model whose latent-space GMM parameters should be initialized.
        z_all (torch.Tensor): Latent representations of shape [N_samples, D], typically extracted from the training set.
        tau_star (torch.Tensor): Teacher soft assignments of shape [N_samples, C], where C is the number of mixture components.
        min_var (float): Minimum variance value used to clamp estimated cluster variances for numerical stability. Defaults to 1e-4.
        min_mass (float): Small constant added to cluster masses to avoid division by zero during estimation. Defaults to 1e-6.

    Returns:
        None
    """
    base = deepof.clustering.model_utils_new.unwrap_dp(model)
    base.eval()

    device = next(base.parameters()).device
    z = z_all.to(device=device, dtype=base.latent_space.gmm_means.dtype)
    tau = tau_star.to(device=device, dtype=z.dtype)
        
    N, D_mix = z.shape
    C = tau.shape[1]

    # Cluster masses and priors
    mass = tau.sum(dim=0) + min_mass
    prior = (mass / mass.sum()).clamp(min=1e-8, max=1.0)

    # Means: (C, D_mix)
    means = (tau.T @ z) / mass.unsqueeze(1)

    # Variances: weighted per-cluster diag variance (C, D_mix)
    diffs = z.unsqueeze(1) - means.unsqueeze(0)                       # [N, C, D_mix]
    vars_ = (tau.unsqueeze(-1) * (diffs ** 2)).sum(dim=0) / mass.unsqueeze(-1)  # [C, D_mix]
    vars_ = vars_.clamp(min=min_var)
    log_vars = vars_.log()

    # Edge-case guard
    tiny = (mass <= 1e-4)
    if tiny.any():
        global_mean = z.mean(dim=0)
        global_var = z.var(dim=0, unbiased=False).clamp(min=min_var)
        means[tiny] = global_mean
        log_vars[tiny] = global_var.log()

    # Write into model buffers/params
    base.latent_space.gmm_means.data.copy_(means)
    base.latent_space.gmm_log_vars.data.copy_(log_vars)
    if hasattr(base.latent_space, "prior"):
        if isinstance(base.latent_space.prior, torch.nn.Parameter):
            base.latent_space.prior.data.copy_(prior)
        else:
            base.latent_space.prior.copy_(prior)

    print("Initialized GMM from teacher τ*: "
          f"mean |μ|={means.norm(dim=1).mean():.3f}, "
          f"mean σ²={vars_.mean():.5f}, "
          f"entropy(π)={-(prior*prior.clamp_min(1e-9).log()).sum().item():.3f}")
    

@torch.no_grad()
def fit_nodes_pca(
    dataset: BatchDictDataset,
    n_components_pos: int = 32,
    n_components_spd: int = 32,
    batch_size: int = 4096,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
):
    """
    Fits two IncrementalPCAs:
      - one on positions (x,y) only
      - one on speeds (the 3rd channel per node)

    Assumes dataset returns x with shape [B, T, N, F] where F>=3 and
    channel 0,1 are (x,y), channel 2 is speed.

    Args:
        dataset (BatchDictDataset): Dataset providing node features and adjacency information.
        n_components_pos (int): Number of PCA components to retain for flattened position features. Defaults to 32.
        n_components_spd (int): Number of PCA components to retain for flattened speed features. Defaults to 32.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 4096.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.
        max_samples (Optional[int]): Maximum number of samples to use during PCA fitting. If None, uses all samples. Defaults to None.

    Returns:
        Tuple[IncrementalPCA, torch.Tensor, IncrementalPCA, torch.Tensor]:
        ipca_pos: Fitted IncrementalPCA object for flattened position features.
        feats_pos_all: Tensor of shape [N_samples, n_components_pos] containing PCA-transformed position features.
        ipca_spd: Fitted IncrementalPCA object for flattened speed features.
        feats_spd_all: Tensor of shape [N_samples, n_components_spd] containing PCA-transformed speed features.
    """

    # ---- Pass 1: partial_fit ----
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )

    ipca_pos = IncrementalPCA(n_components=n_components_pos)
    ipca_spd = IncrementalPCA(n_components=n_components_spd)

    seen = 0
    for batch in loader:
        x, a, *rest = batch
        B, T, N, F = x.shape

        if F < 3: # pragma: no cover
            raise ValueError(f"Expected at least 3 channels (x,y,speed); got F={F}")

        pos = x[..., :2]   # [B,T,N,2]
        spd = x[..., 2:3]  # [B,T,N,1]

        X_pos = pos.reshape(B, -1).float().numpy()  # (B, T*N*2)
        X_spd = spd.view(B, -1).float().numpy()  # (B, T*N*1)

        if (max_samples is not None) and (seen >= max_samples):
            break
        if (max_samples is not None) and (seen + B > max_samples):
            cut = max(1, max_samples - seen)
            X_pos = X_pos[:cut]
            X_spd = X_spd[:cut]
            B = cut

        ipca_pos.partial_fit(X_pos)
        ipca_spd.partial_fit(X_spd)
        seen += B

    # ---- Pass 2: transform all ----
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )

    feats_pos, feats_spd = [], []
    for batch in loader:
        x, a, *rest = batch
        B, T, N, F = x.shape

        pos = x[..., :2]   # [B,T,N,2]
        spd = x[..., 2:3]  # [B,T,N,1]

        X_pos = pos.reshape(B, -1).float().numpy()
        X_spd = spd.view(B, -1).float().numpy()

        Z_pos = ipca_pos.transform(X_pos)  # (B, n_components_pos)
        Z_spd = ipca_spd.transform(X_spd)  # (B, n_components_spd)

        feats_pos.append(torch.from_numpy(Z_pos).float())
        feats_spd.append(torch.from_numpy(Z_spd).float())

    feats_pos_all = torch.cat(feats_pos, dim=0)  # [N, n_components_pos]
    feats_spd_all = torch.cat(feats_spd, dim=0)  # [N, n_components_spd]
    return ipca_pos, feats_pos_all, ipca_spd, feats_spd_all


@torch.no_grad()
def fit_angles_pca(
    dataset_with_angles: BatchDictDataset,
    n_components: int = 32,
    batch_size: int = 8192,
    num_workers: int = 0,
) -> Tuple[IncrementalPCA, torch.Tensor]:
    """
    Fits IncrementalPCA on angle tensors and returns both the fitted ipca and features.
    Mirrors fit_nodes_pca but for angle data.

    Args:
        dataset_with_angles (BatchDictDataset): Dataset providing precomputed angle tensors.
        n_components (int): Number of PCA components to retain for the flattened angle features. Defaults to 32.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 8192.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.

    Returns:
        Tuple[IncrementalPCA, torch.Tensor]:
        ipca: Fitted IncrementalPCA object for the angle features.
        feats_all: Tensor of shape [N_samples, n_components] containing PCA-transformed angle features.
    """
    assert getattr(dataset_with_angles, "return_angles", False), \
        "fit_angles_pca expects a dataset created with return_angles=True."

    # Pass 1: partial_fit
    ipca = IncrementalPCA(n_components=n_components)
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=False, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
    )
    for batch in loader:
        # Expected: (x, a, ang, vid) when return_angles=True
        if len(batch) >= 3:
            ang = batch[2]  # angles tensor
        else: # pragma: no cover
            raise RuntimeError(f"Angles loader must yield at least 3 elements, got {len(batch)}")
        # Flatten: (B, T, K) -> (B, T*K)
        X = ang.view(ang.size(0), -1).float().cpu().numpy()
        ipca.partial_fit(X)

    # Pass 2: transform all
    feats_all = []
    loader = dataset_with_angles.make_loader(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, iterable_for_h5=True,
        pin_memory=False, block_shuffle=False, permute_within_block=False,
        prefetch_factor=0 if num_workers == 0 else 2,
        persistent_workers=(num_workers > 0),
    )
    for batch in loader:
        ang = batch[2]
        X = ang.view(ang.size(0), -1).float().cpu().numpy()
        Z = ipca.transform(X)
        feats_all.append(torch.from_numpy(Z).float())

    return ipca, torch.cat(feats_all, dim=0)


@torch.no_grad()
def extract_pca_edges_view(dataset: BatchDictDataset,
                           n_components: int = 16,
                           batch_size: int = 8192,
                           num_workers: int = 0,
                           max_samples: Optional[int] = None) -> torch.Tensor:
    """
    Returns PCA features [N, n_components] for all samples' edge tensor 'a' (T, E, F_edge),
    in order (shuffle=False), using two passes: partial_fit, then transform.
   
    Args:
        dataset (BatchDictDataset): Dataset providing node and edge features.
        n_components (int): Number of PCA components to retain for the flattened edge features. Defaults to 16.
        batch_size (int): Batch size used for the two-pass IncrementalPCA fitting and transformation. Defaults to 8192.
        num_workers (int): Number of worker processes used by the dataset loader. Defaults to 0.
        max_samples (Optional[int]): Maximum number of samples to use during PCA fitting. If None, uses all samples. Defaults to None.

    Returns:
       torch.Tensor: Tensor of shape [N_samples, n_components] containing PCA-transformed edge features.
    """

    # 1) Pass 1: fit IncrementalPCA on flattened edges
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )

    ipca = IncrementalPCA(n_components=n_components)
    seen = 0
    for batch in loader:
        x, a, *rest = batch  # keep on CPU for PCA
        B, T, E, F_edge = a.shape
        A = a.view(B, T * E * F_edge).float().numpy()  # (B, D_edges)
        if (max_samples is not None) and (seen >= max_samples):
            break
        if (max_samples is not None) and (seen + B > max_samples):
            A = A[:max(1, max_samples - seen), :]
            B = A.shape[0]
        ipca.partial_fit(A)
        seen += B

    # 2) Pass 2: transform all edges -> PCA
    loader = dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=(num_workers > 0),
        block_shuffle=False,
        permute_within_block=False,
    )
    feats = []
    for batch in loader:
        x, a, *rest = batch
        B, T, E, F_edge = a.shape
        A = a.view(B, T * E * F_edge).float().numpy()
        Z = ipca.transform(A)  # (B, n_components)
        feats.append(torch.from_numpy(Z).float())
    feats_all = torch.cat(feats, dim=0)  # CPU [N, n_components]
    return feats_all


def run_turtle_teacher_on_views(views_dict: dict,
                                n_components: int,
                                gamma: float = 6.0,
                                alpha_sample_entropy: float = 1.0,
                                outer_steps: int = 200,
                                inner_steps: int = 200,
                                normalize_feats: bool = True,
                                verbose: bool = True,
                                device: Optional[torch.device] = None,
                                head_temp: float = 0.3,
                                task_temp: float = 0.3,
                                batch_size: int = 2048) -> Tuple[Any, torch.Tensor]:
    """
    Fits a TURTLE teacher on a set of precomputed views and returns the final soft assignments.

    The input views are packed into a TensorDataset, trained in shuffled mini-batches, 
    and then evaluated sequentially to produce tau_star.

    Args:
        views_dict (dict): Dictionary mapping view names to tensors of shape [N, D]. Entries with value None are ignored.
        n_components (int): Number of output components or clusters.
        gamma (float): Strength of the marginal entropy penalty encouraging balanced cluster usage. Defaults to 6.0.
        alpha_sample_entropy (float): Weight of the per-sample entropy term encouraging confident assignments. Defaults to 1.0.
        outer_steps (int): Number of outer optimization steps for the task encoder. Defaults to 200.
        inner_steps (int): Number of inner optimization steps for the per-view heads at each outer step. Defaults to 200.
        normalize_feats (bool): If True, L2-normalizes features before passing them to the per-view heads. Defaults to True.
        verbose (bool): If True, prints training progress during fitting. Defaults to True.
        device (Optional[torch.device]): Device on which the teacher should be trained. Defaults to CPU if None.
        head_temp (float): Temperature used for the per-view head logits. Defaults to 0.3.
        task_temp (float): Temperature used for the task encoder logits. Defaults to 0.3.
        batch_size (int): Batch size used for training and prediction. Defaults to 2048.

    Returns:
        Tuple[Any, torch.Tensor]:
        teacher: Fitted TURTLE teacher object.
        tau_star: Tensor of shape [N, K] containing the final soft assignments in dataset order.
    """
    
    device = device or torch.device("cpu")
    
    # Move data to cpu
    keys = []
    tensors = []
    for k, v in views_dict.items():
        if v is not None:
            keys.append(k)
            tensors.append(v.cpu()) # Ensure CPU
            
    assert len(tensors) > 0, "No active views found."
    
    # Create DataLoader (Shuffling is generally good for the teacher)
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True)

    # Initialize Teacher
    feature_dims = [t.shape[1] for t in tensors] 
    teacher = TurtleTeacher(
        feature_dims=feature_dims,
        n_components=n_components,
        gamma=gamma,
        alpha_sample_entropy=alpha_sample_entropy,
        inner_lr=0.1,
        inner_steps=inner_steps,
        head_wd=1e-4,
        head_temp=head_temp,
        task_temp=task_temp,
        normalize_feats=normalize_feats,
        lr_theta=1e-3,
        device=device.type,
    ).to(device)

    # Fit teacher batch wise
    print(f"--- Fitting TurtleTeacher (batch_size={batch_size}) ---")
    teacher.fit(loader, outer_steps=outer_steps, rho=0.04, verbose=verbose)
    
    # Predict tau star ("ideal" labeling) using the trained teacher
    # We use a sequential loader to ensure output matches dataset order
    print("--- Computing final τ* assignments ---")
    seq_loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    tau_star = teacher.predict(seq_loader)
    
    return teacher, tau_star


class DiscriminativeHead(nn.Module):
    """
    Simple linear head on top of z (latent) to predict C clusters (logits).

    Args:
        latent_dim (int): Dimensionality of the latent input vectors.
        n_components (int): Number of output components or clusters.
    """
    def __init__(self, latent_dim: int, n_components: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_components)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)  # logits
    

def maybe_build_turtle_teacher(  
    *,
    teacher_cfg: TurtleTeacherCfg,  
    common_cfg: CommonFitCfg,
    train_dataset: "BatchDictDataset",  
    preprocessed_train: dict,  
    data_path: str, 
    device: torch.device,
    latent_view: Optional[torch.Tensor] = None, #get's calculated externally from the model

) -> Tuple[Optional[Any], Optional[torch.Tensor], Dict[str, Any]]:
    """
    Builds the requested teacher views, fits the TURTLE teacher if enabled, and returns the resulting teacher together with final soft assignments.

    Depending on the configuration, this function can include a latent view, PCA-based node, edge, and angle views, and a supervised label view.
    All constructed views are returned so they can be reused later.

    Args:
        teacher_cfg (TurtleTeacherCfg): Configuration controlling which views are used and how the teacher is trained.
        common_cfg (CommonFitCfg): Common training configuration, including the number of components.
        train_dataset (BatchDictDataset): Training dataset used to extract or derive teacher views.
        preprocessed_train (dict): Preprocessed training data used when rebuilding angle-based datasets.
        data_path (str): Path to the underlying dataset files.
        device (torch.device): Device on which the teacher should be trained.
        latent_view (Optional[torch.Tensor]): Optional latent representation computed externally, for example by another model. Required if include_latent_view is enabled.

    Returns:
        Tuple[Optional[Any], Optional[torch.Tensor], Dict[str, Any]]:
        teacher: Fitted TURTLE teacher object, or None if the teacher is disabled.
        tau_star: Final soft assignments of shape [N, K], or None if the teacher is disabled.
        views: Dictionary containing the constructed views and any fitted PCA objects for reuse.
    """

    # Early return in case of no teacher being required
    if not teacher_cfg.use_turtle_teacher:
        return None, None, {}

    views: Dict[str, Any] = {
        "z": None,
        "pca_pos": None,
        "pca_spd": None,
        "pca_edges": None,
        "pca_angles": None,
    }

    # Latent view (only for VaDE): include only if explicitly requested
    if teacher_cfg.include_latent_view:
        if latent_view is None: # pragma: no cover
            raise ValueError("include_latent_view=True but latent_view=None")
        views["z"]=latent_view.to(device)

    # Nodes view (PCA pos + PCA spd) ---
    if teacher_cfg.include_nodes_view:
        print("\n--- Building PCA views for teacher (nodes) ---")
        _, pca_pos, _, pca_spd = fit_nodes_pca(
            train_dataset,
            n_components_pos=teacher_cfg.pca_nodes_dim,
            n_components_spd=teacher_cfg.pca_nodes_dim,
            batch_size=teacher_cfg.batch_size_nodes,
            num_workers=0,
        )
        views["pca_pos"] = pca_pos
        views["pca_spd"] = pca_spd

    # Edges view (distances between nodes)
    if teacher_cfg.include_edges_view:
        print("\n--- Building PCA views for teacher (edges) ---")
        pca_edges = extract_pca_edges_view(
            train_dataset, n_components=teacher_cfg.pca_edges_dim, batch_size=teacher_cfg.batch_size_edges, num_workers=0
        )
        views["pca_edges"] = pca_edges

    # Angles view (standardized to fit_angles_pca everywhere)
    if teacher_cfg.include_angles_view:
        print("\n--- Building PCA views for teacher (angles) ---")
        angles_train_dataset = BatchDictDataset(
            preprocessed_train, data_path, "train_",
            force_rebuild=False, h5_chunk_len=common_cfg.batch_size, return_angles=True
        )
        _, pca_angles_train = fit_angles_pca(
            angles_train_dataset, n_components=teacher_cfg.pca_angles_dim, batch_size=teacher_cfg.batch_size_angles, num_workers=0 
        )
        views["pca_angles"] = pca_angles_train

    print("\n--- Running TURTLE teacher on views ---")
    teacher, tau_star = run_turtle_teacher_on_views(
        views_dict=views, n_components=common_cfg.n_components, gamma=teacher_cfg.teacher_gamma,
        alpha_sample_entropy=teacher_cfg.teacher_alpha_sample_entropy, outer_steps=teacher_cfg.teacher_outer_steps,
        inner_steps=teacher_cfg.teacher_inner_steps, normalize_feats=teacher_cfg.teacher_normalize_feats,
        verbose=True, device=device, head_temp=teacher_cfg.teacher_head_temp, task_temp=teacher_cfg.teacher_task_temp,
        batch_size = teacher_cfg.teacher_batch_size,
    )

    return teacher, tau_star.detach(), views

