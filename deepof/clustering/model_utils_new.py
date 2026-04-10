# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""Utility functions for both training autoencoder models in deepof.models and tuning hyperparameters with deepof.hypermodels."""

import os
from typing import Any, NewType, Tuple, Dict, Optional, Mapping
import copy
from dataclasses import dataclass, asdict
import tqdm
from contextlib import nullcontext

from IPython.display import clear_output
import numpy as np
import torch
import torch.nn as nn

from deepof.config import PROGRESS_BAR_FIXED_WIDTH
from deepof.data_loading import get_dt
import deepof.utils
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

    learning_rate: float = 1e-3
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

    # Diagnostics
    diag_max_batches: int = 4
    seed: int = None

    # Tuning
    limit_train_batches: Optional[int] = 1000
    limit_val_batches: Optional[int] = 1000


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
    pca_nodes_dim: int = 32
    pca_edges_dim: int = 32
    pca_angles_dim: int = 32
    batch_size_nodes: int = 4096
    batch_size_edges: int = 8192
    batch_size_angles: int = 8192

    # Refresh
    teacher_refresh_every: Optional[int] = None
    teacher_freeze_at: Optional[int] = 10
    reinit_gmm_on_refresh: bool = False
    teacher_batch_size: int = 2048


@dataclass
class VaDECfg:

    learning_rate_pretrain: float = 1e-3
    gmm_learning_rate: float = 1e-3 
    pretrain_epochs: int = 10

    reg_cat_clusters: float = 0.0
    recluster: bool = False
    freeze_gmm_epochs: int = 0.0
    freeze_decoder_epochs: int = 0.0
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

    kl_annealing_mode: str = "tf_sigmoid"
    kl_max_weight: float = 1.0
    kl_warmup: int = 5
    kl_end_weight: float = 0.2
    kl_cooldown: int = 5

    kl_annealing_mode_pretrain: str = "tf_sigmoid"
    kl_max_weight_pretrain: float = 0.2
    kl_warmup_pretrain: int = 15
    kl_end_weight_pretrain: float = 0.2
    kl_cooldown_pretrain: int = 10






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


def unwrap_dp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def move_to(x, device):
    if isinstance(x, (list, tuple)):
        return type(x)(move_to(xx, device) for xx in x)
    if isinstance(x, Mapping):
        return {k: move_to(v, device) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x


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
        lines.append("ckpt_contains: bundle")
        keys = ["state_dict"]
        if rebuild_spec is not None: keys.append("rebuild_spec")
        if log_summary is not None: keys.append("log_summary")
        lines.append("bundle_keys: " + ", ".join(keys))
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
        m = unwrap_dp(model)
        payload = {"state_dict": m.state_dict()}
        if rebuild_spec is not None:
            payload["rebuild_spec"] = rebuild_spec
        if log_summary is not None:
            payload["log_summary"] = log_summary
        torch.save(payload, ckpt_path)


    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def recompute_edges(
    x: torch.Tensor,           # (B, T, N, 3) with [x,y,speed] per node
    edge_index: torch.Tensor,  # indices pairs of nodes to connect
) -> torch.Tensor:
    """
    Recompute edge distances from node coordinates.

    Returns:
        a: (B, T, E, 1) where a[..., e, 0] is the Euclidean distance between the
           two nodes specified by edge_index[e].
    """
    # vvvvv NEW block vvvvv
    if x.ndim != 4 or x.size(-1) < 2: # pragma: no cover
        raise ValueError(f"x must have shape (B,T,N,>=2). Got {tuple(x.shape)}")
    if edge_index.ndim != 2 or edge_index.size(-1) != 2: # pragma: no cover
        raise ValueError(f"edge_index must have shape (E,2). Got {tuple(edge_index.shape)}")

    coords = x[..., 0:2]  # (B,T,N,2)

    # Ensure edge_index on same device
    if edge_index.device != x.device:
        edge_index = edge_index.to(x.device)

    i = edge_index[:, 0].long()  # (E,)
    j = edge_index[:, 1].long()  # (E,)

    pi = coords.index_select(dim=2, index=i)  # (B,T,E,2)
    pj = coords.index_select(dim=2, index=j)  # (B,T,E,2)

    d2 = (pi - pj).pow(2).sum(dim=-1)         # (B,T,E)
    d = torch.sqrt(torch.clamp(d2, min=1e-12))  # (B,T,E)

    return d.unsqueeze(-1)  # (B,T,E,1)


# Unified checkpoint paths per model/run
def ckpt_paths(model_name: str, common_cfg : CommonFitCfg):
    ckpt_dir = os.path.join(common_cfg.output_path, "models", model_name.lower(), f"run_{common_cfg.run}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_path = os.path.join(ckpt_dir, "best_model_val.pth")
    best_score_path = os.path.join(ckpt_dir, "best_model_score.pth")
    teacher_init_path = os.path.join(ckpt_dir, "model_teacher_init.pth")
    return ckpt_dir, best_val_path, best_score_path, teacher_init_path


def check_model_inputs(
    model_name: Optional[str] = None,
    encoder_type: Optional[str] = None,
    kl_annealing_mode: Optional[str] = None,
    contrastive_similarity_function: Optional[str] = None,
    contrastive_loss_function: Optional[str] = None, 
):  # pragma: no cover
    """
    Checks and validates enum-like input parameters for various plot functions.

    This function acts as a centralized guard to ensure that all categorical
    and list-based inputs are valid before being used in downstream logic.

    Args:
        model_name (str): Name of the model
        encoder_type (str): Type of encode-decoder pair being used
        kl_annealing_mode (str): Which function should be used to increase and decrease KL
        contrastive_similarity_function (str): Which function should be used to calculate similarity between sampels for the contrastive model
        contrastive_loss_function (str): Which function should be used to calculate the loss for the contrastive model
    """    

    # =========================================================================
    # 1. GENERATE LISTS OF VALID OPTIONS
    # =========================================================================
    
    # --- Statically defined options ---
    model_opts = ["VaDE", "VQVAE", "Contrastive"]
    encoder_opts = ["recurrent", "TCN", "transformer"]
    kl_annealing_mode_opts = ["linear","sigmoid","tf_sigmoid"]
    contrastive_similarity_function_opts = ["cosine","dot","euclidean","edit"]
    contrastive_loss_function_ops=["nce","fc", "dlc", "hard_dcl"]

    # =========================================================================
    # 3. CONFIGURE AND RUN VALIDATION CHECKS
    # Format: (param_name, param_value, valid_options, is_list, custom_error)
    # =========================================================================
    validation_checks = [
        ("model_name", model_name, model_opts, False, None, True, False),
        ("encoder_type", encoder_type, encoder_opts, False, None, True, False),
        ("kl_annealing_mode", kl_annealing_mode, kl_annealing_mode_opts, False, None, True, False),
        ("contrastive_similarity_function", contrastive_similarity_function, contrastive_similarity_function_opts, False, None, True, False),
        ("contrastive_loss_function", contrastive_loss_function, contrastive_loss_function_ops, False, None, True, False),
    ]

    for name, value, options, is_list, error_msg, only_one_of_many, can_be_dict in validation_checks:
        deepof.utils.validate_parameter(name, value, options, is_list, error_msg, only_one_of_many, can_be_dict)


def embedding_per_video(
    coordinates: coordinates,
    to_preprocess: table_dict,
    model: str,
    meta_info: dict,
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
            else: # pragma: no cover
                raise AssertionError("No animal IDs found in coordinates._animal_ids.")

        if extract_pair == [""]:
            return ""

        if not isinstance(extract_pair, list) or len(extract_pair) != 2: # pragma: no cover
            raise AssertionError(
                "extract_pair must be a list with two animal ids or [\"\"] in case of a single mouse!"
            )

        id1, id2 = extract_pair
        if id1 not in animal_ids or id2 not in animal_ids: # pragma: no cover
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
                dist_standardize=meta_info['dist_standardize'],
                speed_standardize=meta_info['speed_standardize'] ,
                coord_standardize=meta_info['coord_standardize'],
            )    

        else:

            processed_exp, _, _ = dict_to_preprocess.preprocess(
                coordinates=coordinates,
                scale=scale,
                window_size=window_size,
                window_step=1,
                shuffle=False,
                pretrained_scaler=global_scaler,
                dist_standardize=meta_info['dist_standardize'],
                speed_standardize=meta_info['speed_standardize'] ,
                coord_standardize=meta_info['coord_standardize'],
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
                else: # pragma: no cover
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

    elif softcounts_extraction_method is not None: # pragma: no cover
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


def slice_time_per_sample(
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
    if "state_dict" not in ckpt: # pragma: no cover
        raise RuntimeError(f"Checkpoint at {path} is not a bundle (missing 'state_dict').")
    if "rebuild_spec" not in ckpt: # pragma: no cover
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

    else: # pragma: no cover
        raise ValueError(f"Unknown model_name in rebuild_spec: {model_name}")

    model.to(device)
    model.eval()
    _materialize_encoder(model, tuple(spec["x_shape"]), tuple(spec["a_shape"]), device)
    rep = model.load_state_dict(state, strict=strict)
    model.eval()

    load_report = {"missing": rep.missing_keys, "unexpected": rep.unexpected_keys}
    return model, log_summary, spec, load_report


def load_best_checkpoints(
    model: nn.Module,
    best_path_val: str,
    best_path_score: str,
    device: torch.device,
    save_weights: bool,
) -> Tuple[nn.Module, nn.Module]:
    """
    Loads the best-val and best-score checkpoints into two separate model copies.

    Returns the best-val model and best-score model, both unwrapped from DataParallel.
    If a checkpoint does not exist, the corresponding model retains its current weights.

    Args:
        model (nn.Module): Current model (possibly DataParallel-wrapped).
        best_path_val (str): Path to the best-validation checkpoint.
        best_path_score (str): Path to the best-score checkpoint.
        device (torch.device): Device for loading weights.
        save_weights (bool): Whether weight saving was enabled during training.

    Returns:
    Tuple[nn.Module, nn.Module]: (best_val_model, best_score_model), both unwrapped.
    """
    model_score = copy.deepcopy(model)

    if save_weights and os.path.exists(best_path_val):
        ckpt = torch.load(best_path_val, map_location=device, weights_only=False)
        unwrap_dp(model).load_state_dict(ckpt["state_dict"], strict=False)

    if save_weights and os.path.exists(best_path_score):
        ckpt = torch.load(best_path_score, map_location=device, weights_only=False)
        unwrap_dp(model_score).load_state_dict(ckpt["state_dict"], strict=False)

    return unwrap_dp(model), unwrap_dp(model_score)