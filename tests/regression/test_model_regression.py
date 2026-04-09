# @author lucasmiranda42
# encoding: utf-8
# module deepof

"""

Testing module for deepof.models

"""

import os
import random
from shutil import rmtree

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from types import SimpleNamespace

import deepof.clustering
import deepof.clustering.models_new
import deepof.clustering.model_utils_new


###################
# Regression tests
###################


def _summarize_model(
    model,
    model_type,
    preprocessed_object,
    data_path,
    log_summary,
    batch_size=16,
):
    """
    Run the returned contrastive model on the first validation batch in a deterministic way
    and return a compact summary DataFrame suitable for regression testing.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    preprocessed_train, preprocessed_val = preprocessed_object

    val_dataset = deepof.clustering.models_new.BatchDictDataset(
        preprocessed_val,
        data_path,
        "val_",
        force_rebuild=False,
        h5_chunk_len=batch_size,
        supervised_dict=None,
    )

    val_loader = val_dataset.make_loader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        iterable_for_h5=True,
        pin_memory=False,
        prefetch_factor=0,
        persistent_workers=False,
        block_shuffle=False,
        permute_within_block=False,
    )

    batch = next(iter(val_loader))
    x, a = batch[0].to(device), batch[1].to(device)

    if model_type=="contrastive":
        half_len = x.shape[1] // 2
        starts = (torch.ones([x.shape[0]], device=device) * half_len // 2).int()

        x = deepof.clustering.model_utils_new.slice_time_per_sample(x, starts, half_len)
        a = deepof.clustering.model_utils_new.slice_time_per_sample(a, starts, half_len)


    with torch.no_grad():
        z = model.encoder(x, a)
        z = F.normalize(z, dim=1)

        sim = z @ z.t()
        eye = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
        offdiag = sim[~eye]

        param_sq_sum = 0.0
        for p in model.parameters():
            param_sq_sum += float((p.detach().float() ** 2).sum().item())
        param_norm = param_sq_sum ** 0.5

    summary = pd.DataFrame([{
        "embed_sum": float(z.sum().item()),
        "embed_mean": float(z.mean().item()),
        "embed_std": float(z.std().item()),
        "sim_diag_mean": float(torch.diag(sim).mean().item()),
        "sim_offdiag_mean": float(offdiag.mean().item()) if offdiag.numel() > 0 else 0.0,
        "param_norm": float(param_norm),
        "total_loss_train_start": float(log_summary['train']['total_loss'][0]/10000),
        "total_loss_train_end": float(log_summary['train']['total_loss'][-1]/10000),
        "total_loss_val_start": float(log_summary['val']['total_loss'][0]/10000),
        "total_loss_val_end": float(log_summary['val']['total_loss'][-1]/10000),

    }])

    return summary


@settings(deadline=None)
@pytest.mark.slow
@given(
    encoder_type=st.sampled_from(["recurrent", "transformer"]),
)
def test_vade_full_pipeline_regression(encoder_type):
    """
    Full-pipeline regression test for the vade model with all encoder / decoder pairs.

    TCN pipeline is skipped due to currently not avoidable high randomness
    """
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_multi_topview"),
        video_path=os.path.join(".", "tests", "test_examples", "test_multi_topview", "Videos"),
        table_path=os.path.join(".", "tests", "test_examples", "test_multi_topview", "Tables"),
        animal_ids=["B", "W"],
        bodypart_graph="deepof_11",
        arena="circular-autodetect",
        video_scale="380 mm",
        video_format=".mp4",
        table_format=".h5",
        exp_conditions=None,
    ).create(force=True, test=True)

    # Not strictly needed for fitting, but kept consistent with your other project tests
    prun._exp_conditions = {
        "test": pd.DataFrame({"CSDS": ["test_cond1"]}),
        "test2": pd.DataFrame({"CSDS": ["test_cond2"]}),
    }

    preprocessed_object, meta_info, adj_matrix, _, _ = prun.get_graph_dataset(
        center="Center",
        align="Spine_1",
        window_size=24,
        window_step=1,
        test_videos=1,
        preprocess=True,
        scale="standard",
        dist_standardize="groupwise",
        speed_standardize="groupwise",
        coord_standardize="groupwise",
    )


    output_path = os.path.join(".", "tests", "test_examples", "test_data", "vade_regression_run")
    os.makedirs(output_path, exist_ok=True)

    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg(
        model_name="vade",
        encoder_type=encoder_type,
        batch_size=64,
        latent_dim=4,
        epochs= 2,
        n_components=4,
        output_path=output_path,
        data_path=".",
        log_history=False,
        pretrained=None,
        save_weights=True,   # so returned model_val is the best-val checkpoint
        run=0,
        num_workers=0,
        prefetch_factor=0,
        use_amp=False,
        interaction_regularization=0.0,
        kmeans_loss=0.0,
        diag_max_batches=1,
        kl_annealing_mode="linear",
        kl_max_weight=1.0,
        kl_warmup=1,
        kl_end_weight=1.0,
        kl_cooldown=1,
        seed=seed,
    )

    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg(
        use_turtle_teacher=False,
    )

    vade_cfg = deepof.clustering.model_utils_new.VaDECfg(
        reg_cat_clusters=1.0,
        tf_cluster_weight=1.0,
        nonempty_weight=1.0,
        temporal_cohesion_weight=1.0,
        reg_scatter_weight=1.0,
        repel_weight=1.0,
        kmeans_loss_pretrain = 1.0,
        repel_weight_pretrain = 0.5,
        repel_length_scale_pretrain = 0.5,
        nonempty_weight_pretrain = 2e-2,
        nonempty_p_pretrain = 2.0,
        nonempty_floor_percent_pretrain = 0.05,
    )

    contrastive_cfg = deepof.clustering.model_utils_new.ContrastiveCfg(
    )

    model_val, model_score, teacher_init_model, log_summary = deepof.clustering.models_new.embedding_model_fitting(
        preprocessed_object=preprocessed_object,
        adjacency_matrix=adj_matrix,
        meta_info=meta_info,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg,
        vade_cfg=vade_cfg,
        contrastive_cfg=contrastive_cfg,
        h5_dataset_folder=None,
        shuffle=False,
        device="cpu",
    )

    summary = _summarize_model(
        model=model_val,
        model_type="vade",
        preprocessed_object=preprocessed_object,
        data_path=os.path.join(output_path, "Datasets"),
        log_summary=log_summary,
        batch_size=16,
    )


    ref_dir = os.path.join(".", "tests", "test_examples", "test_data", "regression_tests", "vade" + "_" + encoder_type)
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, "vade_summary.csv")

    if not os.path.exists(ref_path):
        print("\033[33mCreating reference for vade pipeline regression!\033[0m")
        summary.to_csv(ref_path, index=False)
    else:
        print("\033[33mFound reference, comparing...\033[0m")

    ref = pd.read_csv(ref_path)

    # Cleanup project artifact created by .create(force=True, test=True)
    rmtree(os.path.join(".", "tests", "test_examples", "test_multi_topview", "deepof_project")) # created helper project
    rmtree(os.path.join(".", "tests", "test_examples", "test_data", "vade_regression_run")) # created datasets and models for model training

    pd.testing.assert_frame_equal(
        summary,
        ref,
        atol=1e-2 if encoder_type != "TCN" else 1e-1,
        rtol=1e-2 if encoder_type != "TCN" else 1e-1,
        check_dtype=False,
        check_like=True,
    )


@settings(deadline=None)
@pytest.mark.slow
@given(
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_vqvae_full_pipeline_regression(encoder_type):
    """
    Full-pipeline regression test for the vqvae model with all encoder / decoder pairs.

    Less strict test for TCN pipeline as it has small remaining randomness
    """
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_multi_topview"),
        video_path=os.path.join(".", "tests", "test_examples", "test_multi_topview", "Videos"),
        table_path=os.path.join(".", "tests", "test_examples", "test_multi_topview", "Tables"),
        animal_ids=["B", "W"],
        bodypart_graph="deepof_11",
        arena="circular-autodetect",
        video_scale="380 mm",
        video_format=".mp4",
        table_format=".h5",
        exp_conditions=None,
    ).create(force=True, test=True)

    # Not strictly needed for fitting, but kept consistent with your other project tests
    prun._exp_conditions = {
        "test": pd.DataFrame({"CSDS": ["test_cond1"]}),
        "test2": pd.DataFrame({"CSDS": ["test_cond2"]}),
    }

    preprocessed_object, meta_info, adj_matrix, _, _ = prun.get_graph_dataset(
        center="Center",
        align="Spine_1",
        window_size=24,
        window_step=1,
        test_videos=1,
        preprocess=True,
        scale="standard",
        dist_standardize="groupwise",
        speed_standardize="groupwise",
        coord_standardize="groupwise",
    )


    output_path = os.path.join(".", "tests", "test_examples", "test_data", "vqvae_regression_run")
    os.makedirs(output_path, exist_ok=True)

    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg(
        model_name="vqvae",
        encoder_type=encoder_type,
        batch_size=64,
        latent_dim=4,
        epochs= 5 if encoder_type != "TCN" else 2,
        n_components=4,
        output_path=output_path,
        data_path=".",
        log_history=False,
        pretrained=None,
        save_weights=True,   # so returned model_val is the best-val checkpoint
        run=0,
        num_workers=0,
        prefetch_factor=0,
        use_amp=False,
        interaction_regularization=0.0,
        kmeans_loss=0.0,
        diag_max_batches=1,
        kl_annealing_mode="linear",
        kl_max_weight=1.0,
        kl_warmup=1,
        kl_end_weight=1.0,
        kl_cooldown=1,
        seed=seed,
    )

    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg(
        use_turtle_teacher=False,
    )

    vade_cfg = deepof.clustering.model_utils_new.VaDECfg()

    contrastive_cfg = deepof.clustering.model_utils_new.ContrastiveCfg()

    model_val, model_score, teacher_init_model, log_summary = deepof.clustering.models_new.embedding_model_fitting(
        preprocessed_object=preprocessed_object,
        adjacency_matrix=adj_matrix,
        meta_info=meta_info,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg,
        vade_cfg=vade_cfg,
        contrastive_cfg=contrastive_cfg,
        h5_dataset_folder=None,
        shuffle=False,
        device="cpu",
    )

    summary = _summarize_model(
        model=model_val,
        model_type="vqvae",
        preprocessed_object=preprocessed_object,
        data_path=os.path.join(output_path, "Datasets"),
        log_summary=log_summary,
        batch_size=16,
    )

    ref_dir = os.path.join(".", "tests", "test_examples", "test_data", "regression_tests", "vqvae" + "_" + encoder_type)
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, "vqvae_summary.csv")

    if not os.path.exists(ref_path):
        print("\033[33mCreating reference for vqvae pipeline regression!\033[0m")
        summary.to_csv(ref_path, index=False)
    else:
        print("\033[33mFound reference, comparing...\033[0m")

    ref = pd.read_csv(ref_path)

    # Cleanup project artifact created by .create(force=True, test=True)
    rmtree(os.path.join(".", "tests", "test_examples", "test_multi_topview", "deepof_project")) # created helper project
    rmtree(os.path.join(".", "tests", "test_examples", "test_data", "vqvae_regression_run")) # created datasets and models for model training

    pd.testing.assert_frame_equal(
        summary,
        ref,
        atol=1e-5 if encoder_type != "TCN" else 1e-1,
        rtol=1e-5 if encoder_type != "TCN" else 1e-1,
        check_dtype=False,
        check_like=True,
    )


@settings(deadline=None)
@pytest.mark.slow
@given(
    encoder_type=st.sampled_from(["recurrent", "TCN", "transformer"]),
)
def test_contrastive_full_pipeline_regression(encoder_type):
    """
    Full-pipeline regression test for the contrastive model with all encoder / decoder pairs.

    Less strict test for TCN pipeline as it has small remaining randomness
    """
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


    prun = deepof.data.Project(
        project_path=os.path.join(".", "tests", "test_examples", "test_multi_topview"),
        video_path=os.path.join(".", "tests", "test_examples", "test_multi_topview", "Videos"),
        table_path=os.path.join(".", "tests", "test_examples", "test_multi_topview", "Tables"),
        animal_ids=["B", "W"],
        bodypart_graph="deepof_11",
        arena="circular-autodetect",
        video_scale="380 mm",
        video_format=".mp4",
        table_format=".h5",
        exp_conditions=None,
    ).create(force=True, test=True)

    # Not strictly needed for fitting, but kept consistent with your other project tests
    prun._exp_conditions = {
        "test": pd.DataFrame({"CSDS": ["test_cond1"]}),
        "test2": pd.DataFrame({"CSDS": ["test_cond2"]}),
    }

    preprocessed_object, meta_info, adj_matrix, _, _ = prun.get_graph_dataset(
        center="Center",
        align="Spine_1",
        window_size=24,
        window_step=1,
        test_videos=1,
        preprocess=True,
        scale="standard",
        dist_standardize="groupwise",
        speed_standardize="groupwise",
        coord_standardize="groupwise",
    )


    output_path = os.path.join(".", "tests", "test_examples", "test_data", "contrastive_regression_run")
    os.makedirs(output_path, exist_ok=True)

    common_cfg = deepof.clustering.model_utils_new.CommonFitCfg(
        model_name="contrastive",
        encoder_type=encoder_type,
        batch_size=64,
        latent_dim=4,
        epochs= 5 if encoder_type != "TCN" else 2,
        n_components=4,
        output_path=output_path,
        data_path=".",
        log_history=False,
        pretrained=None,
        save_weights=True,   # so returned model_val is the best-val checkpoint
        run=0,
        num_workers=0,
        prefetch_factor=0,
        use_amp=False,
        interaction_regularization=0.0,
        kmeans_loss=0.0,
        diag_max_batches=1,
        kl_annealing_mode="linear",
        kl_max_weight=1.0,
        kl_warmup=1,
        kl_end_weight=1.0,
        kl_cooldown=1,
        seed=seed,
    )

    teacher_cfg = deepof.clustering.model_utils_new.TurtleTeacherCfg(
        use_turtle_teacher=False,
    )

    vade_cfg = deepof.clustering.model_utils_new.VaDECfg()

    contrastive_cfg = deepof.clustering.model_utils_new.ContrastiveCfg(
        temperature=0.1,
        contrastive_similarity_function="cosine",
        contrastive_loss_function="nce",
        beta=0.1,
        tau=0.1,
        aug_min_shift=1,
        aug_max_shift=3,
        aug_p_shift=0.5,
        aug_noise_sigma=0.03,
        aug_p_noise=0.5,
        aug_min_interp=3,
        aug_max_interp=8,
        aug_p_interp=0.5,
        aug_max_rot=30,
        aug_n_rot=3,
        aug_p_rot=0.5,
    )

    model_val, model_score, teacher_init_model, log_summary = deepof.clustering.models_new.embedding_model_fitting(
        preprocessed_object=preprocessed_object,
        adjacency_matrix=adj_matrix,
        meta_info=meta_info,
        common_cfg=common_cfg,
        teacher_cfg=teacher_cfg,
        vade_cfg=vade_cfg,
        contrastive_cfg=contrastive_cfg,
        h5_dataset_folder=None,
        shuffle=False,
        device="cpu",
    )

    summary = _summarize_model(
        model=model_val,
        model_type="contrastive",
        preprocessed_object=preprocessed_object,
        data_path=os.path.join(output_path, "Datasets"),
        log_summary=log_summary,
        batch_size=16,
    )

    ref_dir = os.path.join(".", "tests", "test_examples", "test_data", "regression_tests", "contrastive" + "_" + encoder_type)
    os.makedirs(ref_dir, exist_ok=True)
    ref_path = os.path.join(ref_dir, "contrastive_summary.csv")

    if not os.path.exists(ref_path):
        print("\033[33mCreating reference for contrastive pipeline regression!\033[0m")
        summary.to_csv(ref_path, index=False)
    else:
        print("\033[33mFound reference, comparing...\033[0m")

    ref = pd.read_csv(ref_path)

    # Cleanup project artifact created by .create(force=True, test=True)
    rmtree(os.path.join(".", "tests", "test_examples", "test_multi_topview", "deepof_project")) # created helper project
    rmtree(os.path.join(".", "tests", "test_examples", "test_data", "contrastive_regression_run")) # created datasets and models for model training

    pd.testing.assert_frame_equal(
        summary,
        ref,
        atol=1e-5 if encoder_type != "TCN" else 1e-1,
        rtol=1e-5 if encoder_type != "TCN" else 1e-1,
        check_dtype=False,
        check_like=True,
    )

